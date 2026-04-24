import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# =========================
# 网络配置
# =========================
LOCAL_BIND_IP = "0.0.0.0"
LOCAL_BIND_PORT = 8888
SERVER_IP = "127.0.0.1"
SERVER_PORT = 9000
CONTROL_PORT = 9001  # UI -> server target input side-channel

# =========================
# 手动控制配置
# =========================
MANUAL_MODE_ID = 99
MANUAL_NONE = 0
MANUAL_DRIVE = 1
MANUAL_STOP_HOLD = 2
MANUAL_LINEAR = 0.30
MANUAL_ANGULAR = 0.80


def recv_all(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def safe_depth_to_color(depth_mm: np.ndarray) -> np.ndarray:
    vis_depth = depth_mm.copy()
    vis_depth[vis_depth > 5000] = 5000
    vis_depth = (vis_depth.astype(np.float32) / 5000.0 * 255).astype(np.uint8)
    return cv2.applyColorMap(vis_depth, cv2.COLORMAP_JET)


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def preferred_screen_geometry():
    screens = QApplication.screens()
    if len(screens) >= 2:
        return screens[1].availableGeometry(), 1
    return screens[0].availableGeometry(), 0


def send_target_command(target_name: str) -> str:
    target_name = (target_name or "").strip()
    if not target_name:
        raise ValueError("empty target")
    sock = socket.socket()
    sock.settimeout(2.0)
    try:
        sock.connect((SERVER_IP, CONTROL_PORT))
        sock.sendall(target_name.encode("utf-8"))
        reply = sock.recv(4096)
        return reply.decode("utf-8", errors="ignore").strip() if reply else "NO_REPLY"
    finally:
        sock.close()


def crop_slam_effective_area(
    slam_img: np.ndarray,
    unknown_gray: int = 160,
    tol: int = 10,
    pad: int = 32,
    zoom: float = 1.18,
):
    """
    更稳的雷达图裁剪：
    1. 只保留最大的主体连通区域
    2. 再做适度居中放大
    3. 返回裁剪图和左上角偏移量，供后续叠加目标点使用
    """
    if slam_img is None:
        return None, (0, 0)

    gray = cv2.cvtColor(slam_img, cv2.COLOR_BGR2GRAY)

    # 非未知区域
    valid = (np.abs(gray.astype(np.int16) - unknown_gray) > tol).astype(np.uint8)

    # 去噪和补洞
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    if num_labels <= 1:
        return slam_img, (0, 0)

    # 只取最大的主体区域
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    H, W = slam_img.shape[:2]

    # 初始 bbox + padding
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, W)
    y2 = min(y + h + pad, H)

    # 居中轻微放大
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = (x2 - x1) / zoom
    crop_h = (y2 - y1) / zoom

    crop_w = max(crop_w, 320)
    crop_h = max(crop_h, 320)

    nx1 = max(int(cx - crop_w / 2), 0)
    ny1 = max(int(cy - crop_h / 2), 0)
    nx2 = min(int(cx + crop_w / 2), W)
    ny2 = min(int(cy + crop_h / 2), H)

    if nx2 - nx1 < 60 or ny2 - ny1 < 60:
        return slam_img, (0, 0)

    return slam_img[ny1:ny2, nx1:nx2], (nx1, ny1)


def draw_target_on_slam(
    slam_img: np.ndarray,
    slam_meta: dict,
    target_name: str,
    detected: bool,
    object_xy,
    goal_xy,
    score: float,
    details: dict,
    crop_offset=(0, 0),
    show_points: bool = True,
):
    """
    在已经裁剪后的 slam 图上画目标点、导航点和信息框。
    crop_offset 用来把世界坐标映射到裁剪后的局部图。
    show_points=False 时，只显示右上角信息框，不画点。
    """
    if slam_img is None:
        return None

    out = slam_img.copy()
    if not slam_meta:
        return out

    resolution = slam_meta.get("resolution")
    origin_x = slam_meta.get("origin_x")
    origin_y = slam_meta.get("origin_y")
    width = slam_meta.get("width")
    height = slam_meta.get("height")
    if None in (resolution, origin_x, origin_y, width, height) or resolution <= 1e-9:
        return out

    off_x, off_y = crop_offset

    def world_to_pixel(x, y):
        px = int((x - origin_x) / resolution)
        py = int((y - origin_y) / resolution)
        py = int(height - 1 - py)

        # 映射到裁剪后的局部图
        px -= off_x
        py -= off_y
        return px, py

    # 只有进入 navigation 阶段才画目标物体位置
    if show_points and detected and object_xy is not None:
        ox, oy = object_xy
        px, py = world_to_pixel(ox, oy)
        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.circle(out, (px, py), 7, (0, 255, 255), -1)
            cv2.circle(out, (px, py), 12, (0, 200, 255), 2)
            label = f"{target_name}  {score:.2f}"
            cv2.putText(
                out,
                label,
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 只有进入 navigation 阶段才画导航点
    if show_points and goal_xy is not None:
        gx, gy = goal_xy
        px, py = world_to_pixel(gx, gy)
        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.drawMarker(
                out,
                (px, py),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )

    # 右上角信息框：一直显示
    lines = [
        f"Target: {target_name if target_name else 'N/A'}",
        f"Detected: {'Yes' if detected else 'No'}",
    ]
    if object_xy is not None:
        lines.append(f"Obj XY: ({object_xy[0]:.2f}, {object_xy[1]:.2f})")
    if goal_xy is not None:
        lines.append(f"Goal XY: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
    if detected:
        lines.append(f"Score: {score:.2f}")
    if details:
        for k, v in details.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.3f}")
            else:
                lines.append(f"{k}: {v}")

    font_scale = 0.25
    line_h = 10
    box_w = 90
    box_h = line_h * len(lines) + 16
    x0 = max(out.shape[1] - box_w - 10, 8)
    y0 = 10
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (22, 22, 22), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 255), 1)

    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (x0 + 5, y0 + 10 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return out


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    rgb: np.ndarray = None
    depth: np.ndarray = None
    obs: np.ndarray = None
    val: np.ndarray = None
    slam: np.ndarray = None

    linear_x: float = 0.0
    angular_z: float = 0.0
    interrupted: bool = False
    keys: set = field(default_factory=set)
    connected_robot: bool = False
    connected_server: bool = False
    last_error: str = ""

    target_name: str = ""
    target_detected: bool = False
    target_goal_xy: tuple = None
    target_object_xy: tuple = None
    target_score: float = 0.0
    target_details: dict = field(default_factory=dict)
    slam_meta: dict = field(default_factory=dict)

    # 新增：只有进入 navigation 阶段才画点
    nav_active: bool = False
    target_input_text: str = ""


class ImagePanel(QWidget):
    def __init__(self, title: str, expand: bool = False):
        super().__init__()
        self.title = title
        self.expand = expand
        self._last_img = None

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFixedHeight(36)
        self.title_label.setStyleSheet(
            """
            QLabel {
                background: #1a1a1a;
                color: #f2f2f2;
                font-size: 16px;
                font-weight: 500;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 4px 8px;
            }
            """
        )

        self.image_label = QLabel("等待数据")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 220)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            """
            QLabel {
                background: #0d0d0d;
                color: #bfbfbf;
                border: 1px solid #2e2e2e;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
                font-size: 15px;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def update_image(self, img: np.ndarray):
        self._last_img = img
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        if self._last_img is None:
            self.image_label.setText("等待数据")
            self.image_label.setPixmap(QPixmap())
            return

        pix = bgr_to_pixmap(self._last_img)
        if pix.isNull():
            self.image_label.setText("图像无效")
            return

        self.image_label.setText("")
        mode = Qt.KeepAspectRatioByExpanding if self.expand else Qt.KeepAspectRatio
        scaled = pix.scaled(self.image_label.size(), mode, Qt.SmoothTransformation)

        if self.expand:
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            x = max((scaled.width() - label_w) // 2, 0)
            y = max((scaled.height() - label_h) // 2, 0)
            scaled = scaled.copy(x, y, min(label_w, scaled.width()), min(label_h, scaled.height()))

        self.image_label.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self, shared_state: SharedState):
        super().__init__()
        self.state = shared_state
        self._placed_once = False

        self.setWindowTitle("VLFM Auto Nav UI")
        self.resize(1800, 1000)
        self.setFocusPolicy(Qt.StrongFocus)

        self.rgb_panel = ImagePanel("RGB")
        self.depth_panel = ImagePanel("Depth")
        self.obs_panel = ImagePanel("Obstacle Map")
        self.val_panel = ImagePanel("Value Map")
        self.slam_panel = ImagePanel("Lidar SLAM Map", expand=False)
        self.slam_panel.setMinimumWidth(680)

        self.btn_interrupt = QPushButton("停止")
        self.btn_back = QPushButton("恢复")
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("输入目标物体，例如: couch / potted plant")
        self.target_input.returnPressed.connect(self.on_send_target)
        self.btn_send_target = QPushButton("设置目标")
        self.btn_send_target.clicked.connect(self.on_send_target)
        self.btn_interrupt.clicked.connect(self.on_interrupt)
        self.btn_back.clicked.connect(self.on_back)

        self._btn_interrupt_normal = """
            QPushButton {
                background: #c73b3b;
                color: white;
                border-radius: 10px;
                font-size: 20px;
                font-weight: 700;
                padding: 12px 20px;
                min-height: 56px;
            }
            QPushButton:hover {
                background: #dd4d4d;
            }
        """
        self._btn_interrupt_active = """
            QPushButton {
                background: #8d1f1f;
                color: white;
                border-radius: 10px;
                font-size: 20px;
                font-weight: 700;
                padding: 12px 20px;
                min-height: 56px;
            }
        """
        self._btn_back_style = """
            QPushButton {
                background: #2d6cdf;
                color: white;
                border-radius: 10px;
                font-size: 20px;
                font-weight: 700;
                padding: 12px 20px;
                min-height: 56px;
            }
            QPushButton:hover {
                background: #3f7df0;
            }
        """
        self.btn_interrupt.setStyleSheet(self._btn_interrupt_normal)
        self.btn_back.setStyleSheet(self._btn_back_style)
        self.target_input.setStyleSheet(
            """
            QLineEdit {
                background: #101010;
                color: #f2f2f2;
                border: 1px solid #3a3a3a;
                border-radius: 10px;
                font-size: 18px;
                padding: 12px 14px;
                min-height: 56px;
            }
            """
        )
        self.btn_send_target.setStyleSheet(self._btn_back_style)

        left_grid = QGridLayout()
        left_grid.setSpacing(12)
        left_grid.setContentsMargins(0, 0, 0, 0)
        left_grid.addWidget(self.rgb_panel, 0, 0)
        left_grid.addWidget(self.depth_panel, 0, 1)
        left_grid.addWidget(self.obs_panel, 1, 0)
        left_grid.addWidget(self.val_panel, 1, 1)
        left_widget = QWidget()
        left_widget.setLayout(left_grid)

        input_row = QHBoxLayout()
        input_row.setSpacing(12)
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.addWidget(self.target_input, 1)
        input_row.addWidget(self.btn_send_target, 0)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self.btn_interrupt)
        btn_row.addWidget(self.btn_back)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.slam_panel, 1)
        right_layout.addLayout(input_row, 0)
        right_layout.addLayout(btn_row, 0)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        root = QHBoxLayout()
        root.setSpacing(14)
        root.setContentsMargins(14, 14, 14, 14)
        root.addWidget(left_widget, 1)
        root.addWidget(right_widget, 1)
        central = QWidget()
        central.setLayout(root)
        central.setStyleSheet("background: #151515;")
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(50)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._placed_once:
            self._placed_once = True
            geo, idx = preferred_screen_geometry()
            handle = self.windowHandle()
            screens = QApplication.screens()
            if handle is not None and idx < len(screens):
                handle.setScreen(screens[idx])
            self.setGeometry(geo.adjusted(8, 8, -8, -8))
        self.activateWindow()
        self.raise_()
        self.setFocus()

    def on_send_target(self):
        text = self.target_input.text().strip()
        if not text:
            return
        try:
            reply = send_target_command(text)
            with self.state.lock:
                self.state.last_error = f"Target set: {text}" if reply.startswith("OK") else reply
                self.state.target_input_text = text
            self.target_input.clear()
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f"Set target failed: {e}"

    def on_interrupt(self):
        with self.state.lock:
            self.state.interrupted = True
            self.state.linear_x = 0.0
            self.state.angular_z = 0.0
            self.state.keys.clear()

    def on_back(self):
        with self.state.lock:
            self.state.interrupted = False

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            event.accept()
            return
        with self.state.lock:
            self.state.keys.add(event.key())
        self.update_cmd_from_keys()
        event.accept()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            event.accept()
            return
        with self.state.lock:
            self.state.keys.discard(event.key())
        self.update_cmd_from_keys()
        event.accept()

    def update_cmd_from_keys(self):
        with self.state.lock:
            if self.state.interrupted:
                self.state.linear_x = 0.0
                self.state.angular_z = 0.0
                return

            vx = 0.0
            wz = 0.0
            if Qt.Key_W in self.state.keys:
                vx += MANUAL_LINEAR
            if Qt.Key_S in self.state.keys:
                vx -= MANUAL_LINEAR
            if Qt.Key_A in self.state.keys:
                wz += MANUAL_ANGULAR
            if Qt.Key_D in self.state.keys:
                wz -= MANUAL_ANGULAR
            if Qt.Key_Space in self.state.keys:
                vx = 0.0
                wz = 0.0
            self.state.linear_x = vx
            self.state.angular_z = wz

    def refresh_ui(self):
        with self.state.lock:
            rgb = self.state.rgb
            depth = self.state.depth
            obs = self.state.obs
            val = self.state.val
            slam = self.state.slam
            slam_meta = self.state.slam_meta
            target_name = self.state.target_name
            target_detected = self.state.target_detected
            target_goal_xy = self.state.target_goal_xy
            target_object_xy = self.state.target_object_xy
            target_score = self.state.target_score
            target_details = self.state.target_details
            nav_active = self.state.nav_active
            vx = self.state.linear_x
            wz = self.state.angular_z
            interrupted = self.state.interrupted
            robot_ok = self.state.connected_robot
            server_ok = self.state.connected_server
            last_error = self.state.last_error

        # 先裁原始 slam 图，再叠加目标信息
        slam_crop, crop_offset = crop_slam_effective_area(
            slam,
            unknown_gray=160,
            tol=10,
            pad=32,
            zoom=1.18,
        )

        slam_vis = draw_target_on_slam(
            slam_crop,
            slam_meta,
            target_name,
            target_detected,
            target_object_xy,
            target_goal_xy,
            target_score,
            target_details,
            crop_offset=crop_offset,
            show_points=nav_active,
        )

        self.rgb_panel.update_image(rgb)
        self.depth_panel.update_image(depth)
        self.obs_panel.update_image(obs)
        self.val_panel.update_image(val)
        self.slam_panel.update_image(slam_vis)

        title = (
            f"VLFM Manual Debug UI | "
            f"Robot: {'已连接' if robot_ok else '未连接'} | "
            f"Server: {'已连接' if server_ok else '未连接'} | "
            f"vx={vx:.2f} m/s | wz={wz:.2f} rad/s | "
            f"{'已停止' if interrupted else '手动接管'}"
        )
        if target_name:
            title += f" | Target={target_name}"
        if last_error:
            title += f" | {last_error}"
        self.setWindowTitle(title)
        self.btn_interrupt.setStyleSheet(
            self._btn_interrupt_active if interrupted else self._btn_interrupt_normal
        )


def bridge_loop(shared_state: SharedState):
    while True:
        server_conn = None
        bridge_server = None
        robot_conn = None
        try:
            # 1) 连接推理服务器
            while True:
                try:
                    server_conn = socket.socket()
                    server_conn.connect((SERVER_IP, SERVER_PORT))
                    with shared_state.lock:
                        shared_state.connected_server = True
                        shared_state.last_error = ""
                    print(f"Connected to Inference Server {SERVER_IP}:{SERVER_PORT}")
                    break
                except Exception as e:
                    with shared_state.lock:
                        shared_state.connected_server = False
                        shared_state.last_error = f"Server connect failed: {e}"
                    print(f"Server connect failed: {e}. Retrying in 2s...")
                    time.sleep(2)

            # 2) 等待机器人连接
            bridge_server = socket.socket()
            bridge_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            bridge_server.bind((LOCAL_BIND_IP, LOCAL_BIND_PORT))
            bridge_server.listen(1)
            print(f"Waiting for Robot on {LOCAL_BIND_IP}:{LOCAL_BIND_PORT} ...")
            robot_conn, addr = bridge_server.accept()
            print(f"Robot connected from: {addr}")
            with shared_state.lock:
                shared_state.connected_robot = True
                shared_state.last_error = ""

            while True:
                # A. 从机器人接收 RGB + Depth + Pose + SLAM Map + SLAM Meta
                rgb_header = recv_all(robot_conn, 4)
                if not rgb_header:
                    raise BrokenPipeError("Robot disconnected (rgb header)")
                rgb_size = struct.unpack("I", rgb_header)[0]
                rgb_bytes = recv_all(robot_conn, rgb_size)
                if not rgb_bytes:
                    raise BrokenPipeError("Robot disconnected (rgb data)")
                rgb_np = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

                depth_header = recv_all(robot_conn, 4)
                if not depth_header:
                    raise BrokenPipeError("Robot disconnected (depth header)")
                depth_size = struct.unpack("I", depth_header)[0]
                depth_bytes = recv_all(robot_conn, depth_size)
                if not depth_bytes:
                    raise BrokenPipeError("Robot disconnected (depth data)")
                depth_mm = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                depth_vis_color = safe_depth_to_color(depth_mm) if depth_mm is not None else None

                state_bytes = recv_all(robot_conn, 12)
                if not state_bytes:
                    raise BrokenPipeError("Robot disconnected (state)")

                slam_head = recv_all(robot_conn, 4)
                if not slam_head:
                    raise BrokenPipeError("Robot disconnected (slam header)")
                slam_size = struct.unpack("I", slam_head)[0]
                slam_bytes = recv_all(robot_conn, slam_size) if slam_size > 0 else b""
                slam_img = cv2.imdecode(np.frombuffer(slam_bytes, np.uint8), cv2.IMREAD_COLOR) if slam_bytes else None

                slam_meta_head = recv_all(robot_conn, 4)
                if not slam_meta_head:
                    raise BrokenPipeError("Robot disconnected (slam meta header)")
                slam_meta_size = struct.unpack("I", slam_meta_head)[0]
                slam_meta_bytes = recv_all(robot_conn, slam_meta_size) if slam_meta_size > 0 else b"{}"
                slam_meta = json.loads(slam_meta_bytes.decode("utf-8")) if slam_meta_bytes else {}

                # B. 转发给推理服务器（不包含 SLAM 图）
                payload = (
                    struct.pack("I", rgb_size)
                    + rgb_bytes
                    + struct.pack("I", depth_size)
                    + depth_bytes
                    + state_bytes
                )
                server_conn.sendall(payload)

                # C. 接收推理结果：goal + obs map + val map + obj meta
                goal_data = recv_all(server_conn, 10)
                if not goal_data:
                    raise BrokenPipeError("Server disconnected (goal)")

                obs_head = recv_all(server_conn, 4)
                if not obs_head:
                    raise BrokenPipeError("Server disconnected (obs header)")
                obs_size = struct.unpack("I", obs_head)[0]
                obs_bytes = recv_all(server_conn, obs_size) if obs_size > 0 else b""
                obs_img = cv2.imdecode(np.frombuffer(obs_bytes, np.uint8), cv2.IMREAD_COLOR) if obs_bytes else None

                val_head = recv_all(server_conn, 4)
                if not val_head:
                    raise BrokenPipeError("Server disconnected (val header)")
                val_size = struct.unpack("I", val_head)[0]
                val_bytes = recv_all(server_conn, val_size) if val_size > 0 else b""
                val_img = cv2.imdecode(np.frombuffer(val_bytes, np.uint8), cv2.IMREAD_COLOR) if val_bytes else None

                obj_meta_head = recv_all(server_conn, 4)
                if not obj_meta_head:
                    raise BrokenPipeError("Server disconnected (obj meta header)")
                obj_meta_size = struct.unpack("I", obj_meta_head)[0]
                obj_meta_bytes = recv_all(server_conn, obj_meta_size) if obj_meta_size > 0 else b"{}"
                obj_meta = json.loads(obj_meta_bytes.decode("utf-8")) if obj_meta_bytes else {}

                with shared_state.lock:
                    shared_state.rgb = rgb_np
                    shared_state.depth = depth_vis_color
                    shared_state.obs = obs_img
                    shared_state.val = val_img
                    shared_state.slam = slam_img
                    shared_state.slam_meta = slam_meta
                    shared_state.target_name = obj_meta.get("target_name", "")
                    shared_state.target_detected = obj_meta.get("detected", False)
                    shared_state.target_goal_xy = tuple(obj_meta["goal_xy"]) if obj_meta.get("goal_xy") else None
                    shared_state.target_object_xy = tuple(obj_meta["object_xy"]) if obj_meta.get("object_xy") else None
                    shared_state.target_score = float(obj_meta.get("score", 0.0))
                    # shared_state.target_details = obj_meta.get("details", {})

                    # 只有进入 navigation 阶段才在地图上画点
                    shared_state.nav_active = bool(
                        obj_meta.get("nav_active", shared_state.target_goal_xy is not None)
                    )

                    shared_state.connected_robot = True
                    shared_state.connected_server = True
                    shared_state.last_error = ""
                    vx = shared_state.linear_x
                    wz = shared_state.angular_z
                    interrupted = shared_state.interrupted

                if interrupted:
                    manual_state = MANUAL_STOP_HOLD
                    vx = 0.0
                    wz = 0.0
                elif abs(vx) > 1e-6 or abs(wz) > 1e-6:
                    manual_state = MANUAL_DRIVE
                else:
                    manual_state = MANUAL_NONE

                manual_packet = struct.pack("<BBff", MANUAL_MODE_ID, manual_state, float(vx), float(wz))
                # Combined packet Windows -> robot:
                # [goal_packet 10B][manual_packet 10B]
                robot_conn.sendall(goal_data + manual_packet)

        except Exception as e:
            print(f"Bridge loop error: {e}")
            with shared_state.lock:
                shared_state.connected_robot = False
                shared_state.connected_server = False
                shared_state.last_error = str(e)
        finally:
            try:
                if robot_conn:
                    robot_conn.close()
            except Exception:
                pass
            try:
                if server_conn:
                    server_conn.close()
            except Exception:
                pass
            try:
                if bridge_server:
                    bridge_server.close()
            except Exception:
                pass
            time.sleep(1)


if __name__ == "__main__":
    import sys

    state = SharedState()
    t = threading.Thread(target=bridge_loop, args=(state,), daemon=True)
    t.start()

    app = QApplication(sys.argv)
    win = MainWindow(state)
    win.show()
    sys.exit(app.exec_())