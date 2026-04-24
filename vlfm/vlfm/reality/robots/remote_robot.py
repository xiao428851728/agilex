import numpy as np
import cv2
import struct
import socket
from typing import List, Dict, Tuple, Any
from .base_robot import BaseRobot

# ==================================================
# 🛠️ 硬件参数配置
# ==================================================
CAMERA_INTRINSICS = {
    "fx": 603.9898071289062, 
    "fy": 603.3316650390625,
    "cx": 329.9515075683594, 
    "cy": 248.39097595214844, 
}

CAMERA_POSITION = {
    "x": 0.1,  
    "y": 0.0,
    "z": 0.35  # 你的相机高度
}

CAMERA_TILT = 0.0 

class RemoteRobot(BaseRobot):
    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pose = np.zeros(3) 
        self._sync_success = False

        # ✅✅✅ [日志 1] 启动时打印关键参数，方便截图确认
        print("\n" + "="*40)
        print("🤖 RemoteRobot Config Report")
        print("="*40)
        print(f"📷 Intrinsics (fx, fy): {CAMERA_INTRINSICS['fx']:.2f}, {CAMERA_INTRINSICS['fy']:.2f}")
        print(f"📍 Extrinsics (x, y, z): {CAMERA_POSITION['x']}, {CAMERA_POSITION['y']}, {CAMERA_POSITION['z']} (Height)")
        print(f"📐 Tilt: {CAMERA_TILT} rad")
        print("="*40 + "\n")

    def _recv_all(self, size):
        data = b""
        while len(data) < size:
            chunk = self.conn.recv(size - len(data))
            if not chunk: return None
            data += chunk
        return data

    def sync_data(self) -> bool:
        try:
            # 1. RGB
            head = self._recv_all(4)
            if not head: return False
            rgb_size = struct.unpack("I", head)[0]
            rgb_bytes = self._recv_all(rgb_size)
            self.latest_rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

            # 2. Depth
            head = self._recv_all(4)
            if not head: return False
            depth_size = struct.unpack("I", head)[0]
            depth_bytes = self._recv_all(depth_size)
            # 🚨 关键：IMREAD_UNCHANGED 保持 16位 原始数据
            self.latest_depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

            # ✅✅✅ [日志 2] 打印深度图统计信息，验证单位
            if self.latest_depth is not None:
                d_min = self.latest_depth.min()
                d_max = self.latest_depth.max()
                d_center = self.latest_depth[self.latest_depth.shape[0]//2, self.latest_depth.shape[1]//2]
                print(f"🔍 Depth Check | Min: {d_min}, Max: {d_max}, Center: {d_center} | Dtype: {self.latest_depth.dtype}")
                
                # 自动报警：如果数值很小（比如全是0-10），可能是米单位；如果是几千，是毫米单位
                if d_max < 20 and d_max > 0:
                     print("⚠️ [警告] 深度值看起来像 '米' (Meters)。VLFM PointNavEnv 可能还会除以1000，导致数据归零！")
                elif d_max > 100:
                     print("✅ [正常] 深度值看起来像 '毫米' (mm)。符合预期。")

            # 3. Pose
            pose_bytes = self._recv_all(12)
            if not pose_bytes: return False
            x, y, yaw = struct.unpack("fff", pose_bytes)
            self.latest_pose = np.array([x, y, yaw])

            self._sync_success = True
            return True
        except Exception as e:
            print(f"❌ Sync Error: {e}")
            return False

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        return self.latest_pose[:2], self.latest_pose[2]

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        if not self._sync_success: return {}

        # 构造矩阵逻辑不变...
        rx, ry, ryaw = self.latest_pose
        T_global_base = np.eye(4)
        T_global_base[0:2, 3] = [rx, ry]
        cos_y, sin_y = np.cos(ryaw), np.sin(ryaw)
        T_global_base[0, 0], T_global_base[0, 1] = cos_y, -sin_y
        T_global_base[1, 0], T_global_base[1, 1] = sin_y, cos_y

        T_base_cam = np.eye(4)
        T_base_cam[0, 3] = CAMERA_POSITION["x"]
        T_base_cam[1, 3] = CAMERA_POSITION["y"]
        T_base_cam[2, 3] = CAMERA_POSITION["z"]
        
        if CAMERA_TILT != 0:
            cos_t, sin_t = np.cos(CAMERA_TILT), np.sin(CAMERA_TILT)
            rot_tilt = np.eye(4)
            rot_tilt[0, 0], rot_tilt[0, 2] = cos_t, sin_t
            rot_tilt[2, 0], rot_tilt[2, 2] = -sin_t, cos_t
            T_base_cam = T_base_cam @ rot_tilt

        tf_global_cam = T_global_base @ T_base_cam

        results = {}
        for src in srcs:
            image = self.latest_rgb
            if "depth" in src:
                image = self.latest_depth
            
            results[src] = {
                "image": image,
                "fx": CAMERA_INTRINSICS["fx"],
                "fy": CAMERA_INTRINSICS["fy"],
                "tf_camera_to_global": tf_global_cam
            }
        return results

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        pass