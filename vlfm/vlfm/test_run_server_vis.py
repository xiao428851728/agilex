# vlfm/test_run_server_vis.py
import socket
import struct
import cv2
import numpy as np
import torch
import io
import os
from PIL import Image

from habitat_baselines.common.tensor_dict import TensorDict
from vlfm.test_run_vis import load_policy


# =========================
# 配置：仅保存最后一张地图
# =========================
SAVE_MAP_TO_DISK = True
VIS_DIR = "vlfm_vis"

if SAVE_MAP_TO_DISK:
    os.makedirs(VIS_DIR, exist_ok=True)


# =========================
# 1. 加载 VLFM Policy
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = load_policy(device)
print("✅ VLFM Policy Loaded on", device)

# 根据你的训练设置修改 door 的类别 ID
DOOR_CLASS_ID = 3
objectgoal = torch.tensor([[DOOR_CLASS_ID]], device=device)


# =========================
# 2. recv_all 工具函数
# =========================

def recv_all(conn, size: int) -> bytes:
    """从 conn 精确读取 size 字节"""
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


# =========================
# 3. 启动 9000 端口
# =========================

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("0.0.0.0", 9000))
sock.listen(1)
print("✅ VLFM Unified Server Listening on 9000 ...")

conn, addr = sock.accept()
print("✅ Windows Connected:", addr)


# =========================
# 4. RNN 状态（保持最简）
# =========================

rnn_hidden_states = None
prev_actions = None
masks = torch.ones(1, 1).to(device)


# =========================
# 5. 辅助函数：从 policy 内部 map 对象生成图片并编码
# =========================

def get_maps_from_policy():
    """
    直接从 policy._obstacle_map / policy._value_map 中生成可视化图像（BGR），
    然后返回 (obstacle_bgr, value_bgr)。如果某个不存在，返回 None。
    """
    obstacle_bgr = None
    value_bgr = None

    try:
        obstacle_map_obj = getattr(policy, "_obstacle_map", None)
        if obstacle_map_obj is not None:
            # ObstacleMap.visualize() 返回的是 BGR（内部用 OpenCV 画）
            obstacle_bgr = obstacle_map_obj.visualize()

        value_map_obj = getattr(policy, "_value_map", None)
        if value_map_obj is not None:
            # ValueMap.visualize() 可以传入 obstacle_map，用 explored_area 做遮罩
            obstacle_for_value = obstacle_map_obj if obstacle_bgr is not None else None
            value_bgr = value_map_obj.visualize(obstacle_map=obstacle_for_value)

    except Exception as e:
        print("⚠️ Failed to generate maps from policy:", e)

    return obstacle_bgr, value_bgr


def encode_img_png(img):
    """
    把 BGR 图像编码成 PNG 字节。
    返回 (bytes, size)，如果失败则返回 (b"", 0)。
    """
    if img is None:
        return b"", 0
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return b"", 0
    data = buf.tobytes()
    return data, len(data)


# =========================
# 6. 主推理循环（协议：RGB + Depth + State -> Action + Maps）
# =========================

while True:
    try:
        # ---------- 1) RGB ----------
        header = recv_all(conn, 4)
        if not header:
            print("⚠️ RGB header connection closed")
            break

        rgb_size = struct.unpack("I", header)[0]
        rgb_bytes = recv_all(conn, rgb_size)
        if not rgb_bytes:
            print("⚠️ RGB payload connection closed")
            break

        rgb_np = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        if rgb_np is None:
            print("⚠️ Failed to decode RGB image")
            continue

        rgb_np = cv2.resize(rgb_np, (256, 256))

        # ---------- 2) Depth（使用 PIL 解码 16bit PNG） ----------
        header = recv_all(conn, 4)
        if not header:
            print("⚠️ Depth header connection closed")
            break

        depth_size = struct.unpack("I", header)[0]
        depth_bytes = recv_all(conn, depth_size)
        if not depth_bytes:
            print("⚠️ Depth payload connection closed")
            break

        try:
            depth_img = Image.open(io.BytesIO(depth_bytes))
            depth_mm = np.array(depth_img).astype(np.uint16)
        except Exception as e:
            print("⚠️ Failed to decode depth image with PIL:", e)
            continue

        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_m = cv2.resize(depth_m, (256, 256))

        # ---------- 3) State ----------
        state_bytes = recv_all(conn, 12)
        if not state_bytes:
            print("⚠️ State payload connection closed")
            break

        x, y, yaw = struct.unpack("fff", state_bytes)

        # =========================
        # 构造 VLFM Observations
        # =========================

        rgb_tensor = torch.from_numpy(rgb_np).unsqueeze(0).to(device)                 # [1,256,256,3]
        depth_tensor = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(-1).to(device)  # [1,256,256,1]

        gps = torch.tensor([[x, y]], device=device, dtype=torch.float32)
        compass = torch.tensor([yaw], device=device, dtype=torch.float32)
        heading = torch.tensor([yaw], device=device, dtype=torch.float32)

        observations = TensorDict({
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "gps": gps,
            "compass": compass,
            "heading": heading,
            "objectgoal": objectgoal,
        })

        # =========================
        # VLFM 推理
        # =========================

        with torch.no_grad():
            out = policy.act(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic=True,
            )

        action = int(out.actions.item())
        print(f"🎯 Action = {action}  (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f})")

        # =========================
        # 从 policy 内部生成地图（BGR）
        # =========================

        obstacle_bgr, value_bgr = get_maps_from_policy()

        # 只保存最后一张到磁盘
        if SAVE_MAP_TO_DISK:
            if obstacle_bgr is not None:
                cv2.imwrite(os.path.join(VIS_DIR, "obstacle_latest.png"), obstacle_bgr)
            if value_bgr is not None:
                cv2.imwrite(os.path.join(VIS_DIR, "value_latest.png"), value_bgr)

        # 编码为 PNG 字节
        obstacle_bytes, obstacle_size = encode_img_png(obstacle_bgr)
        value_bytes, value_size = encode_img_png(value_bgr)

        # =========================
        # 回传 Action + Maps
        # =========================
        # 协议：
        # [1字节 action]
        # [4字节 obstacle_size][obstacle_png_bytes]
        # [4字节 value_size][value_png_bytes]
        try:
            conn.sendall(
                struct.pack("B", action) +
                struct.pack("I", obstacle_size) + obstacle_bytes +
                struct.pack("I", value_size) + value_bytes
            )
        except Exception as e:
            print("❌ Send Error:", e)
            break

    except Exception as e:
        print("❌ Server Error:", e)
        break

conn.close()
sock.close()
print("✅ VLFM Unified Server closed")


# 文件路径: vlfm/test_run_server_vis.py
# import socket
# import struct
# import cv2
# import numpy as np
# import torch
# import io
# import os
# from PIL import Image

# from habitat_baselines.common.tensor_dict import TensorDict

# # ✅ 确保这里引用的就是刚才改的那个文件
# from vlfm.test_run_vis import load_policy 

# # =========================
# # 配置
# # =========================
# SAVE_MAP_TO_DISK = True
# VIS_DIR = "vlfm_vis"
# if SAVE_MAP_TO_DISK:
#     os.makedirs(VIS_DIR, exist_ok=True)

# # =========================
# # 辅助函数：中心裁剪 + 缩放 (核心!)
# # =========================
# def crop_center_and_resize(img, target_size=256, is_depth=False):
#     """
#     1. 输入是 640x480
#     2. 裁成 480x480 (保持比例)
#     3. 缩放到 256x256 (适配网络)
#     """
#     h, w = img.shape[:2] # 480, 640
#     short_edge = min(h, w) # 480
    
#     # 计算中心
#     start_x = (w - short_edge) // 2
#     start_y = (h - short_edge) // 2
    
#     # 裁剪
#     cropped = img[start_y:start_y+short_edge, start_x:start_x+short_edge]
    
#     # 缩放 (深度图用最近邻，RGB用线性)
#     interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
#     resized = cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)
    
#     return resized

# def recv_all(conn, size):
#     data = b""
#     while len(data) < size:
#         chunk = conn.recv(size - len(data))
#         if not chunk: return b""
#         data += chunk
#     return data

# def encode_img_png(img):
#     if img is None: return b"", 0
#     ok, buf = cv2.imencode(".png", img)
#     return (buf.tobytes(), len(buf.tobytes())) if ok else (b"", 0)

# def get_maps_from_policy():
#     obs_bgr, val_bgr = None, None
#     try:
#         if hasattr(policy, "_obstacle_map") and policy._obstacle_map:
#             obs_bgr = policy._obstacle_map.visualize()
#         if hasattr(policy, "_value_map") and policy._value_map:
#             mask = policy._obstacle_map if obs_bgr is not None else None
#             val_bgr = policy._value_map.visualize(obstacle_map=mask)
#     except: pass
#     return obs_bgr, val_bgr

# # =========================
# # 加载模型
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# policy = load_policy(device)
# print(f"✅ VLFM Policy Ready on {device}")

# DOOR_CLASS_ID = 3
# objectgoal = torch.tensor([[DOOR_CLASS_ID]], device=device)

# # =========================
# # 服务器启动
# # =========================
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# sock.bind(("0.0.0.0", 9000))
# sock.listen(1)
# print("✅ Listening on 9000...")

# conn, addr = sock.accept()
# print(f"✅ Connected: {addr}")

# rnn_hidden_states = None
# prev_actions = None
# masks = torch.ones(1, 1).to(device)

# # =========================
# # 主循环
# # =========================
# while True:
#     try:
#         # --- 1. 接收 RGB (640x480) ---
#         head = recv_all(conn, 4)
#         if not head: break
#         size = struct.unpack("I", head)[0]
#         data = recv_all(conn, size)
        
#         rgb_raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
#         if rgb_raw is None: continue

#         # ✅ 处理：裁剪+缩放 -> 256x256
#         rgb_np = crop_center_and_resize(rgb_raw, 256, is_depth=False)

#         # --- 2. 接收 Depth (640x480) ---
#         head = recv_all(conn, 4)
#         if not head: break
#         size = struct.unpack("I", head)[0]
#         data = recv_all(conn, size)
        
#         try:
#             depth_pil = Image.open(io.BytesIO(data))
#             depth_raw_mm = np.array(depth_pil).astype(np.uint16)
#         except: continue
        
#         depth_raw_m = depth_raw_mm.astype(np.float32) / 1000.0

#         # ✅ 处理：裁剪+缩放 -> 256x256
#         depth_m = crop_center_and_resize(depth_raw_m, 256, is_depth=True)

#         # --- 3. 接收 State ---
#         data = recv_all(conn, 12)
#         if not data: break
#         x, y, yaw = struct.unpack("fff", data)

#         # =========================
#         # 推理 (输入全是 256x256，不会报错)
#         # =========================
#         rgb_tensor = torch.from_numpy(rgb_np).unsqueeze(0).to(device)
#         depth_tensor = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(-1).to(device)
        
#         gps = torch.tensor([[x, y]], device=device, dtype=torch.float32)
#         compass = torch.tensor([yaw], device=device, dtype=torch.float32)
#         heading = torch.tensor([yaw], device=device, dtype=torch.float32)

#         observations = TensorDict({
#             "rgb": rgb_tensor,
#             "depth": depth_tensor,
#             "gps": gps,
#             "compass": compass,
#             "heading": heading,
#             "objectgoal": objectgoal,
#         })

#         with torch.no_grad():
#             out = policy.act(
#                 observations,
#                 rnn_hidden_states,
#                 prev_actions,
#                 masks,
#                 deterministic=True,
#             )
        
#         action = int(out.actions.item())
#         print(f"🎯 Action {action} | Pos ({x:.2f}, {y:.2f}, {yaw:.2f})")

#         # =========================
#         # 回传
#         # =========================
#         obs_bgr, val_bgr = get_maps_from_policy()
        
#         if SAVE_MAP_TO_DISK and obs_bgr is not None:
#              cv2.imwrite(os.path.join(VIS_DIR, "obstacle_live.png"), obs_bgr)

#         obs_bytes, obs_len = encode_img_png(obs_bgr)
#         val_bytes, val_len = encode_img_png(val_bgr)

#         conn.sendall(
#             struct.pack("B", action) +
#             struct.pack("I", obs_len) + obs_bytes +
#             struct.pack("I", val_len) + val_bytes
#         )

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         break

# conn.close()
# sock.close()
# print("✅ Server Closed")