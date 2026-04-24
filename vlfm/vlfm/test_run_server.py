import socket
import struct
import cv2
import numpy as np
import torch
import io
from PIL import Image

from habitat_baselines.common.tensor_dict import TensorDict
from vlfm.test_run1 import load_policy


# =========================
# 1. 加载 VLFM Policy
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = load_policy(device)
print("✅ VLFM Policy Loaded on", device)

# =========================
# 2. Door ObjectGoal ID（你必须确认）
# =========================
# ⚠️ 一定要改成你训练时 door 对应的类别 ID
DOOR_CLASS_ID = 3
objectgoal = torch.tensor([[DOOR_CLASS_ID]], device=device)


# =========================
# 3. recv_all 工具函数
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
# 4. 启动 9000 端口
# =========================

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("0.0.0.0", 9000))
sock.listen(1)
print("✅ VLFM Unified Server Listening on 9000 ...")

conn, addr = sock.accept()
print("✅ Windows Connected:", addr)


# =========================
# 5. RNN 状态（保持最简）
# =========================

rnn_hidden_states = None
prev_actions = None
masks = torch.ones(1, 1).to(device)


# =========================
# 6. 主推理循环（合并版协议）
# =========================
# 协议：
# [4字节 rgb_size]
# [rgb_jpg_bytes]
# [4字节 depth_size]
# [depth_png_bytes]
# [12字节 state: float32 x, y, yaw]
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


        # ---------- 2) Depth（✅ 使用 PIL 解码 16bit PNG） ----------
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
        # 回传 Action（1 字节）
        # =========================

        conn.send(struct.pack("B", action))


    except Exception as e:
        print("❌ Server Error:", e)
        break


conn.close()
sock.close()
print("✅ VLFM Unified Server closed")
