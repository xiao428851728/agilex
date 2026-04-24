import socket
import cv2
import numpy as np
import struct
import torch
from habitat_baselines.common.tensor_dict import TensorDict


# ✅ 直接复用你现在的 test_run1 里的 load_policy
from vlfm.test_run1 import load_policy


# ===============================
# 1. 启动 VLFM + PointNav
# ===============================

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = load_policy(device)

print("✅ VLFM + 真实 PointNav 已加载完成")

# ===============================
# 2. 启动 Socket 服务
# ===============================

sock = socket.socket()
sock.bind(("0.0.0.0", 9000))
sock.listen(1)

print("✅ VLFM Server Listening on 9000 ...")
conn, addr = sock.accept()
print("✅ Windows Connected:", addr)

# ===============================
# 3. 推理循环
# ===============================

rnn_hidden_states = None
prev_actions = None
masks = torch.ones(1, 1).to(device)

while True:
    # ---------- 接收一帧 JPG ----------
    size_bytes = conn.recv(4)
    if not size_bytes:
        print("⚠️ Connection closed")
        break

    size = struct.unpack("I", size_bytes)[0]

    data = b""
    while len(data) < size:
        data += conn.recv(size - len(data))

    # ---------- 解码图像 ----------
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    img = cv2.resize(img, (256, 256))
    
    # ---------- 构造 VLFM 观测（✅ 只能用 HWC, uint8） ----------

    observations = TensorDict({
        # ✅ 关键修复：直接使用 OpenCV 格式的 HWC + uint8
        "rgb": torch.from_numpy(img).unsqueeze(0).to(device),   # [1, H, W, 3], uint8

        "depth": torch.ones(1, 256, 256, 1, device=device),

        "gps": torch.tensor([[0.0, 0.0]], device=device),
        "compass": torch.tensor([0.0], device=device),
        "heading": torch.tensor([0.0], device=device),

        "objectgoal": torch.tensor([[0]], device=device),
    })

    # ---------- VLFM 推理 ----------
    with torch.no_grad():
        out = policy.act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=True,
        )

    action_id = int(out.actions.item())
    print("🎯 Action =", action_id)

    # ---------- 回传动作 ----------
    conn.send(action_id.to_bytes(1, "big"))
