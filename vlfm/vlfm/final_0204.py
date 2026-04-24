# ==============================================================================
# 1. 核心系统库
# ==============================================================================
import socket
import struct
import cv2
import numpy as np
import torch
import io
import os
from PIL import Image

# ==============================================================================
# 2. Habitat & Hydra 基础库
# ==============================================================================
import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.common.tensor_dict import TensorDict
from gym import spaces

# ==============================================================================
# 3. 注册 VLFM 模块
# ==============================================================================
import vlfm.measurements.traveled_stairs  
import vlfm.policy.habitat_policies   
import vlfm.policy.reality_policies   
import vlfm.mapping.frontier_map          
import vlfm.mapping.obstacle_map          
import vlfm.mapping.value_map             

# =========================
# ⚙️ 配置区域
# =========================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 9000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIS_DIR = "vlfm_vis"
os.makedirs(VIS_DIR, exist_ok=True)

TARGET_SIZE = (224, 224)  
TARGET_OBJECT_ID = 2  

# =========================
# 🛠️ 1. 图像预处理工具
# =========================
def image_resize(img: torch.Tensor, size: tuple, channels_last: bool = False, interpolation_mode: str = "area") -> torch.Tensor:
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if no_batch_dim: img = img.unsqueeze(0)
    
    if channels_last:
        if len(img.shape) == 4: img = img.permute(0, 3, 1, 2)
        else: img = img.permute(0, 1, 4, 2, 3)

    img = torch.nn.functional.interpolate(img.float(), size=size, mode=interpolation_mode)
    
    if channels_last:
        if len(img.shape) == 4: img = img.permute(0, 2, 3, 1)
        else: img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim: img = img.squeeze(dim=0)
    return img

def preprocess_obs_vlfm_style(rgb_np, depth_mm_np, device):
    rgb_tensor = torch.from_numpy(rgb_np).float().to(device)
    depth_m_tensor = torch.from_numpy(depth_mm_np).float().to(device) / 1000.0
    depth_m_tensor = depth_m_tensor.unsqueeze(-1)

    rgb_resized = image_resize(rgb_tensor, TARGET_SIZE, channels_last=True, interpolation_mode="area")
    depth_resized = image_resize(depth_m_tensor, TARGET_SIZE, channels_last=True, interpolation_mode="area")
    depth_norm = torch.clamp(depth_resized, 0.0, 5.0) / 5.0

    return rgb_resized.byte(), depth_norm

# =========================
# 🧠 2. 加载 Policy
# =========================
def load_vlfm_policy():
    print(f"🔥 Initializing VLFM Full System on {DEVICE}...")
    config_path = "/home/sxz/sjd/vlfm/config/experiments/vlfm_objectnav_hm3d.yaml"
    
    config = get_config(config_path, [
        "habitat_baselines.rl.policy.name=HabitatITMPolicyV2",
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=224", 
        "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=224",
    ])
    
    policy_cls = baseline_registry.get_policy(config.habitat_baselines.rl.policy.name)
    
    obs_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
        "objectgoal": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int64),
        "gps": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "compass": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
    })
    action_space = spaces.Discrete(4)
    
    policy = policy_cls.from_config(config, obs_space, action_space)
    policy.to(DEVICE)
    policy.eval()
    print("✅ VLFM Policy loaded!")
    return policy

policy = load_vlfm_policy()

# =========================
# 📡 3. 核心工具函数
# =========================
def recv_all(conn, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk: return b""
        data += chunk
    return data

def get_maps_from_policy(policy_instance):
    """提取可视化地图 (Obstacle & Value)"""
    obs_bgr, val_bgr = None, None
    mapper = None
    if hasattr(policy_instance, "mapper"): mapper = policy_instance.mapper
    elif hasattr(policy_instance, "policy") and hasattr(policy_instance.policy, "mapper"):
        mapper = policy_instance.policy.mapper

    if mapper:
        if hasattr(mapper, "obstacle_map"):
            try: obs_bgr = mapper.obstacle_map.visualize()
            except: pass
        if hasattr(mapper, "value_map"):
            try: 
                val_bgr = mapper.value_map.visualize(obstacle_map=getattr(mapper, "obstacle_map", None))
            except: pass
    return obs_bgr, val_bgr

# ==============================================================================
# ⚡️⚡️ 关键函数：提取你看到的那个 Log 中的 Best Frontier ⚡️⚡️
# ==============================================================================
def get_best_frontier_goal(policy_instance):
    """
    直接从 VLFM 的 ValueMap 中提取 calculated best frontier。
    这就是日志里打印的那个值！
    """
    try:
        # 1. 深入找到 Mapper (处理 Policy 嵌套)
        mapper = None
        if hasattr(policy_instance, "mapper"): 
            mapper = policy_instance.mapper
        elif hasattr(policy_instance, "policy") and hasattr(policy_instance.policy, "mapper"):
            mapper = policy_instance.policy.mapper

        # 2. 提取 best_frontier
        # VLFM 的 mapping/value_map.py 中就是把这个值打印到了终端
        if mapper and hasattr(mapper, "value_map"):
            frontier_tensor = mapper.value_map.best_frontier # 这是一个 Tensor
            
            # 必须转回 CPU 才能读取数值
            if frontier_tensor is not None:
                f_cpu = frontier_tensor.detach().cpu().numpy().flatten()
                
                # 坐标有效性检查 (排除初始化时的 0,0)
                if abs(f_cpu[0]) > 0.01 or abs(f_cpu[1]) > 0.01:
                    # print(f"🔍 DEBUG: Extracted Frontier: {f_cpu}")
                    return True, float(f_cpu[0]), float(f_cpu[1])
                    
    except Exception as e:
        print(f"⚠️ Error extracting frontier: {e}")
        
    return False, 0.0, 0.0

# =========================
# 🚀 4. 主循环
# =========================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(f"🚀 Server listening on {SERVER_PORT}...")

counter = 0

while True:
    print("⏳ Waiting for connection...")
    conn, addr = sock.accept()
    print(f"✅ Robot connected: {addr}")
    
    # 初始化状态
    rnn_hidden_states = torch.zeros(1, policy.net.num_recurrent_layers, 512, device=DEVICE) if hasattr(policy, "net") else torch.zeros(1, 4, 512, device=DEVICE)
    prev_actions = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
    masks = torch.zeros(1, 1, device=DEVICE, dtype=torch.bool)
    object_goal_tensor = torch.tensor([[TARGET_OBJECT_ID]], device=DEVICE, dtype=torch.long)

    try:
        while True:
            # 1. 接收 Header (RGB Size)
            header = recv_all(conn, 4)
            if not header: break
            rgb_size = struct.unpack("I", header)[0]
            rgb_bytes = recv_all(conn, rgb_size)
            rgb_np = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # 2. 接收 Depth Size
            header = recv_all(conn, 4)
            if not header: break
            depth_size = struct.unpack("I", header)[0]
            depth_bytes = recv_all(conn, depth_size)
            depth_img = Image.open(io.BytesIO(depth_bytes))
            depth_mm_np = np.array(depth_img).astype(np.float32)

            # 3. 接收 State (X, Y, Yaw)
            state_bytes = recv_all(conn, 12)
            if not state_bytes: break
            x, y, yaw = struct.unpack("fff", state_bytes)
            
            # 4. 预处理
            rgb_tensor, depth_tensor = preprocess_obs_vlfm_style(rgb_np, depth_mm_np, DEVICE)

            # 5. 构造 Batch
            batch = TensorDict({
                "rgb": rgb_tensor.unsqueeze(0),
                "depth": depth_tensor.unsqueeze(0),
                "objectgoal": object_goal_tensor,
                "gps": torch.tensor([[x, y]], device=DEVICE, dtype=torch.float32),
                "compass": torch.tensor([[yaw]], device=DEVICE, dtype=torch.float32),
                "heading": torch.tensor([[yaw]], device=DEVICE, dtype=torch.float32)
            })
            
            # ==========================================
            # ⚡️ FIX 1: 提前初始化变量，防止报错
            # ==========================================
            discrete_val = 0 
            
            # 6. VLFM 推理 (Update Map)
            with torch.no_grad():
                # policy.act 会更新内部地图
                action_data = policy.act(batch, rnn_hidden_states, prev_actions, masks, deterministic=True)
                
                # 提取动作 (无论是否用到，都要提取以更新 RNN 状态)
                if isinstance(action_data, tuple):
                    actions_tensor, rnn_hidden_states = action_data
                    discrete_val = actions_tensor[0].item()
                else:
                    rnn_hidden_states = action_data.rnn_hidden_states
                    if hasattr(action_data, "actions"):
                        discrete_val = action_data.actions[0].item()

                prev_actions.copy_(torch.tensor([[discrete_val]], device=DEVICE, dtype=torch.long))
                masks.fill_(True)

            # ==========================================================
            # ⚡️⚡️ 决策核心：直接拿 Best Frontier ⚡️⚡️
            # ==========================================================
            
            # A. 调用函数提取 frontier
            has_goal, tx, ty = get_best_frontier_goal(policy)
            
            final_action = discrete_val # 默认行为
            
            if has_goal:
                # 只要有全局 Frontier，就告诉小车 "Action 1 (FORWARD)" 模式，并带上坐标
                final_action = 1 
                print(f"🎯 Use Best Frontier: ({tx:.2f}, {ty:.2f})")
            else:
                # 还没算出来 (Initialize 阶段)，保持 Policy 建议的离散动作 (比如转圈)
                print(f"🔄 Init Mode (Spinning): Action {discrete_val}")
                tx, ty = 0.0, 0.0

            # 7. 准备地图数据
            obs_bgr, val_bgr = get_maps_from_policy(policy)
            counter += 1
            
            obs_bytes = b""
            if obs_bgr is not None: _, buf = cv2.imencode(".png", obs_bgr); obs_bytes = buf.tobytes()
            val_bytes = b""
            if val_bgr is not None: _, buf = cv2.imencode(".png", val_bgr); val_bytes = buf.tobytes()

            # 8. 发送数据包
            # 格式: <B: ActionID, B: Flag, f: x, f: y (共 10 bytes)
            packet = struct.pack("<BBff", final_action, int(has_goal), float(tx), float(ty))
            conn.sendall(packet)
            
            # 发送地图
            conn.sendall(struct.pack("I", len(obs_bytes)) + obs_bytes)
            conn.sendall(struct.pack("I", len(val_bytes)) + val_bytes)

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        if conn: conn.close()