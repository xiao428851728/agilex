# ==============================================================================
# VLFM Socket Server (Full Version)
# - Receives: RGB(jpg/png bytes) + Depth(png bytes) + State(x,y,yaw float32)
# - Runs: VLFM policy.act() to update internal maps + get "Best frontier" goal
# - Sends: ActionID + HasGoal + GoalX + GoalY  (10 bytes) + ObstacleMap PNG + ValueMap PNG
#
# Key fixes vs your current code:
# 1) Enforce input size == obs_space/config (224x224)  [IMPORTANT]
# 2) Depth stays in meters (0~5m), NOT normalized to 0~1  [IMPORTANT]
# 3) BGR->RGB conversion for OpenCV decoded images
# 4) masks init = True (not done)
# 5) robust extraction of "best frontier" from policy object via attribute probing
# 6) consistent little-endian struct unpack for state
# ==============================================================================

import socket
import struct
import cv2
import numpy as np
import torch
import io
import os
from PIL import Image

import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.common.tensor_dict import TensorDict
from gym import spaces

# ---- Manual registry imports (required for @register to take effect) ----
import vlfm.measurements.traveled_stairs
import vlfm.policy.habitat_policies
import vlfm.policy.reality_policies
import vlfm.mapping.frontier_map
import vlfm.mapping.obstacle_map
import vlfm.mapping.value_map

# =========================
# ⚙️ Config
# =========================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 9000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIS_DIR = "vlfm_vis"
os.makedirs(VIS_DIR, exist_ok=True)

TARGET_OBJECT_ID = 2

# ---- Must match config + obs_space ----
TARGET_SIZE_HW = (224, 224)  # (H, W)
DEPTH_MAX_M = 5.0

# =========================
# Resize util (VLFM-style)
# =========================
def image_resize(
    img: torch.Tensor,
    size_hw: tuple,  # (H, W)
    channels_last: bool = False,
    interpolation_mode: str = "area",
) -> torch.Tensor:
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError(f"Bad shape: {img.shape}")

    if no_batch_dim:
        img = img.unsqueeze(0)  # (N, H, W, C) or (N, H, W)

    # NHWC -> NCHW
    if channels_last:
        if len(img.shape) == 4:
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.permute(0, 1, 4, 2, 3)

    img = torch.nn.functional.interpolate(img.float(), size=size_hw, mode=interpolation_mode)

    # NCHW -> NHWC
    if channels_last:
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)
        else:
            img = img.permute(0, 1, 3, 4, 2)

    if no_batch_dim:
        img = img.squeeze(0)

    return img

def preprocess_obs_vlfm_style(bgr_np: np.ndarray, depth_mm_np: np.ndarray, device: str):
    """
    Output:
      rgb_uint8: (224,224,3) uint8 in RGB order
      depth_m  : (224,224,1) float32 meters in [0, DEPTH_MAX_M]
    """
    # 0) BGR -> RGB
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

    # 1) to tensor
    rgb_tensor = torch.from_numpy(rgb_np).to(device=device, dtype=torch.float32)  # HWC
    depth_m = torch.from_numpy(depth_mm_np).to(device=device, dtype=torch.float32) / 1000.0
    depth_m = depth_m.unsqueeze(-1)  # HWC(1)

    # 2) resize to (224,224) using area
    rgb_resized = image_resize(rgb_tensor, TARGET_SIZE_HW, channels_last=True, interpolation_mode="area")
    depth_resized = image_resize(depth_m, TARGET_SIZE_HW, channels_last=True, interpolation_mode="area")

    # 3) clamp depth in meters
    depth_resized = torch.clamp(depth_resized, 0.0, float(DEPTH_MAX_M))

    # RGB back to uint8 (0..255)
    rgb_uint8 = torch.clamp(rgb_resized, 0.0, 255.0).to(torch.uint8)

    return rgb_uint8, depth_resized.to(torch.float32)

# =========================
# Load VLFM policy
# =========================
def load_vlfm_policy():
    print(f"🔥 Initializing VLFM Full System on {DEVICE}...")
    config_path = "/home/sxz/sjd/vlfm/config/experiments/vlfm_objectnav_hm3d.yaml"

    config = get_config(
        config_path,
        [
            "habitat_baselines.rl.policy.name=HabitatITMPolicyV2",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=224",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=224",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=224",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=224",
        ],
    )

    policy_cls = baseline_registry.get_policy(config.habitat_baselines.rl.policy.name)
    print(
        "fov, camera_height:",
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov,
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position,
    )

    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=float(DEPTH_MAX_M), shape=(224, 224, 1), dtype=np.float32),
            "objectgoal": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int64),
            "gps": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "compass": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        }
    )
    action_space = spaces.Discrete(4)

    policy = policy_cls.from_config(config, obs_space, action_space)
    policy.to(DEVICE)
    policy.eval()
    print("✅ VLFM Policy loaded!")
    return policy

policy = load_vlfm_policy()

# =========================
# Socket helpers
# =========================
def recv_all(conn, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return b""
        data += chunk
    return data

# =========================
# Map extraction
# =========================
def get_maps_from_policy(policy_instance):
    """
    Try extract maps from policy or wrapper (.policy)
    Returns: (obstacle_bgr, value_bgr) or (None,None)
    """
    def extract(obj):
        o_bgr, v_bgr = None, None
        if hasattr(obj, "_obstacle_map") and obj._obstacle_map is not None:
            try:
                o_bgr = obj._obstacle_map.visualize()
            except Exception as e:
                print(f"⚠️ Obstacle Map visualize failed: {e}")

        if hasattr(obj, "_value_map") and obj._value_map is not None:
            try:
                obs_map_ref = getattr(obj, "_obstacle_map", None)
                v_bgr = obj._value_map.visualize(obstacle_map=obs_map_ref)
            except Exception as e:
                print(f"⚠️ Value Map visualize failed: {e}")
        return o_bgr, v_bgr

    obs_bgr, val_bgr = extract(policy_instance)
    if obs_bgr is None and val_bgr is None and hasattr(policy_instance, "policy"):
        obs_bgr, val_bgr = extract(policy_instance.policy)
    return obs_bgr, val_bgr

# =========================
# Goal extraction: "Best frontier"
# =========================
def _to_xy(v):
    """Convert tensor/ndarray/list -> (x,y) float"""
    if v is None:
        return None
    if torch.is_tensor(v):
        v = v.detach().float().cpu().view(-1).numpy()
    elif isinstance(v, (list, tuple, np.ndarray)):
        v = np.array(v, dtype=np.float32).reshape(-1)
    else:
        return None
    if v.size < 2:
        return None
    return float(v[0]), float(v[1])

def get_best_frontier_goal(policy_object, debug=True):
    """
    Robustly find the internal "best frontier" (x,y) computed by VLFM.
    Since different VLFM versions use different attribute names, we:
      1) unwrap core policy
      2) search attributes containing keywords: frontier/waypoint/goal/target/wp/nav
      3) pick the first candidate that looks like a 2D point
    """
    core = policy_object.policy if hasattr(policy_object, "policy") else policy_object

    # Common direct names to try first (fast path)
    direct_names = [
        "best_frontier",
        "frontier",
        "frontier_goal",
        "frontier_point",
        "waypoint",
        "nav_goal",
        "goal",
        "target",
        "target_xy",
        "wp",
        "best_wp",
    ]
    for name in direct_names:
        if hasattr(core, name):
            xy = _to_xy(getattr(core, name))
            if xy is not None:
                return True, xy[0], xy[1]

    # Generic probing
    keys = []
    for k in dir(core):
        lk = k.lower()
        if any(s in lk for s in ["frontier", "waypoint", "goal", "target", "wp", "nav"]):
            keys.append(k)

    if debug:
        print("\n====== [DEBUG] candidate fields (goal/frontier) ======")

    for k in sorted(keys):
        try:
            v = getattr(core, k)
            xy = _to_xy(v)
            if debug:
                if torch.is_tensor(v):
                    print(f"{k}: Tensor{tuple(v.shape)} -> {xy}")
                else:
                    print(f"{k}: {type(v).__name__} -> {xy}")
            if xy is not None:
                return True, xy[0], xy[1]
        except Exception:
            continue

    if debug:
        print("====================================================\n")

    return False, 0.0, 0.0

# =========================
# RNN state utilities
# =========================
def init_recurrent_states(pol):
    """
    Try infer recurrent layers + hidden size robustly.
    Falls back to (layers=4, hidden=512) if not available.
    """
    layers = 4
    hidden = 512

    net = getattr(pol, "net", None)
    if net is not None:
        layers = getattr(net, "num_recurrent_layers", layers)
        hidden = getattr(net, "rnn_hidden_size", hidden)
        hidden = getattr(net, "hidden_size", hidden)

    return torch.zeros(1, layers, hidden, device=DEVICE)

# =========================
# Main loop
# =========================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(f"🚀 Server listening on {SERVER_PORT}...")

counter = 0

while True:
    conn, addr = sock.accept()
    print(f"✅ Robot connected: {addr}")

    # Init states
    rnn_hidden_states = init_recurrent_states(policy)
    prev_actions = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
    masks = torch.ones(1, 1, device=DEVICE, dtype=torch.bool)  # not done
    object_goal_tensor = torch.tensor([[TARGET_OBJECT_ID]], device=DEVICE, dtype=torch.long)

    try:
        while True:
            # --- RGB ---
            header = recv_all(conn, 4)
            if not header:
                break
            rgb_size = struct.unpack("<I", header)[0]
            rgb_bytes = recv_all(conn, rgb_size)
            if not rgb_bytes:
                break
            bgr_np = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
            if bgr_np is None:
                print("⚠️ Failed to decode RGB frame.")
                continue

            # --- Depth ---
            header = recv_all(conn, 4)
            if not header:
                break
            depth_size = struct.unpack("<I", header)[0]
            depth_bytes = recv_all(conn, depth_size)
            if not depth_bytes:
                break
            depth_img = Image.open(io.BytesIO(depth_bytes))
            depth_mm_np = np.array(depth_img).astype(np.float32)

            # --- State ---
            state_bytes = recv_all(conn, 12)
            if not state_bytes:
                break
            x, y, yaw = struct.unpack("<fff", state_bytes)

            # --- Preprocess ---
            rgb_tensor, depth_tensor = preprocess_obs_vlfm_style(bgr_np, depth_mm_np, DEVICE)

            batch = TensorDict(
                {
                    "rgb": rgb_tensor.unsqueeze(0),         # [1,224,224,3] uint8 RGB
                    "depth": depth_tensor.unsqueeze(0),     # [1,224,224,1] float meters
                    "objectgoal": object_goal_tensor,
                    "gps": torch.tensor([[x, y]], device=DEVICE, dtype=torch.float32),
                    "compass": torch.tensor([[yaw]], device=DEVICE, dtype=torch.float32),
                    "heading": torch.tensor([[yaw]], device=DEVICE, dtype=torch.float32),
                }
            )

            # --- Act (update maps) ---
            with torch.no_grad():
                action_data = policy.act(batch, rnn_hidden_states, prev_actions, masks, deterministic=True)

                discrete_action = 0  # default STOP
                if isinstance(action_data, tuple):
                    actions_tensor, rnn_hidden_states = action_data
                    # actions_tensor may be shape [1,1] or [1]
                    discrete_action = int(actions_tensor.view(-1)[0].item())
                else:
                    rnn_hidden_states = action_data.rnn_hidden_states
                    discrete_action = int(action_data.actions.view(-1)[0].item())

                prev_actions.fill_(discrete_action)
                masks.fill_(True)

            # --- Goal: read Best frontier from policy internals ---
            # Turn debug=True once to discover which field holds the frontier.
            has_goal, tx, ty = get_best_frontier_goal(policy, debug=False)

            if has_goal:
                print(f"🎯 Best frontier goal: ({tx:.4f}, {ty:.4f}) | Robot: ({x:.2f}, {y:.2f}) | Action: {discrete_action}")
            else:
                print(f"⏳ Exploring... (No frontier field found) | Robot: ({x:.2f}, {y:.2f}) | Action: {discrete_action}")
                tx, ty = 0.0, 0.0

            # --- Maps for debug ---
            obs_bgr, val_bgr = get_maps_from_policy(policy)
            counter += 1
            if obs_bgr is not None:
                cv2.imwrite(f"{VIS_DIR}/obs_{counter}.png", obs_bgr)

            obs_bytes = b""
            if obs_bgr is not None:
                ok, buf = cv2.imencode(".png", obs_bgr)
                if ok:
                    obs_bytes = buf.tobytes()

            val_bytes = b""
            if val_bgr is not None:
                ok, buf = cv2.imencode(".png", val_bgr)
                if ok:
                    val_bytes = buf.tobytes()

            # --- Send packet ---
            # Packet: <BBff  (ActionID, HasGoal, GoalX, GoalY)
            conn.sendall(struct.pack("<BBff", int(discrete_action), int(has_goal), float(tx), float(ty)))
            conn.sendall(struct.pack("<I", len(obs_bytes)) + obs_bytes)
            conn.sendall(struct.pack("<I", len(val_bytes)) + val_bytes)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print("🔌 Client disconnected.")
