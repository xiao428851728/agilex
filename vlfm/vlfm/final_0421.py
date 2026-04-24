# ==============================================================================
# 1. Core system libs
# ==============================================================================
import io
import json
import os
import socket
import struct
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ==============================================================================
# 2. Habitat / Hydra
# ==============================================================================
import habitat  # noqa: F401  # required by VLFM runtime registration side effects
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default import get_config

# ==============================================================================
# 3. Register VLFM modules
# ==============================================================================
import vlfm.mapping.frontier_map  # noqa: F401
import vlfm.mapping.obstacle_map  # noqa: F401
import vlfm.mapping.value_map  # noqa: F401
import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.policy.reality_policies  # noqa: F401

# =========================
# Config
# =========================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 9000
CONTROL_PORT = 9001  # small side-channel for UI target input
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIS_DIR = "vlfm_vis"
os.makedirs(VIS_DIR, exist_ok=True)

TARGET_SIZE = (224, 224)
TARGET_OBJECT_ID = 3
TARGET_NAME = "potted plant"   # initial target, can be switched live from terminal
VLFM_CONFIG_PATH = "/home/sxz/sjd/vlfm/config/experiments/vlfm_objectnav_hm3d.yaml"

# live target switching / memory
_TARGET_LOCK = threading.Lock()
_CURRENT_TARGET_NAME = TARGET_NAME
_REMEMBERED_TARGETS: Dict[str, Tuple[float, float]] = {}


# =========================
# Image preprocessing
# =========================
def image_resize(img: torch.Tensor, size: tuple, channels_last: bool = False, interpolation_mode: str = "area") -> torch.Tensor:
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if no_batch_dim:
        img = img.unsqueeze(0)

    if channels_last:
        if len(img.shape) == 4:
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.permute(0, 1, 4, 2, 3)

    img = torch.nn.functional.interpolate(img.float(), size=size, mode=interpolation_mode)

    if channels_last:
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)
        else:
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)
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
# Policy loading
# =========================
def load_vlfm_policy():
    print(f"Initializing VLFM Full System on {DEVICE}...")
    config = get_config(VLFM_CONFIG_PATH, [
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
    print("VLFM Policy loaded!")
    print("[TARGET] Initial target:", TARGET_NAME)
    print(f"[TARGET] UI control listener will run on port {CONTROL_PORT}")
    return policy


policy = load_vlfm_policy()


# =========================
# Generic helpers
# =========================
def recv_all(conn, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def to_xy_tuple(value):
    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy().astype(np.float32).flatten()
        else:
            arr = np.array(value, dtype=np.float32).flatten()
        if len(arr) >= 2:
            x = float(arr[0])
            y = float(arr[1])
            if np.isfinite(x) and np.isfinite(y):
                return (x, y)
    except Exception:
        pass
    return None


def find_attribute_recursive(obj, attr_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    if depth > 6:
        return None

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    if hasattr(obj, attr_name):
        try:
            return getattr(obj, attr_name)
        except Exception:
            return None

    children = [
        "mapper", "policy", "net", "actor_critic", "module", "action_distribution",
        "agent", "navigator", "_policy",
    ]
    for child_name in children:
        if hasattr(obj, child_name):
            try:
                res = find_attribute_recursive(getattr(obj, child_name), attr_name, depth + 1, visited)
                if res is not None:
                    return res
            except Exception:
                pass
    return None


def set_attribute_recursive(obj, attr_name, value, depth=0, visited=None) -> int:
    if visited is None:
        visited = set()
    if depth > 6:
        return 0

    obj_id = id(obj)
    if obj_id in visited:
        return 0
    visited.add(obj_id)

    updated = 0
    if hasattr(obj, attr_name):
        try:
            setattr(obj, attr_name, value)
            updated += 1
        except Exception:
            pass

    children = [
        "mapper", "policy", "net", "actor_critic", "module", "action_distribution",
        "agent", "navigator", "_policy",
    ]
    for child_name in children:
        if hasattr(obj, child_name):
            try:
                updated += set_attribute_recursive(getattr(obj, child_name), attr_name, value, depth + 1, visited)
            except Exception:
                pass
    return updated


def get_current_target_name() -> str:
    with _TARGET_LOCK:
        return _CURRENT_TARGET_NAME


def set_current_target_name(name: str) -> None:
    global _CURRENT_TARGET_NAME
    with _TARGET_LOCK:
        _CURRENT_TARGET_NAME = name


def remember_target(name: str, xy: Tuple[float, float]) -> None:
    with _TARGET_LOCK:
        _REMEMBERED_TARGETS[name] = xy


def snapshot_remembered_targets() -> Dict[str, Tuple[float, float]]:
    with _TARGET_LOCK:
        return dict(_REMEMBERED_TARGETS)


def handle_target_command(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "EMPTY"
    if raw == ":current":
        return f"CURRENT {get_current_target_name()}"
    if raw == ":list":
        memory = snapshot_remembered_targets()
        if not memory:
            return "LIST EMPTY"
        parts = [f"{k}=({x:.2f},{y:.2f})" for k, (x, y) in memory.items()]
        return "LIST " + "; ".join(parts)

    set_current_target_name(raw)
    print(f"[TARGET] Switched requested target to: {raw}")
    return f"OK {raw}"


def target_control_server_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((SERVER_IP, CONTROL_PORT))
    sock.listen(5)
    print(f"[TARGET] Control listener on {SERVER_IP}:{CONTROL_PORT}")
    while True:
        conn = None
        try:
            conn, addr = sock.accept()
            data = conn.recv(4096)
            if not data:
                continue
            raw = data.decode("utf-8", errors="ignore").strip()
            reply = handle_target_command(raw)
            conn.sendall(reply.encode("utf-8"))
            print(f"[TARGET] Control from {addr}: {raw} -> {reply}")
        except Exception as e:
            print(f"[TARGET] Control socket error: {e}")
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass


def apply_live_target_to_policy(policy_instance) -> Optional[str]:
    requested = get_current_target_name()
    current = find_attribute_recursive(policy_instance, "_target_object")
    if isinstance(current, str) and current == requested:
        return current

    updated = set_attribute_recursive(policy_instance, "_target_object", requested)
    # clear stale direct-goal hints; in explore mode we should use frontiers, not the previous target proxy
    set_attribute_recursive(policy_instance, "server_target_proxy", None)
    set_attribute_recursive(policy_instance, "server_nav_active", False)
    set_attribute_recursive(policy_instance, "server_mode", "explore")
    if updated > 0:
        print(f"[TARGET] Applied live target to policy: {requested}")
    return requested


def get_policy_mode_string(policy_instance):
    v = find_attribute_recursive(policy_instance, "server_mode")
    if isinstance(v, str):
        return v
    return None


def get_nav_active_from_policy(policy_instance, has_goal: bool):
    v = find_attribute_recursive(policy_instance, "server_nav_active")
    if v is not None:
        try:
            return bool(v)
        except Exception:
            pass
    return bool(has_goal)


def get_maps_from_policy(policy_instance):
    obs_bgr, val_bgr = None, None

    def extract_from_obj(obj):
        o_bgr, v_bgr = None, None
        if hasattr(obj, "_obstacle_map") and obj._obstacle_map is not None:
            try:
                o_bgr = obj._obstacle_map.visualize()
            except Exception as e:
                print(f"Obstacle Map visualize failed: {e}")

        if hasattr(obj, "_value_map") and obj._value_map is not None:
            try:
                obs_map_ref = getattr(obj, "_obstacle_map", None)
                v_bgr = obj._value_map.visualize(obstacle_map=obs_map_ref)
            except Exception as e:
                print(f"Value Map visualize failed: {e}")
        return o_bgr, v_bgr

    try:
        obs_bgr, val_bgr = extract_from_obj(policy_instance)
        if obs_bgr is None and val_bgr is None and hasattr(policy_instance, "policy"):
            obs_bgr, val_bgr = extract_from_obj(policy_instance.policy)
    except Exception as e:
        print(f"Map extraction global error: {e}")

    return obs_bgr, val_bgr


def get_best_frontier_goal(policy_instance):
    try:
        nav_active = get_nav_active_from_policy(policy_instance, False)
        target = None
        if nav_active:
            target = find_attribute_recursive(policy_instance, "server_target_proxy")
        if target is None:
            target = find_attribute_recursive(policy_instance, "_last_frontier")
        xy = to_xy_tuple(target)
        if xy is not None and (abs(xy[0]) > 0.01 or abs(xy[1]) > 0.01):
            return True, xy[0], xy[1]
    except Exception as e:
        print(f"Error extracting frontier: {e}")
    return False, 0.0, 0.0


def extract_target_metadata(policy_instance, nav_goal_xy=None):
    """
    Best-effort adapter for different VLFM forks.
    Exposes target/object metadata for the richer Windows UI.
    """
    requested_target_name = get_current_target_name()
    meta = {
        "target_name": requested_target_name,
        "detected": False,
        "goal_xy": list(nav_goal_xy) if nav_goal_xy is not None else None,
        "object_xy": None,
        "score": 0.0,
        "details": {},
        "nav_active": False,
    }

    flag_attrs = [
        "_found_goal", "found_goal", "_object_found", "object_found",
        "_goal_found", "goal_found", "_found_object", "found_object",
        "_target_found", "target_found",
    ]
    for name in flag_attrs:
        v = find_attribute_recursive(policy_instance, name)
        if v is not None:
            try:
                meta["details"]["flag_attr"] = name
                meta["detected"] = bool(v)
                break
            except Exception:
                pass

    xy_attrs = [
        "_last_found_goal", "_last_goal", "_goal_xy", "goal_xy",
        "_last_found_object_xy", "_last_object_xy", "target_xy", "object_xy",
    ]
    for name in xy_attrs:
        v = find_attribute_recursive(policy_instance, name)
        xy = to_xy_tuple(v)
        if xy is not None:
            meta["object_xy"] = [xy[0], xy[1]]
            meta["details"]["xy_attr"] = name
            if not meta["detected"]:
                meta["detected"] = True
            break

    score_attrs = ["goal_score", "target_score", "object_score", "_goal_score", "_target_score"]
    for name in score_attrs:
        v = find_attribute_recursive(policy_instance, name)
        if v is not None:
            try:
                if torch.is_tensor(v):
                    v = float(v.detach().cpu().numpy().flatten()[0])
                else:
                    v = float(np.array(v).flatten()[0])
                if np.isfinite(v):
                    meta["score"] = v
                    meta["details"]["score_attr"] = name
                    break
            except Exception:
                pass

    if nav_goal_xy is not None:
        meta["details"]["nav_goal_source"] = "server_goal_packet"
    memory = snapshot_remembered_targets()
    if requested_target_name in memory:
        rx, ry = memory[requested_target_name]
        meta["details"]["remembered_goal_xy"] = [rx, ry]
    meta["details"]["target_object_id"] = TARGET_OBJECT_ID
    meta["details"]["source"] = "VLFM heuristic adapter"
    return meta


# =========================
# Main loop
# =========================
threading.Thread(target=target_control_server_loop, daemon=True).start()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(f"Server listening on {SERVER_PORT}...")

while True:
    print("Waiting for connection...")
    conn, addr = sock.accept()
    print(f"Bridge connected: {addr}")

    if hasattr(policy, "net") and hasattr(policy.net, "num_recurrent_layers"):
        rnn_hidden_states = torch.zeros(1, policy.net.num_recurrent_layers, 512, device=DEVICE)
    else:
        rnn_hidden_states = torch.zeros(1, 4, 512, device=DEVICE)
    prev_actions = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
    masks = torch.zeros(1, 1, device=DEVICE, dtype=torch.bool)
    object_goal_tensor = torch.tensor([[TARGET_OBJECT_ID]], device=DEVICE, dtype=torch.long)

    try:
        while True:
            header = recv_all(conn, 4)
            if not header:
                break
            rgb_size = struct.unpack("I", header)[0]
            rgb_bytes = recv_all(conn, rgb_size)
            rgb_np = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

            header = recv_all(conn, 4)
            if not header:
                break
            depth_size = struct.unpack("I", header)[0]
            depth_bytes = recv_all(conn, depth_size)
            depth_img = Image.open(io.BytesIO(depth_bytes))
            depth_mm_np = np.array(depth_img).astype(np.float32)

            state_bytes = recv_all(conn, 12)
            if not state_bytes:
                break
            x, y, yaw = struct.unpack("fff", state_bytes)

            apply_live_target_to_policy(policy)

            rgb_tensor, depth_tensor = preprocess_obs_vlfm_style(rgb_np, depth_mm_np, DEVICE)
            batch = TensorDict({
                "rgb": rgb_tensor.unsqueeze(0),
                "depth": depth_tensor.unsqueeze(0),
                "objectgoal": object_goal_tensor,
                "gps": torch.tensor([[x, y]], device=DEVICE, dtype=torch.float32),
                "compass": torch.tensor([[yaw]], device=DEVICE, dtype=torch.float32),
                "heading": torch.tensor([[yaw]], device=DEVICE, dtype=torch.float32),
            })

            discrete_val = 0
            with torch.no_grad():
                action_data = policy.act(batch, rnn_hidden_states, prev_actions, masks, deterministic=True)
                if isinstance(action_data, tuple):
                    actions_tensor, rnn_hidden_states = action_data
                    discrete_val = int(actions_tensor[0].item())
                else:
                    rnn_hidden_states = action_data.rnn_hidden_states
                    if hasattr(action_data, "actions"):
                        discrete_val = int(action_data.actions[0].item())
                prev_actions.copy_(torch.tensor([[discrete_val]], device=DEVICE, dtype=torch.long))
                masks.fill_(True)

            has_goal, tx, ty = get_best_frontier_goal(policy)
            final_action = discrete_val
            nav_active = get_nav_active_from_policy(policy, has_goal)
            if has_goal:
                final_action = 1
                print(f"Use Best Frontier: ({tx:.2f}, {ty:.2f})")
            else:
                tx, ty = 0.0, 0.0
                print(f"Init Mode (Spinning): Action {discrete_val}")

            current_target_name = get_current_target_name()
            if nav_active and has_goal:
                remember_target(current_target_name, (float(tx), float(ty)))

            obs_bgr, val_bgr = get_maps_from_policy(policy)
            obs_bytes = b""
            if obs_bgr is not None:
                _, buf = cv2.imencode(".png", obs_bgr)
                obs_bytes = buf.tobytes()

            val_bytes = b""
            if val_bgr is not None:
                _, buf = cv2.imencode(".png", val_bgr)
                val_bytes = buf.tobytes()

            nav_goal_xy = (float(tx), float(ty)) if has_goal else None
            obj_meta = extract_target_metadata(policy, nav_goal_xy=nav_goal_xy)
            obj_meta["nav_active"] = nav_active

            mode_value = get_policy_mode_string(policy)
            if isinstance(mode_value, str):
                obj_meta.setdefault("details", {})
                obj_meta["details"]["policy_mode"] = mode_value

            obj_meta_json = json.dumps(obj_meta, ensure_ascii=False).encode("utf-8")

            # Reply protocol:
            # [goal_packet 10B][obs_len][obs_png][val_len][val_png][obj_meta_len][obj_meta_json]
            packet = struct.pack("<BBff", final_action, int(has_goal), float(tx), float(ty))
            conn.sendall(packet)
            conn.sendall(struct.pack("I", len(obs_bytes)) + obs_bytes)
            conn.sendall(struct.pack("I", len(val_bytes)) + val_bytes)
            conn.sendall(struct.pack("I", len(obj_meta_json)) + obj_meta_json)

    except Exception as e:
        print(f"Connection Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass
