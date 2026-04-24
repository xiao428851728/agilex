# ========= test_run.py：在无 Habitat / 无 PointNav ckpt 下，用虚拟数据测试 VLFM 输出 action =========

import torch
import numpy as np
from omegaconf import OmegaConf

from habitat_baselines.common.tensor_dict import TensorDict

# 先引入 PointNav 相关模块，用于 monkey patch
import vlfm.policy.utils.pointnav_policy as pointnav_utils
import vlfm.policy.base_objectnav_policy as base_obj


# ========= 1️⃣ 完全替换 PointNav 相关内容：Dummy WrappedPointNavResNetPolicy =========

class DummyActionDistribution:
    def __init__(self):
        pass


class DummyPointNavPolicy:
    """
    伪造的 PointNav 策略：
    - 具有 to / eval / act / action_distribution 接口
    - act 固定返回 “前进” 动作 ID = 1
    """
    def __init__(self):
        self.action_distribution = DummyActionDistribution()

    def to(self, device):
        return self

    def eval(self):
        return self

    def act(self, *args, **kwargs):
        # 固定返回“前进”动作
        return torch.tensor([[1]], dtype=torch.long)


class DummyWrappedPointNavResNetPolicy:
    """
    用来替换原始 WrappedPointNavResNetPolicy，避免访问真实 PointNav 的 net 等属性。
    """
    def __init__(self, *args, **kwargs):
        self.policy = DummyPointNavPolicy()
        # 原实现里会用到 num_recurrent_layers，这里给个 0 即可
        self.num_recurrent_layers = 0

    def act(self, *args, **kwargs):
        return self.policy.act(*args, **kwargs)


def dummy_load_pointnav_policy(*args, **kwargs):
    print("⚠️ Dummy PointNav policy injected (bypassing real checkpoint).")
    return DummyPointNavPolicy()


# ✅ Monkey patch：替换 PointNav 加载与封装类
pointnav_utils.load_pointnav_policy = dummy_load_pointnav_policy
pointnav_utils.WrappedPointNavResNetPolicy = DummyWrappedPointNavResNetPolicy
base_obj.WrappedPointNavResNetPolicy = DummyWrappedPointNavResNetPolicy


# 现在再导入 HabitatITMPolicyV3 和 VLFMPolicyConfig
from vlfm.policy.habitat_policies import HabitatITMPolicyV3, VLFMPolicyConfig


# ========= 2️⃣ 生成虚拟 Habitat 风格的观测 =========

def generate_dummy_observations(batch_size: int = 1) -> TensorDict:
    """
    构造一个最小可用的 Habitat 风格 TensorDict 观测：
    - rgb: (B, H, W, 3)
    - depth: (B, H, W, 1)
    - gps, compass, heading: 占位
    - objectgoal: 目标类别 ID，占位 0
    """
    rgb = torch.rand(batch_size, 256, 256, 3)
    depth = torch.rand(batch_size, 256, 256, 1)
    gps = torch.zeros(batch_size, 2)
    compass = torch.zeros(batch_size, 1)
    heading = torch.zeros(batch_size, 1)

    object_goal = torch.zeros(batch_size, 1, dtype=torch.long)

    obs = TensorDict(
        {
            "rgb": rgb,
            "depth": depth,
            "gps": gps,
            "compass": compass,
            "heading": heading,
            "objectgoal": object_goal,
        },
        batch_size=batch_size,
    )

    return obs


# ========= 3️⃣ 自动补齐 VLFM + Habitat 必要配置 =========

def auto_fill_full_config(cfg):

    # ---------- 补齐 VLFM policy ----------
    cfg.setdefault("habitat_baselines", {})
    cfg.habitat_baselines.setdefault("rl", {})
    cfg.habitat_baselines.rl.setdefault("policy", {})
    policy_cfg = cfg.habitat_baselines.rl.policy

    required_keys = VLFMPolicyConfig.kwaarg_names

    default_map = {
        "text_prompt": "chair",
        "visualize": False,
        "max_steps": 500,
        "use_memory": False,
        "confidence_threshold": 0.5,
        "pointnav_policy_path": "",
        "depth_image_shape": (256, 256),
        "obstacle_map_area_threshold": 0.02,
        "agent_radius": 0.2,
        "min_obstacle_height": 0.05,   # 5cm 以上视为障碍
        "max_obstacle_height": 1.0,    # 1m 以下的物体视为可检测障碍
        "coco_threshold": 0.5,
        "non_coco_threshold": 0.4,
    }

    for k in required_keys:
        if k not in policy_cfg:
            policy_cfg[k] = default_map.get(k, None)
            print(f"⚠️ Auto-filled missing policy key: {k} = {policy_cfg[k]}")

    # ---------- 补齐 habitat.simulator ----------
    cfg.setdefault("habitat", {})
    cfg.habitat.setdefault("simulator", {})
    cfg.habitat.simulator.setdefault("agents", {})
    cfg.habitat.simulator.agents.setdefault("main_agent", {})
    cfg.habitat.simulator.agents.main_agent.setdefault("sim_sensors", {})
    sim_sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors

    sim_sensors.setdefault("rgb_sensor", {})
    sim_sensors.rgb_sensor.setdefault("position", [0.0, 1.5, 0.0])
    sim_sensors.rgb_sensor.setdefault("width", 256)
    sim_sensors.rgb_sensor.setdefault("height", 256)

    sim_sensors.setdefault("depth_sensor", {})
    sim_sensors.depth_sensor.setdefault("min_depth", 0.1)
    sim_sensors.depth_sensor.setdefault("max_depth", 10.0)
    sim_sensors.depth_sensor.setdefault("hfov", 90.0)
    sim_sensors.depth_sensor.setdefault("width", 256)

    print("✅ Dummy habitat.simulator config injected successfully.")

    # ---------- 补齐 habitat_baselines.eval.video_option ----------
    cfg.habitat_baselines.setdefault("eval", {})
    cfg.habitat_baselines.eval.setdefault("video_option", [])

    print("✅ Dummy habitat_baselines.eval.video_option injected successfully.")

    # ---------- 补齐 habitat.dataset ----------
    cfg.habitat.setdefault("dataset", {})
    cfg.habitat.dataset.setdefault("data_path", "data/hm3d_dummy")

    print("✅ Dummy habitat.dataset.data_path injected successfully.")


# ========= 4️⃣ 加载 VLFM Policy =========

def load_policy(cfg_path: str, ckpt_path: str, device: str):

    cfg = OmegaConf.load(cfg_path)

    auto_fill_full_config(cfg)

    # 这里会使用我们已经 monkey patch 的 Dummy WrappedPointNavResNetPolicy
    policy = HabitatITMPolicyV3.from_config(cfg)

    print("⚠️ Skip loading checkpoint for dummy test.")

    # state_dict = torch.load(ckpt_path, map_location=device)
    # policy.load_state_dict(state_dict, strict=False)

    policy.to(device)
    policy.eval()

    print("✅ VLFM Policy Loaded Successfully")
    return policy


# ========= 5️⃣ 主测试流程 =========

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = "config/experiments/vlfm_objectnav_hm3d.yaml"
    ckpt_path = "data/dummy_policy.pth"

    policy = load_policy(cfg_path, ckpt_path, device)

    rnn_hidden_states = None
    prev_actions = None
    masks = torch.ones(1, 1)

    print("\n✅ Start Dummy VLFM Test\n")

    for step in range(100):
        observations = generate_dummy_observations()

        for k in observations.keys():
            if torch.is_tensor(observations[k]):
                observations[k] = observations[k].to(device)

        with torch.no_grad():
            out = policy.act(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic=True,
            )

        action_id = int(out.actions.item())
        print(f"[Step {step}] Action = {action_id}")

    print("\n✅ Dummy VLFM Test Finished")


if __name__ == "__main__":
    main()
