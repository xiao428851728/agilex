import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from habitat_baselines.common.tensor_dict import TensorDict

from vlfm.policy.habitat_policies import HabitatITMPolicyV3


# ===========================
# 1. 生成虚拟观测数据
# ===========================

def generate_dummy_observations():
    """
    构造一个最小可运行的 VLFM + PointNav 观测输入
    """
    H, W = 256, 256

    obs = TensorDict({
        "rgb": torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8),
        "depth": torch.rand(1, H, W,1),
        "gps": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "compass": torch.tensor([0.0], dtype=torch.float32),
        "heading": torch.tensor([0.0], dtype=torch.float32),

        # ObjectNav 目标：chair → id=0
        "objectgoal": torch.tensor([[0]], dtype=torch.long),
    })

    return obs


# ===========================
# 2. 构造最小 VLFM + Habitat 配置
# ===========================

def build_minimal_cfg() -> DictConfig:
    cfg = OmegaConf.create()

    # ---- Habitat Simulator Dummy 配置 ----
    cfg.habitat = {
        "simulator": {
            "agents": {
                "main_agent": {
                    "sim_sensors": {
                        "rgb_sensor": {
                            "position": [0.0, 1.5, 0.0]
                        },
                        "depth_sensor": {
                            "min_depth": 0.1,
                            "max_depth": 5.0,
                            "hfov": 90,
                            "width": 256,
                        },
                    }
                }
            }
        },
        "dataset": {
            "data_path": "hm3d_dummy"
        }
    }

    # ---- Habitat Baselines 配置 ----
    cfg.habitat_baselines = {
        "eval": {
            "video_option": []
        },
        "rl": {
            "policy": {
                # ✅ 真实 PointNav 权重路径（你已验证）
                "pointnav_policy_path": "/home/sxz/sjd/vlfm/data/pointnav_weights.pth",

                # ✅ VLFM 关键参数（补齐缺省）
                "text_prompt": "chair",
                "depth_image_shape": [256, 256],
                "pointnav_stop_radius": 0.3,
                "use_max_confidence": False,
                "object_map_erosion_size": 2,
                "exploration_thresh": 0.7,
                "obstacle_map_area_threshold": 0.02,
                "min_obstacle_height": 0.05,
                "max_obstacle_height": 1.0,
                "hole_area_thresh": 0.01,
                "use_vqa": False,
                "vqa_prompt": None,
                "coco_threshold": 0.2,
                "non_coco_threshold": 0.2,
                "agent_radius": 0.2,
            }
        }
    }

    return cfg


# ===========================
# 3. 加载 VLFM + 真实 PointNav
# ===========================

def load_policy(device="cpu"):
    cfg = build_minimal_cfg()

    print("✅ 使用真实 PointNav 权重：")
    print(cfg.habitat_baselines.rl.policy.pointnav_policy_path)

    policy = HabitatITMPolicyV3.from_config(cfg)
    policy.to(device)
    policy.eval()

    print("✅ VLFM + 真实 PointNav 加载成功")
    return policy


# ===========================
# 4. 主测试流程
# ===========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = load_policy(device)

    print("\n✅ Start REAL PointNav + VLFM Test\n")

    rnn_hidden_states = None
    prev_actions = None
    masks = torch.ones(1, 1)

    for step in range(20):
        observations = generate_dummy_observations()

        # ✅ 只对 Tensor 执行 .to(device)
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

    print("\n✅ REAL PointNav + VLFM Test Finished")


# ===========================
# 5. 入口
# ===========================

if __name__ == "__main__":
    main()

