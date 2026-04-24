# # vlfm/test_run1.py
# import torch
# from omegaconf import OmegaConf, DictConfig
# from habitat_baselines.common.tensor_dict import TensorDict

# from vlfm.policy.habitat_policies import HabitatITMPolicyV3


# def generate_dummy_observations():
#     """
#     构造一个最小可运行的 VLFM + PointNav 观测输入
#     """
#     H, W = 256, 256

#     obs = TensorDict({
#         "rgb": torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8),
#         "depth": torch.rand(1, H, W, 1),
#         "gps": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
#         "compass": torch.tensor([0.0], dtype=torch.float32),
#         "heading": torch.tensor([0.0], dtype=torch.float32),
#         "objectgoal": torch.tensor([[0]], dtype=torch.long),  # 示例：chair -> 0
#     })

#     return obs


# def build_minimal_cfg() -> DictConfig:
#     cfg = OmegaConf.create()

#     # ---- Habitat Simulator Dummy 配置 ----
#     cfg.habitat = {
#         "simulator": {
#             "agents": {
#                 "main_agent": {
#                     "sim_sensors": {
#                         "rgb_sensor": {
#                             "position": [0.0, 1.5, 0.0]
#                         },
#                         "depth_sensor": {
#                             "min_depth": 0.1,
#                             "max_depth": 5.0,
#                             "hfov": 90,
#                             "width": 256,
#                         },
#                     }
#                 }
#             }
#         },
#         "dataset": {
#             "data_path": "hm3d_dummy"
#         }
#     }

#     # ---- Habitat Baselines 配置 ----
#     cfg.habitat_baselines = {
#         "eval": {
#             # 非空即可（是否用到 policy_info 已经不影响我们直接从 map 对象可视化）
#             "video_option": ["disk"]
#         },
#         "rl": {
#             "policy": {
#                 "pointnav_policy_path": "/home/sxz/sjd/vlfm/data/pointnav_weights.pth",

#                 "text_prompt": "chair",
#                 "depth_image_shape": [256, 256],
#                 "pointnav_stop_radius": 0.3,
#                 "use_max_confidence": False,
#                 "object_map_erosion_size": 2,
#                 "exploration_thresh": 0.7,
#                 "obstacle_map_area_threshold": 0.02,
#                 "min_obstacle_height": 0.05,
#                 "max_obstacle_height": 0.3,
#                 "hole_area_thresh": 0.01,
#                 "use_vqa": False,
#                 "vqa_prompt": None,
#                 "coco_threshold": 0.2,
#                 "non_coco_threshold": 0.2,
#                 "agent_radius": 0.2,
#             }
#         }
#     }

#     return cfg


# def load_policy(device: str = "cpu"):
#     cfg = build_minimal_cfg()

#     print("✅ 使用真实 PointNav 权重：")
#     print(cfg.habitat_baselines.rl.policy.pointnav_policy_path)

#     policy = HabitatITMPolicyV3.from_config(cfg)
#     policy.to(device)
#     policy.eval()

#     print("✅ VLFM + 真实 PointNav 加载成功")
#     return policy


# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     policy = load_policy(device)

#     print("\n✅ Start REAL PointNav + VLFM Test\n")

#     rnn_hidden_states = None
#     prev_actions = None
#     masks = torch.ones(1, 1).to(device)

#     for step in range(20):
#         observations = generate_dummy_observations()

#         for k in observations.keys():
#             if torch.is_tensor(observations[k]):
#                 observations[k] = observations[k].to(device)

#         with torch.no_grad():
#             out = policy.act(
#                 observations,
#                 rnn_hidden_states,
#                 prev_actions,
#                 masks,
#                 deterministic=True,
#             )

#         action_id = int(out.actions.item())
#         print(f"[Step {step}] Action = {action_id}")

#     print("\n✅ REAL PointNav + VLFM Test Finished")


# if __name__ == "__main__":
#     main()

# 文件路径: vlfm/test_run_vis.py
import torch
from omegaconf import OmegaConf, DictConfig
from habitat_baselines.common.tensor_dict import TensorDict
from vlfm.policy.habitat_policies import HabitatITMPolicyV3

# ===========================
# 1. 生成虚拟观测数据 (测试用)
# ===========================
def generate_dummy_observations():
    H, W = 256, 256
    obs = TensorDict({
        "rgb": torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8),
        "depth": torch.rand(1, H, W, 1),
        "gps": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "compass": torch.tensor([0.0], dtype=torch.float32),
        "heading": torch.tensor([0.0], dtype=torch.float32),
        "objectgoal": torch.tensor([[0]], dtype=torch.long),
    })
    return obs

# ===========================
# 2. 配置函数 (核心修改区域)
# ===========================
def build_minimal_cfg() -> DictConfig:
    cfg = OmegaConf.create()

    # ---- Habitat Simulator 配置 ----
    cfg.habitat = {
        "simulator": {
            "agents": {
                "main_agent": {
                    "sim_sensors": {
                        "rgb_sensor": {
                            # ✅ 真实高度 (假设0.6米，请根据实测微调)
                            "position": [0.0, 0.6, 0.0],
                            
                            # ✅ 必须是 256 (喂给神经网络)
                            "height": 256,
                            "width": 256,
                            
                            # ✅✅✅ 核心修改：FOV 改为 43 度
                            # 原理：640宽对应56度，裁成480宽后对应约43度
                            "hfov": 43, 
                        },
                        "depth_sensor": {
                            # ✅ 忽略脚下噪点 (解决原地打转)
                            "min_depth": 0.5,
                            "max_depth": 4.0,
                            
                            # ✅ 必须和 RGB 一致
                            "height": 256,
                            "width": 256,
                            "hfov": 43,
                        },
                    }
                }
            }
        },
        "dataset": { "data_path": "hm3d_dummy" }
    }

    # ---- VLFM 策略配置 ----
    cfg.habitat_baselines = {
        "eval": { "video_option": ["chair"] },
        "rl": {
            "policy": {
                "pointnav_policy_path": "/home/sxz/sjd/vlfm/data/pointnav_weights.pth",
                
                # ✅ 提示词优化
                "text_prompt": "chair",
                
                # ✅ 必须是 [256, 256] (神经网络只认这个)
                "depth_image_shape": [256, 256],
                
                "pointnav_stop_radius": 0.5,
                "use_max_confidence": False,
                "object_map_erosion_size": 2,
                "exploration_thresh": 0.7,
                "obstacle_map_area_threshold": 0.02,
                "min_obstacle_height": 0.15,
                "max_obstacle_height": 1.0,
                "hole_area_thresh": 0.01,
                "use_vqa": False,
                "vqa_prompt": None,
                "agent_radius": 0.25,

                # ✅ 关键阈值 (解决"看见了不停"的问题)
                "coco_threshold": 0.18,
                "non_coco_threshold": 0.15,
                "stop_score_thresh": 0.18,
            }
        }
    }
    return cfg

# ===========================
# 3. 加载函数
# ===========================
def load_policy(device="cpu"):
    cfg = build_minimal_cfg()
    print("✅ 使用真实 PointNav 权重与 FOV=43 配置")
    policy = HabitatITMPolicyV3.from_config(cfg)
    policy.to(device)
    policy.eval()
    return policy

# ===========================
# 4. 主函数 (补上了!)
# ===========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(device)

    print("\n✅ Start Mock Test (Use vlfm_server.py for Real Robot)\n")
    
    rnn_hidden_states = None
    prev_actions = None
    masks = torch.ones(1, 1).to(device)

    # 模拟运行几步，验证配置是否报错
    for step in range(5):
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
        print(f"Step {step}: Action {out.actions.item()}")

if __name__ == "__main__":
    main()