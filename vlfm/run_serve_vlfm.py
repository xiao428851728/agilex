import socket
import struct
import cv2
import numpy as np
import torch
import hydra
import os
import sys
from omegaconf import DictConfig

# ==============================================================================
# 1. 路径修复核心代码
# ==============================================================================
# 获取当前脚本所在的绝对路径 (即 ~/sjd/vlfm/)
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出真实的 config 路径 (即 ~/sjd/vlfm/config)
config_path = os.path.join(curr_dir, "config")

# 双重检查：如果找不到，报错提示
if not os.path.exists(config_path):
    print(f"❌ Critical Error: Config directory NOT found at: {config_path}")
    print("   Please make sure you are running this script from the project root.")
    sys.exit(1)
else:
    print(f"✅ Config path resolved: {config_path}")

# ==============================================================================
# 2. 导入模块
# ==============================================================================
from vlfm.reality.robots.remote_robot import RemoteRobot 
from vlfm.reality.objectnav_env import ObjectNavEnv
from vlfm.policy.reality_policies import RealityITMPolicyV2

# 注册必要模块
import vlfm.policy.habitat_policies
import vlfm.mapping.frontier_map
import vlfm.mapping.obstacle_map
import vlfm.mapping.value_map
import vlfm.measurements.traveled_stairs

# ==============================================================================
# 3. 主程序
# ==============================================================================
@hydra.main(version_base=None, config_path=config_path, config_name="experiments/reality")
def main(cfg: DictConfig) -> None:
    print("\n🚀 Starting VLFM Server...")
    print(f"   Target Goal: {cfg.env.goal}")
    
    # 启动 Socket
    HOST = "0.0.0.0"
    PORT = 9000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"⏳ Waiting for connection on {HOST}:{PORT}...")
        
        conn, addr = server_socket.accept()
        print(f"✅ Connected to bridge: {addr}")
        
        # 初始化远程机器人
        robot = RemoteRobot(conn)

        # 初始化环境
        print("🌍 Initializing Environment...")
        env = ObjectNavEnv(
            robot=robot,
            max_body_cam_depth=cfg.env.max_body_cam_depth,
            max_gripper_cam_depth=5.0,
        )
        
        # 初始化策略
        print("🧠 Loading Policy (RealityITMPolicyV2)...")
        policy = RealityITMPolicyV2.from_config(cfg)
        policy.reset()

        # 等待第一帧
        print("⏳ Waiting for first frame from robot...")
        if not robot.sync_data():
            print("❌ Failed to receive first frame. Closing.")
            return

        # 开始导航
        print("🏁 Ready! Navigation loop starting.")
        obs = env.reset(goal=cfg.env.goal)
        
        step_count = 0
        while True:
            # A. 策略计算
            mask = torch.tensor([step_count != 0], dtype=torch.bool, device="cuda")
            action_dict = policy.act(obs, None, None, mask)
            
            # B. 动作解析
            lin = action_dict.get("linear", 0.0)
            ang = action_dict.get("angular", 0.0)
            
            # 简单映射为离散动作 (0:Stop, 1:Fwd, 2:Left, 3:Right)
            action_id = 0
            if lin > 0.05: 
                action_id = 1
            elif ang > 0.1: # 左转
                action_id = 2
            elif ang < -0.1: # 右转
                action_id = 3
            
            print(f"Step {step_count}: Lin={lin:.2f}, Ang={ang:.2f} -> ActionID={action_id}")

            # C. 发送动作 (只发 1 字节，防止粘包)
            try:
                conn.sendall(struct.pack("B", action_id))
            except BrokenPipeError:
                print("❌ Connection lost.")
                break

            # D. 等待下一帧同步
            if not robot.sync_data():
                print("❌ Sync failed.")
                break
            
            # E. 更新观测
            obs = env._get_obs()
            step_count += 1

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals(): conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()