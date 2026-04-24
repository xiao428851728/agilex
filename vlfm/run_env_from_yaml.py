# run_env_from_yaml.py
import argparse
from omegaconf import OmegaConf
from habitat import Env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="/home/sxz/sjd/vlfm/config/pointnav_textinst.yaml",
        help="",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)  

    env = Env(cfg)
    obs = env.reset()
    ep  = env.current_episode

    info = getattr(ep, "info", {}) or {}
    goal_text = (info.get("goal_text") or "").strip()

    print("\n=== Current Episode ===")
    print("Episode ID :", getattr(ep, "episode_id", None))
    print("Scene ID   :", getattr(ep, "scene_id", None))
    print("Start pos  :", getattr(ep, "start_position", None))
    print("Goal pos   :", getattr(ep.goals[0], "position", None))
    print("Text goal  :", goal_text if goal_text else "<EMPTY>")
    try:
        print("Measurements:", list(env._task.measurements.measures.keys()))
        print("Metrics:", env.get_metrics())
    except Exception:
        pass

    env.close()

if __name__ == "__main__":
    main()
