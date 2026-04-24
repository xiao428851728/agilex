# # -*- coding: utf-8 -*-
# import os, glob, json, gzip
# from types import SimpleNamespace
# from typing import Any, Dict, List, Optional

# from habitat.core.registry import registry
# from habitat.core.dataset import Dataset
# from habitat.core.dataset import Episode as CoreEpisode

# def _jload(path: str) -> Dict[str, Any]:
#     if path.endswith(".gz"):
#         with gzip.open(path, "rt", encoding="utf-8") as f:
#             return json.load(f)
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def _as_str(x) -> str:
#     return "" if x is None else str(x)

# def _safe_get_goal_position(scene_blob: Dict[str, Any], join_key: str) -> Optional[List[float]]:
#     goals_map = scene_blob.get("goals", {})
#     g = goals_map.get(join_key)
#     if isinstance(g, dict):
#         pos = g.get("position")
#         if isinstance(pos, list) and len(pos) == 3:
#             return pos
#     return None

# @registry.register_dataset(name="TextInstNav-v1")
# class TextInstNavDataset(Dataset):
#     def __init__(self, config=None):
#         self.config = config
#         self.episodes: List[CoreEpisode] = []

#         data_path = getattr(self.config, "DATA_PATH", None) or getattr(self.config, "data_path", None)
#         if not data_path:
#             raise RuntimeError("[TextInstNavDataset] config.DATA_PATH is required")
#         split_dir = os.path.dirname(data_path) if str(data_path).endswith((".json", ".json.gz")) else str(data_path)

#         val_text_path = os.path.join(split_dir, "val_text.json")
#         if not os.path.exists(val_text_path):
#             val_text_path += ".gz"
#         if not os.path.exists(val_text_path):
#             raise FileNotFoundError(f"[TextInstNavDataset] Not found val_text.json(.gz): {val_text_path}")
#         meta = _jload(val_text_path)
#         self.attribute_data: Dict[str, Dict[str, Any]] = meta.get("attribute_data", {})

#         content_dir = os.path.join(split_dir, "content")
#         if not os.path.isdir(content_dir):
#             raise FileNotFoundError(f"[TextInstNavDataset] Not found content dir: {content_dir}")
#         files = sorted(glob.glob(os.path.join(content_dir, "*.json")) + glob.glob(os.path.join(content_dir, "*.json.gz")))
#         if not files:
#             raise FileNotFoundError(f"[TextInstNavDataset] No scene json under {content_dir}")

#         for fp in files:
#             scene_key = os.path.splitext(os.path.basename(fp))[0]
#             scene_blob = _jload(fp)
#             raw_eps = scene_blob.get("episodes", [])
#             if not isinstance(raw_eps, list):
#                 continue

#             for idx, e in enumerate(raw_eps):
#                 # 必填字段
#                 ep_id = _as_str(e.get("episode_id", f"{scene_key}-{idx:06d}"))
#                 scene_id = e.get("scene_id")
#                 if not scene_id:
#                     raise ValueError(f"[TextInstNav] episode missing scene_id (scene={scene_key}, ep_idx={idx})")
#                 start_pos = e.get("start_position", [0.0, 0.0, 0.0])
#                 start_rot = e.get("start_rotation", [1.0, 0.0, 0.0, 0.0])

#                 ep = CoreEpisode(
#                     episode_id=ep_id,
#                     scene_id=scene_id,
#                     start_position=start_pos,
#                     start_rotation=start_rot,
#                 )

#                 # 其它可选字段
#                 if "scene_dataset_config" in e:
#                     setattr(ep, "scene_dataset_config", e["scene_dataset_config"])
#                 if "object_category" in e:
#                     setattr(ep, "object_category", e["object_category"])

#                 # 若 episode 自带 goals 列表，尽量映射成可访问对象
#                 if isinstance(e.get("goals"), list):
#                     goals_list = []
#                     for g in e["goals"]:
#                         if isinstance(g, dict):
#                             goals_list.append(SimpleNamespace(**g))
#                     if goals_list:
#                         setattr(ep, "goals", goals_list)

#                 # join & 文本合并 & 最小 goal
#                 tid = e.get("goal_object_id") or e.get("target_id") or e.get("goal_view_id") or e.get("instance_id")
#                 info = (getattr(ep, "info", {}) or {})
#                 if tid is not None:
#                     tid = _as_str(tid)
#                     join_key = f"{scene_key}_{tid}"
#                     info["attr_join_key"] = join_key

#                     a = self.attribute_data.get(join_key)
#                     if a:
#                         info["goal_text_intrinsic"] = a.get("intrinsic_attributes", "")
#                         info["goal_text_extrinsic"]  = a.get("extrinsic_attributes", "")
#                         img = a.get("image")
#                         if img:
#                             info["goal_image"] = img

#                     need_goal = not hasattr(ep, "goals") or not getattr(ep, "goals", None)
#                     if need_goal:
#                         pos = _safe_get_goal_position(scene_blob, join_key)
#                         if pos is not None:
#                             minimal_goal = SimpleNamespace(position=pos, object_id=tid)
#                             setattr(ep, "goals", [minimal_goal])
#                 else:
#                     info["attr_join_key"] = None

#                 if not getattr(ep, "object_category", None):
#                     ep.object_category = "textinst"

#                 ep.info = info
#                 self.episodes.append(ep)

#     @classmethod
#     def get_scenes_to_load(cls, config):
#         return None

# -*- coding: utf-8 -*-
import os, json, gzip, glob
from typing import List, Dict, Any
import numpy as np

from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry

def _load_json_maybe_gz(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 仅携带 position 的轻量目标（替代官方 PointGoal）
class _PointGoalLike:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)

def _infer_scene_id(scene_key: str, scenes_dir: str) -> str:
    """
    把 content 文件名里的 <scene_key>（例: 4ok3usBNeis）
    转成 HM3D annotated dataset 下的 scene_id（相对 scene dataset 的路径）：
      hm3d_v0.2/val/<scene_key>/<scene_key>.basis.glb
    注意：这里返回“相对”路径；真正的 dataset config 在 YAML 的 simulator.habitat_sim_v0.scene_dataset 指定。
    """
    return f"hm3d_v0.2/val/{scene_key}/{scene_key}.basis.glb"

def _best_text_for_goal(scene_key: str, obj_name: str, attr_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    从 val_text.json.gz 的 attribute_data 里挑一条最相关的文本（intrinsic/extrinsic）
    规则：优先 scene_id 命中的图片名（前缀 4ok3usBNeis_*），再按包含 object 关键词做个简易打分。
    """
    obj_kw = (obj_name or "").split("_")[0].lower()
    cands = []
    for k, v in attr_map.items():
        if not k.startswith(scene_key + "_"):
            continue
        img = str(v.get("image", "")).lower()
        score = (2 if obj_kw and obj_kw in img else 0) - len(img) * 1e-4
        cands.append((score, v))
    if not cands:
        # 退化：取任何同场景 id 的
        for k, v in attr_map.items():
            if k.startswith(scene_key + "_"):
                cands.append((0.0, v))
    if not cands:
        return {"intrinsic": "", "extrinsic": ""}

    v = sorted(cands, key=lambda x: -x[0])[0][1]
    return {
        "intrinsic": v.get("intrinsic_attributes", "") or "",
        "extrinsic": v.get("extrinsic_attributes", "") or "",
    }

def _cfg_get(obj, key: str, default=None):
    # 兼容 omegaconf.DictConfig / dict
    try:
        if hasattr(obj, key):
            return getattr(obj, key)
    except Exception:
        pass
    try:
        return obj.get(key, default)
    except Exception:
        return default

@registry.register_dataset(name="TextInstNav-v1")
class TextInstNavDatasetV1(Dataset):
    """
    把 instancenav/<split>/content/*.json 与 val_text.json.gz 拼成带 position + 文本描述的 episodes
    - 每条 episode:
        .scene_id   -> hm3d_v0.2/val/<scene>/<scene>.basis.glb  （相对 scene dataset）
        .start_position / .start_rotation
        .goals[0].position
        .info["goal_text_intrinsic"/"goal_text_extrinsic"/"object_name"/"object_category"]
    """

    episodes: List[Episode]

    def __init__(self, config=None, **kwargs):
        # 注意：父类没有自定义 __init__，不要 super().__init__
        self.config = config if config is not None else kwargs.get("config", {})
        self.episodes = []

        # data_path 指向 split 根目录（里面需要有 content/ 和 val_text.json.gz）
        split_root = str(_cfg_get(self.config, "data_path", "")) or ""
        if not split_root:
            raise ValueError("[TextInstNavDatasetV1] config.data_path is empty")

        if not os.path.isdir(split_root):
            split_root = os.path.dirname(split_root)

        content_dir   = os.path.join(split_root, "content")
        val_text_path = os.path.join(split_root, "val_text.json.gz")
        scenes_dir    = str(_cfg_get(self.config, "scenes_dir", "")) or ""

        if not scenes_dir or not os.path.isdir(scenes_dir):
            raise ValueError(f"[TextInstNavDatasetV1] invalid scenes_dir: {scenes_dir}")

        # 文本库（可选）
        attr_map: Dict[str, Dict[str, str]] = {}
        if os.path.exists(val_text_path):
            vt = _load_json_maybe_gz(val_text_path)
            if isinstance(vt, dict):
                attr_map = (vt.get("attribute_data") or {}) if isinstance(vt.get("attribute_data"), dict) else {}
        else:
            print(f"[TextInstNavDatasetV1] warn: {val_text_path} not found; goal text will be empty.")

        files = sorted(glob.glob(os.path.join(content_dir, "*.json"))) + \
                sorted(glob.glob(os.path.join(content_dir, "*.json.gz")))

        if not files:
            print(f"[TextInstNavDatasetV1] warn: no files under {content_dir}")

        built = 0
        for fp in files:
            try:
                data = _load_json_maybe_gz(fp)
            except Exception as e:
                print(f"[TextInstNavDatasetV1] skip {fp}: load error: {e}")
                continue

            goals_blob = data.get("goals")
            if not goals_blob:
                print(f"[TextInstNavDatasetV1] {fp}: no 'goals'; skip")
                continue

            if isinstance(goals_blob, dict):
                goals = list(goals_blob.values())
            elif isinstance(goals_blob, list):
                goals = goals_blob
            else:
                print(f"[TextInstNavDatasetV1] {fp}: goals must be list/dict; skip")
                continue

            scene_key = os.path.splitext(os.path.basename(fp))[0]
            scene_id  = _infer_scene_id(scene_key, scenes_dir)

            for gi, g in enumerate(goals):
                # 目标位置：优先 "position"/"point"
                pos = g.get("position") or g.get("point")
                if pos is None:
                    print(f"[TextInstNavDatasetV1] {fp}#{gi}: missing POINT goal; skip")
                    continue
                if isinstance(pos, dict) and "x" in pos:
                    pos = [pos["x"], pos["y"], pos["z"]]
                pos = list(map(float, pos))

                # 起点：优先 view_points[0].agent_state.position；否则在目标附近偏移
                start = None
                vps = g.get("view_points") or []
                if isinstance(vps, list) and vps:
                    st = (vps[0] or {}).get("agent_state") or {}
                    start = st.get("position")
                if start is None:
                    start = [pos[0] + 0.5, pos[1], pos[2] + 0.5]
                start = list(map(float, start))
                start_quat = [0.0, 0.0, 0.0, 1.0]

                obj_name = str(g.get("object_name", "") or "")
                txt = _best_text_for_goal(scene_key, obj_name, attr_map)

                info_dict = {
                    "goal_text_intrinsic": txt.get("intrinsic", ""),
                    "goal_text_extrinsic": txt.get("extrinsic", ""),
                    "object_name": obj_name,
                    "object_category": g.get("object_category", "") or "",
                }

                # Episode：先用最兼容的构造器创建，再补字段
                try:
                    ep = Episode(
                        episode_id=f"{scene_key}_{gi}",
                        scene_id=scene_id,
                        start_position=start,
                        start_rotation=start_quat,
                        info=info_dict,
                    )
                except TypeError:
                    ep = Episode(
                        episode_id=f"{scene_key}_{gi}",
                        scene_id=scene_id,
                        start_position=start,
                    )
                    try: ep.start_rotation = start_quat
                    except Exception: pass
                    try: ep.info = info_dict
                    except Exception: pass

                # HABITAT 的 DistanceToGoal 只要求 .position
                ep.goals = [_PointGoalLike(pos)]
                self.episodes.append(ep)
                built += 1

        if built == 0:
            print("[TextInstNavDatasetV1] no episodes built; check your content/ and val_text.json.gz")
