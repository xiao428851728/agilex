# /home/sxz/sjd/vlfm/vlfm_extention/text_goal_sensor.py
# -*- coding: utf-8 -*-
from typing import Any, Dict
import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes

@registry.register_sensor(name="TextGoalSensor")
class TextGoalSensor(Sensor):
    """
    把 episode.info['goal_text_intrinsic'] 暴露成 obs["text_goal"]（uint8向量，UTF-8截断编码）
    适配较老的 Habitat：实现 _get_* 系列方法；不要用 @property 覆盖 uuid 等属性。
    """
    _uuid = "text_goal"
    _max_len = 512

    def __init__(self, sim, config, *args, **kwargs):
        try:
            super().__init__(sim=sim, config=config, *args, **kwargs)
        except TypeError:
            super().__init__(config=config, *args, **kwargs)
        self._sim = sim

    # ---- 旧接口要求实现的三个虚方法 ----
    def _get_uuid(self, *args, **kwargs) -> str:
        return getattr(self.config, "uuid", self._uuid)

    def _get_observation_space(self, *args, **kwargs) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=(self._max_len,), dtype=np.uint8)

    def _get_sensor_type(self, *args, **kwargs) -> SensorTypes:
        return SensorTypes.TENSOR

    # ---- 实际观测 ----
    def get_observation(self, observations: Dict[str, Any], episode, *args, **kwargs):
        info = getattr(episode, "info", {}) or {}
        txt = str(info.get("goal_text_intrinsic") or "").strip()
        b = txt.encode("utf-8")[: self._max_len]
        out = np.zeros((self._max_len,), dtype=np.uint8)
        if b:
            out[: len(b)] = np.frombuffer(b, dtype=np.uint8)
        return out
