"""perception_trigger.py — 조건부 VLM 호출 Trigger 로직.

YOLO 탐지 결과를 평가하여 VLM을 호출해야 하는지 결정합니다.
현재 구현된 조건은 ACTION_REQUEST, PROXIMITY, NEW_OBJECT입니다.
"""

import time


class PerceptionTrigger:

    def __init__(self, min_interval: float = 5.0, proximity_threshold: float = 3.0):
        self.min_interval = min_interval
        self.proximity_threshold = proximity_threshold
        self._last_vlm_time = 0.0
        self._prev_ids: set = set()

    def evaluate(
        self, detections: list, action_request: str | None = None
    ) -> str | None:
        """VLM Trigger 조건을 평가합니다.

        반환값이 None이 아니면 VLM을 호출합니다.
        ACTION_REQUEST는 Rate Limit을 우회합니다.
        """
        now = time.time()

        # 1. ACTION_REQUEST — Rate Limit 우회
        if action_request:
            self._last_vlm_time = now
            return f"ACTION_REQUEST:{action_request}"

        # Rate Limit
        if (now - self._last_vlm_time) < self.min_interval:
            return None

        # 2. PROXIMITY — 가까운 사람
        for det in detections:
            if (
                det.get("class") == "person"
                and det.get("range_m") is not None
                and det["range_m"] < self.proximity_threshold
            ):
                self._last_vlm_time = now
                return f"PROXIMITY:person@{det['range_m']:.1f}m"

        # 3. NEW_OBJECT — 새로운 tracking ID
        current_ids = {det.get("id", -1) for det in detections if det.get("id", -1) >= 0}
        new_ids = current_ids - self._prev_ids
        self._prev_ids = current_ids
        if new_ids:
            self._last_vlm_time = now
            return f"NEW_OBJECT:ids={new_ids}"

        return None
