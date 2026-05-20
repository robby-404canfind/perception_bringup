"""perception_cache.py — 스레드 안전 비전 상태 캐시.

/perception/detections를 구독하는 쪽이 update_from_msg()를 호출하고,
scan/find/follow 액션이 snapshot()으로 최신 상태를 읽습니다.
"""

import json
import threading


class PerceptionCache:

    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._lock = threading.RLock()
        self._targets: list = []
        self._frame_w = frame_w
        self._frame_h = frame_h

    def update_from_msg(self, msg_data: str, logger=None):
        """JSON String 메시지에서 상태를 업데이트합니다."""
        try:
            data = json.loads(msg_data) if msg_data else {}
            with self._lock:
                self._targets = data.get("objects", [])
                self._frame_w = int(data.get("frame_w", self._frame_w))
                self._frame_h = int(data.get("frame_h", self._frame_h))
        except Exception as e:
            if logger:
                logger.warn(f"[PerceptionCache] parse failed: {e}")

    def snapshot(self) -> dict:
        """현재 비전 상태의 스냅샷을 반환합니다."""
        with self._lock:
            return {
                "targets": list(self._targets),
                "frame_w": self._frame_w,
                "frame_h": self._frame_h,
            }
