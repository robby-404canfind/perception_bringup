"""perception_context_builder_node.py — YOLO + Depth + VLM 통합 Context publish 노드.

/perception/detections를 구독하여 /perception/context/raw (JSON)와
/perception/context/summary (한국어 요약)를 publish합니다.
VLM Trigger 조건이 충족되면 최신 RGB 프레임으로 VLM을 호출합니다.
별도의 snapshot request를 받으면 현재 RGB 프레임을 sensor_msgs/Image로 publish합니다.
"""

import json
import time
from threading import Thread

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .perception_trigger import PerceptionTrigger
from .vlm_client import VLMClient


class PerceptionContextBuilderNode(Node):

    def __init__(self):
        super().__init__("perception_context_builder")

        self.declare_parameter("vlm_backend", "ollama")
        self.declare_parameter("vlm_model", "qwen2.5vl:7b")
        self.declare_parameter("min_trigger_interval", 5.0)
        self.declare_parameter("image_topic", "/Tiago_Lite/Astra_rgb/image_color")
        self.declare_parameter("detection_topic", "/perception/detections")
        self.declare_parameter("context_raw_topic", "/perception/context/raw")
        self.declare_parameter("context_summary_topic", "/perception/context/summary")
        self.declare_parameter("snapshot_request_topic", "/perception/snapshot/request")
        self.declare_parameter("snapshot_image_topic", "/perception/snapshot/image")
        self.declare_parameter("snapshot_info_topic", "/perception/snapshot/info")
        self.declare_parameter(
            "system2_debug_image_topic", "/perception/system2/debug_image"
        )

        vlm_backend = self.get_parameter("vlm_backend").value
        vlm_model = self.get_parameter("vlm_model").value
        min_interval = self.get_parameter("min_trigger_interval").value

        self.vlm = VLMClient(backend=vlm_backend, model=vlm_model)
        self.trigger = PerceptionTrigger(min_interval=min_interval)
        self.cv_bridge = CvBridge()

        self.latest_cv_image = None
        self.latest_image_header = None
        self._latest_vlm_result: dict | None = None
        self._latest_vlm_time: float = 0.0

        detection_topic = self.get_parameter("detection_topic").value
        context_raw_topic = self.get_parameter("context_raw_topic").value
        context_summary_topic = self.get_parameter("context_summary_topic").value
        snapshot_request_topic = self.get_parameter("snapshot_request_topic").value
        snapshot_image_topic = self.get_parameter("snapshot_image_topic").value
        snapshot_info_topic = self.get_parameter("snapshot_info_topic").value
        system2_debug_image_topic = self.get_parameter(
            "system2_debug_image_topic"
        ).value

        # 구독
        self.create_subscription(String, detection_topic, self._on_detections, 10)
        image_topic = self.get_parameter("image_topic").value
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(String, snapshot_request_topic, self._on_snapshot_req, 10)

        # publish
        self._pub_raw = self.create_publisher(String, context_raw_topic, 10)
        self._pub_summary = self.create_publisher(String, context_summary_topic, 10)
        self._pub_snap_img = self.create_publisher(Image, snapshot_image_topic, 10)
        self._pub_snap_info = self.create_publisher(String, snapshot_info_topic, 10)
        self._pub_debug_img = self.create_publisher(
            Image, system2_debug_image_topic, 10
        )

        self.get_logger().info(
            f"PerceptionContextBuilder 시작 (VLM={vlm_backend}/{vlm_model})"
        )

    def _on_image(self, msg: Image):
        try:
            self.latest_cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_header = msg.header
        except Exception:
            pass

    def _on_detections(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        objects = data.get("objects", [])

        # VLM Trigger 평가
        trigger_reason = self.trigger.evaluate(objects)
        if trigger_reason and self.latest_cv_image is not None:
            Thread(
                target=self._call_vlm,
                args=(self.latest_cv_image.copy(), trigger_reason),
                daemon=True,
            ).start()

        # context/raw 조립
        vlm_scene = self._latest_vlm_result or {}
        vlm_age = (
            round(time.time() - self._latest_vlm_time, 1)
            if self._latest_vlm_result
            else None
        )

        raw = {
            "frame_w": data.get("frame_w", 640),
            "frame_h": data.get("frame_h", 480),
            "objects": objects,
            "vlm_scene": vlm_scene,
            "vlm_triggered": trigger_reason is not None,
            "vlm_age_sec": vlm_age,
        }
        self._pub_raw.publish(String(data=json.dumps(raw, ensure_ascii=False)))

        # context/summary
        summary = self._to_korean_summary(objects, vlm_scene)
        self._pub_summary.publish(String(data=summary))

        self._publish_debug_image(objects, vlm_scene, vlm_age)

    def _call_vlm(self, cv_image: np.ndarray, reason: str):
        self.get_logger().info(f"VLM 호출: {reason}")
        result = self.vlm.describe_scene(cv_image)
        self._latest_vlm_result = result
        self._latest_vlm_time = time.time()
        self.get_logger().info(
            f"VLM 응답: {result.get('scene_summary', 'N/A')}"
        )

    def _on_snapshot_req(self, msg: String):
        if self.latest_cv_image is None:
            self.get_logger().warn("Snapshot 요청이지만 이미지 없음")
            return

        # 최신 RGB 프레임을 원본 Image 메시지로 publish
        snap_msg = self.cv_bridge.cv2_to_imgmsg(self.latest_cv_image, encoding="bgr8")
        if self.latest_image_header is not None:
            snap_msg.header = self.latest_image_header
        self._pub_snap_img.publish(snap_msg)

        # 스냅샷 정보 publish
        try:
            req_data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            req_data = {}

        info = {
            "snapshot_id": req_data.get("snapshot_id", ""),
            "requester": req_data.get("requester", ""),
            "reason": req_data.get("reason", ""),
            "vlm_scene": self._latest_vlm_result or {},
        }
        self._pub_snap_info.publish(String(data=json.dumps(info, ensure_ascii=False)))

    def _publish_debug_image(
        self, objects: list, vlm_scene: dict, vlm_age: float | None
    ) -> None:
        if self.latest_cv_image is None:
            return

        debug_image = self._build_debug_image(
            self.latest_cv_image,
            objects,
            vlm_scene,
            vlm_age,
        )
        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        if self.latest_image_header is not None:
            debug_msg.header = self.latest_image_header
        self._pub_debug_img.publish(debug_msg)

    def _build_debug_image(
        self,
        cv_image: np.ndarray,
        objects: list,
        vlm_scene: dict,
        vlm_age: float | None,
    ) -> np.ndarray:
        debug_image = np.ascontiguousarray(cv_image.copy())
        h, w = debug_image.shape[:2]

        for obj in objects:
            bbox = obj.get("bbox") or {}
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            bw = int(bbox.get("w", 0))
            bh = int(bbox.get("h", 0))
            if bw <= 0 or bh <= 0:
                continue

            x0 = max(0, min(w - 1, x))
            y0 = max(0, min(h - 1, y))
            x1 = max(0, min(w - 1, x + bw))
            y1 = max(0, min(h - 1, y + bh))
            if x1 <= x0 or y1 <= y0:
                continue

            cv2.rectangle(debug_image, (x0, y0), (x1, y1), (0, 220, 255), 2)
            label = self._object_label(obj)
            self._draw_text(debug_image, label, (x0, max(18, y0 - 6)), scale=0.5)

        vlm_text = self._vlm_overlay_text(vlm_scene, vlm_age)
        self._draw_text(debug_image, vlm_text, (8, 24), scale=0.55)
        return debug_image

    @staticmethod
    def _object_label(obj: dict) -> str:
        obj_id = obj.get("id", "?")
        cls = obj.get("class", "unknown")
        confidence = obj.get("confidence")
        conf_text = (
            f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "?"
        )
        range_m = obj.get("range_m")
        range_text = (
            f"{range_m:.2f}m" if isinstance(range_m, (int, float)) else "range=?"
        )
        return f"id={obj_id} {cls} conf={conf_text} {range_text}"

    @staticmethod
    def _vlm_overlay_text(vlm_scene: dict, vlm_age: float | None) -> str:
        age_text = f"{vlm_age:.1f}s" if isinstance(vlm_age, (int, float)) else "n/a"
        if not vlm_scene:
            return f"VLM age={age_text}: none"

        scene_summary = " ".join(str(vlm_scene.get("scene_summary") or "").split())
        if scene_summary and scene_summary.isascii():
            scene_text = scene_summary
        elif scene_summary:
            scene_text = "scene_summary=available"
        else:
            scene_text = "scene_summary=empty"

        social_hints = vlm_scene.get("social_hints") or []
        hint_types = []
        if isinstance(social_hints, list):
            for hint in social_hints:
                if isinstance(hint, dict) and hint.get("type"):
                    hint_type = " ".join(str(hint["type"]).split())
                    if hint_type.isascii():
                        hint_types.append(hint_type)
        hint_text = ",".join(hint_types[:2]) if hint_types else "none"
        overlay = f"VLM age={age_text}: {scene_text}; hints={hint_text}"
        return overlay if len(overlay) <= 88 else overlay[:85] + "..."

    @staticmethod
    def _draw_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float = 0.5,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text = text if text.isascii() else text.encode("ascii", "ignore").decode()
        x, y = origin
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        h, w = image.shape[:2]
        x = max(0, min(w - tw - 6, x))
        y = max(th + 4, min(h - baseline - 4, y))
        top_left = (max(0, x - 3), max(0, y - th - 5))
        bottom_right = (min(w - 1, x + tw + 3), min(h - 1, y + baseline + 3))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _to_korean_summary(objects: list, vlm_scene: dict) -> str:
        parts = []
        for obj in objects:
            obj_id = obj.get("id", "?")
            cls = obj.get("class", "unknown")
            direction = obj.get("direction", "?")
            range_m = obj.get("range_m")
            range_str = f"{range_m}m" if range_m is not None else "거리 미확인"
            parts.append(f"{obj_id}번 {cls}, 화면 {direction}, {range_str}")

        scene_summary = vlm_scene.get("scene_summary")
        if scene_summary:
            parts.append(f"VLM: {scene_summary}")

        return " / ".join(parts) if parts else "감지된 객체 없음"


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionContextBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
