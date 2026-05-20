"""perception_context_builder_node.py — YOLO + Depth + VLM 통합 Context publish 노드.

/perception/detections를 구독하여 /perception/context/raw (JSON)와
/perception/context/summary (한국어 요약)를 publish합니다.
VLM Trigger 조건이 충족되면 snapshot을 캡처하고 VLM을 호출합니다.
"""

import json
import time
from threading import Thread

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from .perception_trigger import PerceptionTrigger
from .vlm_client import VLMClient


class PerceptionContextBuilderNode(Node):

    def __init__(self):
        super().__init__("perception_context_builder")

        self.declare_parameter("vlm_backend", "ollama")
        self.declare_parameter("vlm_model", "qwen2.5vl:7b")
        self.declare_parameter("vlm_mock_mode", False)
        self.declare_parameter("min_trigger_interval", 5.0)
        self.declare_parameter("image_topic", "/Tiago_Lite/Astra_rgb/image_color")
        self.declare_parameter("detection_topic", "/perception/detections")
        self.declare_parameter("context_raw_topic", "/perception/context/raw")
        self.declare_parameter("context_summary_topic", "/perception/context/summary")
        self.declare_parameter("snapshot_request_topic", "/perception/snapshot/request")
        self.declare_parameter("snapshot_image_topic", "/perception/snapshot/image")
        self.declare_parameter("snapshot_info_topic", "/perception/snapshot/info")

        vlm_backend = self.get_parameter("vlm_backend").value
        vlm_model = self.get_parameter("vlm_model").value
        vlm_mock = self.get_parameter("vlm_mock_mode").value
        min_interval = self.get_parameter("min_trigger_interval").value

        self.vlm = VLMClient(
            backend=vlm_backend, model=vlm_model, mock_mode=vlm_mock
        )
        self.trigger = PerceptionTrigger(min_interval=min_interval)
        self.cv_bridge = CvBridge()

        self.latest_cv_image = None
        self._latest_vlm_result: dict | None = None
        self._latest_vlm_time: float = 0.0

        detection_topic = self.get_parameter("detection_topic").value
        context_raw_topic = self.get_parameter("context_raw_topic").value
        context_summary_topic = self.get_parameter("context_summary_topic").value
        snapshot_request_topic = self.get_parameter("snapshot_request_topic").value
        snapshot_image_topic = self.get_parameter("snapshot_image_topic").value
        snapshot_info_topic = self.get_parameter("snapshot_info_topic").value

        # 구독
        self.create_subscription(String, detection_topic, self._on_detections, 10)
        image_topic = self.get_parameter("image_topic").value
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(String, snapshot_request_topic, self._on_snapshot_req, 10)

        # publish
        self._pub_raw = self.create_publisher(String, context_raw_topic, 10)
        self._pub_summary = self.create_publisher(String, context_summary_topic, 10)
        self._pub_snap_img = self.create_publisher(CompressedImage, snapshot_image_topic, 10)
        self._pub_snap_info = self.create_publisher(String, snapshot_info_topic, 10)

        mode_str = "mock" if vlm_mock else f"{vlm_backend}/{vlm_model}"
        self.get_logger().info(f"PerceptionContextBuilder 시작 (VLM={mode_str})")

    def _on_image(self, msg: Image):
        try:
            self.latest_cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
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

        # JPEG 압축 publish
        _, buf = cv2.imencode(
            ".jpg", self.latest_cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        snap_msg = CompressedImage()
        snap_msg.format = "jpeg"
        snap_msg.data = buf.tobytes()
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
