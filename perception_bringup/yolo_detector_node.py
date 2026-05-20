"""yolo_detector_node.py — YOLO26 Detection + Tracking + Depth ROS2 노드.

RGB 이미지를 구독하고, YOLO26 .track()으로 tracked detections를 생성하고,
depth 카메라에서 range_m을 추정하여 /perception/detections에 JSON으로 publish합니다.
"""

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("model", "yolo26n.pt")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("image_topic", "/Tiago_Lite/Astra_rgb/image_color")
        self.declare_parameter("depth_topic", "/Tiago_Lite/Astra_depth/image")
        self.declare_parameter("detection_topic", "/perception/detections")

        model_path = self.get_parameter("model").value
        self.threshold = self.get_parameter("threshold").value
        device = self.get_parameter("device").value

        self.yolo = YOLO(model_path)
        if device != "cpu":
            self.yolo.to(device)

        self.cv_bridge = CvBridge()
        self.latest_depth = None
        self.latest_cv_image = None

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        detection_topic = self.get_parameter("detection_topic").value

        self.create_subscription(Image, image_topic, self._image_cb, image_qos)
        self.create_subscription(Image, depth_topic, self._depth_cb, image_qos)
        self._pub_det = self.create_publisher(String, detection_topic, 10)

        self.get_logger().info(
            f"YoloDetector 시작 (model={model_path}, device={device}, "
            f"thr={self.threshold}, image={image_topic})"
        )

    def _image_cb(self, msg: Image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest_cv_image = cv_image

        results = self.yolo.track(
            source=cv_image,
            persist=True,
            conf=self.threshold,
            verbose=False,
        )

        detections = self._parse_tracked(results[0], cv_image.shape)

        for det in detections:
            det["range_m"] = self._estimate_depth(det["bbox"])

        payload = {
            "frame_w": cv_image.shape[1],
            "frame_h": cv_image.shape[0],
            "objects": detections,
        }
        self._pub_det.publish(String(data=json.dumps(payload)))

    def _depth_cb(self, msg: Image):
        try:
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f"Depth 변환 실패: {e}", throttle_duration_sec=5.0)

    def _parse_tracked(self, result, img_shape: tuple) -> list:
        h, w = img_shape[:2]
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            cx, cy, bw, bh = box.xywh[0].tolist()
            det = {
                "id": int(box.id) if box.id is not None else -1,
                "class": self.yolo.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "center": {"x": round(cx), "y": round(cy)},
                "bbox": {
                    "x": round(cx - bw / 2),
                    "y": round(cy - bh / 2),
                    "w": round(bw),
                    "h": round(bh),
                },
                "direction": self._classify_direction(cx, w),
                "range_m": None,
            }
            detections.append(det)

        return detections

    @staticmethod
    def _classify_direction(cx: float, width: int) -> str:
        if cx < width / 3:
            return "left"
        elif cx > 2 * width / 3:
            return "right"
        return "center"

    def _estimate_depth(self, bbox: dict) -> float | None:
        if self.latest_depth is None:
            return None

        cx = bbox["x"] + bbox["w"] // 2
        cy = bbox["y"] + bbox["h"] // 2
        half = max(bbox["w"], bbox["h"]) // 10 or 5

        dh, dw = self.latest_depth.shape[:2]
        y0, y1 = max(0, cy - half), min(dh, cy + half)
        x0, x1 = max(0, cx - half), min(dw, cx + half)

        if y0 >= y1 or x0 >= x1:
            return None

        roi = self.latest_depth[y0:y1, x0:x1]

        if roi.dtype == np.float32:
            valid = roi[(roi > 0.1) & (roi < 50.0)]
        elif roi.dtype == np.uint16:
            roi_m = roi.astype(np.float32) / 1000.0
            valid = roi_m[(roi_m > 0.1) & (roi_m < 50.0)]
        else:
            return None

        if len(valid) == 0:
            return None
        return round(float(np.median(valid)), 2)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
