"""assess_scene_node.py — /system1/assess_scene ActionServer."""

import json
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from system_interfaces.action import AssessScene

from .actions.assess_scene import exec_assess_scene
from .vlm_client import VLMClient


class AssessSceneNode(Node):

    def __init__(self):
        super().__init__("assess_scene_node")

        self.declare_parameter("vlm_backend", "ollama")
        self.declare_parameter("vlm_model", "qwen2.5vl:7b")
        self.declare_parameter("vlm_timeout_sec", 30.0)
        self.declare_parameter("image_topic", "/Tiago_Lite/Astra_rgb/image_color")
        self.declare_parameter("context_raw_topic", "/perception/context/raw")

        self.callback_group = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()
        self.latest_cv_image = None
        self.latest_context: dict = {}
        self.latest_context_time = 0.0

        self.vlm = VLMClient(
            backend=self.get_parameter("vlm_backend").value,
            model=self.get_parameter("vlm_model").value,
            timeout=float(self.get_parameter("vlm_timeout_sec").value),
        )

        self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self._on_image,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            self.get_parameter("context_raw_topic").value,
            self._on_context,
            10,
            callback_group=self.callback_group,
        )

        self._action_server = ActionServer(
            self,
            AssessScene,
            "/system1/assess_scene",
            self._execute_cb,
            callback_group=self.callback_group,
        )
        self.get_logger().info("AssessSceneNode ActionServer 시작: /system1/assess_scene")

    def _on_image(self, msg: Image):
        try:
            self.latest_cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"assess_scene image 변환 실패: {e}")

    def _on_context(self, msg: String):
        try:
            self.latest_context = json.loads(msg.data) if msg.data else {}
            self.latest_context_time = time.time()
        except json.JSONDecodeError:
            self.latest_context = {}

    def _execute_cb(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(
            f"assess_scene Goal 수신: mission_id={req.mission_id}, "
            f"query={req.query}, timeout={req.timeout_sec}s"
        )

        def _publish_fb(fb: dict):
            msg = AssessScene.Feedback()
            msg.state = fb.get("state", "")
            msg.detail = fb.get("detail", "")
            msg.elapsed_sec = float(fb.get("elapsed_sec", 0.0))
            goal_handle.publish_feedback(msg)

        result_data = exec_assess_scene(
            node=self,
            vlm_client=self.vlm,
            cv_image=self.latest_cv_image,
            context=self.latest_context,
            query=req.query,
            timeout_sec=req.timeout_sec,
            feedback_cb=_publish_fb,
            mission_id=req.mission_id,
            request_id=req.request_id,
        )

        result = AssessScene.Result()
        result.success = bool(result_data["success"])
        result.message = result_data["message"]
        result.assessment_json = json.dumps(
            result_data["assessment"],
            ensure_ascii=False,
        )

        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = AssessSceneNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
