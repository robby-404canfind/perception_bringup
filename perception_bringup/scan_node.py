"""scan_node.py — scan() ActionServer 래퍼 노드.

/system1/scan Action을 수신하고, exec_scan()을 실행하고,
Feedback과 Result를 publish합니다.
"""

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from system_interfaces.action import Scan

from .actions.scan import exec_scan
from .perception_cache import PerceptionCache


class ScanNode(Node):

    def __init__(self):
        super().__init__("scan_node")

        self.callback_group = ReentrantCallbackGroup()
        self.perception_cache = PerceptionCache()
        self.create_subscription(
            String, "/perception/detections",
            lambda msg: self.perception_cache.update_from_msg(msg.data, self.get_logger()),
            10,
            callback_group=self.callback_group,
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.snapshot_pub = self.create_publisher(
            String, "/perception/snapshot/request", 10
        )

        self._action_server = ActionServer(
            self,
            Scan,
            "/system1/scan",
            self._execute_cb,
            callback_group=self.callback_group,
        )
        self.get_logger().info("ScanNode ActionServer 시작: /system1/scan")

    def _execute_cb(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(
            f"scan Goal 수신: mission_id={req.mission_id}, "
            f"duration={req.duration_sec}s, sweep={req.sweep_deg}°, mode=all_objects"
        )

        def _publish_fb(fb: dict):
            msg = Scan.Feedback()
            msg.state = fb.get("state", "")
            msg.detail = fb.get("detail", "")
            msg.elapsed_sec = float(fb.get("elapsed_sec", 0.0))
            msg.objects_found = int(fb.get("objects_found", 0))
            goal_handle.publish_feedback(msg)

        # exec_scan 자체는 blocking이지만, MultiThreadedExecutor가 detection 구독을 계속 처리합니다.
        result_data = exec_scan(
            node=self,
            perception_cache=self.perception_cache,
            cmd_pub=self.cmd_pub,
            sweep_deg=req.sweep_deg,
            duration_sec=req.duration_sec,
            snapshot_pub=self.snapshot_pub,
            feedback_cb=_publish_fb,
        )

        # Result
        result = Scan.Result()
        result.success = result_data["success"]
        class_summary = result_data.get("class_summary", "")
        result.message = (
            f"{len(result_data['objects_found'])}개 발견: {class_summary}"
            if result_data["success"]
            else "미발견"
        )
        result.objects_json = json.dumps(
            result_data["objects_found"],
            ensure_ascii=False,
        )

        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ScanNode()
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
