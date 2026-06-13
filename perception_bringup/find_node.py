"""find_node.py — find() ActionServer 래퍼 노드.

/system1/find Action을 수신한 뒤 exec_find()를 실행하고
Feedback과 Result를 publish합니다. known class local find만 지원합니다.
"""

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from system_interfaces.action import Find

from .actions.find import exec_find
from .perception_cache import PerceptionCache


class FindNode(Node):

    def __init__(self):
        super().__init__("find_node")

        # Action wrapper: find 판단은 exec_find()에 둡니다.
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
            Find,
            "/system1/find",
            self._execute_cb,
            callback_group=self.callback_group,
        )
        self.get_logger().info("FindNode ActionServer 시작: /system1/find")

    def _execute_cb(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(
            f"find Goal 수신: mission_id={req.mission_id}, "
            f"target_class={req.target_class}, timeout={req.timeout_sec}s"
        )

        def _publish_fb(fb: dict):
            msg = Find.Feedback()
            msg.state = fb.get("state", "")
            msg.detail = fb.get("detail", "")
            msg.elapsed_sec = float(fb.get("elapsed_sec", 0.0))
            goal_handle.publish_feedback(msg)

        result_data = exec_find(
            node=self,
            perception_cache=self.perception_cache,
            cmd_pub=self.cmd_pub,
            target_class=req.target_class,
            timeout_sec=req.timeout_sec,
            sweep_deg=req.sweep_deg,
            snapshot_pub=self.snapshot_pub,
            feedback_cb=_publish_fb,
            mission_id=req.mission_id,
            request_id=req.request_id,
        )

        result = Find.Result()
        result.success = result_data["success"]
        result.message = (
            f"{req.target_class} 발견 ({result_data['search_method']})"
            if result_data["success"]
            else f"{req.target_class} 미발견"
        )
        result.object_json = json.dumps(result_data.get("found_object") or {})

        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FindNode()
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
