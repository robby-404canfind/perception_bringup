"""follow_node.py — follow() ActionServer 래퍼 노드.

/system1/follow Action을 수신하고, exec_follow()를 실행하고,
Feedback과 Result를 publish합니다.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from system_interfaces.action import Follow

from .actions.follow import exec_follow
from .perception_cache import PerceptionCache


class FollowNode(Node):

    def __init__(self):
        super().__init__("follow_node")

        # Action wrapper: follow 제어는 exec_follow()에 둡니다.
        self.callback_group = ReentrantCallbackGroup()
        self.perception_cache = PerceptionCache()
        self.create_subscription(
            String, "/perception/detections",
            lambda msg: self.perception_cache.update_from_msg(msg.data, self.get_logger()),
            10,
            callback_group=self.callback_group,
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._action_server = ActionServer(
            self,
            Follow,
            "/system1/follow",
            self._execute_cb,
            callback_group=self.callback_group,
        )
        self.get_logger().info("FollowNode ActionServer 시작: /system1/follow")

    def _execute_cb(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(
            f"follow Goal 수신: mission_id={req.mission_id}, "
            f"target_class={req.target_class}, target_id={req.target_id}, "
            f"distance={req.target_distance_m}m"
        )

        def _publish_fb(fb: dict):
            msg = Follow.Feedback()
            msg.state = fb.get("state", "")
            msg.elapsed_sec = float(fb.get("elapsed_sec", 0.0))
            msg.current_distance_m = float(fb.get("current_distance_m", 0.0))
            msg.target_status = fb.get("target_status", "")
            goal_handle.publish_feedback(msg)

        result_data = exec_follow(
            node=self,
            perception_cache=self.perception_cache,
            cmd_pub=self.cmd_pub,
            target_id=req.target_id,
            target_class=req.target_class,
            target_distance_m=req.target_distance_m,
            max_time_sec=req.max_time_sec,
            feedback_cb=_publish_fb,
        )

        result = Follow.Result()
        result.success = result_data["success"]
        result.message = f"follow {result_data['final_state']}"
        result.final_state = result_data["final_state"]

        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FollowNode()
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
