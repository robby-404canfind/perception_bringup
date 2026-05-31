"""perception.launch.py — Ch04 Perception 스택 실행.

yolo_detector + perception_context_builder를 함께 실행합니다.
find/scan/follow/assess_scene/resolve_target ActionServer 노드는 별도로 실행합니다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("perception_bringup")
    config_file = os.path.join(pkg_share, "config", "perception_config.yaml")

    return LaunchDescription(
        [
            # YOLO Detector 노드
            Node(
                package="perception_bringup",
                executable="yolo_detector",
                name="yolo_detector",
                parameters=[config_file],
                output="screen",
            ),
            # Perception Context Builder 노드
            Node(
                package="perception_bringup",
                executable="perception_context_builder",
                name="perception_context_builder",
                parameters=[config_file],
                output="screen",
            ),
        ]
    )
