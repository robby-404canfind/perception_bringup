from setuptools import find_packages, setup
import os
from glob import glob

package_name = "perception_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("*.pt")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=[
        "setuptools",
        "ultralytics",
        "lap>=0.5.12",
        "opencv-python",
        "openai",
        "pydantic",
        "numpy<1.25",
    ],
    zip_safe=True,
    maintainer="student",
    maintainer_email="todo@todo.com",
    description="Perception 실습 패키지 (Physical AI 커리큘럼 Chapter 04)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "yolo_detector = perception_bringup.yolo_detector_node:main",
            "perception_context_builder = perception_bringup.perception_context_builder_node:main",
            "find_node = perception_bringup.find_node:main",
            "scan_node = perception_bringup.scan_node:main",
            "follow_node = perception_bringup.follow_node:main",
            "assess_scene_node = perception_bringup.assess_scene_node:main",
            "resolve_target_node = perception_bringup.resolve_target_node:main",
        ],
    },
)
