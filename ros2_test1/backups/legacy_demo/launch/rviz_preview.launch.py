from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    arm_share = Path(get_package_share_directory("arm_5_description"))
    robot_description = (arm_share / "urdf" / "arm_5.urdf").read_text()
    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(arm_share / "rviz" / "arm_5.rviz")],
            output="screen",
        ),
    ])
