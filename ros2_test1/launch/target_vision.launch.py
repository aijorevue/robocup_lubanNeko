from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ros2_test1",
            executable="target_vision",
            name="target_vision",
            output="screen",
        ),
    ])
