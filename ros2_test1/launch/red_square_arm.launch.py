from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("calibration_file", default_value=""),
        Node(
            package="ros2_test1",
            executable="red_square_arm",
            name="red_square_arm",
            parameters=[{
                "execute": LaunchConfiguration("execute"),
                "calibration_file": LaunchConfiguration("calibration_file"),
            }],
            output="screen",
        ),
    ])
