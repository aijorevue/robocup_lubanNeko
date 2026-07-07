from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("execute", default_value="false"),
            Node(
                package="ros2_test1",
                executable="target_vision",
                name="red_square_grasp_vision",
                arguments=[
                    "--enable-red-square-grasp",
                    "--enable-arm-preview",
                    "--trigger-kind",
                    "square",
                    "--trigger-color",
                    "red",
                ],
                additional_env={"RED_SQUARE_EXECUTE": LaunchConfiguration("execute")},
                output="screen",
            )
        ]
    )
