from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DESKTOP_ENV = {
    "XDG_RUNTIME_DIR": "/run/user/1000",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
    "DISPLAY": ":0",
    "QT_QPA_PLATFORM": "xcb",
}


def generate_launch_description():
    share_dir = Path(get_package_share_directory("ros2_test1"))
    robot_description = (share_dir / "urdf" / "arm_5.urdf").read_text()
    rviz_config = str(share_dir / "rviz" / "arm_5.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("target_color", default_value="red"),
            DeclareLaunchArgument("target_kind", default_value="square"),
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    "pkill -f '/ros2_test1/[t]arget_vision|python3 -m ros2_test1.[t]arget_vision' || true",
                ],
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            TimerAction(
                period=1.0,
                actions=[
                    Node(
                        package="ros2_test1",
                        executable="target_vision",
                        name="red_square_grasp_vision",
                        arguments=[
                            "--enable-red-square-grasp",
                            "--enable-arm-preview",
                            "--no-web",
                            "--width",
                            "960",
                            "--height",
                            "540",
                            "--fps",
                            "25",
                            "--window-x",
                            "60",
                            "--window-y",
                            "60",
                            "--window-width",
                            "620",
                            "--window-height",
                            "520",
                            "--preview-id1",
                            "550",
                            "--preview-id2",
                            "550",
                            "--grasp-id1-ready",
                            "550",
                            "--grasp-id2-ready",
                            "550",
                            "--servo-angle-gap-deg",
                            "20",
                            "--grasp-command-interval",
                            "1.8",
                            "--grasp-stable-frames",
                            "3",
                            "--grasp-center-deadband-px",
                            "45",
                            "--grasp-id2-pixel-gain",
                            "0.04",
                            "--grasp-id1-pixel-gain-y",
                            "0.02",
                            "--grasp-max-step-ticks",
                            "6",
                            "--post-center-retreat-mm",
                            "50",
                            "--post-center-down-mm",
                            "30",
                            "--trigger-kind",
                            LaunchConfiguration("target_kind"),
                            "--trigger-color",
                            LaunchConfiguration("target_color"),
                        ],
                        additional_env={
                            **DESKTOP_ENV,
                            "RED_SQUARE_EXECUTE": LaunchConfiguration("execute"),
                        },
                        output="screen",
                    )
                ],
            ),
            TimerAction(
                period=1.5,
                actions=[
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        arguments=["-d", rviz_config, "--geometry", "620x660+650+35"],
                        additional_env=DESKTOP_ENV,
                        output="screen",
                    )
                ],
            ),
        ]
    )
