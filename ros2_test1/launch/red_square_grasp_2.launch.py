from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown, TimerAction
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
                        executable="target_vision_2",
                        name="red_square_grasp_direct_2",
                        arguments=[
                            "--enable-red-square-grasp",
                            "--enable-arm-preview",
                            "--no-web",
                            "--width",
                            "800",
                            "--height",
                            "600",
                            "--fps",
                            "30",
                            "--detect-every-n-frames",
                            "1",
                            "--detection-scale",
                            "0.75",
                            "--window-x",
                            "0",
                            "--window-y",
                            "60",
                            "--window-width",
                            "620",
                            "--window-height",
                            "520",
                            "--preview-id1",
                            "550",
                            "--preview-id2",
                            "300",
                            "--preview-id4",
                            "1000",
                            "--preview-id6",
                            "510",
                            "--grasp-id1-ready",
                            "550",
                            "--grasp-id2-ready",
                            "300",
                            "--grasp-id4-closed",
                            "1000",
                            "--grasp-id4-open",
                            "1500",
                            "--servo-angle-gap-deg",
                            "0",
                            "--grasp-command-interval",
                            "0.15",
                            "--grasp-stable-frames",
                            "1",
                            "--camera-gripper-offset-mm",
                            "50",
                            "--camera-gripper-vertical-offset-mm",
                            "0",
                            "--trigger-kind",
                            "square",
                            "--trigger-color",
                            "red",
                        ],
                        additional_env={
                            **DESKTOP_ENV,
                            "RED_SQUARE_EXECUTE": LaunchConfiguration("execute"),
                        },
                        on_exit=Shutdown(reason="target vision 2.0 window closed"),
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
