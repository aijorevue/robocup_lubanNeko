"""Shared launch assembly for the RK direct grasp app."""

from __future__ import annotations

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

COMMON_CAMERA_ARGS = [
    "--enable-red-square-grasp",
    "--enable-arm-preview",
    "--no-web",
    "--width",
    "800",
    "--height",
    "600",
    "--fps",
    "30",
]

COMMON_GRASP_ARGS = [
    "--preview-id1",
    "550",
    "--preview-id2",
    "300",
    "--grasp-id1-ready",
    "550",
    "--grasp-id2-ready",
    "300",
    "--grasp-center-deadband-px",
    "30",
    "--trigger-kind",
    LaunchConfiguration("target_kind"),
    "--trigger-color",
    LaunchConfiguration("target_color"),
]

MODE_PROFILES = {
    "rk": {
        "executable": "target_vision",
        "node_name": "red_square_grasp_rk_direct_vision",
        "detect_every": "1",
        "preview_id4": "1120",
        "preview_id6": "500",
        "grasp_id4_closed": "1120",
        "grasp_id4_open": "1700",
        "servo_angle_gap_deg": "20",
        "grasp_command_interval": "0.18",
        "grasp_stable_frames": "4",
        "grasp_id2_pixel_gain": "0.13",
        "grasp_id6_pixel_gain": "0.11",
        "grasp_id6_max_step_ticks": "18",
        "grasp_id1_pixel_gain_y": "0",
        "grasp_max_step_ticks": "28",
        "post_center_retreat_mm": "52",
        "post_center_down_mm": "150",
        "timer_period": 1.0,
        "rviz_period": 1.5,
        "extra_args": [
            "--direct-servo-bus",
            "--direct-arm-uart",
            "/dev/ttyS9",
            "--direct-zp-uart",
            "/dev/ttyS0",
            "--direct-arm-time-ms",
            "600",
            "--direct-zp-time-ms",
            "350",
            "--direct-gripper-time-ms",
            "180",
            "--post-center-direct-descend",
            "--qr-template-every",
            "1",
            "--grasp-retrigger-cooldown",
            "1.2",
            "--detection-smoothing-alpha",
            "1.0",
            "--detection-smoothing-match-px",
            "90",
            "--fresh-detection-on-lock",
            "--grasp-id2-center-max",
            "464",
            "--chassis-link",
            "--chassis-uart",
            "auto",
            "--station-no-target-timeout",
            "5.0",
        ],
        "kill_before_start": True,
    },
}


def generate_grasp_launch(mode):
    profile = MODE_PROFILES[mode]
    share_dir = Path(get_package_share_directory("ros2_test1"))
    robot_description = (share_dir / "urdf" / "arm_5.urdf").read_text()
    rviz_config = str(share_dir / "rviz" / "arm_5.rviz")

    actions = [
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("target_color", default_value="red"),
        DeclareLaunchArgument("target_kind", default_value="square"),
    ]
    if profile["kill_before_start"]:
        actions.append(
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    "pkill -f '/ros2_test1/[t]arget_vision|python3 -m ros2_test1.[t]arget_vision' || true",
                ],
                output="screen",
            )
        )

    actions.extend(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            TimerAction(
                period=profile["timer_period"],
                actions=[_vision_node(profile)],
            ),
            TimerAction(
                period=profile["rviz_period"],
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
    return LaunchDescription(actions)


def _vision_node(profile):
    args = [
        *COMMON_CAMERA_ARGS,
        "--detect-every-n-frames",
        profile["detect_every"],
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
        *COMMON_GRASP_ARGS,
        "--preview-id4",
        profile["preview_id4"],
        "--preview-id6",
        profile["preview_id6"],
        "--grasp-id4-closed",
        profile["grasp_id4_closed"],
        "--grasp-id4-open",
        profile["grasp_id4_open"],
        "--servo-angle-gap-deg",
        profile["servo_angle_gap_deg"],
        "--grasp-command-interval",
        profile["grasp_command_interval"],
        "--grasp-stable-frames",
        profile["grasp_stable_frames"],
        "--grasp-id2-pixel-gain",
        profile["grasp_id2_pixel_gain"],
        "--grasp-id6-pixel-gain",
        profile["grasp_id6_pixel_gain"],
        "--grasp-id6-max-step-ticks",
        profile["grasp_id6_max_step_ticks"],
        "--grasp-id1-pixel-gain-y",
        profile["grasp_id1_pixel_gain_y"],
        "--grasp-max-step-ticks",
        profile["grasp_max_step_ticks"],
        "--post-center-retreat-mm",
        profile["post_center_retreat_mm"],
        "--post-center-down-mm",
        profile["post_center_down_mm"],
        *profile["extra_args"],
    ]
    return Node(
        package="ros2_test1",
        executable=profile["executable"],
        name=profile["node_name"],
        arguments=args,
        additional_env={
            **DESKTOP_ENV,
            "RED_SQUARE_EXECUTE": LaunchConfiguration("execute"),
        },
        on_exit=Shutdown(reason="target vision window closed"),
        output="screen",
    )
