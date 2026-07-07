import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    urdf_path = '/home/cat/ros2_ws/ros2_test1/ros2_test1/my_arm.urdf'
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),
        Node(
            package='ros2_test1',
            executable='face_detector.py',
            name='ball_detector'
        ),
    ])
