from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stm32_bridge',
            executable='stm32_bridge_node',
            name='stm32_bridge',
            parameters=[{
                'port': '/dev/ttyUSB0',
                'baud': 115200,
            }],
            output='screen',
        ),
    ])
