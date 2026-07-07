测试串口通信：终端一：
source /opt/ros/humble/setup.bash
source /home/cat/ros2_ws/install/setup.bash
ros2 run stm32_bridge stm32_bridge_node

终端二：
source /opt/ros/humble/setup.bash
source /home/cat/ros2_ws/install/setup.bash
ros2 topic pub /servo_cmd std_msgs/msg/String "{data: '1:1500'}" -1# robocup-up
