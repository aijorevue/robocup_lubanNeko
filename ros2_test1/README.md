# Ros2_test1

Clean target-vision package for the RK3588S camera.

The active program is `ros2_test1.target_vision`. It detects only:

- yellow, red, and blue balls
- red and blue squares
- red and blue rings
- QR codes whose decoded content matches `R` or `B`, using `assets/red.png` and `assets/blue.png` as templates/fallbacks

Run from the board:

```bash
cd ~/ros2_ws/ros2_test1
python3 -m ros2_test1.target_vision
```

After building the ROS 2 package:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_test1
source install/setup.bash
ros2 run ros2_test1 target_vision
```

The HDMI screen shows a live OpenCV window. A browser preview is also available on port `8080`.

Old experiments and generated files are under `archive/`.
