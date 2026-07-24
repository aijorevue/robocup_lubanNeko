# ros2_test1 Mechanical Arm Vision Grasp

Current project entry:

```bash
RED_SQUARE_EXECUTE=true /home/cat/ros2_ws/start_target_vision.sh
```

Default mode is RK-only direct UART. RCT6 mode is still available:

```bash
SERVO_MODE=rc RED_SQUARE_EXECUTE=true /home/cat/ros2_ws/start_target_vision.sh
```

## Active Runtime Files

- `start_target_vision.sh`: unified desktop launcher
- `launch/red_square_grasp_rk_direct.launch.py`: RK-only entry
- `launch/red_square_grasp_rc.launch.py`: RCT6 entry
- `ros2_test1/launch_common.py`: shared RK/RC launch assembly
- `ros2_test1/target_vision.py`: RK-only visual grasp logic
- `ros2_test1/target_vision_rc.py`: RCT6 visual grasp logic
- `ros2_test1/arm_kinematics_common.py`: shared IK and joint mapping
- `ros2_test1/arm_kinematics.py`: RK calibration profile
- `ros2_test1/arm_kinematics_rc.py`: RCT6 calibration profile
- `ros2_test1/detectors/`: color and shape detection helpers
- `ros2_test1/assets/red.png`, `ros2_test1/assets/blue.png`: QR templates

Old backups and demos are kept under `backups/` and are not installed as active console scripts.

More detail is in `ARCHITECTURE.md`.
