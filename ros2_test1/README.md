# ros2_test1 Mechanical Arm Vision Grasp

Current project entry:

```bash
RED_SQUARE_EXECUTE=true /home/cat/ros2_ws/start_target_vision.sh
```

The default profile is RK direct UART with the H7 chassis link enabled. For a
manual arm/camera session without H7, use:

```bash
VISION_PROFILE=standalone RED_SQUARE_EXECUTE=true /home/cat/ros2_ws/start_target_vision.sh
```

## Active Runtime Files

- `start_target_vision.sh`: unified desktop launcher
- `start_chassis_arm_link.sh`: compatibility wrapper for the linked profile
- `launch/red_square_chassis_rk_direct.launch.py`: H7-linked RK entry
- `launch/red_square_grasp_rk_direct.launch.py`: RK-only entry
- `ros2_test1/launch_common.py`: shared RK launch assembly
- `ros2_test1/target_vision.py`: RK-only visual grasp logic
- `ros2_test1/chassis_link.py`: versioned H7 task protocol and reconnect logic
- `ros2_test1/field_mode.py`: mirrored red/blue target policy
- `ros2_test1/arm_kinematics_common.py`: shared IK and joint mapping
- `ros2_test1/arm_kinematics.py`: RK calibration profile
- `ros2_test1/detectors/`: color and shape detection helpers
- `ros2_test1/assets/red.png`, `ros2_test1/assets/blue.png`: QR templates

Old backups and demos are kept under `backups/` and are not installed as active console scripts.

More detail is in `ARCHITECTURE.md`.

## H7 Link Safety Contract

- RK sends `RK,ARM,READY` only after the camera has produced a frame and both
  direct servo UARTs are writable (`/dev/ttyS9` for 85kg and `/dev/ttyS0` for
  ZP).
- `DISC_CATCH` keeps its station-specific 2-second no-target timer. A camera
  outage does not pause that timer; the arm returns home before RK reports
  `DONE` or `ERR` to H7.
- `PLATFORM_PICK` starts its 2-second target watch after the ready-pose motion
  allowance, not at the instant H7 sends `START`.
- `COLUMN_CATCH` remains active until H7 sends the matching sequence `STOP`.
  STOP, RESET, process exit, and servo faults all attempt the home contract.
- The direct servo UARTs are opened exclusively while the vision process is
  running, so `servo_position` cannot silently compete with the autonomous
  controller.
