# Mechanical Arm Vision Grasp Architecture

This document describes the active project layout on the RK board:

```text
/home/cat/ros2_ws
|-- start_target_vision.sh                 # unified desktop launcher
|-- target_vision_desktop.log              # launcher and app log
`-- ros2_test1
    |-- launch
    |   |-- red_square_grasp_rk_direct.launch.py  # RK-only direct UART version
    |   `-- red_square_chassis_rk_direct.launch.py # H7-linked RK version
    |-- ros2_test1
    |   |-- target_vision.py              # active RK-only visual grasp
    |   |-- launch_common.py              # shared launch construction
    |   |-- arm_kinematics_common.py      # shared IK/math implementation
    |   |-- arm_kinematics.py             # RK IK/calibration mapping
    |   |-- assets/red.png                # QR template, still required
    |   `-- assets/blue.png               # QR template, still required
    |-- urdf/arm_5.urdf
    |-- rviz/arm_5.rviz
    `-- backups                           # old backups and legacy demos only
```

## Start Commands

Default RK-only direct UART mode:

```bash
RED_SQUARE_EXECUTE=true /home/cat/ros2_ws/start_target_vision.sh
```

Useful target overrides:

```bash
VISION_TARGET_COLOR=red VISION_TARGET_KIND=any RED_SQUARE_EXECUTE=true /home/cat/ros2_ws/start_target_vision.sh
```

## Active Modes

RK-only mode uses:

- launch: `red_square_grasp_rk_direct.launch.py`
- app: `target_vision.py`
- 85kg UART: `/dev/ttyS9`
- ZP UART: `/dev/ttyS0`
- no real servo readback; it uses last commanded targets plus explicit wait time

## Servo Roles

- `ID1`: arm height, lower tick means lower arm
- `ID2`: image Y correction and reach, target above center means increase ID2
- `ID4`: lower splitter, yellow ball closes, other targets open to 1600
- `ID5`: catcher, opens during retreat before dropping the object
- `ID6`: image X correction, target right of center means decrease ID6
- `ID7`: gripper, closed 1120, open 1700

## RK-only Grasp Flow

1. Start app and expand to ready pose: `ID1=550`, `ID2=300`.
2. Detect red square, red ring, or red QR target each frame.
3. Use visual `dx/dy` to center target:
   - target right: decrease `ID6`
   - target left: increase `ID6`
   - target above: increase `ID2`
   - target below: decrease `ID2`
4. When centered and stable, open `ID7`.
5. Keep the locked centered `ID6` while descending.
6. Use IK/direct descend to move `ID1/ID2` to the grasp pose.
7. Close `ID7`.
8. Retreat with `ID1/ID2`, then return `ID6` toward center.
9. Open `ID5` before dropping; after `ID7` opens and waits 0.3s, close `ID5`.
10. Return to ready and continue detection.

## Files To Avoid Editing First

Start with these when changing behavior:

- `ros2_test1/ros2_test1/target_vision.py` for RK-only behavior
- `ros2_test1/launch/red_square_grasp_rk_direct.launch.py` for RK launch parameters
- `ros2_test1/ros2_test1/launch_common.py` for shared RK launch defaults
- `ros2_test1/ros2_test1/arm_kinematics_common.py` for shared IK/math
- `ros2_test1/ros2_test1/arm_kinematics.py` for RK IK/calibration

## Legacy Area

Historical one-off demos and old generated backups are kept under:

```text
/home/cat/ros2_ws/ros2_test1/backups/
|-- launch       # old launch file versions
|-- python       # old Python file versions
`-- legacy_demo  # old demo entry points removed from setup.py
```

These files are preserved for reference, but they are no longer installed as ROS console scripts and are not used by the unified startup command.
