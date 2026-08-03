"""RK-only Arm_5 kinematics profile."""

from __future__ import annotations

from .arm_kinematics_common import ArmKinematicsProfile, build_exports


_PROFILE = ArmKinematicsProfile(
    HOME_ID1_TICK=480,
    HOME_ID2_TICK=10,
    READY_ID1_TICK=550,
    READY_ID2_TICK=300,
    GRIPPER_CLOSED_TICK=1120,
    GRIPPER_OPEN_TICK=1500,
    BASE_YAW_CENTER_TICK=500,
    BASE_YAW_HOME_TICK=500,
    BASE_YAW_READY_TICK=500,
    MIN_ANGLE_GAP_DEG=20.0,
)

globals().update(build_exports(_PROFILE))

__all__ = [name for name in globals() if name.isupper() and not name.startswith("_")] + [
    "KINEMATICS_PROFILE",
    "KINEMATICS_MODEL",
    "id6_radians",
    "id1_degrees",
    "id2_degrees",
    "id2_tick_from_degrees",
    "angle_gap_degrees",
    "has_safe_angle_gap",
    "enforce_angle_gap",
    "gripper_position_mm",
    "solve_gripper_position",
    "joint_positions",
]
