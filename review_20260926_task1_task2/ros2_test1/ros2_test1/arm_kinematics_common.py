"""Shared Arm_5 servo mapping and planar kinematics.

Mode-specific modules keep their own calibrated constants and bind them to this
shared implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

JOINT_NAMES = [
    "servo_base_yaw",
    "servo_blue",
    "blue_chain_passive",
    "servo_orange",
    "orange_chain_passive",
    "extension_level",
    "gripper_left",
    "gripper_right",
]


@dataclass(frozen=True)
class ArmKinematicsProfile:
    HOME_ID1_TICK: int
    HOME_ID2_TICK: int
    READY_ID1_TICK: int
    READY_ID2_TICK: int
    GRIPPER_CLOSED_TICK: int
    GRIPPER_OPEN_TICK: int
    BASE_YAW_CENTER_TICK: int
    BASE_YAW_HOME_TICK: int
    BASE_YAW_READY_TICK: int
    MIN_ANGLE_GAP_DEG: float
    BASE_YAW_TICKS_PER_REVOLUTION: float = 1000.0
    ID1_ZERO_DEG_TICK: float = 460.0
    ID1_NINETY_DEG_TICK: float = 80.0
    ID2_NINETY_DEG_TICK: float = 250.0
    ID2_FORWARD_180_DEG_TICK: float = 640.0
    ID1_SAFE_LIMITS: tuple[int, int] = (150, 710)
    ID2_SAFE_LIMITS: tuple[int, int] = (0, 769)
    BASE_HEIGHT_MM: float = 35.0
    BLUE_ACTIVE_LENGTH_MM: float = 238.0
    ORANGE_EXTENSION_LENGTH_MM: float = 190.6
    LEVEL_TIP_LENGTH_MM: float = 60.0


class ArmKinematicsModel:
    def __init__(self, profile: ArmKinematicsProfile):
        self.profile = profile

    def id6_radians(self, tick):
        return (float(tick) - self.profile.BASE_YAW_CENTER_TICK) * (
            2.0 * math.pi / self.profile.BASE_YAW_TICKS_PER_REVOLUTION
        )

    def id1_degrees(self, tick):
        physical_degrees = (self.profile.ID1_ZERO_DEG_TICK - tick) * (
            90.0
            / (self.profile.ID1_ZERO_DEG_TICK - self.profile.ID1_NINETY_DEG_TICK)
        )
        return 180.0 - physical_degrees

    def id2_degrees(self, tick):
        return (self.profile.ID2_FORWARD_180_DEG_TICK - tick) * (
            90.0
            / (
                self.profile.ID2_FORWARD_180_DEG_TICK
                - self.profile.ID2_NINETY_DEG_TICK
            )
        )

    def id2_tick_from_degrees(self, degrees):
        return self.profile.ID2_FORWARD_180_DEG_TICK - degrees * (
            (
                self.profile.ID2_FORWARD_180_DEG_TICK
                - self.profile.ID2_NINETY_DEG_TICK
            )
            / 90.0
        )

    def angle_gap_degrees(self, id1_tick, id2_tick):
        return abs(self.id1_degrees(id1_tick) - self.id2_degrees(id2_tick))

    def has_safe_angle_gap(self, id1_tick, id2_tick, minimum_degrees=None):
        if minimum_degrees is None:
            minimum_degrees = self.profile.MIN_ANGLE_GAP_DEG
        if minimum_degrees <= 0.0:
            return True
        return self.angle_gap_degrees(id1_tick, id2_tick) > minimum_degrees

    def enforce_angle_gap(self, id1_tick, id2_tick, id2_limits, minimum_degrees=None):
        if minimum_degrees is None:
            minimum_degrees = self.profile.MIN_ANGLE_GAP_DEG
        id1_tick = int(round(id1_tick))
        id2_tick = int(round(id2_tick))
        if self.has_safe_angle_gap(id1_tick, id2_tick, minimum_degrees):
            return id1_tick, id2_tick

        id1_angle = self.id1_degrees(id1_tick)
        id2_angle = self.id2_degrees(id2_tick)
        required_gap = minimum_degrees + 0.1
        target_angle = (
            id1_angle - required_gap
            if id2_angle <= id1_angle
            else id1_angle + required_gap
        )
        candidate = int(round(self.id2_tick_from_degrees(target_angle)))
        candidate = max(id2_limits[0], min(id2_limits[1], candidate))
        direction = -1 if self.id2_degrees(candidate) >= id1_angle else 1
        while not self.has_safe_angle_gap(id1_tick, candidate, minimum_degrees):
            next_candidate = candidate + direction
            if next_candidate < id2_limits[0] or next_candidate > id2_limits[1]:
                raise ValueError("ID1/ID2 cannot satisfy the configured angle gap")
            candidate = next_candidate
        return id1_tick, candidate

    def gripper_position_mm(self, id1_tick, id2_tick):
        orange = math.radians(self.id1_degrees(id1_tick))
        blue = math.radians(self.id2_degrees(id2_tick))
        x_mm = (
            self.profile.BLUE_ACTIVE_LENGTH_MM * math.cos(blue)
            - self.profile.ORANGE_EXTENSION_LENGTH_MM * math.cos(orange)
            + self.profile.LEVEL_TIP_LENGTH_MM
        )
        z_mm = (
            self.profile.BASE_HEIGHT_MM
            + self.profile.BLUE_ACTIVE_LENGTH_MM * math.sin(blue)
            - self.profile.ORANGE_EXTENSION_LENGTH_MM * math.sin(orange)
        )
        return x_mm, z_mm

    def joint_positions(self, id1_tick, id2_tick, id4_tick, id6_tick=None):
        if id6_tick is None:
            id6_tick = self.profile.BASE_YAW_CENTER_TICK
        orange = math.radians(self.id1_degrees(id1_tick))
        blue = math.radians(self.id2_degrees(id2_tick))
        gripper_range = (
            self.profile.GRIPPER_OPEN_TICK - self.profile.GRIPPER_CLOSED_TICK
        )
        gripper_ratio = max(
            0.0,
            min(
                1.0,
                (id4_tick - self.profile.GRIPPER_CLOSED_TICK)
                / float(gripper_range),
            ),
        )
        return [
            self.id6_radians(id6_tick),
            blue,
            orange - blue,
            orange,
            blue - orange,
            -orange,
            -0.05 - 0.75 * gripper_ratio,
            0.05 + 0.75 * gripper_ratio,
        ]


def build_exports(profile: ArmKinematicsProfile):
    model = ArmKinematicsModel(profile)
    exports = asdict(profile)
    exports.update(
        {
            "JOINT_NAMES": JOINT_NAMES,
            "KINEMATICS_PROFILE": profile,
            "KINEMATICS_MODEL": model,
            "id6_radians": model.id6_radians,
            "id1_degrees": model.id1_degrees,
            "id2_degrees": model.id2_degrees,
            "id2_tick_from_degrees": model.id2_tick_from_degrees,
            "angle_gap_degrees": model.angle_gap_degrees,
            "has_safe_angle_gap": model.has_safe_angle_gap,
            "enforce_angle_gap": model.enforce_angle_gap,
            "gripper_position_mm": model.gripper_position_mm,
            "joint_positions": model.joint_positions,
        }
    )
    return exports
