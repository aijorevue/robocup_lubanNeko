"""Shared Arm_5 servo mapping and planar kinematics.

Mode-specific modules keep their own calibrated constants and bind them to this
shared implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


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

    def solve_gripper_position(
        self,
        target_x_mm,
        target_z_mm,
        seed_id1,
        seed_id2,
        id1_limits=None,
        id2_limits=None,
        minimum_gap_degrees=None,
    ):
        if id1_limits is None:
            id1_limits = self.profile.ID1_SAFE_LIMITS
        if id2_limits is None:
            id2_limits = self.profile.ID2_SAFE_LIMITS
        if minimum_gap_degrees is None:
            minimum_gap_degrees = self.profile.MIN_ANGLE_GAP_DEG

        coarse_id1 = np.arange(id1_limits[0], id1_limits[1] + 1, 4, dtype=float)
        coarse_id2 = np.arange(id2_limits[0], id2_limits[1] + 1, 4, dtype=float)
        grid_id1, grid_id2 = np.meshgrid(coarse_id1, coarse_id2, indexing="ij")
        best = self._best_grid_target(
            grid_id1,
            grid_id2,
            target_x_mm,
            target_z_mm,
            seed_id1,
            seed_id2,
            minimum_gap_degrees,
        )

        refine_id1 = np.arange(
            max(id1_limits[0], best[0] - 5),
            min(id1_limits[1], best[0] + 5) + 1,
            dtype=float,
        )
        refine_id2 = np.arange(
            max(id2_limits[0], best[1] - 5),
            min(id2_limits[1], best[1] + 5) + 1,
            dtype=float,
        )
        grid_id1, grid_id2 = np.meshgrid(refine_id1, refine_id2, indexing="ij")
        return self._best_grid_target(
            grid_id1,
            grid_id2,
            target_x_mm,
            target_z_mm,
            seed_id1,
            seed_id2,
            minimum_gap_degrees,
        )

    def _best_grid_target(
        self,
        grid_id1,
        grid_id2,
        target_x_mm,
        target_z_mm,
        seed_id1,
        seed_id2,
        minimum_gap_degrees,
    ):
        orange = np.deg2rad(self.id1_degrees(grid_id1))
        blue = np.deg2rad(self.id2_degrees(grid_id2))
        x_mm = (
            self.profile.BLUE_ACTIVE_LENGTH_MM * np.cos(blue)
            - self.profile.ORANGE_EXTENSION_LENGTH_MM * np.cos(orange)
            + self.profile.LEVEL_TIP_LENGTH_MM
        )
        z_mm = (
            self.profile.BASE_HEIGHT_MM
            + self.profile.BLUE_ACTIVE_LENGTH_MM * np.sin(blue)
            - self.profile.ORANGE_EXTENSION_LENGTH_MM * np.sin(orange)
        )
        gap = np.abs(self.id1_degrees(grid_id1) - self.id2_degrees(grid_id2))
        position_error = np.square(x_mm - target_x_mm) + np.square(
            z_mm - target_z_mm
        )
        seed_cost = 0.002 * (
            np.square(grid_id1 - seed_id1) + np.square(grid_id2 - seed_id2)
        )
        if minimum_gap_degrees <= 0.0:
            cost = position_error + seed_cost
        else:
            cost = np.where(
                gap > minimum_gap_degrees, position_error + seed_cost, np.inf
            )
        index = np.unravel_index(np.argmin(cost), cost.shape)
        id1_tick = int(grid_id1[index])
        id2_tick = int(grid_id2[index])
        error_mm = math.sqrt(float(position_error[index]))
        return id1_tick, id2_tick, error_mm

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
            "solve_gripper_position": model.solve_gripper_position,
            "joint_positions": model.joint_positions,
        }
    )
    return exports
