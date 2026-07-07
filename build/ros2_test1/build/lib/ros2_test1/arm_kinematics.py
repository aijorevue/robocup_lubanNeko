"""Arm_5 servo mapping and planar kinematics shared by vision and RViz."""

from __future__ import annotations

import math

import numpy as np


HOME_ID1_TICK = 403
HOME_ID2_TICK = 281
READY_ID1_TICK = 550
READY_ID2_TICK = 550
GRIPPER_CLOSED_TICK = 500
GRIPPER_OPEN_TICK = 900

ID1_SAFE_LIMITS = (150, 710)
ID2_SAFE_LIMITS = (252, 769)
MIN_ANGLE_GAP_DEG = 20.0

BASE_HEIGHT_MM = 35.0
BLUE_ACTIVE_LENGTH_MM = 238.0
ORANGE_EXTENSION_LENGTH_MM = 190.6
LEVEL_TIP_LENGTH_MM = 60.0

JOINT_NAMES = [
    "servo_blue",
    "blue_chain_passive",
    "servo_orange",
    "orange_chain_passive",
    "extension_level",
    "gripper_left",
    "gripper_right",
]


def id1_degrees(tick):
    return 180.0 + (tick - 470.0) * (90.0 / 360.0)


def id2_degrees(tick):
    return 90.0 + (tick - 510.0) * (90.0 / -388.0)


def id2_tick_from_degrees(degrees):
    return 510.0 + (degrees - 90.0) / (90.0 / -388.0)


def angle_gap_degrees(id1_tick, id2_tick):
    return abs(id1_degrees(id1_tick) - id2_degrees(id2_tick))


def has_safe_angle_gap(id1_tick, id2_tick, minimum_degrees=MIN_ANGLE_GAP_DEG):
    return angle_gap_degrees(id1_tick, id2_tick) > minimum_degrees


def enforce_angle_gap(id1_tick, id2_tick, id2_limits, minimum_degrees=MIN_ANGLE_GAP_DEG):
    id1_tick = int(round(id1_tick))
    id2_tick = int(round(id2_tick))
    if has_safe_angle_gap(id1_tick, id2_tick, minimum_degrees):
        return id1_tick, id2_tick

    id1_angle = id1_degrees(id1_tick)
    id2_angle = id2_degrees(id2_tick)
    required_gap = minimum_degrees + 0.1
    target_angle = id1_angle - required_gap if id2_angle <= id1_angle else id1_angle + required_gap
    candidate = int(round(id2_tick_from_degrees(target_angle)))
    candidate = max(id2_limits[0], min(id2_limits[1], candidate))
    direction = -1 if id2_degrees(candidate) >= id1_angle else 1
    while not has_safe_angle_gap(id1_tick, candidate, minimum_degrees):
        next_candidate = candidate + direction
        if next_candidate < id2_limits[0] or next_candidate > id2_limits[1]:
            raise ValueError("ID1/ID2 cannot satisfy the configured angle gap")
        candidate = next_candidate
    return id1_tick, candidate


def gripper_position_mm(id1_tick, id2_tick):
    orange = math.radians(id1_degrees(id1_tick))
    blue = math.radians(id2_degrees(id2_tick))
    x_mm = (
        BLUE_ACTIVE_LENGTH_MM * math.cos(blue)
        - ORANGE_EXTENSION_LENGTH_MM * math.cos(orange)
        + LEVEL_TIP_LENGTH_MM
    )
    z_mm = (
        BASE_HEIGHT_MM
        + BLUE_ACTIVE_LENGTH_MM * math.sin(blue)
        - ORANGE_EXTENSION_LENGTH_MM * math.sin(orange)
    )
    return x_mm, z_mm


def solve_gripper_position(
    target_x_mm,
    target_z_mm,
    seed_id1,
    seed_id2,
    id1_limits=ID1_SAFE_LIMITS,
    id2_limits=ID2_SAFE_LIMITS,
    minimum_gap_degrees=MIN_ANGLE_GAP_DEG,
):
    coarse_id1 = np.arange(id1_limits[0], id1_limits[1] + 1, 4, dtype=float)
    coarse_id2 = np.arange(id2_limits[0], id2_limits[1] + 1, 4, dtype=float)
    grid_id1, grid_id2 = np.meshgrid(coarse_id1, coarse_id2, indexing="ij")
    best = _best_grid_target(
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
    return _best_grid_target(
        grid_id1,
        grid_id2,
        target_x_mm,
        target_z_mm,
        seed_id1,
        seed_id2,
        minimum_gap_degrees,
    )


def _best_grid_target(
    grid_id1,
    grid_id2,
    target_x_mm,
    target_z_mm,
    seed_id1,
    seed_id2,
    minimum_gap_degrees,
):
    orange = np.deg2rad(id1_degrees(grid_id1))
    blue = np.deg2rad(id2_degrees(grid_id2))
    x_mm = (
        BLUE_ACTIVE_LENGTH_MM * np.cos(blue)
        - ORANGE_EXTENSION_LENGTH_MM * np.cos(orange)
        + LEVEL_TIP_LENGTH_MM
    )
    z_mm = (
        BASE_HEIGHT_MM
        + BLUE_ACTIVE_LENGTH_MM * np.sin(blue)
        - ORANGE_EXTENSION_LENGTH_MM * np.sin(orange)
    )
    gap = np.abs(id1_degrees(grid_id1) - id2_degrees(grid_id2))
    position_error = np.square(x_mm - target_x_mm) + np.square(z_mm - target_z_mm)
    seed_cost = 0.002 * (np.square(grid_id1 - seed_id1) + np.square(grid_id2 - seed_id2))
    cost = np.where(gap > minimum_gap_degrees, position_error + seed_cost, np.inf)
    index = np.unravel_index(np.argmin(cost), cost.shape)
    id1_tick = int(grid_id1[index])
    id2_tick = int(grid_id2[index])
    error_mm = math.sqrt(float(position_error[index]))
    return id1_tick, id2_tick, error_mm


def joint_positions(id1_tick, id2_tick, id4_tick):
    orange = math.radians(id1_degrees(id1_tick))
    blue = math.radians(id2_degrees(id2_tick))
    gripper_ratio = max(0.0, min(1.0, (id4_tick - GRIPPER_CLOSED_TICK) / 400.0))
    return [
        blue,
        orange - blue,
        orange,
        blue - orange,
        -orange,
        -0.05 - 0.75 * gripper_ratio,
        0.05 + 0.75 * gripper_ratio,
    ]
