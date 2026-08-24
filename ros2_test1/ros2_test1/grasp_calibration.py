"""Measured task-two grasp mapping for the 85KG ID1/ID2 arm pair."""

from __future__ import annotations


GRASP_MIN_DISTANCE_CM = 7.0
GRASP_MAX_DISTANCE_CM = 30.0

# Measured letter-block poses. Rings use the same pose at the measured depth.
GRASP_DISTANCE_TICKS = (
    (10.0, 540, 450),
    (11.0, 530, 470),
    (12.0, 520, 490),
    (13.0, 510, 510),
    (14.0, 493, 507),
    (15.0, 477, 503),
    (16.0, 460, 500),
    (18.5, 460, 490),
    (19.0, 453, 488),
    (20.0, 438, 483),
    (21.5, 415, 475),
    (25.0, 363, 458),
    (30.0, 288, 433),
)

NEAR_ID2_START_DISTANCE_CM = 7.0
NEAR_ID2_END_DISTANCE_CM = 9.0
NEAR_ID2_START_TICK = 540
NEAR_ID2_END_TICK = 530


def calibrated_grasp_ticks(distance_cm: float) -> tuple[int, int]:
    """Interpolate the measured pose for 7-30 cm.

    The first measured segment (10-11 cm) is linearly extrapolated down to
    7 cm until dedicated 7-10 cm calibration points are available.
    """

    distance_cm = float(distance_cm)
    if not GRASP_MIN_DISTANCE_CM <= distance_cm <= GRASP_MAX_DISTANCE_CM:
        raise ValueError(
            f"grasp distance {distance_cm:.1f}cm outside "
            f"{GRASP_MIN_DISTANCE_CM:.0f}-{GRASP_MAX_DISTANCE_CM:.0f}cm"
        )

    if distance_cm < GRASP_DISTANCE_TICKS[0][0]:
        lower, upper = GRASP_DISTANCE_TICKS[:2]
    else:
        lower = upper = None
        for candidate_lower, candidate_upper in zip(
            GRASP_DISTANCE_TICKS, GRASP_DISTANCE_TICKS[1:]
        ):
            if candidate_lower[0] <= distance_cm <= candidate_upper[0]:
                lower, upper = candidate_lower, candidate_upper
                break
        if lower is None:
            lower = upper = GRASP_DISTANCE_TICKS[-1]

    span = upper[0] - lower[0]
    ratio = 0.0 if span <= 0.0 else (distance_cm - lower[0]) / span
    # Apply the current mechanical calibration offset after interpolation.
    # This affects only distance-derived descent poses, not fixed poses.
    id1 = round(lower[1] + (upper[1] - lower[1]) * ratio) + 50
    id2 = round(lower[2] + (upper[2] - lower[2]) * ratio)
    if NEAR_ID2_START_DISTANCE_CM <= distance_cm <= NEAR_ID2_END_DISTANCE_CM:
        near_ratio = (
            (distance_cm - NEAR_ID2_START_DISTANCE_CM)
            / (NEAR_ID2_END_DISTANCE_CM - NEAR_ID2_START_DISTANCE_CM)
        )
        id2 = round(
            NEAR_ID2_START_TICK
            + (NEAR_ID2_END_TICK - NEAR_ID2_START_TICK) * near_ratio
        )
    return int(id1), int(id2)
