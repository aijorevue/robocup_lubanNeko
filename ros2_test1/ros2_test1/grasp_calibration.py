"""Measured task-two grasp mapping for the 85KG ID1/ID2 arm pair."""

from __future__ import annotations


GRASP_MIN_DISTANCE_CM = 10.0
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


def calibrated_grasp_ticks(distance_cm: float) -> tuple[int, int]:
    """Linearly interpolate the measured ID1/ID2 pose for 10-30 cm."""

    distance_cm = float(distance_cm)
    if not GRASP_MIN_DISTANCE_CM <= distance_cm <= GRASP_MAX_DISTANCE_CM:
        raise ValueError(
            f"grasp distance {distance_cm:.1f}cm outside "
            f"{GRASP_MIN_DISTANCE_CM:.0f}-{GRASP_MAX_DISTANCE_CM:.0f}cm"
        )

    for lower, upper in zip(GRASP_DISTANCE_TICKS, GRASP_DISTANCE_TICKS[1:]):
        if lower[0] <= distance_cm <= upper[0]:
            span = upper[0] - lower[0]
            ratio = 0.0 if span <= 0.0 else (distance_cm - lower[0]) / span
            id1 = round(lower[1] + (upper[1] - lower[1]) * ratio)
            id2 = round(lower[2] + (upper[2] - lower[2]) * ratio)
            return int(id1), int(id2)

    return GRASP_DISTANCE_TICKS[-1][1], GRASP_DISTANCE_TICKS[-1][2]
