from .common import build_mask, detect_color_balls


COLOR_NAME = "blue"
# Keep blue separate from the green/cyan field and low-saturation purple-blue
# chassis parts visible at the lower-right edge of the main camera.
COLOR_RANGES = [((96, 160, 24), (115, 255, 255))]
# Ball main pass only; preserve the shared ring mask and shadow recovery.
BALL_COLOR_RANGES = [((96, 160, 24), (119, 255, 255))]
MASK_SETTINGS = {"kernel_size": 3, "close_iterations": 2}
BALL_SETTINGS = {
    # Reject small blue clutter and partial edge structures. Formal task-one
    # detection uses the original 800x600 source frame.
    "min_area": 9000,
    # Perspective and partial overlap distort the farther ball to about 0.29
    # circularity and 0.43 fill in the real task-one camera view.
    "min_circularity": 0.28,
    "min_radius": 14,
    "min_fill": 0.40,
    "min_center_fill": 0.42,
    # Keep large square artifacts rejected while preserving small distant
    # circular balls whose polygon approximation has only a few vertices.
    "square_filter_min_area": 10000,
}
DRAW_COLOR = (255, 0, 0)
# Ball-only recovery calibrated on the shadow photo set. Keep mask() unchanged
# because task-two ring detection imports it through MASK_BUILDERS.
SHADOW_BALL_COLOR_RANGES = [((96, 50, 15), (120, 255, 255))]
SHADOW_BALL_SETTINGS = {
    **BALL_SETTINGS,
    "min_circularity": 0.68,
    "min_fill": 0.64,
    "min_center_fill": 0.52,
    "square_filter_min_area": BALL_SETTINGS["min_area"],
}


def mask(hsv):
    return build_mask(hsv, COLOR_RANGES, MASK_SETTINGS)


def detect(hsv, rings):
    results = detect_color_balls(
        hsv,
        COLOR_NAME,
        BALL_COLOR_RANGES,
        MASK_SETTINGS,
        BALL_SETTINGS,
        rings,
    )
    recovered = detect_color_balls(
        hsv,
        COLOR_NAME,
        SHADOW_BALL_COLOR_RANGES,
        MASK_SETTINGS,
        SHADOW_BALL_SETTINGS,
        rings,
    )
    # Keep existing localization/depth inputs; add only a previously missing
    # ball. Both passes keep the formal area/radius and ring-overlap checks.
    for candidate in recovered:
        x, y, width, height = candidate["bbox"]
        duplicate = False
        for existing in results:
            ex, ey, ew, eh = existing["bbox"]
            intersection = (
                max(0, min(x + width, ex + ew) - max(x, ex))
                * max(0, min(y + height, ey + eh) - max(y, ey))
            )
            union = width * height + ew * eh - intersection
            if intersection / max(1, union) >= 0.30:
                duplicate = True
                break
        if not duplicate:
            results.append(candidate)
    return results
