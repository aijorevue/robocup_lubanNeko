from .common import build_mask, detect_color_balls


COLOR_NAME = "blue"
# Keep blue separate from the green/cyan field and low-saturation purple-blue
# chassis parts visible at the lower-right edge of the main camera.
COLOR_RANGES = [((96, 160, 24), (115, 255, 255))]
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


def mask(hsv):
    return build_mask(hsv, COLOR_RANGES, MASK_SETTINGS)


def detect(hsv, rings):
    return detect_color_balls(
        hsv,
        COLOR_NAME,
        COLOR_RANGES,
        MASK_SETTINGS,
        BALL_SETTINGS,
        rings,
    )
