from .common import build_mask, detect_color_balls


COLOR_NAME = "yellow"
COLOR_RANGES = [((15, 55, 55), (42, 255, 255))]
MASK_SETTINGS = {"kernel_size": 3, "close_iterations": 1}
BALL_SETTINGS = {
    "min_area": 120,
    "min_circularity": 0.58,
    "min_radius": 7,
    "min_fill": 0.50,
    "min_center_fill": 0.36,
}
DRAW_COLOR = (0, 255, 255)


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
