from .common import build_mask, detect_color_balls


COLOR_NAME = "red"
COLOR_RANGES = [
    ((0, 80, 70), (10, 255, 255)),
    ((165, 80, 70), (180, 255, 255)),
]
MASK_SETTINGS = {"kernel_size": 3, "close_iterations": 1}
BALL_SETTINGS = {
    "min_area": 900,
    "min_circularity": 0.68,
    "min_radius": 8,
    "min_fill": 0.58,
    "min_center_fill": 0.45,
}
DRAW_COLOR = (0, 0, 255)


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
