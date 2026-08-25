from .common import build_mask, detect_color_balls


COLOR_NAME = "red"
COLOR_RANGES = [
    ((0, 80, 80), (10, 255, 255)),
    ((160, 80, 80), (180, 255, 255)),
]
MASK_SETTINGS = {"kernel_size": 5, "close_iterations": 2}
BALL_SETTINGS = {
    "min_area": 180,
    "min_circularity": 0.68,
    "min_radius": 8,
    "min_fill": 0.64,
    "min_center_fill": 0.52,
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
