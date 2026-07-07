from .common import build_mask, detect_color_balls


COLOR_NAME = "blue"
COLOR_RANGES = [((95, 50, 50), (135, 255, 255))]
MASK_SETTINGS = {"kernel_size": 5, "close_iterations": 2}
BALL_SETTINGS = {
    "min_area": 180,
    "min_circularity": 0.68,
    "min_radius": 8,
    "min_fill": 0.64,
    "min_center_fill": 0.52,
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
