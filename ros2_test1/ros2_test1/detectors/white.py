from .common import build_mask, detect_white_balls


COLOR_NAME = "white"
COLOR_RANGES = [((0, 0, 150), (180, 58, 255))]
MASK_SETTINGS = {"kernel_size": 5, "close_iterations": 2}
BALL_SETTINGS = {
    "min_area": 220,
    "min_circularity": 0.55,
    "min_radius": 10,
    "min_fill": 0.56,
    "min_center_fill": 0.48,
    "max_non_white_ratio": 0.06,
    "max_surrounding_non_white_ratio": 0.18,
    "max_mean_saturation": 45.0,
    "min_mean_value": 165.0,
}
DRAW_COLOR = (220, 220, 220)


def mask(hsv):
    return build_mask(hsv, COLOR_RANGES, MASK_SETTINGS)


def detect(hsv, occupied_detections, non_white_mask):
    return detect_white_balls(
        hsv,
        COLOR_NAME,
        COLOR_RANGES,
        MASK_SETTINGS,
        BALL_SETTINGS,
        occupied_detections,
        non_white_mask,
    )
