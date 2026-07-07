from .common import build_mask, detect_white_balls


COLOR_NAME = "white"
COLOR_RANGES = [((0, 0, 170), (180, 48, 255))]
MASK_SETTINGS = {"kernel_size": 3, "close_iterations": 1}
BALL_SETTINGS = {
    "min_area": 170,
    "min_circularity": 0.66,
    "min_radius": 8,
    "min_fill": 0.58,
    "min_center_fill": 0.38,
    "max_non_white_ratio": 0.08,
    "max_mean_saturation": 38.0,
    "min_mean_value": 185.0,
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
