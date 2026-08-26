from .common import build_mask, detect_color_balls


COLOR_NAME = "blue"
# Keep the hue floor above the green field surface.  The previous value of 80
# merged the ball with the field into one contour under the main camera.
COLOR_RANGES = [((90, 24, 24), (150, 255, 255))]
MASK_SETTINGS = {"kernel_size": 3, "close_iterations": 2}
BALL_SETTINGS = {
    # The two far task-one balls measure about 4.9k and 6.5k px^2 in the
    # 800x600 source frame. Keep smaller blue clutter rejected while allowing
    # those valid balls through the existing shape and fill checks.
    # Full-resolution task-one detection is retained so this floor is
    # applied to the source image rather than to a 0.75-scaled frame.
    "min_area": 4500,
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
