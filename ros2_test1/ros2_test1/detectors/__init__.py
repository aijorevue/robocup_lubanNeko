from . import blue, red, white, yellow


BALL_DETECTORS = (yellow, red, blue)
BALL_COLORS = tuple(detector.COLOR_NAME for detector in BALL_DETECTORS)
SHAPE_COLORS = (red.COLOR_NAME, blue.COLOR_NAME)
DRAW_COLORS = {
    yellow.COLOR_NAME: yellow.DRAW_COLOR,
    red.COLOR_NAME: red.DRAW_COLOR,
    blue.COLOR_NAME: blue.DRAW_COLOR,
    white.COLOR_NAME: white.DRAW_COLOR,
}
MASK_BUILDERS = {
    yellow.COLOR_NAME: yellow.mask,
    red.COLOR_NAME: red.mask,
    blue.COLOR_NAME: blue.mask,
    white.COLOR_NAME: white.mask,
}
WHITE_DETECTOR = white
