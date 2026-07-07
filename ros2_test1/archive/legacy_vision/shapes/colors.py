"""HSV color ranges for 5 colors"""

COLORS = {
    'yellow': {'low': (20, 80, 80), 'high': (35, 255, 255), 'bgr': (0, 255, 255)},
    'blue':   {'low': (95, 50, 50), 'high': (135, 255, 255), 'bgr': (255, 0, 0)},
    'green':  {'low': (40, 80, 80), 'high': (80, 255, 255), 'bgr': (0, 255, 0)},
    'white':  {'low': (0, 0, 200), 'high': (180, 30, 255), 'bgr': (255, 255, 255)},
    'red':    {
        'low1': (0, 80, 80), 'high1': (10, 255, 255),
        'low2': (160, 80, 80), 'high2': (180, 255, 255),
        'bgr': (0, 0, 255), 'dual': True
    },
}

ALL_COLORS = ['yellow', 'blue', 'green', 'red', 'white']
