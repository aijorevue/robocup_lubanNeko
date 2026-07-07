"""Ball detector: solid colored circles"""
import cv2
import numpy as np
from ..colors import COLORS, ALL_COLORS
from ..detect_utils import color_mask, get_contours, circularity, get_center_radius


def detect_balls(hsv, colors=None):
    if colors is None: colors = ALL_COLORS
    results = []
    for cname in colors:
        cfg = COLORS[cname]
        mask = color_mask(hsv, cname)
        cnts = get_contours(mask, min_area=150)
        for c in cnts:
            circ = circularity(c)
            if circ < 0.75: continue
            if cv2.contourArea(c) < 200: continue
            x, y, r = get_center_radius(c)
            if r < 10: continue
            results.append({'center':(x,y), 'radius':r, 'color':cname,
                           'bgr':cfg['bgr'], 'type':'ball', 'circularity':round(circ,2)})
    return results
