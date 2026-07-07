"""Square detector: colored rectangular objects with 4 corners"""
import cv2
import numpy as np
from ..colors import COLORS, ALL_COLORS
from ..detect_utils import color_mask, get_contours, approx_poly, circularity


def detect_squares(hsv, colors=None):
    if colors is None: colors = ALL_COLORS
    results = []
    for cname in colors:
        cfg = COLORS[cname]
        mask = color_mask(hsv, cname)
        cnts = get_contours(mask, min_area=500)
        for c in cnts:
            circ = circularity(c)
            if circ > 0.7: continue
            poly = approx_poly(c, epsilon_factor=0.03)
            if len(poly) != 4: continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), angle = rect
            if min(w, h) < 15: continue
            area_ratio = cv2.contourArea(c) / (w * h) if w * h > 0 else 0
            if area_ratio < 0.7: continue
            results.append({'center':(int(cx),int(cy)), 'width':int(w), 'height':int(h),
                           'angle':round(angle,1), 'color':cname, 'bgr':cfg['bgr'],
                           'type':'square', 'fill':round(area_ratio,2)})
    return results
