"""Ring detector: colored circular ring with a hole"""
import cv2
import numpy as np
from ..colors import COLORS, ALL_COLORS
from ..detect_utils import color_mask, circularity, get_center_radius


def detect_rings(hsv, colors=None):
    if colors is None: colors = ALL_COLORS
    results = []
    for cname in colors:
        cfg = COLORS[cname]
        mask = color_mask(hsv, cname)
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None: continue
        used_holes = set()
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < 300: continue
            circ = circularity(c)
            if circ < 0.65: continue
            x, y, outer_r = get_center_radius(c)
            if outer_r < 15: continue

            has_hole = False; inner_r = 0
            if hierarchy[0][i][2] != -1:
                child_idx = hierarchy[0][i][2]
                child = cnts[child_idx]
                child_area = cv2.contourArea(child)
                child_circ = circularity(child)
                cx, cy, cr = get_center_radius(child)

                # 子轮廓必须在父轮廓内部且半径足够小,形状接近圆
                dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if (child_circ > 0.5 and cr > 5 and
                    dist < outer_r * 0.3 and cr < outer_r * 0.7):
                    has_hole = True
                    inner_r = cr
                    used_holes.add(child_idx)

            if has_hole:
                results.append({'center': (x, y), 'outer_r': outer_r, 'inner_r': inner_r,
                                'color': cname, 'bgr': cfg['bgr'], 'type': 'ring'})
    return results
