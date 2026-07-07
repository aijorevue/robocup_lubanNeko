"""Shared detection utilities"""
import cv2
import numpy as np
from .colors import COLORS


def make_mask(hsv, low, high):
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def color_mask(hsv, cname):
    """统一颜色掩码，自动处理 red 双段"""
    cfg = COLORS[cname]
    if cfg.get('dual'):
        m1 = make_mask(hsv, cfg['low1'], cfg['high1'])
        m2 = make_mask(hsv, cfg['low2'], cfg['high2'])
        return cv2.bitwise_or(m1, m2)
    return make_mask(hsv, cfg['low'], cfg['high'])


def get_contours(mask, min_area=100):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= min_area]


def circularity(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0: return 0
    return 4 * np.pi * area / (peri * peri)


def get_center_radius(contour):
    (x, y), r = cv2.minEnclosingCircle(contour)
    return int(x), int(y), int(r)


def approx_poly(contour, epsilon_factor=0.04):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon_factor * peri, True)
