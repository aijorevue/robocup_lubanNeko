"""Task-two ring geometry/depth shared with the inspected standalone App.

Kept separate so formal task-one ball/ring exclusion is unaffected.
"""
import cv2
import numpy as np

RING_SIZE_MM = 55.0
RING_DISTANCE_EXTRA_CM = 0.5

def detect_rings(frame: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = []
    ranges = {"red": [((0, 80, 70), (10, 255, 255)),
                       ((165, 80, 70), (180, 255, 255))],
              "blue": [((95, 50, 50), (135, 255, 255))]}
    for color, limits in ranges.items():
        mask = None
        for low, high in limits:
            part = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            child_index = hierarchy[0][index][2]
            if child_index < 0:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
            (cx, cy), outer = cv2.minEnclosingCircle(contour)
            child = contours[child_index]
            inner_area = cv2.contourArea(child)
            (_, _), inner = cv2.minEnclosingCircle(child)
            if area < 300 or circularity < 0.55 or outer < 14 or inner < 5:
                continue
            ratio = inner / outer if outer else 0
            if not 0.20 <= ratio <= 0.75 or inner_area < 40:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
            area_percent = (max(float(w * h), float(area)) / frame_area) * 100.0
            distance_cm = (-1.6072186919749336 + RING_DISTANCE_EXTRA_CM
                           + 31.628878020276648 * RING_SIZE_MM / 42.67
                           / max(0.001, float(area_percent) ** 0.5))
            output.append({"kind": "ring", "color": color,
                           "center": (int(cx), int(cy)),
                           "radius": int(outer), "outer_radius": int(outer),
                           "inner_radius": int(inner), "bbox": (x, y, w, h),
                           "score": circularity, "distance_cm": distance_cm,
                           "depth_cm": distance_cm, "depth_mm": distance_cm * 10.0,
                           "area_percent": area_percent})
    return output

