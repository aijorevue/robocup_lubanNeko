import cv2
import numpy as np


def build_mask(hsv, color_ranges, mask_settings):
    mask = None
    for low, high in color_ranges:
        part = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = part if mask is None else cv2.bitwise_or(mask, part)

    kernel = np.ones(
        (mask_settings["kernel_size"], mask_settings["kernel_size"]),
        np.uint8,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=mask_settings["close_iterations"],
    )
    return mask


def external_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def circularity(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    return 0 if peri == 0 else 4 * np.pi * area / (peri * peri)


def looks_like_square(contour, min_area=180, min_fill=0.60, max_aspect=1.85):
    area = cv2.contourArea(contour)
    if area < min_area:
        return False

    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return False

    poly = cv2.approxPolyDP(contour, 0.035 * peri, True)
    if len(poly) < 4 or len(poly) > 6:
        return False

    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) < 12:
        return False

    rect_area = w * h
    fill = area / rect_area if rect_area > 0 else 0
    aspect = max(w, h) / min(w, h) if min(w, h) else 999
    return fill >= min_fill and aspect <= max_aspect


def center_fill(mask, cx, cy, radius):
    radius = max(3, int(radius * 0.28))
    center_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(center_mask, (int(cx), int(cy)), radius, 255, -1)
    center_area = np.count_nonzero(center_mask)
    if center_area == 0:
        return 0.0
    colored_area = np.count_nonzero(cv2.bitwise_and(mask, center_mask))
    return colored_area / center_area


def overlaps_existing_ring(bbox, color, rings):
    x, y, w, h = bbox
    area = max(1, w * h)
    for ring in rings:
        if ring.get("color") != color:
            continue
        rx, ry, rw, rh = ring["bbox"]
        ix0 = max(x, rx)
        iy0 = max(y, ry)
        ix1 = min(x + w, rx + rw)
        iy1 = min(y + h, ry + rh)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        overlap = (ix1 - ix0) * (iy1 - iy0) / area
        if overlap > 0.45:
            return True
    return False


def detect_color_balls(hsv, color_name, color_ranges, mask_settings, ball_settings, rings):
    results = []
    mask = build_mask(hsv, color_ranges, mask_settings)
    contours = external_contours(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < ball_settings["min_area"]:
            continue
        if looks_like_square(contour, min_area=ball_settings["min_area"]):
            continue
        contour_circularity = circularity(contour)
        if contour_circularity < ball_settings["min_circularity"]:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < ball_settings["min_radius"]:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if overlaps_existing_ring((x, y, w, h), color_name, rings):
            continue

        fill = area / (np.pi * radius * radius) if radius > 0 else 0
        contour_center_fill = center_fill(mask, cx, cy, radius)
        if (
            fill < ball_settings["min_fill"]
            or contour_center_fill < ball_settings["min_center_fill"]
        ):
            continue

        results.append(
            {
                "kind": "ball",
                "color": color_name,
                "center": (int(cx), int(cy)),
                "radius": int(radius),
                "bbox": (x, y, w, h),
                "score": round(contour_circularity, 2),
            }
        )
    return results
