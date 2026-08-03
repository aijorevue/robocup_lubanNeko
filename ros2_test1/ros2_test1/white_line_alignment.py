"""Detect the field reference strip used after the DISC entry arc."""

from __future__ import annotations

import cv2
import numpy as np


class WhiteLineAlignmentDetector:
    def __init__(
        self,
        saturation_max=80,
        value_min=175,
        roi_x=(0.02, 0.98),
        roi_y=(0.18, 0.88),
    ):
        self.saturation_max = int(saturation_max)
        self.value_min = int(value_min)
        self.roi_x = roi_x
        self.roi_y = roi_y

    @staticmethod
    def _fitted_line(contour, frame_width, frame_height):
        component = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
        center_band = np.zeros_like(component)
        band_x0 = int(frame_width * 0.25)
        band_x1 = int(frame_width * 0.75)
        center_band[:, band_x0:band_x1] = component[:, band_x0:band_x1]
        ys, xs = np.nonzero(center_band)
        if len(xs) < 100:
            return None, None
        points = np.column_stack((xs, ys)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(
            points, cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1)
        if vx < 0:
            vx, vy = -vx, -vy
        angle_deg = float(np.degrees(np.arctan2(vy, vx)))
        if abs(float(vx)) < 1e-6:
            return angle_deg, None
        y_at_center = float(
            y0 + (vy / vx) * (frame_width * 0.5 - x0)
        )
        return angle_deg, y_at_center

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((0, 0, self.value_min), dtype=np.uint8),
            np.array((179, self.saturation_max, 255), dtype=np.uint8),
        )
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        x0 = int(width * self.roi_x[0])
        x1 = int(width * self.roi_x[1])
        y0 = int(height * self.roi_y[0])
        y1 = int(height * self.roi_y[1])
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            rect = cv2.minAreaRect(contour)
            length = float(max(rect[1]))
            thickness = float(min(rect[1]))
            if length < width * 0.15:
                continue
            if thickness < height * 0.01 or thickness > height * 0.12:
                continue
            if length / max(1.0, thickness) < 6.0:
                continue
            if area < width * height * 0.003:
                continue
            candidates.append(
                (area, contour, rect, (x, y, box_width, box_height))
            )

        if not candidates:
            return None

        area, contour, rect, bounds = max(candidates, key=lambda item: item[0])
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return None
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
        length = float(max(rect[1]))
        thickness = float(min(rect[1]))
        angle_deg, y_at_center = self._fitted_line(
            contour, width, height
        )
        if y_at_center is None:
            return None
        return {
            "center_x": center_x,
            "center_y": center_y,
            "angle_deg": angle_deg,
            "y_at_center": y_at_center,
            "length": length,
            "thickness": thickness,
            "area": area,
            "bounds": bounds,
            "frame_width": width,
            "frame_height": height,
        }
