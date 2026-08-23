"""Detect the field reference strip used after the DISC entry arc."""

from __future__ import annotations

import cv2
import numpy as np


class WhiteLineAlignmentDetector:
    def __init__(
        self,
        saturation_max=95,
        value_min=160,
        roi_x=(0.02, 0.98),
        roi_y=(0.15, 0.92),
        max_hold_frames=4,
    ):
        self.saturation_max = int(saturation_max)
        self.value_min = int(value_min)
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.max_hold_frames = max(0, int(max_hold_frames))
        self._last_measurement = None
        self._missed_frames = 0

    def reset_tracking(self):
        """Discard stale geometry when a new white-line phase begins."""
        self._last_measurement = None
        self._missed_frames = 0

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

    @staticmethod
    def _local_contrast(gray, x, y, box_width, box_height):
        """Compare the bright band with narrow strips immediately around it."""
        height, width = gray.shape[:2]
        cx0 = max(0, int(x + box_width * 0.10))
        cx1 = min(width, int(x + box_width * 0.90))
        if cx1 <= cx0:
            return 0.0
        band_y0 = max(0, int(y + box_height * 0.20))
        band_y1 = min(height, int(y + box_height * 0.80))
        thickness = max(3, int(box_height * 0.65))
        above_y0 = max(0, band_y0 - thickness)
        below_y1 = min(height, band_y1 + thickness)
        band = gray[band_y0:band_y1, cx0:cx1]
        above = gray[above_y0:band_y0, cx0:cx1]
        below = gray[band_y1:below_y1, cx0:cx1]
        if band.size == 0 or (above.size == 0 and below.size == 0):
            return 0.0
        outside_values = []
        if above.size:
            outside_values.append(float(np.median(above)))
        if below.size:
            outside_values.append(float(np.median(below)))
        return float(np.median(band)) - float(np.median(outside_values))

    def _candidate_score(self, candidate, gray, width, height):
        area, contour, rect, bounds = candidate
        x, y, box_width, box_height = bounds
        length = float(max(rect[1]))
        thickness = float(min(rect[1]))
        span = min(1.0, box_width / max(1.0, width))
        slenderness = min(1.0, length / max(1.0, 7.0 * thickness))
        contrast = self._local_contrast(
            gray, x, y, box_width, box_height
        )
        contrast_score = float(np.clip((contrast - 5.0) / 55.0, 0.0, 1.0))
        lower_region_score = float(np.clip(y / max(1.0, height), 0.0, 1.0))
        component = np.zeros((box_height, box_width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)
        horizontal_coverage = float(
            np.count_nonzero(np.any(component > 0, axis=0)) /
            max(1.0, box_width)
        )
        fill_ratio = float(np.count_nonzero(component) /
                           max(1.0, box_width * box_height))
        coverage_score = float(np.clip((horizontal_coverage - 0.55) / 0.35,
                                       0.0, 1.0))
        strip_fill_score = float(
            np.clip(1.0 - abs(fill_ratio - 0.48) / 0.30, 0.0, 1.0)
        )
        score = (
            2.8 * span
            + 2.0 * slenderness
            + 2.4 * contrast_score
            + 0.8 * lower_region_score
            + 1.8 * coverage_score
            + 1.0 * strip_fill_score
        )
        if self._last_measurement is not None:
            previous_y = float(self._last_measurement["y_at_center"])
            previous_angle = float(self._last_measurement["angle_deg"])
            angle, y_at_center = self._fitted_line(contour, width, height)
            if y_at_center is not None:
                y_delta = abs(y_at_center - previous_y) / max(1.0, height)
                angle_delta = abs(angle - previous_angle) / 45.0
                score += 2.5 * max(0.0, 1.0 - min(1.0, y_delta / 0.30))
                score += 1.2 * max(0.0, 1.0 - min(1.0, angle_delta))
        return score

    def _fallback_measurement(self, frame):
        """Find a long ground strip when exposure defeats the strict mask."""
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0 = int(width * self.roi_x[0])
        x1 = int(width * self.roi_x[1])
        # The reference strip is on the ground. Keeping the fallback below
        # the upper third prevents the overhead white box from winning.
        y0 = max(int(height * 0.28), int(height * self.roi_y[0]))
        y1 = min(height, int(height * 0.96))
        if x1 <= x0 or y1 <= y0:
            return None

        value = hsv[:, :, 2]
        roi_value = value[y0:y1, x0:x1]
        loose_value = max(115, int(np.percentile(roi_value, 58)))
        loose_saturation = min(150, self.saturation_max + 45)
        bright_mask = cv2.inRange(
            hsv,
            np.array((0, 0, loose_value), dtype=np.uint8),
            np.array((179, loose_saturation, 255), dtype=np.uint8),
        )

        # Top-hat emphasizes a bright strip against a locally darker floor,
        # so the fallback remains useful when global brightness changes.
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        top_hat = cv2.morphologyEx(
            gray_blur,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (61, 21)),
        )
        roi_top_hat = top_hat[y0:y1, x0:x1]
        top_hat_value = max(10, int(np.percentile(roi_top_hat, 88)))
        local_bright_mask = cv2.inRange(
            top_hat,
            top_hat_value,
            255,
        )

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_or(bright_mask, local_bright_mask)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        best_score = -1.0
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if box_width < width * 0.40 or box_height > height * 0.16:
                continue
            if area < width * height * 0.0008:
                continue

            rect = cv2.minAreaRect(contour)
            length = float(max(rect[1]))
            thickness = float(min(rect[1]))
            if thickness < height * 0.004:
                continue
            if length / max(1.0, thickness) < 4.0:
                continue

            angle_deg, y_at_center = self._fitted_line(
                contour, width, height
            )
            if (
                y_at_center is None
                or y_at_center < height * 0.28
                or y_at_center >= height * 0.96
                or abs(angle_deg) > 32.0
            ):
                continue

            component = np.zeros((box_height, box_width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)
            fill_ratio = float(
                np.count_nonzero(component) /
                max(1.0, box_width * box_height)
            )
            horizontal_coverage = float(
                np.count_nonzero(np.any(component > 0, axis=0)) /
                max(1.0, box_width)
            )
            if horizontal_coverage < 0.55 or fill_ratio > 0.82:
                continue

            contrast = self._local_contrast(
                gray, x, y, box_width, box_height
            )
            if contrast < 5.0:
                continue

            score = (
                3.0 * min(1.0, box_width / max(1.0, width))
                + 2.0 * min(1.0, length / max(1.0, 7.0 * thickness))
                + 2.0 * horizontal_coverage
                + 1.5 * float(np.clip((contrast - 5.0) / 55.0, 0.0, 1.0))
                + 0.8 * float(np.clip(y_at_center / height, 0.0, 1.0))
            )
            if score > best_score:
                best = (contour, rect, (x, y, box_width, box_height), area)
                best_score = score

        if best is None:
            return None

        contour, rect, bounds, area = best
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
            "held": False,
        }

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        value_eq = clahe.apply(hsv[:, :, 2])
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        x0 = int(width * self.roi_x[0])
        x1 = int(width * self.roi_x[1])
        y0 = int(height * self.roi_y[0])
        y1 = int(height * self.roi_y[1])
        roi_mask[y0:y1, x0:x1] = 255
        roi_value = value_eq[y0:y1, x0:x1]
        roi_gray = gray_eq[y0:y1, x0:x1]
        adaptive_value = max(self.value_min, int(np.percentile(roi_value, 72)))
        adaptive_gray = max(self.value_min, int(np.percentile(roi_gray, 74)))
        white_mask = cv2.inRange(
            hsv,
            np.array((0, 0, adaptive_value), dtype=np.uint8),
            np.array((179, self.saturation_max, 255), dtype=np.uint8),
        )
        gray_mask = cv2.inRange(gray_eq, adaptive_gray, 255)
        mask = cv2.bitwise_or(white_mask, gray_mask)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
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
            if length < width * 0.25:
                continue
            if thickness < height * 0.005 or thickness > height * 0.10:
                continue
            # Reject bright structures attached to the top of the search ROI.
            # This is geometric, so it remains valid when the line moves in Y.
            if y <= y0 + height * 0.04 and box_height > height * 0.06:
                continue
            # A large bright box can be long and rectangular too, but it is
            # substantially thicker and more solid than the field strip.
            if box_height > height * 0.12:
                continue
            component = np.zeros((box_height, box_width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)
            fill_ratio = float(np.count_nonzero(component) /
                               max(1.0, box_width * box_height))
            if box_height > height * 0.07 and fill_ratio > 0.68:
                continue
            if length / max(1.0, thickness) < 4.5:
                continue
            if area < width * height * 0.0015:
                continue
            if self._local_contrast(
                gray, x, y, box_width, box_height
            ) < 12.0:
                continue
            # The field strip is a long, continuous band crossing most of
            # the lower view. Short bright seams and the upper white box do
            # not have this span/coverage combination.
            if box_width < width * 0.45:
                continue
            component = np.zeros((box_height, box_width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)
            horizontal_coverage = float(
                np.count_nonzero(np.any(component > 0, axis=0)) /
                max(1.0, box_width)
            )
            fill_ratio = float(
                np.count_nonzero(component) /
                max(1.0, box_width * box_height)
            )
            if horizontal_coverage < 0.65:
                continue
            if box_width > width * 0.55 and fill_ratio > 0.58:
                continue
            angle_deg, y_at_center = self._fitted_line(
                contour, width, height
            )
            if (
                y_at_center is None
                or not height * 0.22 <= y_at_center < height * 0.94
                or abs(angle_deg) > 25.0
            ):
                continue
            candidates.append(
                (area, contour, rect, (x, y, box_width, box_height))
            )

        if not candidates:
            fallback = self._fallback_measurement(frame)
            if fallback is not None:
                self._missed_frames = 0
                self._last_measurement = fallback
                return fallback
            self._missed_frames += 1
            if (
                self._last_measurement is not None
                and self._missed_frames <= self.max_hold_frames
            ):
                held = dict(self._last_measurement)
                held["held"] = True
                return held
            self._last_measurement = None
            return None

        area, contour, rect, bounds = max(
            candidates,
            key=lambda item: self._candidate_score(
                item, gray, width, height
            ),
        )
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
        if (
            y_at_center is None
            or not 0.0 <= y_at_center < float(height)
            or abs(angle_deg) > 25.0
        ):
            return None
        measurement = {
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
            "held": False,
        }
        self._missed_frames = 0
        self._last_measurement = measurement
        return measurement
