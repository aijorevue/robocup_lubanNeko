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
        self._line_edge_history = []
        self._task2_edge_history = []

    def reset_tracking(self):
        """Discard stale geometry when a new white-line phase begins."""
        self._last_measurement = None
        self._missed_frames = 0
        self._line_edge_history = []
        self._task2_edge_history = []

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
    def _fit_task2_band_edges(contour, bounds, frame_width, frame_height):
        """Fit the strip's two edges and derive its geometric center line."""
        x, y, box_width, box_height = bounds
        if box_width < 20 or box_height < 3:
            return None
        component = np.zeros((box_height, box_width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)

        xs = []
        tops = []
        bottoms = []
        for local_x in range(box_width):
            rows = np.flatnonzero(component[:, local_x])
            if len(rows) < 2:
                continue
            xs.append(float(x + local_x))
            tops.append(float(y + rows[0]))
            bottoms.append(float(y + rows[-1]))
        if len(xs) < max(30, int(box_width * 0.55)):
            return None

        xs = np.asarray(xs, dtype=np.float64)
        tops = np.asarray(tops, dtype=np.float64)
        bottoms = np.asarray(bottoms, dtype=np.float64)

        # The right edge is the end of the continuous white strip. Ignore
        # isolated bright marks attached or adjacent to the strip end.
        support = bottoms - tops + 1.0
        support_threshold = max(3.0, float(np.median(support)) * 0.45)
        valid = support >= support_threshold
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 30:
            return None
        breaks = np.flatnonzero(np.diff(valid_indices) > 3)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(valid_indices) - 1]
        run = max(
            range(len(starts)),
            key=lambda index: (
                ends[index] - starts[index] + 1,
                ends[index],
            ),
        )
        keep_start = int(starts[run])
        keep_end = int(ends[run]) + 1
        xs = xs[keep_start:keep_end]
        tops = tops[keep_start:keep_end]
        bottoms = bottoms[keep_start:keep_end]
        if len(xs) < 30:
            return None

        def robust_fit(values):
            keep = np.ones(values.shape, dtype=bool)
            for _ in range(3):
                if np.count_nonzero(keep) < 12:
                    return None
                slope, intercept = np.polyfit(xs[keep], values[keep], 1)
                residual = np.abs(values - (slope * xs + intercept))
                median = float(np.median(residual[keep]))
                mad = float(np.median(np.abs(residual[keep] - median)))
                limit = max(2.0, median + 3.0 * max(1.0, mad))
                next_keep = residual <= limit
                if np.array_equal(next_keep, keep):
                    break
                keep = next_keep
            if np.count_nonzero(keep) < max(24, int(len(xs) * 0.55)):
                return None
            slope, intercept = np.polyfit(xs[keep], values[keep], 1)
            return float(slope), float(intercept), keep

        top_fit = robust_fit(tops)
        bottom_fit = robust_fit(bottoms)
        if top_fit is None or bottom_fit is None:
            return None
        top_slope, top_intercept, top_keep = top_fit
        bottom_slope, bottom_intercept, bottom_keep = bottom_fit
        center_slope = (top_slope + bottom_slope) * 0.5
        center_intercept = (top_intercept + bottom_intercept) * 0.5
        center_x = frame_width * 0.5
        y_at_center = center_slope * center_x + center_intercept
        top_center = top_slope * center_x + top_intercept
        bottom_center = bottom_slope * center_x + bottom_intercept
        thickness = float(np.median(bottoms - tops))
        if thickness <= 0.0 or bottom_center <= top_center:
            return None
        x_left = float(xs[0])
        x_right = float(xs[-1])
        return {
            "angle_deg": float(np.degrees(np.arctan(center_slope))),
            "y_at_center": float(y_at_center),
            "thickness": thickness,
            "top_edge": (
                top_slope,
                top_intercept,
            ),
            "bottom_edge": (
                bottom_slope,
                bottom_intercept,
            ),
            "center_line": (
                (x_left, center_slope * x_left + center_intercept),
                (x_right, center_slope * x_right + center_intercept),
            ),
            "edge_polygon": (
                (x_left, top_slope * x_left + top_intercept),
                (x_right, top_slope * x_right + top_intercept),
                (x_right, bottom_slope * x_right + bottom_intercept),
                (x_left, bottom_slope * x_left + bottom_intercept),
            ),
            "edge_support": int(
                min(np.count_nonzero(top_keep), np.count_nonzero(bottom_keep))
            ),
            "left_edge_x": x_left,
            "right_edge_x": x_right,
        }

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
        area, contour, rect, bounds, *extra = candidate
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
            edge_geometry = extra[0] if extra else None
            if edge_geometry is not None:
                angle = edge_geometry["angle_deg"]
                y_at_center = edge_geometry["y_at_center"]
            else:
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

            edge_geometry = self._fit_task2_band_edges(
                contour, (x, y, box_width, box_height), width, height
            )
            if edge_geometry is None:
                angle_deg, y_at_center = self._fitted_line(
                    contour, width, height
                )
            else:
                angle_deg = edge_geometry["angle_deg"]
                y_at_center = edge_geometry["y_at_center"]
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
            edge_geometry = self._fit_task2_band_edges(
                contour, (x, y, box_width, box_height), width, height
            )
            if edge_geometry is None:
                angle_deg, y_at_center = self._fitted_line(
                    contour, width, height
                )
            else:
                angle_deg = edge_geometry["angle_deg"]
                y_at_center = edge_geometry["y_at_center"]
            if (
                y_at_center is None
                or not height * 0.22 <= y_at_center < height * 0.94
                or abs(angle_deg) > 25.0
            ):
                continue
            candidates.append(
                (
                    area,
                    contour,
                    rect,
                    (x, y, box_width, box_height),
                    edge_geometry,
                )
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
            self._line_edge_history = []
            return None

        area, contour, rect, bounds, edge_geometry = max(
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
        if edge_geometry is None:
            thickness = float(min(rect[1]))
            angle_deg, y_at_center = self._fitted_line(
                contour, width, height
            )
        else:
            thickness = float(edge_geometry["thickness"])
            angle_deg = float(edge_geometry["angle_deg"])
            y_at_center = float(edge_geometry["y_at_center"])
        self._line_edge_history.append(
            (y_at_center, angle_deg, thickness, edge_geometry)
        )
        self._line_edge_history = self._line_edge_history[-3:]
        y_at_center = float(
            np.median([item[0] for item in self._line_edge_history])
        )
        angle_deg = float(
            np.median([item[1] for item in self._line_edge_history])
        )
        thickness = float(
            np.median([item[2] for item in self._line_edge_history])
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
        if edge_geometry is not None:
            measurement.update(
                {
                    "center_line": edge_geometry["center_line"],
                    "edge_polygon": edge_geometry["edge_polygon"],
                    "edge_support": edge_geometry["edge_support"],
                }
            )
        self._missed_frames = 0
        self._last_measurement = measurement
        return measurement

    def detect_task2(self, frame):
        """Detect the lower ground strip used by formal task two.

        The generic detector has a deliberately permissive fallback for task
        one.  That fallback can accept the broad, slanted underside of the
        overhead box when the real strip is bright but solid.  Task two has a
        stable view: the reference strip is below the box, thin, and spans
        most of the image.  Keep this path separate so task one is unchanged.
        """
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        value_eq = clahe.apply(hsv[:, :, 2])
        x0 = int(width * self.roi_x[0])
        x1 = int(width * self.roi_x[1])
        y0 = int(height * self.roi_y[0])
        y1 = int(height * self.roi_y[1])
        roi_mask = np.zeros((height, width), dtype=np.uint8)
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
        mask = cv2.bitwise_and(
            cv2.bitwise_or(white_mask, gray_mask), roi_mask
        )
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

        candidates = []
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            rect = cv2.minAreaRect(contour)
            length = float(max(rect[1]))
            thickness = float(min(rect[1]))
            edge_geometry = self._fit_task2_band_edges(
                contour, (x, y, box_width, box_height), width, height
            )
            if edge_geometry is None:
                continue
            angle_deg = edge_geometry["angle_deg"]
            y_at_center = edge_geometry["y_at_center"]

            # The actual task-two strip is below the box and above the arm.
            # These bounds reject both the box edge and lower arm structures.
            if not height * 0.30 <= y_at_center <= height * 0.68:
                continue
            if box_width < width * 0.45 or box_width > width * 0.92:
                continue
            if box_height > height * 0.075:
                continue
            if not height * 0.005 <= thickness <= height * 0.070:
                continue
            if length / max(1.0, thickness) < 4.5:
                continue
            if area < width * height * 0.0010:
                continue
            if abs(angle_deg) > 25.0:
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
            if horizontal_coverage < 0.65:
                continue

            # Prefer the lowest thin strip in the valid band.  This is what
            # separates the ground line from the larger slanted box edge.
            vertical_score = float(
                np.clip(
                    (y_at_center - height * 0.30) /
                    max(1.0, height * 0.38),
                    0.0,
                    1.0,
                )
            )
            span_score = float(np.clip(box_width / max(1.0, width), 0.0, 1.0))
            thin_score = float(
                np.clip(
                    1.0 - abs(thickness / max(1.0, height) - 0.035) / 0.035,
                    0.0,
                    1.0,
                )
            )
            contrast = self._local_contrast(
                gray, x, y, box_width, box_height
            )
            contrast_score = float(np.clip((contrast - 5.0) / 55.0, 0.0, 1.0))
            score = (
                4.0 * vertical_score
                + 2.0 * span_score
                + 2.0 * thin_score
                + 1.5 * horizontal_coverage
                + 1.5 * contrast_score
            )
            candidates.append(
                (
                    score,
                    contour,
                    rect,
                    (x, y, box_width, box_height),
                    area,
                    edge_geometry,
                )
            )

        if not candidates:
            self._missed_frames += 1
            if (
                self._last_measurement is not None
                and self._missed_frames <= self.max_hold_frames
            ):
                held = dict(self._last_measurement)
                held["held"] = True
                return held
            self._last_measurement = None
            self._task2_edge_history = []
            return None

        _, contour, rect, bounds, area, edge_geometry = max(
            candidates, key=lambda candidate: candidate[0]
        )
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return None
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
        length = float(max(rect[1]))
        thickness = float(edge_geometry["thickness"])
        angle_deg = float(edge_geometry["angle_deg"])
        y_at_center = float(edge_geometry["y_at_center"])
        self._task2_edge_history.append(
            (y_at_center, angle_deg, thickness, edge_geometry)
        )
        self._task2_edge_history = self._task2_edge_history[-3:]
        y_at_center = float(
            np.median([item[0] for item in self._task2_edge_history])
        )
        angle_deg = float(
            np.median([item[1] for item in self._task2_edge_history])
        )
        thickness = float(
            np.median([item[2] for item in self._task2_edge_history])
        )
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
            "center_line": edge_geometry["center_line"],
            "edge_polygon": edge_geometry["edge_polygon"],
            "edge_support": edge_geometry["edge_support"],
            "left_edge_x": edge_geometry["left_edge_x"],
            "right_edge_x": edge_geometry["right_edge_x"],
        }
        self._missed_frames = 0
        self._last_measurement = measurement
        return measurement
