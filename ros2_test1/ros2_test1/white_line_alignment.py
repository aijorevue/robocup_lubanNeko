"""Detect the field reference strip used after the DISC entry arc."""

from __future__ import annotations

import cv2
import numpy as np


TASK3_ABSOLUTE_WHITE_VALUE_MIN = 180
TASK3_ABSOLUTE_WHITE_SATURATION_MAX = 100
TASK3_SCANLINE_MIN_ABSOLUTE_WHITE_RATIO = 0.08
TASK3_MIN_ABSOLUTE_WHITE_PIXEL_RATIO = 0.06
TASK3_MIN_ABSOLUTE_WHITE_COLUMN_RATIO = 0.20
TASK3_RIGHT_EDGE_MAX_GAP = 3
TASK3_RIGHT_EDGE_MIN_RUN_RATIO = 0.05
TASK3_RIGHT_EDGE_MIN_PIXEL_RATIO = 0.055
TASK3_RIGHT_EDGE_MAX_STEP_RATIO = 0.05
TASK3_RIGHT_EDGE_CONFIRM_FRAMES = 2


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
        self._task3_edge_history = []
        self._task3_last_measurement = None
        self._task3_missed_frames = 0
        self._task3_pending_right_edge = None
        self._task3_pending_right_edge_frames = 0

    def reset_tracking(self):
        """Discard stale geometry when a new white-line phase begins."""
        self._last_measurement = None
        self._missed_frames = 0
        self._line_edge_history = []
        self._task2_edge_history = []
        self._task3_edge_history = []
        self._task3_last_measurement = None
        self._task3_missed_frames = 0
        self._task3_pending_right_edge = None
        self._task3_pending_right_edge_frames = 0

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
    def _task2_absolute_white_recovery(
        white_mask, roi_mask, frame_width, frame_height, roi_bounds
    ):
        """Recover a thin tape band when glare joins it to a broad contour."""
        mask = cv2.bitwise_and(white_mask, roi_mask)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)),
        )
        roi_x0, roi_x1, roi_y0, roi_y1 = roi_bounds
        x_min = max(0, roi_x0)
        x_max = min(frame_width, roi_x1)
        y_min = max(roi_y0, int(frame_height * 0.22))
        y_max = min(roi_y1 - 1, int(frame_height * 0.68))
        if x_max <= x_min or y_max < y_min:
            return None
        rows = []
        for y in range(y_min, y_max + 1):
            row = mask[y, x_min:x_max] > 0
            changes = np.flatnonzero(
                np.diff(np.pad(row.astype(np.int8), (1, 1)))
            )
            if len(changes) < 2:
                continue
            starts, ends = changes[::2], changes[1::2]
            best = int(np.argmax(ends - starts))
            left = int(starts[best] + x_min)
            right = int(ends[best] + x_min)
            if (
                right - left >= frame_width * 0.40
                and left <= frame_width * 0.20
                and right >= frame_width * 0.55
            ):
                rows.append((y, left, right))
        groups = []
        for row in rows:
            if not groups or row[0] > groups[-1][-1][0] + 1:
                groups.append([])
            groups[-1].append(row)

        best_result = None
        best_score = 0.0
        for group in groups:
            if not (
                max(5, int(frame_height * 0.008))
                <= len(group)
                <= frame_height * 0.10
            ):
                continue
            band_top, band_bottom = group[0][0], group[-1][0]
            left = int(np.median([row[1] for row in group]))
            right = int(np.median([row[2] for row in group]))
            scan_top = max(y_min, band_top - int(frame_height * 0.10))
            scan_bottom = min(y_max + 1, band_bottom + int(frame_height * 0.10) + 1)
            columns = []
            for x in range(left, right):
                column = mask[scan_top:scan_bottom, x] > 0
                changes = np.flatnonzero(
                    np.diff(np.pad(column.astype(np.int8), (1, 1)))
                )
                options = []
                for start, end in zip(changes[::2], changes[1::2]):
                    top, bottom = int(start + scan_top), int(end + scan_top)
                    overlap = min(bottom, band_bottom + 1) - max(top, band_top)
                    if 3 <= bottom - top <= frame_height * 0.10 and overlap >= 3:
                        options.append((overlap, top, bottom))
                if options:
                    _, top, bottom = max(options)
                    columns.append((x, top, bottom))
            if not columns:
                continue
            runs = []
            for column in columns:
                if not runs or column[0] > runs[-1][-1][0] + max(
                    6, int(frame_width * 0.06)
                ):
                    runs.append([])
                runs[-1].append(column)
            run = max(runs, key=len)
            xs = np.asarray([item[0] for item in run], dtype=np.float64)
            tops = np.asarray([item[1] for item in run], dtype=np.float64)
            bottoms = np.asarray([item[2] for item in run], dtype=np.float64)
            span = float(xs[-1] - xs[0] + 1)
            if (
                span < frame_width * 0.35
                or span > frame_width * 0.92
                or len(xs) < span * 0.70
            ):
                continue

            def robust_fit(values):
                keep = np.ones(values.shape, dtype=bool)
                for _ in range(3):
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
                continue
            top_slope, top_intercept, top_keep = top_fit
            bottom_slope, bottom_intercept, bottom_keep = bottom_fit
            center_slope = (top_slope + bottom_slope) * 0.5
            center_intercept = (top_intercept + bottom_intercept) * 0.5
            y_at_center = center_slope * (frame_width * 0.5) + center_intercept
            thickness = float(np.median(bottoms - tops))
            angle_deg = float(np.degrees(np.arctan(center_slope)))
            support = int(
                min(np.count_nonzero(top_keep), np.count_nonzero(bottom_keep))
            )
            if (
                not y_min <= y_at_center <= y_max
                or not frame_height * 0.005 <= thickness <= frame_height * 0.10
                or span / max(1.0, thickness) < 8.0
                or abs(angle_deg) > 25.0
                or support < max(30, int(span * 0.45))
            ):
                continue
            x_left, x_right = float(xs[0]), float(xs[-1])
            result = {
                "angle_deg": angle_deg,
                "y_at_center": float(y_at_center),
                "thickness": thickness,
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
                "edge_support": support,
                "left_edge_x": x_left,
                "right_edge_x": x_right,
                "area": float(np.sum(bottoms - tops)),
                "bounds": (
                    int(x_left),
                    int(np.min(tops)),
                    int(span),
                    int(np.max(bottoms) - np.min(tops)),
                ),
            }
            score = span * len(group)
            if score > best_score:
                best_score = score
                best_result = result
        return best_result

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
            if horizontal_coverage < 0.55:
                continue
            # A clean field strip can become almost fully filled after the
            # morphology pass. Keep that thin, long geometry while still
            # rejecting solid bright boxes and chassis structures.
            is_solid_reference_strip = (
                box_height <= height * 0.06
                and length / max(1.0, thickness) >= 12.0
                and horizontal_coverage >= 0.80
            )
            if fill_ratio > 0.82 and not is_solid_reference_strip:
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
            local_contrast = self._local_contrast(
                gray, x, y, box_width, box_height
            )
            edge_geometry = self._fit_task2_band_edges(
                contour, (x, y, box_width, box_height), width, height
            )
            has_embedded_reference_strip = bool(
                edge_geometry is not None
                and box_width >= width * 0.45
                and height * 0.005 <= edge_geometry["thickness"] <= height * 0.06
                and height * 0.22 <= edge_geometry["y_at_center"] < height * 0.94
                and abs(edge_geometry["angle_deg"]) <= 25.0
                and local_contrast >= 12.0
            )
            effective_thickness = (
                float(edge_geometry["thickness"])
                if has_embedded_reference_strip
                else thickness
            )
            if length < width * 0.25:
                continue
            if (
                effective_thickness < height * 0.005
                or effective_thickness > height * 0.10
            ):
                continue
            # Reject bright structures attached to the top of the search ROI.
            # This is geometric, so it remains valid when the line moves in Y.
            if (
                not has_embedded_reference_strip
                and y <= y0 + height * 0.04
                and box_height > height * 0.06
            ):
                continue
            # A large bright box can be long and rectangular too, but it is
            # substantially thicker and more solid than the field strip.
            if (
                not has_embedded_reference_strip
                and box_height > height * 0.12
            ):
                continue
            component = np.zeros((box_height, box_width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)
            fill_ratio = float(np.count_nonzero(component) /
                               max(1.0, box_width * box_height))
            if (
                not has_embedded_reference_strip
                and box_height > height * 0.07
                and fill_ratio > 0.68
            ):
                continue
            if length / max(1.0, effective_thickness) < 4.5:
                continue
            if area < width * height * 0.0015:
                continue
            if local_contrast < 12.0:
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
            # Do not reject a genuinely clean white strip merely because its
            # thresholded contour is solid. Large bright structures remain
            # excluded by their height/aspect geometry and the ROI checks.
            is_solid_reference_strip = (
                box_height <= height * 0.06
                and length / max(1.0, thickness) >= 12.0
                and horizontal_coverage >= 0.80
                and local_contrast >= 12.0
            )
            if (
                box_width > width * 0.55
                and fill_ratio > 0.58
                and not is_solid_reference_strip
            ):
                continue
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

        # The real strip enters from the left, while the wood box occupies
        # the upper-left part of the frame and can touch the strip in the
        # threshold mask. Keep a right-side preference, but leave enough of
        # the strip in view for the width gate: at 0.28 a valid line can be
        # cropped to just below the 45% width requirement and become a false
        # NOT_FOUND. The upper box remains rejected by its height/thickness
        # gates below.
        right_window_x = int(width * 0.20)
        preferred_mask = mask.copy()
        preferred_mask[:, :right_window_x] = 0
        preferred_contours, _ = cv2.findContours(
            preferred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        full_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Evaluate both views. A small bright fragment in the preferred
        # window must not prevent the complete physical strip from being
        # considered, while the cropped view still separates a strip that
        # touches the upper wooden box on the far left.
        contours = list(preferred_contours) + list(full_contours)

        candidates = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            rect = cv2.minAreaRect(contour)
            length = float(max(rect[1]))
            raw_thickness = float(min(rect[1]))
            edge_geometry = self._fit_task2_band_edges(
                contour, (x, y, box_width, box_height), width, height
            )
            if edge_geometry is None:
                continue
            angle_deg = edge_geometry["angle_deg"]
            y_at_center = edge_geometry["y_at_center"]
            fitted_thickness = float(edge_geometry["thickness"])
            fitted_span = float(
                edge_geometry["right_edge_x"] - edge_geometry["left_edge_x"] + 1.0
            )
            edge_support = int(edge_geometry["edge_support"])

            # The actual task-two strip is below the box and above the arm.
            # These bounds reject both the box edge and lower arm structures.
            # At the start of the formal approach the tape can still be high
            # in the image. The upper box remains below this gate only in the
            # threshold mask; its broad/non-slender geometry is rejected by
            # the remaining checks.
            if not height * 0.22 <= y_at_center <= height * 0.68:
                continue
            if fitted_span < width * 0.45 or fitted_span > width * 0.92:
                continue
            # Sparse highlights above or below the tape can attach to its
            # contour and greatly inflate minAreaRect/boundingRect. The edge
            # fitter has already isolated the longest continuously supported
            # band, so use its robust thickness and span for the strip gates.
            # Keep only a generous raw-height sanity check for pathological
            # components; the upper box is still rejected by Y and fitted
            # thickness rather than by a fragile outer rectangle.
            if box_height > height * 0.22:
                continue
            if not height * 0.005 <= fitted_thickness <= height * 0.10:
                continue
            if fitted_span / max(1.0, fitted_thickness) < 8.0:
                continue
            if edge_support < max(30, int(fitted_span * 0.45)):
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
                    (y_at_center - height * 0.22) /
                    max(1.0, height * 0.46),
                    0.0,
                    1.0,
                )
            )
            span_score = float(np.clip(fitted_span / max(1.0, width), 0.0, 1.0))
            thin_score = float(
                np.clip(
                    1.0
                    - abs(fitted_thickness / max(1.0, height) - 0.035) / 0.035,
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
                    raw_thickness,
                )
            )

        if not candidates:
            recovery = self._task2_absolute_white_recovery(
                white_mask, roi_mask, width, height, (x0, x1, y0, y1)
            )
            if recovery is not None:
                self._task2_edge_history.append(
                    (
                        recovery["y_at_center"],
                        recovery["angle_deg"],
                        recovery["thickness"],
                        recovery,
                    )
                )
                self._task2_edge_history = self._task2_edge_history[-3:]
                measurement = {
                    "center_x": (
                        recovery["left_edge_x"] + recovery["right_edge_x"]
                    ) * 0.5,
                    "center_y": recovery["y_at_center"],
                    "angle_deg": float(np.median(
                        [item[1] for item in self._task2_edge_history]
                    )),
                    "y_at_center": float(np.median(
                        [item[0] for item in self._task2_edge_history]
                    )),
                    "length": (
                        recovery["right_edge_x"] - recovery["left_edge_x"] + 1.0
                    ),
                    "thickness": float(np.median(
                        [item[2] for item in self._task2_edge_history]
                    )),
                    "area": recovery["area"],
                    "bounds": recovery["bounds"],
                    "frame_width": width,
                    "frame_height": height,
                    "held": False,
                    "center_line": recovery["center_line"],
                    "edge_polygon": recovery["edge_polygon"],
                    "edge_support": recovery["edge_support"],
                    "left_edge_x": recovery["left_edge_x"],
                    "right_edge_x": recovery["right_edge_x"],
                }
                self._missed_frames = 0
                self._last_measurement = measurement
                return measurement
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

        _, contour, rect, bounds, area, edge_geometry, _ = max(
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

    @staticmethod
    def _task3_absolute_white_edge(
        hsv,
        y0,
        y1,
        x0,
        x1,
        previous_right=None,
    ):
        """Find a sustained true-white terminal run in a task-three band.

        Enhanced-gray masks are useful for recovering a dim strip, but they
        also turn dark-cloth reflections into bright candidates.  The right
        endpoint is therefore derived from absolute HSV-white support only;
        a few missing columns are tolerated inside a run, while an isolated
        bright fragment cannot become the endpoint.
        """
        height, width = hsv.shape[:2]
        y0 = max(0, int(y0))
        y1 = min(height, int(y1))
        x0 = max(0, int(x0))
        x1 = min(width, int(x1))
        if x1 <= x0 or y1 <= y0:
            return None

        sample = hsv[y0:y1, x0:x1]
        absolute_white = (
            (sample[:, :, 2] >= TASK3_ABSOLUTE_WHITE_VALUE_MIN)
            & (sample[:, :, 1] <= TASK3_ABSOLUTE_WHITE_SATURATION_MAX)
        )
        support = np.count_nonzero(absolute_white, axis=0).astype(np.int32)
        band_height = max(1, y1 - y0)
        minimum_support = max(2, int(round(band_height * 0.12)))
        valid = support >= minimum_support

        runs = []
        run_start = None
        last_valid = None
        gap = 0
        for index, is_valid in enumerate(valid):
            if is_valid:
                if run_start is None:
                    run_start = index
                last_valid = index
                gap = 0
                continue
            if run_start is None:
                continue
            gap += 1
            if gap > TASK3_RIGHT_EDGE_MAX_GAP:
                runs.append((run_start, last_valid))
                run_start = None
                last_valid = None
                gap = 0
        if run_start is not None:
            runs.append((run_start, last_valid))

        minimum_run = max(
            24,
            int(round((x1 - x0) * TASK3_RIGHT_EDGE_MIN_RUN_RATIO)),
        )
        candidates = []
        for start, end in runs:
            run_width = int(end - start + 1)
            if run_width < minimum_run:
                continue
            run_support = support[start:end + 1]
            pixel_ratio = float(
                np.count_nonzero(absolute_white[:, start:end + 1])
                / max(1, run_width * band_height)
            )
            if pixel_ratio < TASK3_RIGHT_EDGE_MIN_PIXEL_RATIO:
                continue
            edge_x = float(x0 + end)
            width_score = run_width / max(1.0, float(x1 - x0))
            density_score = float(
                np.mean(run_support) / max(1.0, float(band_height))
            )
            continuity_score = 0.0
            if previous_right is not None:
                continuity_score = max(
                    0.0,
                    1.0
                    - abs(edge_x - float(previous_right))
                    / max(60.0, float(width) * 0.25),
                )
            score = (
                2.4 * width_score
                + 3.0 * pixel_ratio
                + 1.2 * density_score
                + 1.0 * continuity_score
            )
            candidates.append(
                (
                    score,
                    run_width,
                    edge_x,
                    x0 + start,
                    pixel_ratio,
                    density_score,
                )
            )

        if not candidates:
            return None
        _, run_width, edge_x, left_x, pixel_ratio, density_score = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return {
            "left_edge_x": float(left_x),
            "right_edge_x": edge_x,
            "run_width": float(run_width),
            "white_pixel_ratio": pixel_ratio,
            "white_density": density_score,
        }

    def _task3_scanline_fallback(
        self, frame, threshold_floor=135, saturation_limit=150
    ):
        """Extract a partial horizontal strip without merging the upper box.

        At close range the threshold mask can connect the strip to the wooden
        box above it.  Contour geometry then describes the box instead of the
        strip and the normal task-three gate rejects both.  Scan individual
        rows, select the terminal horizontal run, and fit the centerline only
        inside that thin band.  This is intentionally task-three-only.
        """
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        value_eq = clahe.apply(hsv[:, :, 2])
        x0 = max(0, int(width * 0.02))
        x1 = min(width, int(width * 0.98))
        # The formal target is centered near Y=300 in a 600px frame. Reject
        # the upper bright hardware band before it can become a white-line
        # candidate; keeping this guard task-three-only avoids changing the
        # task-one/task-two camera contracts.
        y0 = max(int(height * 0.24), int(height * 0.15))
        # Exclude the lower vehicle/arm highlight from the task-three scan.
        # The formal reference is Y=300 in a 600px frame, so the extra lower
        # margin is unnecessary and can only admit chassis structure.
        y1 = min(height, int(height * 0.64))
        if x1 <= x0 or y1 <= y0:
            return None

        roi_value = value_eq[y0:y1, x0:x1]
        roi_gray = gray_eq[y0:y1, x0:x1]
        adaptive_value = max(
            int(threshold_floor), int(np.percentile(roi_value, 60))
        )
        adaptive_gray = max(
            int(threshold_floor), int(np.percentile(roi_gray, 62))
        )
        bright = cv2.inRange(
            hsv,
            np.array((0, 0, adaptive_value), dtype=np.uint8),
            np.array((179, int(saturation_limit), 255), dtype=np.uint8),
        )
        gray_mask = cv2.inRange(gray_eq, adaptive_gray, 255)
        mask = cv2.bitwise_or(bright, gray_mask)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        row_runs = []
        close_kernel = np.ones((1, max(9, min(17, width // 45))), np.uint8)
        for row_y in range(y0, y1):
            row = cv2.morphologyEx(
                mask[row_y:row_y + 1], cv2.MORPH_CLOSE, close_kernel
            )[0] > 0
            indices = np.flatnonzero(row)
            if indices.size == 0:
                continue
            runs = []
            start = previous = int(indices[0])
            for index in indices[1:]:
                index = int(index)
                if index - previous > 3:
                    runs.append((start, previous))
                    start = index
                previous = index
            runs.append((start, previous))
            run_start, run_end = max(
                runs, key=lambda run: run[1] - run[0] + 1
            )
            run_width = run_end - run_start + 1
            if run_width >= max(80, int(width * 0.12)):
                row_runs.append((row_y, run_width, run_start, run_end))

        if not row_runs:
            return None

        # Build groups before selecting the terminal edge.  A dark-cloth
        # reflection can extend farther right than the real strip, so a
        # frame-global terminal-edge preference is unsafe here.
        groups = []
        for item in row_runs:
            if not groups or item[0] - groups[-1][-1][0] > 2:
                groups.append([item])
            else:
                groups[-1].append(item)
        max_group_height = max(24, int(height * 0.14))
        groups = [group for group in groups if group]
        if not groups:
            return None
        reference_y = height * 0.50
        if self._task3_last_measurement is not None:
            reference_y = float(
                self._task3_last_measurement.get("y_at_center", reference_y)
            )

        def group_score(candidate):
            # Do not use the globally rightmost enhanced-gray run.  A gray
            # reflection can extend farther right than the actual strip.
            # Use the terminal row plateau to isolate the horizontal strip's
            # Y band, but derive its X endpoint only from absolute-white
            # support within that band.
            maximum_end = max(item[3] for item in candidate)
            edge_tolerance = max(12, int(width * 0.03))
            terminal_rows = [
                item for item in candidate
                if item[3] >= maximum_end - edge_tolerance
            ]
            if not terminal_rows:
                terminal_rows = candidate
            group_y = float(np.median([item[0] for item in terminal_rows]))
            max_width = max(item[1] for item in terminal_rows)
            group_left = max(0, int(min(item[2] for item in terminal_rows)))
            group_right = min(width - 1, int(max(item[3] for item in terminal_rows)))
            group_top = max(y0, terminal_rows[0][0] - 3)
            group_bottom = min(y1 - 1, terminal_rows[-1][0] + 3)
            sample = hsv[group_top:group_bottom + 1, group_left:group_right + 1]
            absolute_white_ratio = float(np.mean(
                (sample[:, :, 2] >= TASK3_ABSOLUTE_WHITE_VALUE_MIN)
                & (sample[:, :, 1] <= TASK3_ABSOLUTE_WHITE_SATURATION_MAX)
            )) if sample.size else 0.0
            white_edge = self._task3_absolute_white_edge(
                hsv,
                group_top,
                group_bottom + 1,
                group_left,
                group_right + 1,
                previous_right=(
                    self._task3_last_measurement.get("right_edge_x")
                    if self._task3_last_measurement is not None else None
                ),
            )
            if (
                absolute_white_ratio < TASK3_SCANLINE_MIN_ABSOLUTE_WHITE_RATIO
                or white_edge is None
                or len(candidate) < 3
            ):
                return -1.0e6
            y_score = max(
                -1.0,
                1.0 - abs(group_y - reference_y) / max(1.0, height * 0.30),
            )
            width_score = min(
                1.0,
                white_edge["run_width"] / max(1.0, width * 0.45),
            )
            row_score = min(1.0, len(candidate) / max(1.0, height * 0.03))
            white_score = float(np.clip(
                (absolute_white_ratio - 0.08) / 0.22, 0.0, 1.0
            ))
            # Absolute HSV evidence is deliberately dominant: the black
            # cloth reflection passes the enhanced-gray mask but has no
            # pixels in this white band, while the real strip does.
            return (
                4.0 * white_score
                + 2.2 * y_score
                + 1.4 * width_score
                + 0.3 * row_score
                + 1.2 * white_edge["white_density"]
            )

        # Prefer the strip near the formal reference or the last accepted
        # geometry over a wider but unrelated bright structure above it.
        scored_groups = [
            (group_score(group), group) for group in groups
        ]
        scored_groups = [
            item for item in scored_groups
            if item[0] > 0.0
        ]
        if not scored_groups:
            return None
        _, group = max(scored_groups, key=lambda item: item[0])

        maximum_end = max(item[3] for item in group)
        edge_tolerance = max(12, int(width * 0.03))
        terminal_rows = [
            item for item in group
            if item[3] >= maximum_end - edge_tolerance
        ]
        if not terminal_rows:
            terminal_rows = group
        band_top = max(y0, terminal_rows[0][0] - 3)
        band_bottom = min(y1 - 1, terminal_rows[-1][0] + 3)
        group_left = max(0, int(min(item[2] for item in terminal_rows)))
        group_right = min(width - 1, int(max(item[3] for item in terminal_rows)))
        white_edge = self._task3_absolute_white_edge(
            hsv,
            band_top,
            band_bottom + 1,
            group_left,
            group_right + 1,
            previous_right=(
                self._task3_last_measurement.get("right_edge_x")
                if self._task3_last_measurement is not None else None
            ),
        )
        if white_edge is None:
            return None
        right_edge_x = float(white_edge["right_edge_x"])
        left_edge_x = float(white_edge["left_edge_x"])
        if right_edge_x - left_edge_x < width * 0.15:
            return None

        # Fit the center of the bright band column by column.  Restricting the
        # fit to the selected terminal band prevents the overhead structure
        # from biasing the angle and Y measurement.
        fit_x0 = max(x0, int(width * 0.28), int(left_edge_x))
        fit_x1 = min(x1 - 1, int(round(right_edge_x)))
        points = []
        for column_x in range(fit_x0, fit_x1 + 1):
            column = mask[band_top:band_bottom + 1, column_x]
            ys = np.flatnonzero(column)
            if ys.size:
                points.append((column_x, band_top + float(np.median(ys))))
        if len(points) < max(20, int(width * 0.08)):
            return None

        coordinates = np.asarray(points, dtype=np.float32)
        slope, intercept = np.polyfit(coordinates[:, 0], coordinates[:, 1], 1)
        angle_deg = float(np.degrees(np.arctan(float(slope))))
        y_at_center = float(intercept + slope * (width * 0.5))
        if not height * 0.24 <= y_at_center <= height * 0.64:
            return None
        if abs(angle_deg) > 25.0:
            return None

        center_line = (
            (float(fit_x0), float(intercept + slope * fit_x0)),
            (float(fit_x1), float(intercept + slope * fit_x1)),
        )
        return {
            "center_x": float((fit_x0 + fit_x1) * 0.5),
            "center_y": y_at_center,
            "angle_deg": angle_deg,
            "y_at_center": y_at_center,
            "length": float(fit_x1 - fit_x0 + 1),
            "thickness": float(band_bottom - band_top + 1),
            "area": float(np.count_nonzero(mask[band_top:band_bottom + 1])),
            "bounds": (
                int(round(left_edge_x)), band_top,
                int(round(right_edge_x - left_edge_x + 1)),
                band_bottom - band_top + 1,
            ),
            "frame_width": width,
            "frame_height": height,
            "held": False,
            "center_line": center_line,
            "edge_support": len(points),
            "right_edge_x": right_edge_x,
        }

    @staticmethod
    def _task3_local_recovery(frame, previous):
        """Recover the same strip from a short local search after a miss.

        The normal task-three scan can reject a real strip when one exposure
        gap breaks its terminal run or when the chassis moves a few pixels
        between queries.  This recovery remains constrained to the previous
        strip geometry, so bright chassis parts and the upper box cannot become
        a new target.
        """
        if frame is None or frame.size == 0 or previous is None:
            return None

        height, width = frame.shape[:2]
        previous_y = float(previous.get("y_at_center", height * 0.5))
        previous_right = float(previous.get("right_edge_x", width * 0.75))
        bounds = previous.get("bounds") or (width * 0.20, previous_y, width * 0.45, 20)
        previous_left = float(bounds[0])
        previous_width = max(40.0, float(bounds[2]))
        y_margin = max(42, int(height * 0.09))
        y0 = max(int(height * 0.24), int(round(previous_y - y_margin)))
        y1 = min(int(height * 0.64), int(round(previous_y + y_margin)))
        x0 = max(0, int(round(previous_left - max(55, width * 0.07))))
        x1 = min(width, int(round(previous_right + max(55, width * 0.07))))
        if x1 <= x0 or y1 <= y0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        value = hsv[:, :, 2]
        roi_value = value[y0:y1, x0:x1]
        roi_gray = gray_eq[y0:y1, x0:x1]
        value_floor = max(112, int(np.percentile(roi_value, 45)))
        gray_floor = max(112, int(np.percentile(roi_gray, 48)))
        bright = cv2.inRange(
            hsv,
            np.array((0, 0, value_floor), dtype=np.uint8),
            np.array((179, 175, 255), dtype=np.uint8),
        )
        gray_mask = cv2.inRange(gray_eq, gray_floor, 255)
        mask = cv2.bitwise_or(bright, gray_mask)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)),
        )

        row_runs = []
        minimum_width = max(60, int(previous_width * 0.28), int(width * 0.10))
        for row_y in range(y0, y1):
            row = mask[row_y] > 0
            indices = np.flatnonzero(row[x0:x1])
            if indices.size == 0:
                continue
            runs = []
            start = previous_index = int(indices[0])
            for index in indices[1:]:
                index = int(index)
                if index - previous_index > 4:
                    runs.append((start, previous_index))
                    start = index
                previous_index = index
            runs.append((start, previous_index))
            for run_start, run_end in runs:
                run_start += x0
                run_end += x0
                run_width = run_end - run_start + 1
                if run_width < minimum_width:
                    continue
                edge_delta = abs(float(run_end) - previous_right)
                y_delta = abs(float(row_y) - previous_y)
                score = (
                    run_width / max(1.0, previous_width)
                    - edge_delta / max(40.0, width * 0.16)
                    - y_delta / max(20.0, y_margin)
                )
                row_runs.append((score, row_y, run_start, run_end))

        if not row_runs:
            return None
        best_score = max(item[0] for item in row_runs)
        selected = [item for item in row_runs if item[0] >= best_score - 0.16]
        selected = sorted(selected, key=lambda item: item[1])
        groups = []
        for item in selected:
            if not groups or item[1] - groups[-1][-1][1] > 2:
                groups.append([item])
            else:
                groups[-1].append(item)
        if not groups:
            return None
        group = max(
            groups,
            key=lambda candidate: (
                len(candidate),
                max(item[3] - item[2] + 1 for item in candidate),
            ),
        )
        if len(group) < 2:
            return None

        band_top = max(y0, group[0][1] - 3)
        band_bottom = min(y1 - 1, group[-1][1] + 3)
        left_edge_x = float(np.median([item[2] for item in group]))
        right_edge_x = float(np.median([item[3] for item in group]))
        if right_edge_x - left_edge_x < max(70.0, width * 0.12):
            return None

        points = []
        for column_x in range(int(left_edge_x), int(right_edge_x) + 1):
            column = mask[band_top:band_bottom + 1, column_x]
            ys = np.flatnonzero(column)
            if ys.size:
                points.append((column_x, band_top + float(np.median(ys))))
        if len(points) < max(20, int(width * 0.06)):
            return None
        coordinates = np.asarray(points, dtype=np.float32)
        slope, intercept = np.polyfit(coordinates[:, 0], coordinates[:, 1], 1)
        angle_deg = float(np.degrees(np.arctan(float(slope))))
        y_at_center = float(intercept + slope * (width * 0.5))
        if not height * 0.24 <= y_at_center <= height * 0.64:
            return None
        if abs(angle_deg) > 25.0:
            return None
        return {
            "center_x": (left_edge_x + right_edge_x) * 0.5,
            "center_y": y_at_center,
            "angle_deg": angle_deg,
            "y_at_center": y_at_center,
            "length": right_edge_x - left_edge_x + 1,
            "thickness": float(band_bottom - band_top + 1),
            "area": float(np.count_nonzero(mask[band_top:band_bottom + 1])),
            "bounds": (
                int(round(left_edge_x)), band_top,
                int(round(right_edge_x - left_edge_x + 1)),
                band_bottom - band_top + 1,
            ),
            "frame_width": width,
            "frame_height": height,
            "held": False,
            "center_line": (
                (left_edge_x, float(intercept + slope * left_edge_x)),
                (right_edge_x, float(intercept + slope * right_edge_x)),
            ),
            "edge_support": len(points),
            "right_edge_x": right_edge_x,
        }

    @staticmethod
    def _task3_has_absolute_white_support(frame, measurement):
        """Require continuous true-white pixels along the fitted strip."""
        if frame is None or frame.size == 0 or measurement is None:
            return False
        bounds = measurement.get("bounds")
        center_line = measurement.get("center_line")
        if bounds is None or center_line is None:
            return False

        height, width = frame.shape[:2]
        x, y, box_width, box_height = [int(round(value)) for value in bounds]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        if x1 - x0 < max(60, int(width * 0.08)) or y1 <= y0:
            return False

        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        absolute_white = (
            (hsv[:, :, 2] >= TASK3_ABSOLUTE_WHITE_VALUE_MIN)
            & (hsv[:, :, 1] <= TASK3_ABSOLUTE_WHITE_SATURATION_MAX)
        )
        line_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        (line_x0, line_y0), (line_x1, line_y1) = center_line
        thickness = float(measurement.get("thickness", box_height))
        support_thickness = int(np.clip(max(7.0, thickness * 1.3), 7, 37))
        cv2.line(
            line_mask,
            (int(round(line_x0)) - x0, int(round(line_y0)) - y0),
            (int(round(line_x1)) - x0, int(round(line_y1)) - y0),
            255,
            support_thickness,
        )
        support = line_mask > 0
        supported_columns = np.any(support, axis=0)
        support_column_count = int(np.count_nonzero(supported_columns))
        if support_column_count <= 0:
            return False
        white_support = absolute_white & support
        white_column_ratio = float(
            np.count_nonzero(np.any(white_support, axis=0))
            / support_column_count
        )
        white_pixel_ratio = float(
            np.count_nonzero(white_support) / max(1, np.count_nonzero(support))
        )
        return (
            white_column_ratio >= TASK3_MIN_ABSOLUTE_WHITE_COLUMN_RATIO
            and white_pixel_ratio >= TASK3_MIN_ABSOLUTE_WHITE_PIXEL_RATIO
        )

    def _task3_measurement_is_continuous(self, measurement):
        """Reject implausible frame-to-frame jumps to another bright object."""
        previous = self._task3_last_measurement
        if previous is None:
            return True
        height = float(measurement.get("frame_height", 600.0))
        width = float(measurement.get("frame_width", 800.0))
        y_delta = abs(
            float(measurement.get("y_at_center", 0.0))
            - float(previous.get("y_at_center", 0.0))
        )
        edge_delta = abs(
            float(measurement.get("right_edge_x", 0.0))
            - float(previous.get("right_edge_x", 0.0))
        )
        angle_delta = abs(
            float(measurement.get("angle_deg", 0.0))
            - float(previous.get("angle_deg", 0.0))
        )
        return (
            y_delta <= max(45.0, height * 0.10)
            and edge_delta <= max(70.0, width * 0.10)
            and angle_delta <= 8.0
        )

    def _task3_accept_measurement(self, frame, measurement):
        """Accept only continuous geometry or recover the last valid strip."""
        if measurement is not None and frame is not None and frame.size:
            refined_right_edge = self._task3_refine_right_boundary(
                frame, measurement
            )
            measurement = dict(measurement)
            measurement["right_edge_x"] = refined_right_edge
        if (
            measurement is not None
            and not self._task3_has_absolute_white_support(frame, measurement)
        ):
            measurement = None
        if measurement is None:
            return self._task3_recover_last()
        if self._task3_measurement_is_continuous(measurement):
            return self._task3_stabilize(measurement)

        # A bright upper structure must not replace a tracked strip in one
        # frame. Search locally around the previous geometry before holding it.
        if self._task3_last_measurement is not None:
            recovered = self._task3_local_recovery(
                frame, self._task3_last_measurement
            )
            if (
                recovered is not None
                and self._task3_measurement_is_continuous(recovered)
            ):
                return self._task3_stabilize(recovered)
        return self._task3_recover_last()

    def _task3_recover_last(self):
        """Keep a confirmed result for only a few detector cycles."""
        self._task3_missed_frames += 1
        self._task3_pending_right_edge = None
        self._task3_pending_right_edge_frames = 0
        if (
            self._task3_last_measurement is not None
            and self._task3_missed_frames <= self.max_hold_frames
        ):
            held = dict(self._task3_last_measurement)
            held["held"] = True
            return held
        self._task3_last_measurement = None
        self._task3_edge_history = []
        return None

    def detect_task3(self, frame):
        """Detect the BLUE task-three strip with a recoverable fallback.

        Task three begins farther from the strip than task two. Perspective,
        exposure, and partial frame entry can therefore make the strict
        task-two band limits reject a real strip. Keep the task-two detector
        as the first choice, then use a task-three-only relaxed candidate gate
        that still requires a long, thin, continuous bright band and a
        measurable right boundary.
        """
        # Prefer the task-three scanline geometry before contour candidates.
        # In the close/mixed view the real strip can be joined to the upper
        # box in the binary mask, while a lower arm/vehicle highlight remains
        # a separate, thick contour.  Letting that contour win first produces
        # a stable but wrong Y/RX result and prevents the existing scanline
        # recovery from running.
        scanline = self._task3_scanline_fallback(frame)
        if scanline is None:
            # A lower-contrast retry is still constrained by the same narrow
            # ROI and thin horizontal-band geometry.
            scanline = self._task3_scanline_fallback(
                frame, threshold_floor=118, saturation_limit=175
            )
        if scanline is None and self._task3_last_measurement is not None:
            scanline = self._task3_local_recovery(
                frame, self._task3_last_measurement
            )
        if scanline is not None:
            return self._task3_accept_measurement(frame, scanline)

        height, width = frame.shape[:2]
        task3_max_line_y = height * 0.64

        strict = self.detect_task2(frame)
        if strict is not None and float(strict.get("y_at_center", 0.0)) > \
                task3_max_line_y:
            # The lower vehicle/arm structure can satisfy task-two's wider
            # vertical gate. It is outside the task-three line search band.
            strict = None
        if (
            strict is not None
            and not self._task3_has_absolute_white_support(frame, strict)
        ):
            strict = None
        if strict is not None and not strict.get("held", False):
            # The strict edge fitter intentionally drops short support runs.
            # That is useful for task two, but at the farther task-three view
            # it can turn the middle of a real strip into a false right edge.
            # Re-run the task-three fallback when RX ends well before the
            # accepted component's actual right side.
            bounds = strict.get("bounds")
            strict_right = strict.get("right_edge_x")
            if bounds is not None and strict_right is not None:
                bounds_x, _, bounds_width, _ = bounds
                if float(strict_right) >= float(
                    bounds_x + max(12.0, bounds_width * 0.80)
                ):
                    strict = dict(strict)
                    strict["right_edge_x"] = self._task3_refine_right_boundary(
                        frame, strict
                    )
                    return self._task3_accept_measurement(frame, strict)
        self._task2_edge_history = []
        if frame is None or frame.size == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        value_eq = clahe.apply(hsv[:, :, 2])
        x0 = int(width * self.roi_x[0])
        x1 = int(width * self.roi_x[1])
        y0 = int(height * 0.24)
        # Keep the lower vehicle body out of the task-three candidate ROI.
        # The line target is centered at Y=300 in the 600px frame; pixels
        # below this guard are reserved for the chassis and arm structure.
        y1 = min(height, int(task3_max_line_y))
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y0:y1, x0:x1] = 255
        roi_value = value_eq[y0:y1, x0:x1]
        roi_gray = gray_eq[y0:y1, x0:x1]
        adaptive_value = max(145, int(np.percentile(roi_value, 67)))
        adaptive_gray = max(145, int(np.percentile(roi_gray, 69)))
        white_mask = cv2.inRange(
            hsv,
            np.array((0, 0, adaptive_value), dtype=np.uint8),
            np.array((179, min(135, self.saturation_max + 40), 255), dtype=np.uint8),
        )
        gray_mask = cv2.inRange(gray_eq, adaptive_gray, 255)
        mask = cv2.bitwise_and(
            cv2.bitwise_or(white_mask, gray_mask), roi_mask
        )
        # The real strip can touch the lower edge of the overhead box in the
        # image. Any vertical close bridges those separate objects and turns
        # the strip into a large box contour. Keep this close horizontal-only;
        # the later horizontal close handles exposure gaps along the strip.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        # At the farther task-three distance the strip can be split by small
        # exposure gaps. Bridge only along its long axis so separate vertical
        # bright objects are not merged into the candidate.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)),
        )

        # The real strip enters from the left, while the wood box occupies
        # the upper-left part of the frame and can touch the strip in the
        # threshold mask. Prefer a right-side strip segment where the box is
        # absent. This path is task-three-only and keeps the full mask as a
        # fallback for views where the segment is not visible yet.
        right_window_x = int(width * 0.28)
        preferred_mask = mask.copy()
        preferred_mask[:, :right_window_x] = 0
        contours, _ = cv2.findContours(
            preferred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        using_preferred_window = bool(contours)
        if not contours:
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
            min_width_ratio = 0.16 if using_preferred_window else 0.30
            if box_width < width * min_width_ratio or box_width > width * 0.98:
                continue
            if box_height > height * (0.12 if using_preferred_window else 0.14):
                continue
            min_area_ratio = 0.00025 if using_preferred_window else 0.0006
            if area < width * height * min_area_ratio:
                continue
            if thickness < height * 0.003 or thickness > height * 0.105:
                continue
            if length / max(1.0, thickness) < 3.5:
                continue

            component = np.zeros((box_height, box_width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(component, [shifted], -1, 255, cv2.FILLED)
            absolute_white = (
                (hsv[y:y + box_height, x:x + box_width, 2]
                 >= TASK3_ABSOLUTE_WHITE_VALUE_MIN)
                & (hsv[y:y + box_height, x:x + box_width, 1]
                   <= TASK3_ABSOLUTE_WHITE_SATURATION_MAX)
            )
            component_pixels = component > 0
            absolute_white_ratio = float(
                np.count_nonzero(absolute_white & component_pixels)
                / max(1, np.count_nonzero(component_pixels))
            )
            if absolute_white_ratio < TASK3_SCANLINE_MIN_ABSOLUTE_WHITE_RATIO:
                continue
            horizontal_coverage = float(
                np.count_nonzero(np.any(component > 0, axis=0)) /
                max(1.0, box_width)
            )
            fill_ratio = float(
                np.count_nonzero(component) /
                max(1.0, box_width * box_height)
            )
            max_fill_ratio = 0.94 if using_preferred_window else 0.86
            if horizontal_coverage < 0.50 or fill_ratio > max_fill_ratio:
                continue

            edge_geometry = self._fit_task2_band_edges(
                contour, (x, y, box_width, box_height), width, height
            )
            if edge_geometry is None:
                angle_deg, y_at_center = self._fitted_line(
                    contour, width, height
                )
                right_edge_x = float(x + box_width - 1)
            else:
                angle_deg = edge_geometry["angle_deg"]
                y_at_center = edge_geometry["y_at_center"]
                right_edge_x = self._task3_right_boundary_x(
                    component, x
                )
            if (
                y_at_center is None
                or not height * 0.24 <= y_at_center <= task3_max_line_y
                or abs(angle_deg) > 32.0
            ):
                continue
            contrast = self._local_contrast(
                gray, x, y, box_width, box_height
            )
            if contrast < 2.0:
                continue
            if right_edge_x < x + max(8.0, box_width * 0.65):
                continue
            score = (
                3.0 * min(1.0, box_width / max(1.0, width))
                + 2.0 * min(1.0, length / max(1.0, 7.0 * thickness))
                + 2.0 * horizontal_coverage
                + 1.5 * float(np.clip((contrast - 2.0) / 45.0, 0.0, 1.0))
                + 0.8 * float(np.clip(y_at_center / height, 0.0, 1.0))
            )
            if self._task3_edge_history:
                previous_right = float(self._task3_edge_history[-1][3])
                edge_delta = abs(right_edge_x - previous_right)
                score += 1.8 * max(
                    0.0,
                    1.0 - min(1.0, edge_delta / max(30.0, width * 0.22)),
                )
            candidates.append(
                (score, contour, rect, (x, y, box_width, box_height),
                 area, edge_geometry, right_edge_x)
            )

        if not candidates:
            scanline = self._task3_scanline_fallback(frame)
            if scanline is None:
                scanline = self._task3_scanline_fallback(
                    frame, threshold_floor=118, saturation_limit=175
                )
            if scanline is None and self._task3_last_measurement is not None:
                scanline = self._task3_local_recovery(
                    frame, self._task3_last_measurement
                )
            if scanline is not None:
                return self._task3_accept_measurement(frame, scanline)
            return self._task3_recover_last()

        _, contour, rect, bounds, area, edge_geometry, right_edge_x = max(
            candidates, key=lambda candidate: candidate[0]
        )
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return None
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
        length = float(max(rect[1]))
        thickness = float(min(rect[1]))
        if edge_geometry is None:
            angle_deg, y_at_center = self._fitted_line(
                contour, width, height
            )
        else:
            angle_deg = float(edge_geometry["angle_deg"])
            y_at_center = float(edge_geometry["y_at_center"])
        if y_at_center is None:
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
            "center_line": (
                edge_geometry["center_line"]
                if edge_geometry is not None
                else ((float(bounds[0]), y_at_center),
                      (float(bounds[0] + bounds[2] - 1), y_at_center))
            ),
            "edge_support": (
                edge_geometry["edge_support"]
                if edge_geometry is not None else 0
            ),
            "right_edge_x": right_edge_x,
        }
        return self._task3_accept_measurement(frame, measurement)

    def _task3_stabilize(self, measurement):
        """Apply task-three-only temporal stability to a fresh measurement.

        A new right endpoint is not allowed to replace the tracked endpoint
        on one frame.  This is independent from the existing three-sample
        median: that median can follow a wrong bright structure when it
        survives for several frames.
        """
        raw_right_edge = float(measurement["right_edge_x"])
        previous = self._task3_last_measurement
        accepted_right_edge = raw_right_edge
        confirmed_new_edge = False
        if previous is not None:
            previous_right_edge = float(
                previous.get("right_edge_x", raw_right_edge)
            )
            width = float(measurement.get("frame_width", 800.0))
            jump_limit = max(18.0, width * TASK3_RIGHT_EDGE_MAX_STEP_RATIO)
            if abs(raw_right_edge - previous_right_edge) > jump_limit:
                pending = self._task3_pending_right_edge
                if (
                    pending is None
                    or abs(raw_right_edge - float(pending))
                    > max(10.0, jump_limit * 0.5)
                ):
                    self._task3_pending_right_edge = raw_right_edge
                    self._task3_pending_right_edge_frames = 1
                else:
                    self._task3_pending_right_edge_frames += 1
                if (
                    self._task3_pending_right_edge_frames
                    < TASK3_RIGHT_EDGE_CONFIRM_FRAMES
                ):
                    accepted_right_edge = previous_right_edge
                else:
                    self._task3_pending_right_edge = None
                    self._task3_pending_right_edge_frames = 0
                    confirmed_new_edge = True
            else:
                self._task3_pending_right_edge = None
                self._task3_pending_right_edge_frames = 0
        else:
            self._task3_pending_right_edge = None
            self._task3_pending_right_edge_frames = 0

        measurement = dict(measurement)
        measurement["right_edge_held"] = accepted_right_edge != raw_right_edge
        measurement["right_edge_x"] = accepted_right_edge
        if confirmed_new_edge:
            # Do not let the old endpoint median delay a transition that has
            # already passed the explicit two-frame confirmation.
            self._task3_edge_history = []
        self._task3_edge_history.append(
            (
                float(measurement["y_at_center"]),
                float(measurement["angle_deg"]),
                float(measurement["thickness"]),
                float(measurement["right_edge_x"]),
            )
        )
        self._task3_edge_history = self._task3_edge_history[-3:]
        y_at_center = float(
            np.median([item[0] for item in self._task3_edge_history])
        )
        angle_deg = float(
            np.median([item[1] for item in self._task3_edge_history])
        )
        thickness = float(
            np.median([item[2] for item in self._task3_edge_history])
        )
        right_edge_x = float(
            np.median([item[3] for item in self._task3_edge_history])
        )
        result = dict(measurement)
        result.update(
            {
                "y_at_center": y_at_center,
                "angle_deg": angle_deg,
                "thickness": thickness,
                "right_edge_x": right_edge_x,
                "held": False,
            }
        )
        bounds = result.get("bounds")
        if bounds is not None:
            x, _, box_width, _ = bounds
            slope = float(np.tan(np.radians(angle_deg)))
            center_x = result["frame_width"] * 0.5
            result["center_line"] = (
                (float(x), y_at_center + slope * (float(x) - center_x)),
                (
                    float(x + box_width - 1),
                    y_at_center + slope * (float(x + box_width - 1) - center_x),
                ),
            )
        self._missed_frames = 0
        self._last_measurement = result
        self._task3_missed_frames = 0
        self._task3_last_measurement = result
        return result

    @staticmethod
    def _task3_refine_right_boundary(frame, measurement):
        """Measure the terminal white support inside the accepted strip box."""
        bounds = measurement.get("bounds")
        if bounds is None:
            return float(measurement.get("right_edge_x", 0.0))
        x, y, box_width, box_height = [int(round(value)) for value in bounds]
        height, width = frame.shape[:2]
        x0 = max(0, x)
        x1 = min(width, x + box_width)
        y0 = max(0, y)
        y1 = min(height, y + box_height)
        if x1 <= x0 or y1 <= y0:
            return float(measurement.get("right_edge_x", x))

        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        absolute_white_edge = WhiteLineAlignmentDetector._task3_absolute_white_edge(
            hsv,
            0,
            y1 - y0,
            0,
            x1 - x0,
            previous_right=(
                float(measurement.get("right_edge_x")) - x0
                if measurement.get("right_edge_x") is not None else None
            ),
        )
        if absolute_white_edge is not None:
            return float(x0 + absolute_white_edge["right_edge_x"])
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        bright = ((value >= 125) & (saturation <= 145)).astype(np.uint8)
        bright = cv2.morphologyEx(
            bright * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(5, min(25, box_width // 8)), 3)
            ),
        )
        angle = float(np.radians(measurement.get("angle_deg", 0.0)))
        slope = float(np.tan(angle))
        half_band = max(3, min(18, int(round(measurement.get("thickness", 8.0) * 0.9))))
        support = np.zeros(x1 - x0, dtype=np.int32)
        center_x = width * 0.5
        for local_x in range(x1 - x0):
            absolute_x = x0 + local_x
            center_y = float(measurement["y_at_center"]) + slope * (
                absolute_x - center_x
            )
            center_y -= y0
            band_y0 = max(0, int(round(center_y - half_band)))
            band_y1 = min(bright.shape[0], int(round(center_y + half_band + 1)))
            if band_y1 > band_y0:
                support[local_x] = int(np.count_nonzero(bright[band_y0:band_y1, local_x]))

        nonzero = support[support > 0]
        if nonzero.size == 0:
            return float(measurement.get("right_edge_x", x))
        valid = support >= max(2, int(np.median(nonzero) * 0.30))
        valid_image = (valid.astype(np.uint8) * 255).reshape(1, -1)
        valid_image = cv2.morphologyEx(
            valid_image,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, min(17, box_width // 10)), 1)),
        )
        valid = valid_image[0] > 0
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return float(measurement.get("right_edge_x", x))

        terminal_end = int(valid_indices[-1])
        terminal_start = terminal_end
        while terminal_start > 0 and valid[terminal_start - 1]:
            terminal_start -= 1
        minimum_run = max(8, int((x1 - x0) * 0.03))
        if terminal_end - terminal_start + 1 < minimum_run:
            runs = []
            start = int(valid_indices[0])
            previous = start
            for current in valid_indices[1:]:
                current = int(current)
                if current - previous > 3:
                    runs.append((start, previous))
                    start = current
                previous = current
            runs.append((start, previous))
            _, terminal_end = max(
                runs,
                key=lambda run: (run[1] - run[0] + 1, run[1]),
            )
        return float(x0 + terminal_end)

    @staticmethod
    def _task3_right_boundary_x(component, offset_x):
        """Return the last sustained white-band column, not a sub-run end."""
        support = np.count_nonzero(component, axis=0)
        nonzero = support[support > 0]
        if nonzero.size == 0:
            return float(offset_x)
        threshold = max(2.0, float(np.median(nonzero)) * 0.35)
        valid = support >= threshold
        # Walk from the right and tolerate small exposure gaps. A single
        # bright pixel or an isolated middle run cannot become the boundary.
        last = int(len(valid) - 1)
        while last >= 0 and not valid[last]:
            last -= 1
        if last < 0:
            return float(offset_x)
        # Keep the terminal run only when it has enough width to represent
        # the actual strip edge. If the rightmost pixels are a short bright
        # fragment, fall back to the end of the longest sustained run.
        runs = []
        run_start = last
        gap = 0
        for index in range(last - 1, -1, -1):
            if valid[index]:
                if gap > 3:
                    runs.append((run_start, last))
                    run_start = index
                else:
                    run_start = index
                gap = 0
            else:
                gap += 1
        runs.append((run_start, last))
        minimum_run = max(8, int(len(valid) * 0.03))
        terminal_start, terminal_end = runs[0]
        if terminal_end - terminal_start + 1 >= minimum_run:
            return float(offset_x + terminal_end)
        best_start, best_end = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, run[1]),
        )
        return float(offset_x + best_end)
