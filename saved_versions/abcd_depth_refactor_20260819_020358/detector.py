"""Detect white quadrilateral blocks carrying the RoboCup A/B/C/D letters.

The detector deliberately owns no camera and has no machine-specific paths. It
can therefore run inside the existing RK camera process or as a ROS 2 topic
node without competing for the camera device.
"""

from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # YOLO is optional; the template pipeline still works.
    YOLO = None


LETTERS = ("A", "B", "C", "D")


class ABCDDetector:
    """Find fully visible white letter blocks in a BGR image."""

    def __init__(self, config_path=None):
        self.min_white_value = 105
        self.max_white_saturation = 86
        self.min_candidate_area = 500.0
        self.max_candidate_area_ratio = 0.72
        self.min_side_px = 28.0
        self.max_aspect_ratio = 2.35
        self.min_rectangularity = 0.54
        self.min_glyph_occupancy = 0.025
        self.max_glyph_occupancy = 0.46
        self.min_confidence = 0.42
        self.yolo_weights_path = os.environ.get("ABCD_YOLO_WEIGHTS", "")
        self.yolo_confidence = 0.45
        self._yolo_model = None
        self._load_config(config_path)
        self.templates = self._build_templates()
        self.template_bank = self._build_template_bank()

    def _load_config(self, config_path):
        """Load optional package config without requiring a workspace path."""

        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config" / "letter_detector.yaml"
        path = Path(config_path)
        if not path.is_file():
            return
        try:
            import yaml

            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (ImportError, OSError, UnicodeError, ValueError):
            return
        parameters = (
            document.get("abcd_detector", {})
            .get("ros__parameters", {})
        )
        for name in (
            "min_white_value",
            "max_white_saturation",
            "min_candidate_area",
            "max_candidate_area_ratio",
            "min_side_px",
            "max_aspect_ratio",
            "min_rectangularity",
            "min_glyph_occupancy",
            "max_glyph_occupancy",
            "min_confidence",
            "yolo_confidence",
        ):
            if name in parameters:
                try:
                    setattr(self, name, float(parameters[name]))
                except (TypeError, ValueError):
                    pass
        if "yolo_weights_path" in parameters:
            try:
                self.yolo_weights_path = str(parameters["yolo_weights_path"])
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _build_templates():
        templates = {}
        for letter in LETTERS:
            template_path = (
                Path(__file__).resolve().parent / "assets" / "letters" / f"{letter}.png"
            )
            if template_path.is_file():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None and template.size:
                    templates[letter] = ABCDDetector._normalize_glyph(template)
                    continue

            canvas = np.zeros((120, 120), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2.75
            thickness = 8
            (width, height), baseline = cv2.getTextSize(
                letter, font, scale, thickness
            )
            origin = (
                max(0, (120 - width) // 2),
                max(height, (120 + height) // 2 - baseline // 2),
            )
            cv2.putText(
                canvas,
                letter,
                origin,
                font,
                scale,
                255,
                thickness,
                cv2.LINE_AA,
            )
            templates[letter] = ABCDDetector._normalize_glyph(canvas)
        return templates

    def _build_template_bank(self):
        bank = {}
        for letter, template in self.templates.items():
            variants = [template]
            if letter == "D":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                variants.append(cv2.GaussianBlur(template, (3, 3), 0))
                variants.append(cv2.dilate(template, kernel))
                variants.append(cv2.erode(template, kernel))
            bank[letter] = variants
        return bank

    @staticmethod
    def _normalize_glyph(mask):
        binary = np.asarray(mask, dtype=np.uint8)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(binary) > binary.size // 2:
            binary = cv2.bitwise_not(binary)
        points = cv2.findNonZero(binary)
        if points is None:
            return np.zeros((64, 64), dtype=np.uint8)
        x, y, width, height = cv2.boundingRect(points)
        cropped = binary[y : y + height, x : x + width]
        scale = min(56.0 / max(1, width), 56.0 / max(1, height))
        resized = cv2.resize(
            cropped,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        normalized = np.zeros((64, 64), dtype=np.uint8)
        offset_x = (64 - resized.shape[1]) // 2
        offset_y = (64 - resized.shape[0]) // 2
        normalized[
            offset_y : offset_y + resized.shape[0],
            offset_x : offset_x + resized.shape[1],
        ] = resized
        return normalized

    def detect(self, frame):
        """Return controller-compatible detections for A/B/C/D targets."""

        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return []
        height, width = frame.shape[:2]
        if height < 32 or width < 32:
            return []

        results = self._detect_yolo(frame)
        results.extend(self._detect_white_blocks(frame))
        results.extend(self._detect_dark_letters(frame))
        return self._dedupe_detections(results)

    def _detect_yolo(self, frame):
        if YOLO is None or not self.yolo_weights_path:
            return []
        weights_path = Path(self.yolo_weights_path)
        if not weights_path.is_file():
            return []
        if self._yolo_model is None:
            try:
                self._yolo_model = YOLO(str(weights_path))
            except Exception:
                return []

        detections = []
        try:
            prediction = self._yolo_model.predict(
                frame,
                conf=float(self.yolo_confidence),
                verbose=False,
            )
        except Exception:
            return []

        for result in prediction:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_index = int(box.cls[0]) if hasattr(box.cls, "__iter__") else int(box.cls)
                label = str(names.get(cls_index, cls_index)).upper()
                if label not in LETTERS:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x0, y0, x1, y1 = [int(round(value)) for value in xyxy]
                confidence = float(box.conf[0]) if hasattr(box.conf, "__iter__") else float(box.conf)
                detections.append(
                    {
                        "kind": "letter",
                        "letter": label,
                        "color": "yolo",
                        "source": "abcd_detector_yolo",
                        "confidence": round(confidence * 100.0, 1),
                        "glyph_occupancy": 0.0,
                        "center": (int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))),
                        "box": np.array(
                            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                            dtype=np.int32,
                        ),
                        "bbox": (x0, y0, x1 - x0, y1 - y0),
                        "projected_area": float(max(0, x1 - x0) * max(0, y1 - y0)),
                        "fully_visible": True,
                        "angle": 0.0,
                    }
                )
        return detections

    def _detect_white_blocks(self, frame):
        height, width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array((0, 0, int(self.min_white_value)), dtype=np.uint8),
            np.array((180, int(self.max_white_saturation), 255), dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        candidate_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        candidate_mask = cv2.morphologyEx(
            candidate_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        contours, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = float(height * width)
        edge_margin = max(5, int(min(width, height) * 0.012))
        results = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if area < self.min_candidate_area or area > frame_area * self.max_candidate_area_ratio:
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rect_w, rect_h), angle = rect
            short_side = min(rect_w, rect_h)
            long_side = max(rect_w, rect_h)
            if short_side < self.min_side_px or short_side <= 0.0:
                continue
            if long_side / short_side > self.max_aspect_ratio:
                continue
            rectangularity = area / max(1.0, rect_w * rect_h)
            if rectangularity < self.min_rectangularity:
                continue
            box = self._ordered_box(cv2.boxPoints(rect))
            x, y, box_width, box_height = cv2.boundingRect(box.astype(np.int32))
            fully_visible = (
                x > edge_margin
                and y > edge_margin
                and x + box_width < width - edge_margin
                and y + box_height < height - edge_margin
            )
            rectified = self._rectify(frame, box, 128)
            letter, confidence, occupancy = self._classify(rectified)
            if letter is None:
                continue
            if confidence < max(self.min_confidence, 0.45):
                continue
            results.append(
                {
                    "kind": "letter",
                    "letter": letter,
                    "color": "white",
                    "source": "abcd_detector",
                    "confidence": round(float(confidence) * 100.0, 1),
                    "glyph_occupancy": round(float(occupancy), 4),
                    "center": (int(round(cx)), int(round(cy))),
                    "box": box.astype(np.int32),
                    "bbox": (int(x), int(y), int(box_width), int(box_height)),
                    "projected_area": max(area, float(rect_w * rect_h)),
                    "fully_visible": fully_visible,
                    "angle": round(float(angle), 1),
                }
            )
        return results

    def _detect_dark_letters(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = min(145, max(70, int(np.percentile(gray, 28)) + 20))
        dark_mask = cv2.inRange(gray, 0, threshold)
        dark_mask = cv2.morphologyEx(
            dark_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)),
        )
        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        min_h = max(30, int(height * 0.045))
        max_h = int(height * 0.55)
        min_w = max(16, int(width * 0.018))
        max_w = int(width * 0.45)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_height < min_h or box_height > max_h:
                continue
            if box_width < min_w or box_width > max_w:
                continue
            aspect = box_width / float(box_height)
            if not 0.25 <= aspect <= 1.35:
                continue
            area = float(cv2.contourArea(contour))
            fill = area / max(1.0, float(box_width * box_height))
            if not 0.08 <= fill <= 0.82:
                continue

            pad = max(8, int(max(box_width, box_height) * 0.18))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(width, x + box_width + pad)
            y1 = min(height, y + box_height + pad)
            roi = frame[y0:y1, x0:x1]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if float(roi_gray.mean()) < 100.0:
                continue
            letter, confidence, occupancy = self._classify_dark_letter(roi)
            if letter is None:
                continue
            results.append(
                {
                    "kind": "letter",
                    "letter": letter,
                    "color": "black",
                    "source": "abcd_detector_dark_glyph",
                    "confidence": round(float(confidence) * 100.0, 1),
                    "glyph_occupancy": round(float(occupancy), 4),
                    "center": (int(round(x + box_width / 2.0)), int(round(y + box_height / 2.0))),
                    "box": np.array(
                        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        dtype=np.int32,
                    ),
                    "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                    "projected_area": max(area, float(box_width * box_height)),
                    "fully_visible": True,
                    "angle": 0.0,
                }
            )
        return results

    def _classify_dark_letter(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, glyph = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        glyph = cv2.morphologyEx(
            glyph,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        )
        glyph = self._normalize_glyph(glyph)
        glyph_binary = glyph > 127
        occupancy = float(np.count_nonzero(glyph_binary)) / glyph_binary.size
        if not 0.025 <= occupancy <= 0.58:
            return None, 0.0, occupancy

        scores = []
        for letter in LETTERS:
            scores.append((self._best_template_score(glyph, glyph_binary, letter), letter))
        scores.sort(reverse=True)
        best_score, best_letter = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        confidence = max(0.0, min(1.0, best_score + 0.18 * (best_score - runner_up)))
        if best_score < (0.45 if best_letter == "D" else 0.50):
            return None, confidence, occupancy
        return best_letter, confidence, occupancy

    @staticmethod
    def _dedupe_detections(detections):
        def rect(det):
            x, y, width, height = det["bbox"]
            return (x, y, x + width, y + height)

        def iou(a, b):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter <= 0:
                return 0.0
            area_a = max(1, ax1 - ax0) * max(1, ay1 - ay0)
            area_b = max(1, bx1 - bx0) * max(1, by1 - by0)
            return inter / float(area_a + area_b - inter)

        sorted_detections = sorted(
            detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True
        )
        kept = []
        for detection in sorted_detections:
            detection_rect = rect(detection)
            if any(iou(detection_rect, rect(existing)) > 0.25 for existing in kept):
                continue
            kept.append(detection)
        kept.sort(key=lambda item: (item["center"][0], item["center"][1]))
        return kept

    @staticmethod
    def _ordered_box(points):
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered

    @staticmethod
    def _rectify(frame, box, side):
        destination = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(box.astype(np.float32), destination)
        return cv2.warpPerspective(frame, matrix, (side, side))

    def _classify(self, rectified):
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        inner = gray[14:-14, 14:-14]
        inner = cv2.GaussianBlur(inner, (3, 3), 0)
        _, glyph = cv2.threshold(
            inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        glyph = cv2.morphologyEx(
            glyph,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        glyph = self._normalize_glyph(glyph)
        glyph_binary = glyph > 127
        occupancy = float(np.count_nonzero(glyph_binary)) / glyph_binary.size
        if not self.min_glyph_occupancy <= occupancy <= self.max_glyph_occupancy:
            return None, 0.0, occupancy

        scores = []
        for letter in LETTERS:
            scores.append((self._best_template_score(glyph, glyph_binary, letter), letter))
        scores.sort(reverse=True)
        best_score, best_letter = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        confidence = max(0.0, min(1.0, best_score + 0.18 * (best_score - runner_up)))
        if best_score < self.min_confidence:
            return None, confidence, occupancy
        return best_letter, confidence, occupancy

    def _best_template_score(self, glyph, glyph_binary, letter):
        templates = self.template_bank.get(letter, [self.templates[letter]])
        best_score = 0.0
        for template in templates:
            template_binary = template > 127
            intersection = np.count_nonzero(glyph_binary & template_binary)
            union = np.count_nonzero(glyph_binary | template_binary)
            iou = intersection / max(1, union)
            correlation = cv2.matchTemplate(
                glyph,
                template,
                cv2.TM_CCOEFF_NORMED,
            )[0, 0]
            correlation = max(0.0, float(correlation))
            weight_corr, weight_iou = (0.72, 0.28) if letter == "D" else (0.65, 0.35)
            score = weight_corr * correlation + weight_iou * iou
            if score > best_score:
                best_score = score
        return best_score
