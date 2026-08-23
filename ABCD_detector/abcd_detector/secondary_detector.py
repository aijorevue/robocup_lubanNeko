"""Loose two-letter detector used only by the task-two secondary camera."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


LETTERS = ("A", "B", "C", "D")


class SecondaryLetterDetector:
    """Detect distant black Times New Roman glyphs without requiring a border."""

    def __init__(self):
        self.min_confidence = 0.38
        self.templates = self._load_templates()
        self.template_bank = self._build_template_bank()

    def detect(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return []
        height, width = frame.shape[:2]
        if height < 64 or width < 64:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        mask = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            7,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        frame_area = float(width * height)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if not max(24, int(height * 0.045)) <= box_height <= int(height * 0.34):
                continue
            if not max(12, int(width * 0.014)) <= box_width <= int(width * 0.24):
                continue
            aspect = box_width / float(box_height)
            fill = area / max(1.0, float(box_width * box_height))
            if not 0.24 <= aspect <= 1.55 or not 0.10 <= fill <= 0.80:
                continue
            if area < max(180.0, frame_area * 0.00035):
                continue

            # Distant serif glyphs can split into several strokes. The wider
            # context is intentional and differs from the strict main-camera ROI.
            pad = max(12, int(round(max(box_width, box_height) * 0.34)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(width, x + box_width + pad)
            y1 = min(height, y + box_height + pad)
            roi = frame[y0:y1, x0:x1]
            letter, confidence, occupancy = self._classify(roi)
            if letter is None or confidence < self.min_confidence:
                continue
            detections.append(
                {
                    "kind": "letter",
                    "letter": letter,
                    "color": "black",
                    "source": "secondary_times_new_roman",
                    "confidence": round(confidence * 100.0, 1),
                    "glyph_occupancy": round(occupancy, 4),
                    "center": (int(x + box_width / 2), int(y + box_height / 2)),
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "box": np.array(
                        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        dtype=np.int32,
                    ),
                    "projected_area": max(area, float(box_width * box_height)),
                    "fully_visible": x0 > 0 and y0 > 0 and x1 < width and y1 < height,
                    "angle": 0.0,
                }
            )
        return self._dedupe(detections)

    def _classify(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4)).apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Otsu is deliberately the only classification mask here. Adaptive
        # and fixed masks made the distant A fluctuate to B as exposure moved.
        glyph = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        glyph = self._normalize_glyph(glyph)
        glyph_binary = glyph > 127
        occupancy = float(np.count_nonzero(glyph_binary)) / glyph_binary.size
        if not 0.025 <= occupancy <= 0.60:
            return None, 0.0, occupancy
        scores = sorted(
            (
                (self._best_template_score(glyph, glyph_binary, letter), letter)
                for letter in LETTERS
            ),
            reverse=True,
        )
        score, letter = scores[0]
        runner_up = scores[1][0]
        confidence = max(0.0, min(1.0, score + 0.20 * (score - runner_up)))
        return letter, confidence, occupancy

    def _load_templates(self):
        root = Path(__file__).resolve().parent / "assets" / "letters"
        templates = {}
        for letter in LETTERS:
            image = cv2.imread(str(root / f"{letter}.png"), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"secondary letter template missing: {letter}")
            templates[letter] = self._normalize_glyph(image)
        return templates

    def _build_template_bank(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return {
            letter: (
                template,
                cv2.GaussianBlur(template, (3, 3), 0),
                cv2.dilate(template, kernel),
                cv2.erode(template, kernel),
            )
            for letter, template in self.templates.items()
        }

    def _best_template_score(self, glyph, glyph_binary, letter):
        best = 0.0
        candidate_signature = self._glyph_signature(glyph_binary)
        for template in self.template_bank[letter]:
            template_binary = template > 127
            intersection = np.count_nonzero(glyph_binary & template_binary)
            union = np.count_nonzero(glyph_binary | template_binary)
            iou = intersection / max(1, union)
            correlation = max(
                0.0,
                float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0]),
            )
            template_signature = self._glyph_signature(template_binary)
            projection = self._projection_similarity(
                candidate_signature[2], template_signature[2]
            )
            shape = self._shape_similarity(candidate_signature[0], template_signature[0])
            hole = 1.0 if candidate_signature[1] == template_signature[1] else 0.35
            score = (
                0.45 * correlation
                + 0.20 * iou
                + 0.18 * projection
                + 0.12 * shape
                + 0.05 * hole
            )
            best = max(best, score)
        return best

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
        output = np.zeros((64, 64), dtype=np.uint8)
        x0 = (64 - resized.shape[1]) // 2
        y0 = (64 - resized.shape[0]) // 2
        output[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return output

    @staticmethod
    def _glyph_signature(binary):
        mask = np.asarray(binary, dtype=np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, 0, (np.zeros(16), np.zeros(16))
        outer_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        outer = contours[outer_index]
        holes = 0
        if hierarchy is not None:
            child = hierarchy[0][outer_index][2]
            while child >= 0:
                holes += 1
                child = hierarchy[0][child][0]
        row = cv2.resize(
            mask.mean(axis=1, keepdims=True), (1, 16), interpolation=cv2.INTER_AREA
        ).ravel()
        col = cv2.resize(
            mask.mean(axis=0, keepdims=True), (16, 1), interpolation=cv2.INTER_AREA
        ).ravel()
        return outer, holes, (row / 255.0, col / 255.0)

    @staticmethod
    def _projection_similarity(left, right):
        values = []
        for a, b in zip(left, right):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            if np.std(a) < 1e-5 or np.std(b) < 1e-5:
                values.append(0.0)
            else:
                values.append(max(0.0, float(np.corrcoef(a, b)[0, 1])))
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _shape_similarity(left, right):
        if left is None or right is None:
            return 0.0
        distance = cv2.matchShapes(left, right, cv2.CONTOURS_MATCH_I1, 0.0)
        return max(0.0, 1.0 - min(1.0, float(distance) * 2.5))

    @staticmethod
    def _dedupe(detections):
        kept = []
        for item in sorted(
            detections, key=lambda value: float(value["confidence"]), reverse=True
        ):
            x, y = item["center"]
            if any(
                np.hypot(x - existing["center"][0], y - existing["center"][1])
                < 0.45 * max(item["bbox"][2], item["bbox"][3])
                for existing in kept
            ):
                continue
            kept.append(item)
        kept.sort(key=lambda item: item["center"][0])
        return kept
