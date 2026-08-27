"""Loose two-letter detector used only by the task-two secondary camera."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


LETTERS = ("A", "B", "C", "D")
SECONDARY_TIGHT_DARK_THRESHOLD = 70
SECONDARY_TIGHT_MIN_CONFIDENCE = 0.50
SECONDARY_TIGHT_MIN_SIZE_FRACTION = 0.06
# The low-contrast secondary camera compresses C/D strokes into the card
# background. Keep the existing threshold as the first pass, then score a
# narrow threshold ladder for C/D recovery without changing the A/B path.
SECONDARY_TIGHT_THRESHOLDS = (70, 80, 90, 100, 110, 120, 130, 140)
# Some secondary-camera cards are large enough to be detected as one contour,
# but their white border and board background corrupt the tight glyph crop.
# Restrict the fallback to the card interior and use it only for weak results.
SECONDARY_CARD_INNER_TRIMS = (0.16, 0.18, 0.20)
SECONDARY_CARD_THRESHOLDS = (80, 100, 120, 140)
SECONDARY_CARD_MIN_CONFIDENCE = 0.50
SECONDARY_CARD_OVERRIDE_MAX_BASE_CONFIDENCE = 0.46
SECONDARY_CARD_MIN_AREA = 800.0


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
            # Low-contrast cards can make the padded Otsu window absorb the
            # wall and card edge. Reclassify weak C/D candidates in the tight
            # contour box so the serif's dark body is scored independently.
            tight = (None, 0.0, 0.0)
            min_tight_size = max(
                32, int(round(min(height, width) * SECONDARY_TIGHT_MIN_SIZE_FRACTION))
            )
            if min(box_width, box_height) >= min_tight_size:
                tight = self._classify_tight(
                    frame[y : y + box_height, x : x + box_width]
                )
            tight_letter, tight_confidence, tight_occupancy = tight
            if (
                tight_letter in {"C", "D"}
                and tight_confidence >= SECONDARY_TIGHT_MIN_CONFIDENCE
                and (
                    letter not in {"C", "D"}
                    or tight_confidence > confidence + 0.05
                )
            ):
                letter, confidence, occupancy = tight
            card = (None, 0.0, 0.0)
            if min(box_width, box_height) >= min_tight_size:
                card = self._classify_card_interior(
                    frame[y : y + box_height, x : x + box_width]
                )
            card_letter, card_confidence, card_occupancy = card
            if (
                card_letter in {"C", "D"}
                and card_confidence >= SECONDARY_CARD_MIN_CONFIDENCE
                and confidence < SECONDARY_CARD_OVERRIDE_MAX_BASE_CONFIDENCE
            ):
                letter, confidence, occupancy = card
            if letter is None or confidence < self.min_confidence:
                continue
            min_candidate_size = max(
                32, int(round(min(height, width) * SECONDARY_TIGHT_MIN_SIZE_FRACTION))
            )
            if (
                min(box_width, box_height) < min_candidate_size
                and confidence < SECONDARY_TIGHT_MIN_CONFIDENCE
            ):
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
        # The printed C/D cards remain visible as white rectangles even when
        # their dark strokes fragment under exposure changes. Use that stable
        # card geometry as a C/D-only recovery path.
        for x, y, box_width, box_height in self._white_card_boxes(frame):
            card_letter, card_confidence, card_occupancy = (
                self._classify_card_interior(
                    frame[y : y + box_height, x : x + box_width]
                )
            )
            if (
                card_letter not in {"C", "D"}
                or card_confidence < SECONDARY_CARD_MIN_CONFIDENCE
            ):
                continue
            detections.append(
                {
                    "kind": "letter",
                    "letter": card_letter,
                    "color": "black",
                    "source": "secondary_card_interior",
                    "confidence": round(card_confidence * 100.0, 1),
                    "glyph_occupancy": round(card_occupancy, 4),
                    "center": (int(x + box_width / 2), int(y + box_height / 2)),
                    "bbox": (int(x), int(y), int(box_width), int(box_height)),
                    "box": np.array(
                        [[x, y], [x + box_width, y],
                         [x + box_width, y + box_height], [x, y + box_height]],
                        dtype=np.int32,
                    ),
                    "projected_area": float(box_width * box_height),
                    "fully_visible": x > 0 and y > 0
                    and x + box_width < width and y + box_height < height,
                    "angle": 0.0,
                }
            )
        return self._dedupe(detections)

    @staticmethod
    def _white_card_boxes(frame):
        """Find moderate white card rectangles without using black strokes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 150], dtype=np.uint8),
            np.array([180, 120, 255], dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = frame.shape[:2]
        boxes = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if area < SECONDARY_CARD_MIN_AREA:
                continue
            if not 0.65 <= box_width / float(max(1, box_height)) <= 1.80:
                continue
            if not 0.05 <= box_width / float(width) <= 0.30:
                continue
            if not 0.05 <= box_height / float(height) <= 0.30:
                continue
            fill = area / float(max(1, box_width * box_height))
            if fill < 0.25:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * perimeter, True)
            if not 4 <= len(approx) <= 8:
                continue
            boxes.append((x, y, box_width, box_height))
        return boxes

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
        return self._classify_glyph(glyph)

    def _classify_tight(self, roi):
        """Classify a weak card candidate from its dark serif strokes."""
        if roi is None or roi.size == 0:
            return None, 0.0, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4)).apply(gray)
        results = []
        for threshold in SECONDARY_TIGHT_THRESHOLDS:
            _, dark_strokes = cv2.threshold(
                gray,
                threshold,
                255,
                cv2.THRESH_BINARY_INV,
            )
            variants = (
                dark_strokes,
                cv2.morphologyEx(
                    dark_strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
                ),
            )
            for variant in variants:
                result = self._classify_glyph(self._normalize_glyph(variant))
                if result[0] is not None:
                    results.append(result)
        if not results:
            return None, 0.0, 0.0
        return max(results, key=lambda result: result[1])

    def _classify_card_interior(self, roi):
        """Recover C/D when a detected card includes its white border."""
        if roi is None or roi.size == 0:
            return None, 0.0, 0.0
        height, width = roi.shape[:2]
        if height < 24 or width < 24:
            return None, 0.0, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        results = []
        for trim in SECONDARY_CARD_INNER_TRIMS:
            x0 = int(round(width * trim))
            y0 = int(round(height * trim))
            x1 = width - x0
            y1 = height - y0
            crop = gray[y0:y1, x0:x1]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            enhanced = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(4, 4)
            ).apply(crop)
            # The raw crop is the reliable path for the present card print;
            # keeping the equalized image out of this fallback also keeps the
            # per-frame cost bounded for the live secondary stream.
            for source in (crop,):
                for threshold in SECONDARY_CARD_THRESHOLDS:
                    _, dark_strokes = cv2.threshold(
                        source,
                        threshold,
                        255,
                        cv2.THRESH_BINARY_INV,
                    )
                    result = self._classify_glyph(
                        self._normalize_glyph(dark_strokes)
                    )
                    if result[0] in {"C", "D"}:
                        results.append(result)
        if not results:
            return None, 0.0, 0.0
        return max(results, key=lambda result: result[1])

    def _classify_glyph(self, glyph):
        glyph_binary = glyph > 127
        occupancy = float(np.count_nonzero(glyph_binary)) / glyph_binary.size
        if not 0.025 <= occupancy <= 0.60:
            return None, 0.0, occupancy
        candidate_signature = self._glyph_signature(glyph_binary)
        scores = sorted(
            (
                (
                    self._best_template_score(
                        glyph, glyph_binary, candidate_signature, letter
                    ),
                    letter,
                )
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
            letter: tuple(
                (
                    variant,
                    variant > 127,
                    self._glyph_signature(variant > 127),
                )
                for variant in (
                    template,
                    cv2.GaussianBlur(template, (3, 3), 0),
                    cv2.dilate(template, kernel),
                    cv2.erode(template, kernel),
                )
            )
            for letter, template in self.templates.items()
        }

    def _best_template_score(self, glyph, glyph_binary, candidate_signature, letter):
        best = 0.0
        for template, template_binary, template_signature in self.template_bank[letter]:
            intersection = np.count_nonzero(glyph_binary & template_binary)
            union = np.count_nonzero(glyph_binary | template_binary)
            iou = intersection / max(1, union)
            correlation = max(
                0.0,
                float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0]),
            )
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
