import unittest
from pathlib import Path

import cv2
import numpy as np

from abcd_detector.detector import (
    ABCDDetector,
    TASK2_B_CENTER_BAR_MIN_RATIO,
    TASK3_MAX_CENTER_Y_RATIO,
)


ASSET_DIR = Path(__file__).resolve().parents[1] / "abcd_detector" / "assets" / "letters"


def letter_block(letter, size=240):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (24, 24, 24)
    cv2.rectangle(image, (36, 36), (size - 36, size - 36), (242, 242, 242), -1)
    letter_img = cv2.imread(str(ASSET_DIR / f"{letter}.png"))
    if letter_img is None:
        raise AssertionError(f"missing rulebook letter template: {letter}")
    letter_img = cv2.resize(letter_img, (156, 156), interpolation=cv2.INTER_AREA)
    image[42:198, 42:198] = letter_img
    return image


def letter_page(letter, size=360):
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    letter_img = cv2.imread(str(ASSET_DIR / f"{letter}.png"))
    if letter_img is None:
        raise AssertionError(f"missing rulebook letter template: {letter}")
    letter_img = cv2.resize(letter_img, (180, 180), interpolation=cv2.INTER_AREA)
    image[90:270, 90:270] = letter_img
    return image


def rotated_green_block(letter, angle, size=320):
    background = np.full((size, size, 3), (70, 145, 70), dtype=np.uint8)
    block = letter_block(letter, size=220)
    matrix = cv2.getRotationMatrix2D((110, 110), angle, 1.0)
    rotated = cv2.warpAffine(
        block, matrix, (220, 220), borderValue=(70, 145, 70)
    )
    background[50:270, 50:270] = rotated
    return background


class ABCDDetectorTests(unittest.TestCase):
    def test_task2_center_bar_corrects_ambiguous_b_without_changing_ordinary(self):
        class AmbiguousBDDetector(ABCDDetector):
            def _best_template_score(self, glyph, glyph_binary, letter):
                return {"A": 0.18, "B": 0.50, "C": 0.22, "D": 0.54}[letter]

        detector = AmbiguousBDDetector()
        rectified = cv2.resize(
            cv2.imread(str(ASSET_DIR / "B.png")),
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )
        ordinary, _, _ = detector._classify(rectified, inset=2)
        task2, _, _ = detector._classify(
            rectified, inset=2, task2_bd_check=True
        )
        self.assertEqual(ordinary, "D")
        self.assertEqual(task2, "B")

    def test_task2_center_bar_rejects_true_d_shape(self):
        detector = ABCDDetector()
        self.assertGreaterEqual(
            detector._task2_center_bar_ratio(detector.templates["B"] > 127),
            TASK2_B_CENTER_BAR_MIN_RATIO,
        )
        self.assertLess(
            detector._task2_center_bar_ratio(detector.templates["D"] > 127),
            TASK2_B_CENTER_BAR_MIN_RATIO,
        )

    def test_task2_detects_each_letter_without_changing_task3_path(self):
        detector = ABCDDetector()
        for letter in "ABCD":
            task2_detections = detector.detect_task2(letter_block(letter))
            task3_detections = detector.detect_task3_rotated(
                rotated_green_block(letter, 17)
            )
            self.assertEqual(task2_detections[0]["letter"], letter)
            self.assertEqual(task3_detections[0]["letter"], letter)

    def test_task3_bd_structure_correction_is_bidirectional(self):
        class AmbiguousBDetector(ABCDDetector):
            def _score_rotation_batch(self, normalized_glyphs):
                count = len(normalized_glyphs)
                return {
                    "A": np.full(count, 0.18, dtype=np.float32),
                    "B": np.full(count, 0.54, dtype=np.float32),
                    "C": np.full(count, 0.22, dtype=np.float32),
                    "D": np.full(count, 0.58, dtype=np.float32),
                }

        class AmbiguousDDetector(ABCDDetector):
            def _score_rotation_batch(self, normalized_glyphs):
                count = len(normalized_glyphs)
                return {
                    "A": np.full(count, 0.18, dtype=np.float32),
                    "B": np.full(count, 0.58, dtype=np.float32),
                    "C": np.full(count, 0.22, dtype=np.float32),
                    "D": np.full(count, 0.54, dtype=np.float32),
                }

        b_rectified = cv2.resize(
            cv2.imread(str(ASSET_DIR / "B.png")),
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )
        d_rectified = cv2.resize(
            cv2.imread(str(ASSET_DIR / "D.png")),
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )
        b_letter, _, _, _, _ = AmbiguousBDetector()._classify_rotation_invariant(
            b_rectified, inset=2
        )
        d_letter, _, _, _, _ = AmbiguousDDetector()._classify_rotation_invariant(
            d_rectified, inset=2
        )
        self.assertEqual(b_letter, "B")
        self.assertEqual(d_letter, "D")

    def test_task3_rejects_candidate_below_global_view_limit(self):
        detector = ABCDDetector()
        frame = rotated_green_block("B", 17)
        local_center_y = frame.shape[0] // 2
        full_height = 600
        origin_y = int(full_height * TASK3_MAX_CENTER_Y_RATIO) - local_center_y + 1
        detections = detector.detect_task3_rotated(
            frame,
            frame_shape=(full_height, frame.shape[1], 3),
            frame_origin=(0, origin_y),
        )
        self.assertEqual(detections, [])

    def test_main_camera_serif_b_with_narrow_border(self):
        frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "main_serif_b.jpg"))
        self.assertIsNotNone(frame)
        detections = ABCDDetector().detect(frame)
        matches = [
            d for d in detections
            if d["letter"] == "B"
            and 280 <= d["center"][0] <= 440
            and 230 <= d["center"][1] <= 410
        ]
        self.assertTrue(matches, detections)
        self.assertGreater(matches[0]["confidence"], 80.0)

    def test_detects_each_letter_as_white_letter(self):
        detector = ABCDDetector()
        for letter in "ABCD":
            detections = detector.detect(letter_block(letter))
            self.assertEqual(len(detections), 1, letter)
            self.assertEqual(detections[0]["kind"], "letter")
            self.assertEqual(detections[0]["letter"], letter)
            self.assertEqual(detections[0]["color"], "white")
            self.assertGreater(detections[0]["area_percent"], 0.0)
            self.assertIsNotNone(detections[0]["distance_cm"])
            self.assertAlmostEqual(
                detections[0]["depth_cm"],
                detections[0]["distance_cm"],
            )
            self.assertAlmostEqual(
                detections[0]["depth_mm"],
                detections[0]["distance_cm"] * 10.0,
            )

    def test_ignores_plain_white_square_without_glyph(self):
        detector = ABCDDetector()
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (200, 200), (238, 238, 238), -1)
        self.assertEqual(detector.detect(image), [])

    def test_detects_each_letter_on_white_page(self):
        detector = ABCDDetector()
        for letter in "ABCD":
            detections = detector.detect(letter_page(letter))
            self.assertEqual(len(detections), 1, letter)
            self.assertEqual(detections[0]["kind"], "letter")
            self.assertEqual(detections[0]["letter"], letter)
            self.assertIsNotNone(detections[0]["distance_cm"])

    def test_task3_rotation_classifier_handles_random_in_plane_angles(self):
        detector = ABCDDetector()
        for letter in "ABCD":
            for angle in (0, 17, 43, 90, 137, 180, 223, 270, 319):
                with self.subTest(letter=letter, angle=angle):
                    detections = detector.detect_task3_rotated(
                        rotated_green_block(letter, angle)
                    )
                    self.assertTrue(detections, (letter, angle))
                    self.assertEqual(detections[0]["letter"], letter)
                    self.assertGreaterEqual(
                        detections[0]["classification_margin"], 0.035
                    )


if __name__ == "__main__":
    unittest.main()
