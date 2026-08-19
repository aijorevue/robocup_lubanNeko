import unittest
from pathlib import Path

import cv2
import numpy as np

from abcd_detector.detector import ABCDDetector


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


class ABCDDetectorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
