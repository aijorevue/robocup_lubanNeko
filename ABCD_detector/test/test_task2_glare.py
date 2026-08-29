"""Task-two regressions from the 2026-10-15 glossy-field camera examples."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from abcd_detector.detector import ABCDDetector


ROOT = Path(__file__).parent
ASSETS = ROOT.parent / "abcd_detector" / "assets" / "letters"
CASES = (
    ("b_oblique", "B"), ("b_upright", "B"),
    ("b_small_side", "B"), ("d_rotated", "D"), ("b_oblique_side", "B"),
)


class Task2GlareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ABCDDetector()

    def test_supplied_cards_at_original_and_camera_resolution(self):
        for name, letter in CASES:
            image = cv2.imread(str(ROOT / "fixtures" / "task2_glare" / (name + ".png")))
            self.assertIsNotNone(image, name)
            for size in (None, (800, 600)):
                frame = image if size is None else cv2.resize(image, size)
                with self.subTest(name=name, size=size):
                    detections = self.detector.detect_task2(frame)
                    self.assertEqual([d["letter"] for d in detections], [letter])
                    self.assertTrue(detections[0]["depth_valid"])
                    self.assertEqual(detections[0]["depth_source"], "task2_white_block")

    def test_supplied_cards_with_jpeg_and_blur(self):
        for name, letter in CASES:
            frame = cv2.resize(cv2.imread(str(ROOT / "fixtures" / "task2_glare" / (name + ".png"))), (800, 600))
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            for variant in (cv2.imdecode(encoded, 1), cv2.GaussianBlur(frame, (3, 3), 0)):
                with self.subTest(name=name):
                    detections = self.detector.detect_task2(variant)
                    self.assertEqual([d["letter"] for d in detections], [letter])
                    self.assertTrue(detections[0]["depth_valid"])

    def test_abcd_rotations_on_weakly_saturated_green(self):
        for letter in "ABCD":
            card = cv2.resize(cv2.imread(str(ASSETS / (letter + ".png"))), (150, 150))
            for angle in range(0, 360, 15):
                frame = np.full((400, 400, 3), (165, 200, 170), np.uint8)
                frame[125:275, 125:275] = card
                matrix = cv2.getRotationMatrix2D((200, 200), angle, 1.0)
                frame = cv2.warpAffine(frame, matrix, (400, 400), borderValue=(165, 200, 170))
                with self.subTest(letter=letter, angle=angle):
                    detections = self.detector.detect_task2(frame)
                    self.assertEqual([d["letter"] for d in detections], [letter])

    def test_blank_cards_glare_and_holes_are_not_letters(self):
        for background in ((165, 200, 170), (195, 210, 195), (80, 150, 90)):
            frame = np.full((400, 500, 3), background, np.uint8)
            cv2.rectangle(frame, (0, 360), (499, 399), (240, 240, 240), -1)
            cv2.circle(frame, (80, 150), 30, (20, 20, 20), -1)
            cv2.circle(frame, (420, 150), 30, (20, 20, 20), -1)
            for angle in (0, 23, 67):
                test = frame.copy()
                box = cv2.boxPoints(((250, 200), (130, 130), angle)).astype(np.int32)
                cv2.fillConvexPoly(test, box, (240, 240, 240))
                with self.subTest(background=background, angle=angle):
                    self.assertEqual(self.detector.detect_task2(test), [])

    def test_recovered_card_does_not_hide_another_legacy_candidate(self):
        frame = np.full((400, 600, 3), (70, 145, 70), np.uint8)
        for x, letter, tint in ((80, "B", (240, 240, 240)), (360, "D", (165, 180, 165))):
            glyph = cv2.resize(cv2.imread(str(ASSETS / (letter + ".png")), 0), (150, 150))
            patch = np.repeat(glyph[:, :, None], 3, axis=2)
            patch[glyph > 180] = tint
            frame[120:270, x:x + 150] = patch
        detections = self.detector.detect_task2(frame)
        self.assertEqual(sorted(d["letter"] for d in detections), ["B", "D"])
        self.assertTrue(all(d["depth_valid"] for d in detections))


if __name__ == "__main__":
    unittest.main()
