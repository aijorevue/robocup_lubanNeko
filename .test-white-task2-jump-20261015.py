"""Replay-only regression tests: no cameras, serial ports, ROS, or motion."""
import argparse
import ast
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import time
import unittest

import cv2
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(y=200, gray=None, brightness=245, seed=42):
    background = np.random.default_rng(seed).integers(45, 85, (600, 800), dtype=np.uint8)
    image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (16, y-10), (445, y+10), (brightness,)*3, -1)
    if gray is not None:
        cv2.rectangle(image, (20, 305), (490, 329), (gray,)*3, -1)
    return image


class RenameLegacyCalls(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "detect_task2":
            node.attr = "_detect_band"
        return node


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class WhiteJumpTests(unittest.TestCase):
    def test_gray_structure_does_not_beat_white_tape(self):
        for gray in (120, 140, 150, 160, 170, 180, 200):
            with self.subTest(gray=gray):
                image = frame(gray=gray)
                before = baseline.WhiteLineAlignmentDetector().detect_task2(image)
                after = candidate.WhiteLineAlignmentDetector().detect_task2(image)
                self.assertAlmostEqual(before["y_at_center"], 317, delta=1)
                self.assertAlmostEqual(after["y_at_center"], 200, delta=1)

    def test_gray_only_scene_is_not_a_reference_strip(self):
        image = frame(gray=120)
        image[180:221, 0:460] = 60
        self.assertIsNotNone(baseline.WhiteLineAlignmentDetector().detect_task2(image))
        self.assertIsNone(candidate.WhiteLineAlignmentDetector().detect_task2(image))

    def test_clean_white_tape_across_valid_y_band(self):
        for y in (145, 180, 200, 240, 300, 380, 395):
            with self.subTest(y=y):
                result = candidate.WhiteLineAlignmentDetector().detect_task2(frame(y))
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result["y_at_center"], y, delta=1)

    def test_low_brightness_white_with_raw_support_is_retained(self):
        for value in (165, 180, 200, 245):
            with self.subTest(value=value):
                result = candidate.WhiteLineAlignmentDetector().detect_task2(frame(brightness=value))
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result["y_at_center"], 200, delta=1)

    def test_first_large_jump_returns_not_found(self):
        detector = candidate.WhiteLineAlignmentDetector()
        self.assertIsNotNone(detector.detect_task2(frame()))
        self.assertIsNone(detector.detect_task2(frame(310)))
        self.assertAlmostEqual(detector._task2_reference_last["y_at_center"], 200, delta=1)

    def test_repeated_frame_cannot_confirm_large_jump(self):
        detector = candidate.WhiteLineAlignmentDetector()
        detector.detect_task2(frame())
        changed = frame(310)
        self.assertIsNone(detector.detect_task2(changed))
        for _ in range(4):
            self.assertIsNone(detector.detect_task2(changed))
        accepted = detector.detect_task2(changed.copy())
        self.assertIsNotNone(accepted)
        self.assertAlmostEqual(accepted["y_at_center"], 310, delta=1)

    def test_transient_jump_does_not_pollute_the_median(self):
        detector = candidate.WhiteLineAlignmentDetector()
        for _ in range(3):
            detector.detect_task2(frame())
        self.assertIsNone(detector.detect_task2(frame(310)))
        result = detector.detect_task2(frame())
        self.assertAlmostEqual(result["y_at_center"], 200, delta=1)

    def test_lost_line_hold_is_bounded_to_four_calls(self):
        detector = candidate.WhiteLineAlignmentDetector()
        detector.detect_task2(frame())
        for index in range(5):
            result = detector.detect_task2(np.zeros((600, 800, 3), np.uint8))
            if index < 4:
                self.assertTrue(result["held"])
            else:
                self.assertIsNone(result)

    def test_max_hold_zero_does_not_reuse_missing_strip(self):
        detector = candidate.WhiteLineAlignmentDetector(max_hold_frames=0)
        detector.detect_task2(frame())
        self.assertIsNone(detector.detect_task2(np.zeros((600, 800, 3), np.uint8)))

    def test_reset_discards_pending_and_confirmed_measurement(self):
        detector = candidate.WhiteLineAlignmentDetector()
        detector.detect_task2(frame())
        detector.detect_task2(frame(310))
        detector.reset_tracking()
        self.assertIsNone(detector._task2_reference_last)
        self.assertIsNone(detector._task2_reference_pending_frame)
        result = detector.detect_task2(frame(310))
        self.assertAlmostEqual(result["y_at_center"], 310, delta=1)

    def test_null_and_empty_images(self):
        detector = candidate.WhiteLineAlignmentDetector()
        self.assertIsNone(detector.detect_task2(None))
        self.assertIsNone(detector.detect_task2(np.zeros((0, 0, 3), np.uint8)))

    def test_reported_centerline_agrees_with_filtered_coordinates(self):
        detector = candidate.WhiteLineAlignmentDetector()
        for y in (200, 210, 220, 230):
            result = detector.detect_task2(frame(y))
            (x0, y0), (x1, y1) = result["center_line"]
            predicted_y = y0 + (y1-y0)*(400-x0)/(x1-x0)
            self.assertAlmostEqual(predicted_y, result["y_at_center"], places=6)
            self.assertAlmostEqual(x1, result["right_edge_x"], places=6)

    def test_custom_roi_is_respected(self):
        detector = candidate.WhiteLineAlignmentDetector(roi_y=(0.6, 0.92))
        self.assertIsNone(detector.detect_task2(frame()))

    def test_shared_helpers_and_task1_task3_ast_are_preserved(self):
        def methods(path):
            tree = ast.parse(Path(path).read_bytes())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
            return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        old = methods(args.baseline)
        new = methods(args.candidate)
        for name, method in old.items():
            if name in {"__init__", "reset_tracking", "detect_task2"}:
                continue
            expected = RenameLegacyCalls().visit(method) if name in {"detect", "detect_task3"} else method
            self.assertEqual(ast.dump(expected), ast.dump(new[name]), name)

    def test_task1_task3_replay_identical_on_synthetic_sequences(self):
        corpus = [frame(y, gray, seed=i) for i, (y, gray) in enumerate(
            [(145, None), (200, 120), (300, 180), (240, None), (380, None)])]
        corpus += [np.zeros((600, 800, 3), np.uint8)]*6
        for method in ("detect", "detect_task3"):
            old = baseline.WhiteLineAlignmentDetector()
            new = candidate.WhiteLineAlignmentDetector()
            for index, image in enumerate(corpus):
                self.assertEqual(normalize(getattr(old, method)(image)),
                                 normalize(getattr(new, method)(image)), (method, index))

    def test_task1_task3_do_not_call_task2_reference_policy(self):
        for method in ("detect", "detect_task3"):
            detector = candidate.WhiteLineAlignmentDetector()
            def forbidden(_):
                raise AssertionError("task-two-only reference policy leaked")
            detector.detect_task2 = forbidden
            getattr(detector, method)(frame(gray=120))

    def test_historical_scene_results_are_preserved(self):
        self.assertTrue(args.evidence)
        for path in args.evidence:
            image = cv2.imread(path)
            self.assertIsNotNone(image, path)
            for method in ("detect", "detect_task2", "detect_task3"):
                old = baseline.WhiteLineAlignmentDetector()
                new = candidate.WhiteLineAlignmentDetector()
                before = getattr(old, method)(image)
                after = getattr(new, method)(image)
                if method == "detect_task2":
                    self.assertIsNotNone(after)
                    for key in ("y_at_center", "angle_deg", "right_edge_x"):
                        self.assertAlmostEqual(before[key], after[key], places=5)
                else:
                    self.assertEqual(normalize(before), normalize(after), (path, method))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evidence", nargs="+", required=True)
    args = parser.parse_args()
    baseline = load(args.baseline, "baseline_white_line")
    candidate = load(args.candidate, "candidate_white_line")
    print("OpenCV", cv2.__version__)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WhiteJumpTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    for method in ("detect_task2", "detect_task3"):
        for name, module in (("baseline", baseline), ("candidate", candidate)):
            detector = module.WhiteLineAlignmentDetector()
            sample = frame(gray=120)
            started = time.perf_counter()
            for _ in range(10):
                getattr(detector, method)(sample.copy())
            print("replay_mean_ms", method, name, round((time.perf_counter()-started)*100, 2))
    raise SystemExit(not outcome.wasSuccessful())
