"""Hardware-free integration tests against the board's actual formal package.

Usage: TASK2_CANDIDATE_PATH=/path/to/staged.py python3 -B test_alignment.py
No test opens a camera/serial port or calls the formal application entrypoint.
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import types
import unittest
from unittest import mock

import cv2
import numpy as np

candidate = Path(os.environ["TASK2_CANDIDATE_PATH"])
spec = importlib.util.spec_from_file_location("task2_candidate", candidate)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
vision, platform = app.load_formal()


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Reader:
    def __init__(self, frame=None, fresh=True, clock=None):
        self.frame = np.zeros((600, 800, 3), np.uint8) if frame is None else frame
        self.fresh = fresh
        self.clock = clock
        self.closed = False
        self.error = None
        self.ok = True

    def latest(self):
        if self.clock is not None:
            self.clock.now += 0.1
        return self.ok, self.frame.copy() if self.fresh else self.frame, 0, self.error

    def close(self):
        self.closed = True


class Bridge:
    enabled = write_enabled = assumed_feedback = last_command_ok = True
    arm_time_ms = 500
    aux_time_ms = splitter_time_ms = gripper_time_ms = 200
    arm_fd = 123
    status = "fake: no hardware"

    def __init__(self):
        self.commands = []

    def send_targets(self, **kwargs):
        self.commands.append((dict(kwargs), self.arm_time_ms, self.gripper_time_ms))
        return self.status


def make_target(kind="letter", label="D", depth=10, center=(400, 300)):
    return {"kind": kind, "letter": label, "color": label if kind == "ring" else "white",
            "center": center, "bbox": (center[0]-20, center[1]-20, 40, 40),
            "confidence": 90, "score": 0.9, "distance_cm": depth}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.sleep = mock.patch.object(app.time, "sleep", lambda seconds: None)
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()
        self.output.__exit__(None, None, None)

    def controller(self):
        bridge = Bridge()
        controller = app.build_controller(vision, platform, bridge, "red")
        return controller, bridge

    def prepared_task(self, slot=1):
        controller, bridge = self.controller()
        task = controller.platform_task
        clock = Clock()
        task.clock = clock
        task.selected = ("C", "D")
        task.high_ready = True
        task.begin_slot("red", slot)
        task.discard = 0
        return controller, bridge, task, clock

    def test_uses_real_formal_callbacks(self):
        controller, _ = self.controller()
        task = controller.platform_task
        for name in ("pose", "retreat_pose", "gripper", "center", "ring_place_id6",
                     "ring_place_id2", "ring_place_id1", "ring_place_id12",
                     "ring_return_high_id1", "ring_return_high_id2", "ring_return_high_id6",
                     "letter_place_id6", "letter_place_id2", "letter_place_id1",
                     "letter_place_id12", "letter_return_high_id1",
                     "letter_return_high_id2", "letter_return_high_id6"):
            formal_name = "_platform_" + name
            self.assertIs(getattr(task, name).__func__,
                          getattr(vision.TargetGraspController, formal_name))

    def test_default_window_and_slots_4_5(self):
        edge = make_target(center=(40, 300))
        for slot, accepted in ((1, False), (3, False), (4, True), (5, True), (6, False)):
            _, _, task, _ = self.prepared_task(slot)
            for _ in range(3):
                task._detect([edge], (600, 800, 3))
            self.assertEqual(task.target_key is not None, accepted, slot)
        self.assertEqual(platform.TARGET_WINDOW_SIZE_PX, 520)

    def test_selected_letter_and_own_ring_only(self):
        _, _, task, _ = self.prepared_task()
        self.assertFalse(task._allowed(make_target(label="A")))
        self.assertTrue(task._allowed(make_target(label="D")))
        self.assertFalse(task._allowed(make_target("ring", "blue")))
        self.assertTrue(task._allowed(make_target("ring", "red")))

    def test_formal_detection_uses_task2_and_rejects_opponent_ring_overlap(self):
        detector = vision.TargetDetector()
        detector.letter_detector = mock.Mock()
        detector.letter_detector.detect_task2.return_value = [make_target()]
        with mock.patch.object(vision, "detect_platform_rings", return_value=[]):
            accepted = detector.detect(Reader().frame, vision.DETECTION_MODE_PLATFORM_TARGETS,
                                       "red", ("C", "D"))
        self.assertEqual([d["letter"] for d in accepted], ["D"])
        with mock.patch.object(vision, "detect_platform_rings",
                               return_value=[make_target("ring", "blue")]):
            rejected = detector.detect(Reader().frame, vision.DETECTION_MODE_PLATFORM_TARGETS,
                                       "red", ("C", "D"))
        self.assertFalse(any(d.get("kind") == "letter" for d in rejected))
        detector.letter_detector.detect.assert_not_called()

    def test_letter_quota_shared_across_slots(self):
        _, _, task, _ = self.prepared_task()
        task.letter_success_counts["D"] = 2
        for slot in (1, 4, 8):
            task.begin_slot("red", slot)
            self.assertFalse(task._allowed(make_target(label="D")))

    def test_production_calibration_and_action_order(self):
        for kind, label, pose in (("letter", "D", (457, 478)), ("ring", "red", (447, 488))):
            _, _, task, _ = self.prepared_task()
            for _ in range(3):
                task._detect([make_target(kind, label, 25)], (600, 800, 3))
            labels = [action[0] for action in task.actions]
            descend = next(action for action in task.actions if action[0] == "DESCEND")
            self.assertEqual(descend[2][0][:2], pose)
            self.assertEqual(labels[:4], ["GRIPPER_OPEN", "POST_OPEN_ID2_RETREAT",
                                          "DESCEND", "GRIPPER_CLOSE"])
            self.assertIn("RETURN_INTERMEDIATE_HIGH", labels)
            self.assertTrue(labels[-1].endswith("HIGH_ID6"))

    def test_depth_matrix_reuses_production_solver_for_letters_and_rings(self):
        for kind, label in (("letter", "D"), ("ring", "red")):
            for depth in (7, 8, 9, 10, 12, 13, 14, 15, 16, 19, 20.5, 24, 25, 30):
                with self.subTest(kind=kind, depth=depth):
                    _, _, task, _ = self.prepared_task()
                    with mock.patch.object(platform, "calibrated_grasp_ticks",
                                           wraps=platform.calibrated_grasp_ticks) as solver:
                        for _ in range(3):
                            task._detect([make_target(kind, label, depth)], (600, 800, 3))
                    solver.assert_called_once_with(
                        float(depth), id1_offset_ticks=platform.PLATFORM_GRASP_ID1_OFFSET_TICKS,
                        target_kind=kind)
                    self.assertEqual(task.stage, "platform_actions")
                    self.assertEqual(task.finish_reason, "PICKED_" + kind.upper())

    def test_center_limit_skips_not_descends(self):
        _, _, task, _ = self.prepared_task(4)
        task.center_id2 = platform.CENTER_ID2_RANGE[0]
        task.center_id6 = platform.CENTER_ID6_RANGE[0]
        for _ in range(3):
            task._detect([make_target(center=(700, 500))], (600, 800, 3))
        self.assertEqual(task.finish_reason, "SKIPPED:CENTER_LIMIT")
        self.assertNotIn("DESCEND", [item[0] for item in task.actions])

    def test_first_pair_after_warmup_uses_formal_preselection(self):
        detector = mock.Mock()
        detector.detect.return_value = [make_target(label="C", center=(200, 300)),
                                        make_target(label="D", center=(600, 300))]
        pair = app.lock_pair(Reader(), detector, platform, headless=True)
        self.assertEqual(pair, ("C", "D"))
        self.assertEqual(detector.detect.call_count, 2)

    def test_duplicate_frame_not_fresh(self):
        reader = Reader(fresh=False)
        frame = app.next_frame(reader, None)
        self.assertIsNone(app.next_frame(reader, frame))

    def test_slot_runner_uses_formal_mode_and_discards(self):
        _, bridge, task, clock = self.prepared_task()
        detector = mock.Mock()
        detector.detect.return_value = [make_target(depth=15)]
        result = app.run_slot(task, Reader(clock=clock), detector, vision, "red", 4,
                              headless=True)
        self.assertEqual(result, "PICKED_LETTER")
        self.assertTrue(task.high_ready)
        self.assertEqual(task.letter_success_counts["D"], 1)
        self.assertGreaterEqual(detector.detect.call_count, 11)
        self.assertEqual(detector.detect.call_args.kwargs["mode"],
                         vision.DETECTION_MODE_PLATFORM_TARGETS)
        self.assertEqual(detector.detect.call_args.kwargs["target_letters"], ("C", "D"))
        self.assertTrue(bridge.commands)

    def test_slot_empty_camera_still_times_out(self):
        _, _, task, clock = self.prepared_task()
        reader = Reader(clock=clock)
        reader.ok = False
        result = app.run_slot(task, reader, mock.Mock(), vision, "red", 1, headless=True)
        self.assertEqual(result, "SKIPPED:MAIN_TARGET_OR_DEPTH_TIMEOUT")
        self.assertTrue(task.high_ready)

    def test_camera_only_never_constructs_servo_or_h7(self):
        args = app.build_parser().parse_args(["--field", "red", "--headless"])
        reader = Reader()
        with mock.patch.object(app, "open_reader", return_value=reader), \
             mock.patch.object(app, "lock_pair", return_value=("C", "D")), \
             mock.patch.object(vision, "HiwonderSingleBusServoBridge") as bridge, \
             mock.patch.object(app, "H7Link") as h7:
            self.assertEqual(app.run(args), 0)
        bridge.assert_not_called()
        h7.assert_not_called()
        self.assertTrue(reader.closed)

    def test_busy_main_camera_prevents_all_motion(self):
        args = app.build_parser().parse_args(["--field", "red", "--headless", "--execute-h7"])
        with mock.patch.object(app, "open_reader", side_effect=[Reader(), RuntimeError("CAMERA_BUSY")]), \
             mock.patch.object(app, "lock_pair", return_value=("C", "D")), \
             mock.patch.object(vision, "HiwonderSingleBusServoBridge") as bridge, \
             mock.patch.object(app, "H7Link") as h7:
            self.assertEqual(app.run(args), 1)
        bridge.assert_not_called()
        h7.assert_not_called()

    def test_camera_ownership_check_fail_closed(self):
        with mock.patch.object(app.subprocess, "run", side_effect=[
                types.SimpleNamespace(returncode=3), types.SimpleNamespace(returncode=2)]):
            with self.assertRaisesRegex(RuntimeError, "OWNERSHIP_CHECK_FAILED"):
                app.preflight_camera("/fake-camera")

    def service(self, reader=None):
        sent = []
        service = app.Task2WhiteLineService(vision, reader or Reader(), sent.append)
        return service, sent

    def query(self, service, seq=1, phase="TASK2_AFTER_SECONDARY_SHIFT"):
        service.enqueue(f"VISION,WHITE_LINE,QUERY,SEQ,{seq},PHASE,{phase}")

    def test_white_line_uses_task2_normalizes_and_transmits_rx(self):
        service, sent = self.service(Reader(np.zeros((480, 640, 3), np.uint8)))
        detector = mock.Mock()
        detector.detect_task2.return_value = {"y_at_center": 300.1, "angle_deg": 1.25,
            "frame_width": 800, "frame_height": 600, "right_edge_x": 612}
        service.detector = detector
        self.query(service)
        service.process()
        self.assertEqual(detector.detect_task2.call_args.args[0].shape, (600, 800, 3))
        detector.detect.assert_not_called()
        self.assertIn("Y10,3001,A100,125,W,800,H,600,RX,612", sent[-1])

    def test_white_line_rejects_missing_phase_and_wrong_task(self):
        service, sent = self.service()
        service.enqueue("VISION,WHITE_LINE,QUERY,SEQ,1")
        self.query(service, phase="TASK1_AFTER_ARC")
        service.process()
        self.assertEqual(len(sent), 2)
        self.assertTrue(all("ERR,REASON,INVALID_PHASE" in line for line in sent))

    def test_white_line_retains_newest_query_and_resets_new_seq(self):
        service, sent = self.service()
        service.detector = mock.Mock()
        service.detector.detect_task2.return_value = None
        self.query(service, 1)
        self.query(service, 2)
        service.process()
        self.assertEqual(sent, ["RK,VISION,WHITE_LINE,NOT_FOUND,SEQ,2"])
        self.query(service, 2)
        service.process()
        self.assertEqual(service.detector.reset_tracking.call_count, 1)
        self.query(service, 3)
        service.process()
        self.assertEqual(service.detector.reset_tracking.call_count, 2)

    def test_white_line_no_extra_time_based_cache(self):
        reader = Reader()
        reader.frame[:] = (30, 90, 40)
        reader.frame[295:316, 32:641] = 255
        service, sent = self.service(reader)
        self.query(service)
        service.process()
        self.assertIn(",FOUND,", sent[-1])
        reader.frame[:] = 0
        for index in range(5):
            self.query(service)
            service.process()
            self.assertIn(",FOUND," if index < 4 else ",NOT_FOUND,", sent[-1])

    def test_white_line_missing_camera_does_not_reuse_old_result(self):
        reader = Reader()
        reader.ok = False
        service, sent = self.service(reader)
        self.query(service)
        service.process()
        self.assertEqual(sent[-1], "RK,VISION,WHITE_LINE,NOT_FOUND,SEQ,1")

    def test_new_white_line_sequence_does_not_hold_previous_strip(self):
        reader = Reader()
        reader.frame[:] = (30, 90, 40)
        reader.frame[295:316, 32:641] = 255
        service, sent = self.service(reader)
        self.query(service, 1)
        service.process()
        self.assertIn(",FOUND,", sent[-1])
        reader.frame[:] = 0
        self.query(service, 2)
        service.process()
        self.assertEqual(sent[-1], "RK,VISION,WHITE_LINE,NOT_FOUND,SEQ,2")

    def test_h7_batch_preserves_statuses_and_services_white_queries_during_ack(self):
        service = mock.Mock()
        h7 = app.H7Link("/unused", service)
        h7.fd = 999
        payload = (b"VISION,WHITE_LINE,QUERY,SEQ,1,PHASE,TASK2_AFTER_SECONDARY_SHIFT\r\n"
                   b"H7,TEST,TASK2,ACK,SEQ,10,FIELD,RED\r\n"
                   b"H7,TEST,TASK2,RUNNING,SEQ,10,FIELD,RED\r\n"
                   b"H7,TEST,TASK2,DONE,SEQ,10,FIELD,RED\r\n")
        with mock.patch.object(app.select, "select", return_value=([999], [], [])), \
             mock.patch.object(app.os, "read", return_value=payload) as read:
            for state in ("ACK", "RUNNING", "DONE"):
                self.assertIsNotNone(h7.wait_status(10, state, 1, "RED"))
        self.assertEqual(read.call_count, 1)
        service.enqueue.assert_called_once()
        service.process.assert_called_once()

    def test_status_rejects_wrong_sequence_and_field(self):
        fields = "H7,TEST,TASK2,DONE,SEQ,10,FIELD,BLUE".split(",")
        self.assertFalse(app.H7Link._status_matches(fields, 10, {"DONE"}, "RED"))
        self.assertFalse(app.H7Link._status_matches(fields, 11, {"DONE"}, "BLUE"))
        self.assertTrue(app.H7Link._status_matches(fields, 10, {"DONE"}, "BLUE"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
