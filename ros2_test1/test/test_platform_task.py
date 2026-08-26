"""Hardware-free task-two regressions: timed writes and real CDC parser."""
import inspect
import unittest
from unittest.mock import Mock, patch

from ros2_test1.platform_task import (
    PlatformTask, HIGH, LETTER_PLACE, RING_PLACE,
    POST_OPEN_ID2_RETREAT_TICKS_BY_KIND, PLATFORM_GRIPPER_CLOSED,
    PLATFORM_GRIPPER_OPEN,
    TARGET_WINDOW_SIZE_PX, TARGET_WINDOW_MIN_AREA_FRACTION,
    PLATFORM_LETTER_PLACE_TIME_MS,
    _target_in_center_window,
)
from ros2_test1.chassis_link import ChassisArmLink
from ros2_test1.target_vision import TargetGraspController
from ros2_test1.target_vision import TargetDetector, detection_process_worker
import cv2
import numpy as np


def letter(value='A', center=(400, 300), depth=20):
    return dict(kind='letter', letter=value, center=center,
                bbox=(center[0] - 20, center[1] - 20, 40, 40),
                confidence=90, distance_cm=depth, observed=True)


class Fixture:
    def __init__(self):
        self.now = 0.0
        self.writes = []
        self.task = PlatformTask(
            self.pose,
            self.gripper,
            self.center,
            ring_place_id6=self.ring_place_id6,
            ring_place_id2=self.ring_place_id2,
            ring_place_id1=self.ring_place_id1,
            ring_return_high_id1=self.ring_return_high_id1,
            ring_return_high_id2=self.ring_return_high_id2,
            ring_return_high_id6=self.ring_return_high_id6,
            letter_place_id6=self.letter_place_id6,
            letter_place_id2=self.letter_place_id2,
            letter_place_id1=self.letter_place_id1,
            letter_return_high_id1=self.letter_return_high_id1,
            letter_return_high_id2=self.letter_return_high_id2,
            letter_return_high_id6=self.letter_return_high_id6,
            clock=lambda: self.now,
        )

    def pose(self, value, raising):
        self.writes.append(('pose', value, raising))
        return 0.72

    def gripper(self, value):
        self.writes.append(('gripper', value))
        return 0.35

    def center(self, id2, id6):
        self.writes.append(('center', id2, id6))
        return 0.12

    def ring_place_id6(self, id6):
        self.writes.append(('ring_id6', id6))
        return 0.5

    def ring_place_id2(self, id2):
        self.writes.append(('ring_id2', id2))
        return 0.5

    def ring_place_id1(self, id1):
        self.writes.append(('ring_id1', id1))
        return 0.7

    def ring_return_high_id1(self, id1):
        self.writes.append(('ring_high_id1', id1))
        return 0.6

    def ring_return_high_id2(self, id2):
        self.writes.append(('ring_high_id2', id2))
        return 0.6

    def ring_return_high_id6(self, id6):
        self.writes.append(('ring_high_id6', id6))
        return 0.6

    def letter_place_id6(self, id6):
        self.writes.append(('letter_id6', id6))
        return 0.5

    def letter_place_id2(self, id2):
        self.writes.append(('letter_id2', id2))
        return 0.5

    def letter_place_id1(self, id1):
        self.writes.append(('letter_id1', id1))
        return 0.5

    def letter_return_high_id1(self, id1):
        self.writes.append(('letter_high_id1', id1))
        return 0.6

    def letter_return_high_id2(self, id2):
        self.writes.append(('letter_high_id2', id2))
        return 0.6

    def letter_return_high_id6(self, id6):
        self.writes.append(('letter_high_id6', id6))
        return 0.6

    def ready(self):
        self.task.begin_preselect()
        for _ in range(28):
            self.task.preselect([letter('A', (100, 200)), letter('C', (600, 200))])
        assert self.task.stage == 'platform_raise'
        self.now = self.task.deadline
        self.task.tick()
        self.task.done = None

    def feed(self, target, count=11):
        for _ in range(count):
            self.task.tick([target], (600, 800, 3), fresh=True)

    def finish(self):
        for _ in range(20):
            if self.task.stage != 'platform_actions':
                break
            self.now = max(self.now, self.task.deadline)
            self.task.tick()


class TestPlatformTask(unittest.TestCase):
    def test_formal_offset_preserves_other_calibration_callers(self):
        from ros2_test1.grasp_calibration import calibrated_grasp_ticks

        self.assertEqual(calibrated_grasp_ticks(20), (488, 483))
        self.assertEqual(
            calibrated_grasp_ticks(20, id1_offset_ticks=40), (478, 483),
        )

    def test_retreat_floor_after_visual_centering(self):
        for kind in ('letter', 'ring'):
            with self.subTest(kind=kind):
                f = Fixture(); f.ready(); f.writes.clear()
                f.task.begin_slot('red')
                f.task.center_id2 = 460
                f.task.center_id6 = 365
                target = letter() if kind == 'letter' else dict(
                    letter(), kind='ring', color='red', score=.9,
                )
                f.feed(target); f.finish()
                self.assertEqual(f.writes[1], ('pose', (650, 450, 365), False))
                self.assertEqual(f.writes[2], ('pose', (478, 483, 365), False))
                self.assertEqual(f.task.done, 'PICKED_' + kind.upper())

    def test_center_window_requires_400px_and_80_percent_bbox_overlap(self):
        self.assertEqual(TARGET_WINDOW_SIZE_PX, 400)
        self.assertEqual(TARGET_WINDOW_MIN_AREA_FRACTION, 0.80)
        self.assertTrue(_target_in_center_window(
            letter(center=(400, 300)), (600, 800, 3)))
        self.assertFalse(_target_in_center_window(
            letter(center=(650, 300)), (600, 800, 3)))
        self.assertFalse(_target_in_center_window(
            dict(letter(), bbox=(580, 280, 100, 40)), (600, 800, 3)))

    def test_pair_first_and_high_settle_barrier(self):
        f = Fixture()
        f.task.begin_preselect()
        for _ in range(40):
            f.task.preselect([letter('A')])
        self.assertEqual(f.writes, [])
        for _ in range(8):
            f.task.preselect([letter('C')])
        self.assertEqual(f.writes, [])
        for _ in range(6):
            f.task.preselect([letter('A', (100, 200)), letter('C', (600, 200))])
        self.assertEqual(f.writes, [('pose', HIGH, True)])
        f.now = f.task.deadline - .001
        f.task.tick()
        self.assertIsNone(f.task.done)
        f.now += .002
        f.task.tick()
        self.assertEqual(f.task.done, 'PRESELECT_DONE:A:C')

    def test_stale_and_duplicate_letter_pair_never_lock(self):
        for stale in (True, False):
            f = Fixture()
            f.task.begin_preselect()
            for _ in range(60):
                f.task.preselect([letter(), letter('C' if stale else 'A', (600, 300))], fresh=not stale)
            self.assertFalse(f.writes)
            f.now = 25
            f.task.tick()
            self.assertEqual(f.task.error, 'SECONDARY_PAIR_TIMEOUT')

    def test_no_slot_without_preselect_and_high(self):
        f = Fixture()
        f.task.begin_slot('red')
        self.assertEqual(f.task.error, 'PRESELECT_HIGH_NOT_READY')
        self.assertFalse(f.writes)

    def test_platform_uses_the_secondary_pair_not_a_default_letter_set(self):
        f = Fixture()
        f.task.begin_preselect()
        for _ in range(28):
            f.task.preselect([letter('C', (100, 200)), letter('D', (600, 200))])
        self.assertEqual(f.task.selected, ('C', 'D'))
        f.now = f.task.deadline
        f.task.tick()
        f.task.done = None
        f.task.begin_slot('red')
        for _ in range(9):
            f.task.tick([letter('A')], (600, 800, 3), fresh=True)
        self.assertIsNone(f.task.target_key)
        self.assertFalse(any(write[0] == 'pose' and write[1] != HIGH
                             for write in f.writes))
        for _ in range(3):
            f.task.tick([letter('D')], (600, 800, 3), fresh=True)
        self.assertEqual(f.task.target_key, ('letter', 'D'))

    def test_letter_and_both_field_ring_exact_actions(self):
        self.assertEqual(HIGH, (650, 600, 420))
        self.assertEqual(LETTER_PLACE, (500, 350, 670))
        self.assertEqual(RING_PLACE, (520, 340, 185))
        self.assertEqual(POST_OPEN_ID2_RETREAT_TICKS_BY_KIND,
                         {'letter': 30, 'ring': 50})
        for kind, field, placement in [('letter', 'red', LETTER_PLACE),
                                       ('ring', 'red', RING_PLACE),
                                       ('ring', 'blue', RING_PLACE)]:
            with self.subTest(kind=kind, field=field):
                f = Fixture(); f.ready(); f.writes.clear()
                f.task.begin_slot(field)
                target = letter() if kind == 'letter' else dict(letter(), kind='ring', color=field, score=.9)
                f.feed(target)
                f.finish()
                retreat_ticks = POST_OPEN_ID2_RETREAT_TICKS_BY_KIND[kind]
                expected = [
                    ('gripper', PLATFORM_GRIPPER_OPEN),
                    ('pose', (650, 600 - retreat_ticks, 420), False),
                    ('pose', (478, 483, 420), False),
                    ('gripper', PLATFORM_GRIPPER_CLOSED), ('pose', HIGH, True),
                ]
                expected += (
                    [
                        ('ring_id6', 185), ('ring_id2', 340),
                        ('ring_id1', 520), ('gripper', PLATFORM_GRIPPER_OPEN),
                        ('gripper', PLATFORM_GRIPPER_CLOSED), ('ring_high_id1', 650),
                        ('ring_high_id2', 600), ('ring_high_id6', 420),
                    ]
                    if kind == 'ring'
                    else [
                        ('letter_id6', 670), ('letter_id2', 350),
                        ('letter_id1', 500), ('gripper', PLATFORM_GRIPPER_OPEN),
                        ('gripper', PLATFORM_GRIPPER_CLOSED), ('letter_high_id1', 650),
                        ('letter_high_id2', 600), ('letter_high_id6', 420),
                    ]
                )
                self.assertEqual(f.writes, expected)
                self.assertEqual(f.task.done, 'PICKED_' + kind.upper())
                self.assertTrue(f.task.high_ready)
                if kind == 'letter':
                    self.assertEqual(
                        [item for item in f.writes if item[0].startswith('letter_')],
                        [
                            ('letter_id6', 670), ('letter_id2', 350),
                            ('letter_id1', 500), ('letter_high_id1', 650),
                            ('letter_high_id2', 600), ('letter_high_id6', 420),
                        ],
                    )
                    self.assertEqual(PLATFORM_LETTER_PLACE_TIME_MS, 500)

    def test_near_grasp_adjustments_apply_only_from_7_to_10cm(self):
        cases = [
            ('letter', 7.0, 610, 520, 570),
            ('letter', 9.0, 590, 510, 570),
            ('letter', 10.0, 580, 430, 570),
            ('ring', 7.0, 610, 540, 520),
            ('ring', 9.0, 590, 530, 520),
        ]
        for kind, depth, expected_descent_id1, expected_descent_id2, expected_retreat_id2 in cases:
            with self.subTest(kind=kind, depth=depth):
                f = Fixture(); f.ready(); f.writes.clear()
                f.task.begin_slot('red')
                target = letter(depth=depth) if kind == 'letter' else dict(
                    letter(depth=depth), kind='ring', color='red', score=.9,
                )
                f.feed(target); f.finish()
                self.assertEqual(f.writes[1], (
                    'pose', (650, expected_retreat_id2, 420), False,
                ))
                self.assertEqual(f.writes[2], (
                    'pose', (expected_descent_id1, expected_descent_id2, 420), False,
                ))
                self.assertIn(('gripper', PLATFORM_GRIPPER_CLOSED), f.writes)

    def test_last_high_must_settle_before_done_and_no_retract_while_waiting(self):
        f = Fixture(); f.ready(); f.writes.clear()
        f.task.begin_slot('red'); f.feed(letter())
        while len(f.writes) < 13:
            f.now = max(f.now, f.task.deadline); f.task.tick()
        self.assertEqual(f.writes[-1], ('letter_high_id6', HIGH[2]))
        self.assertIsNone(f.task.done)
        f.now = f.task.deadline - .001; f.task.tick()
        self.assertIsNone(f.task.done)
        f.now += .002; f.task.tick()
        self.assertTrue(f.task.done)
        writes = list(f.writes)
        f.now += 100; f.task.tick()
        self.assertEqual(f.writes, writes)

    def test_unselected_letter_and_opponent_ring_skip_after_high(self):
        for field in ('red', 'blue'):
            for target in (letter('B'), dict(letter(), kind='ring', color='blue' if field == 'red' else 'red', score=.9)):
                f = Fixture(); f.ready(); f.writes.clear(); f.task.begin_slot(field)
                f.feed(target); f.finish()
                f.now = f.task.timeout
                f.task.tick()
                f.finish()
                self.assertEqual(f.writes, [('pose', HIGH, True)])
                self.assertEqual(f.task.done, 'SKIPPED:MAIN_TARGET_OR_DEPTH_TIMEOUT')

    def test_missing_invalid_and_nonfinite_depth_never_descend(self):
        for depth in (None, 'bad', 6.9, 31, float('nan'), float('inf')):
            with self.subTest(depth=depth):
                f = Fixture(); f.ready(); f.writes.clear(); f.task.begin_slot('red')
                f.feed(letter(depth=depth), 20)
                self.assertFalse(f.writes)
                f.now = f.task.timeout; f.task.tick(); f.finish()
                self.assertFalse(any(write[0] == 'pose' and write[1] != HIGH
                                     for write in f.writes))
                self.assertIsNone(f.task.done)

    def test_no_frame_times_out_without_grab(self):
        f = Fixture(); f.ready(); f.writes.clear(); f.task.begin_slot('red')
        for _ in range(20):
            f.task.tick([letter()], (600, 800, 3), fresh=False)
        self.assertFalse(f.writes)
        f.now = f.task.timeout; f.task.tick(); f.finish()
        self.assertEqual(f.writes, [('pose', HIGH, True)])

    def test_small_center_correction_and_then_calibrated_depth(self):
        f = Fixture(); f.ready(); f.writes.clear(); f.task.begin_slot('red')
        f.feed(letter(center=(470, 370)))
        self.assertEqual(f.writes, [('center', 593, 415)])
        f.now = f.task.deadline
        f.feed(letter(), 4); f.finish()
        self.assertIn(('pose', (478, 483, 415), False), f.writes)

    def test_write_failure_at_every_action_never_completes(self):
        for failure_index in range(13):
            f = Fixture(); f.ready(); f.writes.clear(); f.task.begin_slot('red')
            f.feed(letter())
            actions = list(f.task.actions)
            def fail(*args):
                raise OSError('injected serial fault')
            label, callback, args = actions[failure_index]
            actions[failure_index] = (label, fail, args)
            from collections import deque
            f.task.actions = deque(actions)
            f.finish()
            self.assertIsNone(f.task.done)
            self.assertEqual(f.task.stage, 'platform_fault')
            self.assertEqual(len(f.writes), failure_index)

    def test_reset_cancels_queued_motion(self):
        f = Fixture(); f.ready(); f.task.begin_slot('red'); f.feed(letter())
        f.task.tick(); f.task.reset(); old = list(f.writes)
        f.now += 100; f.task.tick()
        self.assertEqual(f.writes, old)
        self.assertFalse(f.task.selected)

    def test_second_station_reuses_selected_pair(self):
        f = Fixture(); f.ready()
        for value in ('A', 'C'):
            f.task.begin_slot('red'); f.feed(letter(value)); f.finish()
            self.assertEqual(f.task.done, 'PICKED_LETTER')
        self.assertEqual(f.task.selected, ('A', 'C'))


class TestControllerAndProtocol(unittest.TestCase):
    def make_controller(self):
        bridge = Mock(enabled=True, write_enabled=True, assumed_feedback=True,
                      last_command_ok=True, arm_time_ms=600, gripper_time_ms=210,
                      zp_time_ms=300, status='fake write ok')
        preview = Mock(id6=410)
        kwargs = {name: 1 for name, param in inspect.signature(TargetGraspController).parameters.items()
                  if param.default is inspect.Parameter.empty}
        kwargs.update(enabled=True, servo_bridge=bridge, arm_preview=preview,
                      id1_ready=446, id2_ready=227, id7_closed=370, id7_open=520,
                      id1_limits=(150, 710), id2_limits=(0, 769), angle_gap_degrees=20,
                      startup_sequence=False, one_shot=False)
        controller = TargetGraspController(**kwargs)
        controller.startup_stage = 'complete'
        return controller, bridge

    def test_real_controller_bridge_physical_ids_and_timing(self):
        c, bridge = self.make_controller()
        with patch('ros2_test1.target_vision.time.sleep'):
            c.begin_platform_preselect()
            self.assertFalse(bridge.send_targets.called)
            for _ in range(28):
                c.update_platform_preselect([letter('A', (100, 200)), letter('C', (600, 200))])
        self.assertIsNone(c.consume_chassis_station_done())
        self.assertEqual((c.id1, c.id2, c.id6, c.id5, c.splitter_id4), (650, 600, 420, 600, 300))
        c.platform_task.deadline = 0
        c.update_chassis_station('PLATFORM_PICK', [], (600, 800, 3), False)
        self.assertEqual(c.consume_chassis_station_done(), 'PRESELECT_DONE:A:C')
        before = bridge.send_targets.call_count
        c.update(None, (600, 800, 3))
        self.assertEqual(bridge.send_targets.call_count, before)
        c.begin_chassis_station('PLATFORM_PICK')
        for _ in range(11):
            c.update_chassis_station('PLATFORM_PICK', [letter()], (600, 800, 3))
        for _ in range(20):
            c.platform_task.deadline = 0
            c.update_chassis_station('PLATFORM_PICK', [], (600, 800, 3), False)
        self.assertEqual(c.consume_chassis_station_done(), 'PICKED_LETTER')
        self.assertEqual(
            (c.id1, c.id2, c.id6, c.id7),
            (650, 600, 420, PLATFORM_GRIPPER_CLOSED),
        )
        self.assertEqual(bridge.gripper_time_ms, 210)
        pulses = [call.kwargs['id4'] for call in bridge.send_targets.call_args_list
                  if set(call.kwargs) == {'id4'}]
        self.assertEqual(
            pulses,
            [PLATFORM_GRIPPER_OPEN, PLATFORM_GRIPPER_CLOSED,
             PLATFORM_GRIPPER_OPEN, PLATFORM_GRIPPER_CLOSED],
        )
        self.assertTrue(any(call.kwargs.get('id6') == 670 for call in bridge.send_targets.call_args_list))

    def test_real_parser_sequence_retries_do_not_restart_preselect_or_slot(self):
        link = ChassisArmLink(False, 'unused', 115200, 5)
        lines = []
        link.send_line = lambda line: lines.append(line) or True
        link.ready_to_run = True
        preselect = 'ARM,PLATFORM_PICK,PRESELECT,SEQ,41,FIELD,BLUE,COUNT,2'
        link._handle_line(preselect)
        link._handle_line(preselect)
        self.assertEqual(len(link.consume_preselects()), 1)
        self.assertFalse(any('PRESELECT_DONE' in line for line in lines))
        f = Fixture(); f.ready()
        link.finish_preselect(f.task.selected)
        link._handle_line(preselect)
        self.assertFalse(link.consume_preselects())
        self.assertIn('PRESELECT_DONE,SEQ,41', lines[-1])
        self.assertIn('FIELD,BLUE', lines[-1])
        start = 'ARM,PLATFORM_PICK,START,SEQ,42,FIELD,BLUE,SLOT,1'
        link._handle_line(start); link._handle_line(start)
        self.assertEqual(len(link.consume_platform_slots()), 1)
        f.task.begin_slot('blue'); f.feed(dict(letter(), kind='ring', color='blue', score=.9)); f.finish()
        link.finish_active(f.task.done)
        link._handle_line(start)
        self.assertFalse(link.consume_platform_slots())
        self.assertIn('DONE,SEQ,42', lines[-1])
        self.assertIn('SLOT,1', lines[-1])

    def test_real_ring_detector_output_reaches_grasp_for_both_fields(self):
        detector = TargetDetector()
        for field, color in [('red', (0, 0, 255)), ('blue', (255, 0, 0))]:
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.circle(frame, (400, 300), 70, color, -1)
            cv2.circle(frame, (400, 300), 35, (0, 0, 0), -1)
            detections, _ = detection_process_worker(frame, 1.0, platform=True)
            rings = [d for d in detections if d['kind'] == 'ring']
            self.assertEqual(len(rings), 1)
            self.assertEqual(rings[0]['color'], field)
            self.assertNotIn('confidence', rings[0])
            self.assertGreater(rings[0]['distance_cm'], 10)
            self.assertLess(rings[0]['distance_cm'], 30)
            detector.draw(frame, rings)  # New ring output must also render.
            f = Fixture(); f.ready(); f.task.begin_slot(field)
            f.feed(rings[0]); f.finish()
            self.assertEqual(f.task.done, 'PICKED_RING')


if __name__ == '__main__':
    unittest.main()
