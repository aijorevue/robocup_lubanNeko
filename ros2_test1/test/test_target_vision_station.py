import unittest
from unittest import mock

import numpy as np
import cv2

from ros2_test1 import target_vision


class FakeArmPreview:
    def __init__(self):
        self.id6 = target_vision.BASE_YAW_CENTER_TICK
        self.targets = None
        self.messages = []

    def set_targets(self, id1, id2, id4, id6):
        self.targets = (id1, id2, id4, id6)

    def publish(self, message, *_args):
        self.messages.append(message)


class FakeServoBridge:
    def __init__(self):
        self.enabled = True
        self.write_enabled = True
        self.assumed_feedback = True
        self.arm_time_ms = 1
        self.gripper_time_ms = 1
        self.aux_time_ms = 1
        self.splitter_time_ms = 1
        self.last_command_ok = True
        self.status = "fake ready"
        self.fail_writes = False
        self.ports_open = True
        self.sent = []

    def send_targets(self, **targets):
        self.sent.append(targets)
        self.last_command_ok = not self.fail_writes
        self.status = "fake write failed" if self.fail_writes else "fake write ok"
        return self.status

    def ready_for_commands(self):
        return (
            self.enabled
            and self.write_enabled
            and self.ports_open
            and self.last_command_ok
        )


def make_controller(bridge=None, field_mode=target_vision.FieldMode.RED):
    bridge = bridge or FakeServoBridge()
    preview = FakeArmPreview()
    controller = target_vision.TargetGraspController(
        True,
        bridge,
        preview,
        550,
        300,
        370,
        520,
        35,
        3,
        0.2,
        0.0,
        0.1,
        0.08,
        30,
        0.0,
        0.0,
        50.0,
        0.0,
        5.0,
        24,
        target_vision.ID1_SAFE_LIMITS,
        target_vision.ID2_SAFE_LIMITS,
        20.0,
        startup_sequence=False,
        one_shot=False,
        camera_gripper_vertical_offset_mm=0.0,
        max_lateral_offset_mm=300.0,
        max_one_shot_ik_error_mm=30.0,
        post_center_retreat_mm=52.0,
        post_center_down_mm=150.0,
        post_center_ik_error_mm=30.0,
        field_mode=field_mode,
    )
    controller.startup_stage = "complete"
    controller.algorithm_stage = "centering"
    return controller, bridge, preview


class ChassisStationSafetyTests(unittest.TestCase):
    def test_blue_disc_detector_rejects_dark_cyan_ball(self):
        detector = target_vision.TargetDetector()
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Cyan-blue motion is deliberately outside the tightened blue mask.
        cv2.circle(frame, (400, 300), 42, (150, 95, 35), -1)
        detections = detector.detect(
            frame,
            mode=target_vision.DETECTION_MODE_DISC_BALLS,
            field_name="blue",
        )
        balls = [d for d in detections if d.get("kind") == "ball"]
        self.assertFalse(any(d.get("color") == "blue" for d in balls))

    def test_mixed_bridge_routes_85kg_targets_to_htd85_binary(self):
        bridge = target_vision.HiwonderSingleBusServoBridge(
            "/dev/missing", 115200, enabled=False, write_enabled=False
        )
        bridge.enabled = True
        bridge.write_enabled = True
        bridge.arm_fd = 10
        writes = []
        bridge._write_payload = lambda payload, repeat=None: writes.append((bridge.arm_fd, payload))
        bridge.send_targets(id1=650, id2=500, id6=420)

        self.assertTrue(bridge.last_command_ok)
        self.assertEqual(
            [
                (10, target_vision.htd85_move_packet(1, 650, bridge.arm_time_ms)),
                (10, target_vision.htd85_move_packet(2, 500, bridge.arm_time_ms)),
                (10, target_vision.htd85_move_packet(6, 420, bridge.arm_time_ms)),
            ],
            writes,
        )

    def test_disc_prep_high_keeps_id6_at_420_and_is_idempotent(self):
        controller, bridge, _ = make_controller()
        prep = {"task": "DISC_CATCH", "id1": 650, "id2": 600}
        with mock.patch.object(target_vision.time, "sleep") as sleep_mock:
            controller.prepare_chassis_station_high(prep)
            first_command_count = len(bridge.sent)
            controller.prepare_chassis_station_high(prep)

        self.assertEqual(controller.id6, 420)
        self.assertEqual(bridge.sent[-2]["id1"], 650)
        self.assertEqual(bridge.sent[-2]["id6"], 420)
        self.assertNotIn("id2", bridge.sent[-2])
        self.assertEqual(bridge.sent[-1], {"id2": 600})
        sleep_mock.assert_called_once_with(target_vision.ARM_JOINT_SEQUENCE_DELAY_S)
        self.assertEqual(len(bridge.sent), first_command_count)

    def test_chassis_ready_requires_writable_servo_link(self):
        controller, bridge, _ = make_controller()
        self.assertTrue(controller.ready_for_chassis_link(True))
        bridge.write_enabled = False
        self.assertFalse(controller.ready_for_chassis_link(True))
        self.assertTrue(controller.ready_for_chassis_link(False))

    def test_chassis_ready_rejects_stale_command_success_after_port_loss(self):
        controller, bridge, _ = make_controller()
        bridge.last_command_ok = True
        bridge.ports_open = False

        self.assertFalse(controller.ready_for_chassis_link(True))

    def test_direct_bridge_missing_ports_clears_stale_success(self):
        bridge = target_vision.HiwonderSingleBusServoBridge(
            "/dev/missing", 115200, enabled=False, write_enabled=False
        )
        bridge.enabled = True
        bridge.write_enabled = True
        bridge.last_command_ok = True

        bridge.send_targets(id1=550)

        self.assertFalse(bridge.last_command_ok)
        self.assertFalse(bridge.ready_for_commands())

    def test_locked_grasp_advances_without_fresh_camera_frame(self):
        controller, bridge, _ = make_controller()
        controller.locked_target = {"kind": "letter", "letter": "A", "color": "white"}
        controller.algorithm_stage = "close"
        controller.id4 = controller.id4_open
        controller.update(None, (600, 800, 3), detection_fresh=False)

        self.assertEqual(bridge.sent[-1]["id4"], controller.id4_closed)
        self.assertEqual(controller.algorithm_stage, "close_wait")

    def test_disc_no_frame_timeout_retracts_and_completes(self):
        controller, bridge, _ = make_controller()
        clock = [100.0]
        with mock.patch.object(target_vision.time, "monotonic", lambda: clock[0]), mock.patch.object(
            target_vision.time, "sleep"
        ):
            controller.begin_chassis_station("DISC_CATCH")
            clock[0] = 100.2
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=False
            )
            clock[0] = 100.5
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=False
            )
            clock[0] = 100.8
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=False
            )
            clock[0] = 111.0
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=False
            )

        self.assertIsNone(controller.active_chassis_station)
        self.assertEqual(
            controller.consume_chassis_station_done(),
            "NO_RED_OR_YELLOW_BALL_5.0S",
        )
        high_first = next(
            item for item in bridge.sent
            if item.get("id1") == target_vision.DISC_CATCH_PREP_ID1_TICK
            and item.get("id6") == target_vision.DISC_CATCH_ID6_TICK
            and "id2" not in item
        )
        high_second_index = next(
            index for index, item in enumerate(bridge.sent)
            if item == {"id2": target_vision.DISC_CATCH_PREP_ID2_TICK}
        )
        home_id2_index = next(
            index for index, item in enumerate(bridge.sent)
            if item.get("id2") == target_vision.HOME_ID2_TICK
        )
        self.assertEqual(high_first["id1"], 650)
        self.assertEqual(high_first["id6"], 420)
        self.assertLess(high_second_index, home_id2_index)
        self.assertEqual(bridge.sent[-1]["id1"], target_vision.HOME_ID1_TICK)
        self.assertNotIn("id2", bridge.sent[-1])
        self.assertEqual(
            bridge.sent[-1]["id5"],
            target_vision.CATCHER_HOME_TICK,
        )


    def test_disc_red_field_rejects_blue_ball(self):
        controller, _bridge, _ = make_controller(field_mode=target_vision.FieldMode.RED)

        self.assertIsNone(
            controller._disc_catch_ball_visible([{"kind": "ball", "color": "blue", "center": (320, 240), "area_percent": 1.0}])
        )

    def test_disc_blue_field_rejects_red_ball_and_accepts_blue_first_frame(self):
        controller, _bridge, _ = make_controller(field_mode=target_vision.FieldMode.BLUE)
        red_ball = {"kind": "ball", "color": "red", "center": (320, 240), "area_percent": 1.0}
        blue_ball = {"kind": "ball", "color": "blue", "center": (322, 241), "area_percent": 1.0}

        self.assertIsNone(controller._disc_catch_ball_visible([red_ball]))
        self.assertIs(controller._disc_catch_ball_visible([blue_ball]), blue_ball)

    def test_disc_blue_field_does_not_filter_by_previous_target_position(self):
        controller, _bridge, _ = make_controller(field_mode=target_vision.FieldMode.BLUE)
        blue_ball = {
            "kind": "ball",
            "color": "blue",
            "center": (320, 240),
            "area_percent": 1.0,
        }
        controller.disc_pulse_done = True
        controller.disc_last_pulsed_color = "blue"
        controller.disc_last_pulsed_center = (320.0, 240.0)

        self.assertIs(controller._disc_catch_ball_visible([blue_ball]), blue_ball)
        moved_ball = dict(blue_ball, center=(410, 240))
        self.assertIs(controller._disc_catch_ball_visible([moved_ball]), moved_ball)

    def test_disc_blue_field_waits_for_target_clear_after_cooldown(self):
        controller, bridge, _ = make_controller(field_mode=target_vision.FieldMode.BLUE)
        controller.disc_pulse_done = True
        controller.disc_last_pulsed_color = "blue"
        controller.disc_last_pulsed_center = (320.0, 240.0)
        blue_ball = {
            "kind": "ball",
            "color": "blue",
            "center": (320, 240),
            "area_percent": 1.0,
        }
        controller.active_chassis_station = "DISC_CATCH"
        controller.chassis_station_stage = "disc_close_wait"
        controller.chassis_station_deadline = 100.0
        controller.chassis_station_no_target_deadline = 100000000000.0
        controller.disc_blue_channel_hold_deadline = 100.5

        clock = [100.0]
        with mock.patch.object(target_vision.time, "monotonic", lambda: clock[0]):
            result = controller.update_chassis_station(
                "DISC_CATCH", [blue_ball], (600, 800, 3), detection_fresh=True
            )
            self.assertIn("blue-field ID17 closed", result)
            self.assertEqual(controller.chassis_station_stage, "disc_blue_channel_wait")
            self.assertEqual(controller.chassis_station_deadline, 100.5)

            sent_before_cooldown = len(bridge.sent)
            clock[0] = 100.49
            controller.update_chassis_station(
                "DISC_CATCH", [blue_ball], (600, 800, 3), detection_fresh=True
            )
            self.assertEqual(len(bridge.sent), sent_before_cooldown)
            self.assertTrue(controller.disc_pulse_done)

            clock[0] = 100.5
            result = controller.update_chassis_station(
                "DISC_CATCH", [blue_ball], (600, 800, 3), detection_fresh=True
            )
            self.assertEqual(len(bridge.sent), sent_before_cooldown)
            self.assertEqual(controller.chassis_station_stage, "disc_detect")
            self.assertIn("waiting for blue target to clear", result)
            self.assertTrue(controller.disc_pulse_done)
            self.assertEqual(controller.disc_last_pulsed_color, "blue")

            clock[0] = 100.51
            result = controller.update_chassis_station(
                "DISC_CATCH", [blue_ball], (600, 800, 3), detection_fresh=True
            )
            self.assertIn("already pulsed", result)
            self.assertEqual(len(bridge.sent), sent_before_cooldown)

            for _ in range(3):
                controller.update_chassis_station(
                    "DISC_CATCH", [], (600, 800, 3), detection_fresh=True
                )
            controller.update_chassis_station(
                "DISC_CATCH", [blue_ball], (600, 800, 3), detection_fresh=True
            )

        self.assertEqual(controller.chassis_station_stage, "disc_open_wait")
        self.assertEqual(bridge.sent[-1]["id4"], controller.id7_open)

    def test_disc_blue_field_yellow_holds_channel_then_returns_to_blue(self):
        controller, bridge, _ = make_controller(field_mode=target_vision.FieldMode.BLUE)
        controller.active_chassis_station = "DISC_CATCH"
        controller.chassis_station_stage = "disc_close_wait"
        controller.chassis_station_deadline = 200.0
        controller.chassis_station_no_target_deadline = 100000000000.0
        controller.disc_pulse_done = True
        controller.disc_last_pulsed_color = "yellow"
        controller.disc_last_pulsed_center = (320.0, 240.0)
        controller.disc_blue_channel_hold_deadline = 200.2

        clock = [200.0]
        with mock.patch.object(target_vision.time, "monotonic", lambda: clock[0]):
            result = controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=True
            )
            self.assertEqual(controller.chassis_station_deadline, 200.2)

            sent_before_switch = len(bridge.sent)
            clock[0] = 200.19
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=True
            )
            self.assertEqual(len(bridge.sent), sent_before_switch)
            clock[0] = 200.2
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=True
            )

        self.assertEqual(len(bridge.sent), sent_before_switch + 1)
        self.assertEqual(bridge.sent[-1]["splitter_id4"], target_vision.DISC_CATCH_SPLITTER_FIELD_TICK)
        self.assertEqual(controller.chassis_station_stage, "disc_detect")
        self.assertFalse(controller.disc_pulse_done)
        self.assertIsNone(controller.disc_last_pulsed_color)

    def test_disc_close_completion_immediately_rearms_same_color(self):
        controller, bridge, _ = make_controller(field_mode=target_vision.FieldMode.RED)
        red_ball = {
            "kind": "ball",
            "color": "red",
            "center": (320, 240),
            "area_percent": 1.0,
        }
        controller.active_chassis_station = "DISC_CATCH"
        controller.chassis_station_stage = "disc_close_wait"
        controller.chassis_station_deadline = 100.0
        controller.chassis_station_no_target_deadline = 110.0
        controller.disc_pulse_done = True
        controller.disc_last_pulsed_color = "red"

        clock = [100.0]
        with mock.patch.object(target_vision.time, "monotonic", lambda: clock[0]):
            result = controller.update_chassis_station(
                "DISC_CATCH", [red_ball], (600, 800, 3), detection_fresh=True
            )
            self.assertIn("ready for next target frame", result)
            self.assertFalse(controller.disc_pulse_done)
            self.assertIsNone(controller.disc_last_pulsed_color)
            self.assertEqual(controller.chassis_station_stage, "disc_detect")

            clock[0] = 100.01
            controller.update_chassis_station(
                "DISC_CATCH", [red_ball], (600, 800, 3), detection_fresh=True
            )

        self.assertEqual(controller.chassis_station_stage, "disc_open_wait")
        self.assertEqual(
            bridge.sent[-1]["splitter_id4"],
            target_vision.DISC_CATCH_SPLITTER_FIELD_TICK,
        )
        self.assertEqual(bridge.sent[-1]["id4"], controller.id7_open)

    def test_disc_station_starts_with_app_low_pose(self):
        controller, bridge, _ = make_controller()
        controller.id1 = target_vision.DISC_CATCH_PREP_ID1_TICK
        controller.id2 = target_vision.DISC_CATCH_PREP_ID2_TICK
        controller.id6 = target_vision.DISC_CATCH_ID6_TICK
        with mock.patch.object(target_vision.time, "monotonic", return_value=100.0), mock.patch.object(
            target_vision.time, "sleep"
        ) as sleep_mock:
            controller.begin_chassis_station("DISC_CATCH")

        self.assertEqual(controller.chassis_station_stage, "disc_app_low_settle")
        self.assertEqual(controller.id1, target_vision.DISC_CATCH_READY_ID1_TICK)
        self.assertEqual(controller.id2, target_vision.DISC_CATCH_READY_ID2_TICK)
        self.assertEqual(controller.id6, target_vision.DISC_CATCH_ID6_TICK)
        self.assertEqual(controller.id5, target_vision.DISC_CATCH_CATCHER_READY_TICK)
        self.assertEqual(controller.splitter_id4, target_vision.DISC_CATCH_SPLITTER_READY_TICK)
        self.assertEqual(bridge.sent[-2]["id2"], 550)
        self.assertNotIn("id1", bridge.sent[-2])
        self.assertEqual(bridge.sent[-2]["id4"], 370)
        self.assertEqual(
            bridge.sent[-2]["id5"],
            target_vision.DISC_CATCH_CATCHER_READY_TICK,
        )
        self.assertEqual(bridge.sent[-2]["splitter_id4"], target_vision.SPLITTER_RETRACT_TICK)
        self.assertEqual(bridge.sent[-1]["id1"], 550)
        self.assertEqual(bridge.sent[-1]["id6"], 420)
        self.assertNotIn("id2", bridge.sent[-1])
        sleep_mock.assert_called_once_with(target_vision.ARM_JOINT_SEQUENCE_DELAY_S)

    def test_control_fault_attempts_home_before_reporting_error(self):
        controller, bridge, _ = make_controller()
        with mock.patch.object(target_vision.time, "sleep"):
            controller.begin_chassis_station("PLATFORM_PICK")
            result = controller.abort_chassis_station("SERVO_OR_CONTROL_FAULT")

        self.assertIn("shutdown contracted", result)
        self.assertIsNone(controller.active_chassis_station)
        self.assertEqual(
            controller.consume_chassis_station_error(),
            "SERVO_OR_CONTROL_FAULT",
        )
        self.assertEqual(controller.algorithm_stage, "centering")
        self.assertEqual(bridge.sent[-1]["id6"], target_vision.BASE_YAW_CENTER_TICK)

    def test_failed_home_is_reported_as_station_error(self):
        controller, bridge, _ = make_controller()
        controller.begin_chassis_station("PLATFORM_PICK")
        bridge.fail_writes = True
        with mock.patch.object(target_vision.time, "sleep"):
            controller.abort_chassis_station("SERVO_OR_CONTROL_FAULT")

        self.assertEqual(
            controller.consume_chassis_station_error(),
            "SERVO_OR_CONTROL_FAULT_HOME_FAILED",
        )
        self.assertEqual(controller.algorithm_stage, "fault")

    def test_splitter_write_failure_enters_fault_for_station_abort(self):
        controller, bridge, _ = make_controller()
        controller.begin_chassis_station("DISC_CATCH")
        bridge.fail_writes = True
        controller._send_splitter_id4(1600, "test splitter")

        self.assertEqual(controller.algorithm_stage, "fault")
        self.assertIn("automatic motion stopped", controller.status)

    def test_column_ready_uses_standalone_task_three_pose(self):
        controller, bridge, _ = make_controller()
        with mock.patch.object(target_vision.time, "sleep"):
            controller.begin_chassis_station("COLUMN_CATCH")

        self.assertEqual(
            (controller.id1, controller.id2, controller.id6),
            (650, 500, 420),
        )
        self.assertEqual(controller.id5, target_vision.TASK1_ID15_RETRACT_TICK)
        self.assertEqual(controller.splitter_id4, target_vision.SPLITTER_RETRACT_TICK)
        self.assertEqual(controller.id7, 370)
        self.assertEqual(bridge.sent[-2], {
            "id1": 650,
            "id6": 420,
            "id4": 370,
            "id5": target_vision.TASK1_ID15_RETRACT_TICK,
            "splitter_id4": target_vision.SPLITTER_RETRACT_TICK,
        })
        self.assertEqual(bridge.sent[-1], {"id2": 500})

    def test_column_centering_uses_seven_and_five_tick_steps(self):
        controller, bridge, _ = make_controller()
        controller.active_chassis_station = "COLUMN_CATCH"
        controller.chassis_station_stage = "column_centering"
        controller.id1, controller.id2, controller.id6 = 650, 500, 420
        result, _message = controller._visual_center_step(
            {"center": (500, 400)},
            (600, 800, 3),
            now=1.0,
            can_preview_step=True,
            label="COLUMN_CATCH center",
        )

        self.assertFalse(result)
        self.assertEqual((controller.id2, controller.id6), (493, 415))
        self.assertEqual(bridge.sent[-1], {"id2": 493, "id6": 415})
        self.assertEqual(bridge.arm_time_ms, 1)

    def test_column_stop_retracts_before_done(self):
        controller, bridge, _ = make_controller()
        with mock.patch.object(target_vision.time, "sleep"):
            controller.begin_chassis_station("COLUMN_CATCH")
            controller.stop_chassis_station("COLUMN_CATCH")

        self.assertEqual(
            controller.consume_chassis_station_done(),
            "STOPPED_BY_CHASSIS",
        )
        self.assertIsNone(controller.active_chassis_station)
        self.assertEqual(bridge.sent[-1]["id4"], 370)
        self.assertEqual(bridge.sent[-1]["splitter_id4"], target_vision.SPLITTER_RETRACT_TICK)


def test_platform_letter_policy_matches_selected_letters():
    controller, _bridge, _ = make_controller()
    policy = controller.target_policy
    letter = {"kind": "letter", "letter": "B", "color": "white"}
    ring = {"kind": "ring", "color": policy.platform_ring_color}
    assert policy.matches_platform_target(letter, "letter", {"A", "B"})
    assert not policy.matches_platform_target(letter, "letter", {"A", "C"})
    assert policy.matches_platform_target(ring, "ring", {"A", "C"})

if __name__ == "__main__":
    unittest.main()
