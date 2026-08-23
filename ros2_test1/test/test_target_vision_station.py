import unittest
from unittest import mock

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
        self.zp_time_ms = 1
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
        1300,
        1710,
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
    def test_mixed_bridge_routes_85kg_targets_to_htd85_binary(self):
        bridge = target_vision.DirectBusServoBridge(
            "/dev/missing", 115200, enabled=False, write_enabled=False
        )
        bridge.enabled = True
        bridge.write_enabled = True
        bridge.arm_fd = 10
        bridge.zp_fd = 11
        writes = []
        bridge._write_payload = lambda fd, payload, repeat=None: writes.append((fd, payload))
        bridge.send_targets(id1=600, id2=500, id6=670)

        self.assertTrue(bridge.last_command_ok)
        self.assertEqual(
            [
                (10, target_vision.htd85_move_packet(1, 600, bridge.arm_time_ms)),
                (10, target_vision.htd85_move_packet(2, 500, bridge.arm_time_ms)),
                (10, target_vision.htd85_move_packet(6, 670, bridge.arm_time_ms)),
            ],
            writes,
        )

    def test_disc_prep_high_keeps_id6_at_670_and_is_idempotent(self):
        controller, bridge, _ = make_controller()
        prep = {"task": "DISC_CATCH", "id1": 600, "id2": 600}
        with mock.patch.object(target_vision.time, "sleep") as sleep_mock:
            controller.prepare_chassis_station_high(prep)
            first_command_count = len(bridge.sent)
            controller.prepare_chassis_station_high(prep)

        self.assertEqual(controller.id6, 670)
        self.assertEqual(bridge.sent[-2]["id1"], 600)
        self.assertEqual(bridge.sent[-2]["id6"], 670)
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
        bridge = target_vision.DirectBusServoBridge(
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
        self.assertEqual(bridge.sent[-2]["id2"], target_vision.HOME_ID2_TICK)
        self.assertEqual(bridge.sent[-1]["id1"], target_vision.HOME_ID1_TICK)
        self.assertNotIn("id2", bridge.sent[-1])
        self.assertEqual(
            bridge.sent[-1]["id5"],
            target_vision.DISC_CATCH_CATCHER_READY_TICK,
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
        self.assertEqual(bridge.sent[-2]["id4"], 1300)
        self.assertEqual(bridge.sent[-2]["id5"], 1110)
        self.assertEqual(bridge.sent[-2]["splitter_id4"], 1300)
        self.assertEqual(bridge.sent[-1]["id1"], 460)
        self.assertEqual(bridge.sent[-1]["id6"], 670)
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
        self.assertEqual(bridge.sent[-1]["id4"], 1300)
        self.assertEqual(bridge.sent[-1]["splitter_id4"], target_vision.SPLITTER_YELLOW_TICK)


def test_platform_letter_policy_matches_selected_letters(self):
    controller, _bridge, _ = make_controller()
    policy = controller.target_policy
    letter = {"kind": "letter", "letter": "B", "color": "white"}
    ring = {"kind": "ring", "color": policy.platform_ring_color}
    self.assertTrue(policy.matches_platform_target(letter, "letter", {"A", "B"}))
    self.assertFalse(policy.matches_platform_target(letter, "letter", {"A", "C"}))
    self.assertTrue(policy.matches_platform_target(ring, "ring", {"A", "C"}))

if __name__ == "__main__":
    unittest.main()
