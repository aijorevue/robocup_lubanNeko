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
        self.sent = []

    def send_targets(self, **targets):
        self.sent.append(targets)
        self.last_command_ok = not self.fail_writes
        self.status = "fake write failed" if self.fail_writes else "fake write ok"
        return self.status


def make_controller(bridge=None):
    bridge = bridge or FakeServoBridge()
    preview = FakeArmPreview()
    controller = target_vision.RedSquareGraspController(
        True,
        bridge,
        preview,
        550,
        300,
        1120,
        1700,
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
        10.0,
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
    )
    controller.startup_stage = "complete"
    controller.algorithm_stage = "centering"
    return controller, bridge, preview


class ChassisStationSafetyTests(unittest.TestCase):
    def test_chassis_ready_requires_writable_servo_link(self):
        controller, bridge, _ = make_controller()
        self.assertTrue(controller.ready_for_chassis_link(True))
        bridge.write_enabled = False
        self.assertFalse(controller.ready_for_chassis_link(True))
        self.assertTrue(controller.ready_for_chassis_link(False))

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
            clock[0] = 102.36
            controller.update_chassis_station(
                "DISC_CATCH", [], (600, 800, 3), detection_fresh=False
            )

        self.assertIsNone(controller.active_chassis_station)
        self.assertEqual(
            controller.consume_chassis_station_done(),
            "NO_RED_OR_YELLOW_BALL_2.0S",
        )
        self.assertEqual(bridge.sent[-1]["id1"], target_vision.HOME_ID1_TICK)
        self.assertEqual(bridge.sent[-1]["id2"], target_vision.HOME_ID2_TICK)
        self.assertEqual(bridge.sent[-1]["id5"], target_vision.CATCHER_HOME_TICK)

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
        self.assertEqual(bridge.sent[-1]["id4"], 1120)
        self.assertEqual(bridge.sent[-1]["splitter_id4"], 800)


if __name__ == "__main__":
    unittest.main()
