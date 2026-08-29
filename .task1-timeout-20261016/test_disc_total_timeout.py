"""Exercise the real task-one controller with fake servos and a monotonic clock."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import ros2_test1

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ros2_test1.target_vision", ROOT / "target_vision.py"
)
tv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tv
spec.loader.exec_module(tv)
ros2_test1.target_vision = tv
spec = importlib.util.spec_from_file_location("station_helpers", ROOT / "station_helpers.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class DiscTotalTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.0
        for patch in (mock.patch.object(tv.time, "monotonic", lambda: self.clock),
                      mock.patch.object(tv.time, "sleep")):
            patch.start()
            self.addCleanup(patch.stop)

    def begin(self, field=tv.FieldMode.RED):
        c, bridge, _ = helpers.make_controller(field_mode=field)
        c.begin_chassis_station("DISC_CATCH")
        return c, bridge

    def tick(self, c, fresh=False):
        return c.update_chassis_station("DISC_CATCH", [], (600, 800, 3), detection_fresh=fresh)

    def enter_detect(self, c):
        self.clock = c.chassis_station_deadline
        self.tick(c)
        self.assertEqual(c.chassis_station_stage, "disc_detect")
        return self.clock

    def assert_home_order(self, bridge):
        high = next(i for i, x in enumerate(bridge.sent) if x.get("id1") == tv.PLATFORM_HIGH_POSE[0])
        home26 = next(i for i, x in enumerate(bridge.sent) if x.get("id2") == tv.HOME_ID2_TICK and x.get("id6") == tv.BASE_YAW_HOME_TICK)
        home1 = next(i for i, x in enumerate(bridge.sent) if x.get("id1") == tv.HOME_ID1_TICK)
        self.assertLess(high, home26)
        self.assertLess(home26, home1)

    def test_timer_starts_only_after_low_pose_settles(self):
        c, _ = self.begin()
        self.assertIsNone(c.disc_catch_total_deadline)
        self.clock = c.chassis_station_deadline - 0.001
        self.tick(c)
        self.assertIsNone(c.disc_catch_total_deadline)
        started = self.enter_detect(c)
        self.assertEqual(c.disc_catch_total_deadline, started + 30.0)

    def test_continuous_ball_does_not_extend_budget_red_or_blue(self):
        for field in (tv.FieldMode.RED, tv.FieldMode.BLUE):
            with self.subTest(field=field):
                c, bridge = self.begin(field)
                started = self.enter_detect(c)
                deadline = c.disc_catch_total_deadline
                ball = {"color": "red" if field == tv.FieldMode.RED else "blue", "center": (400, 300)}
                bridge.sent.clear()
                with mock.patch.object(c, "_disc_catch_ball_visible", return_value=ball):
                    for n in range(1, 300):
                        self.clock = started + n / 10.0
                        self.tick(c, fresh=True)
                        self.assertEqual(c.disc_catch_total_deadline, deadline)
                        self.assertEqual(c.active_chassis_station, "DISC_CATCH")
                    self.clock = deadline
                    self.tick(c, fresh=True)
                self.assertEqual(c.consume_chassis_station_done(), "TOTAL_TIMEOUT_30.0S")
                self.assertIsNone(c.active_chassis_station)
                self.assertIsNone(c.disc_catch_total_deadline)
                self.assert_home_order(bridge)
                sent = len(bridge.sent)
                self.tick(c)
                self.assertEqual(len(bridge.sent), sent)

    def test_all_pulse_and_cooldown_stages_respect_total_timeout(self):
        for stage in ("disc_detect", "disc_open", "disc_open_wait", "disc_close", "disc_close_confirm", "disc_close_wait", "disc_blue_channel_hold", "disc_yellow_splitter_reset", "disc_yellow_cooldown"):
            with self.subTest(stage=stage):
                c, bridge = self.begin()
                self.enter_detect(c)
                c.chassis_station_stage = stage
                c.chassis_station_deadline = self.clock + 1000
                self.clock = c.disc_catch_total_deadline
                self.tick(c, fresh=False)
                self.assertEqual(c.consume_chassis_station_done(), "TOTAL_TIMEOUT_30.0S")

    def test_four_second_no_ball_exit_is_preserved(self):
        c, bridge = self.begin()
        started = self.enter_detect(c)
        bridge.sent.clear()
        self.clock = started + 3.999
        self.tick(c)
        self.assertEqual(c.active_chassis_station, "DISC_CATCH")
        self.clock = started + 4.0
        self.tick(c)
        self.assertEqual(c.consume_chassis_station_done(), "NO_RED_OR_YELLOW_BALL_4.0S")
        self.assert_home_order(bridge)

    def test_home_failure_never_reports_done(self):
        c, bridge = self.begin()
        self.enter_detect(c)
        bridge.fail_writes = True
        self.clock = c.disc_catch_total_deadline
        self.tick(c)
        self.assertIsNone(c.consume_chassis_station_done())
        self.assertEqual(c.consume_chassis_station_error(), "HOME_FAILED")

    def test_new_run_and_reset_discard_old_deadline(self):
        c, _ = self.begin()
        self.enter_detect(c)
        c.reset_from_chassis()
        self.assertIsNone(c.disc_catch_total_deadline)
        self.clock += 100
        c.begin_chassis_station("DISC_CATCH")
        self.assertIsNone(c.disc_catch_total_deadline)
        started = self.enter_detect(c)
        self.assertEqual(c.disc_catch_total_deadline, started + 30)

    def test_other_stations_do_not_use_disc_deadline(self):
        c, _ = self.begin()
        c.disc_catch_total_deadline = self.clock - 1
        with mock.patch.object(c.platform_task, "tick") as tick, mock.patch.object(c, "_sync_platform_task", return_value="platform unchanged"), mock.patch.object(c, "_update_column_catch_station", return_value="column unchanged"), mock.patch.object(c, "_finish_chassis_station_after_retract") as finish:
            self.assertEqual(c.update_chassis_station("PLATFORM_PICK", [], (600, 800, 3)), "platform unchanged")
            self.assertEqual(c.update_chassis_station("COLUMN_CATCH", [], (600, 800, 3)), "column unchanged")
            tick.assert_called_once()
            finish.assert_not_called()

    def test_pre_timeout_servo_sequence_matches_previous_source(self):
        old_spec = importlib.util.spec_from_file_location("ros2_test1.target_vision_before", ROOT / "target_vision.before.py")
        old = importlib.util.module_from_spec(old_spec)
        sys.modules[old_spec.name] = old
        old_spec.loader.exec_module(old)
        for field in (tv.FieldMode.RED, tv.FieldMode.BLUE):
            for no_ball in (False, True):
                with self.subTest(field=field, no_ball=no_ball):
                    traces = []
                    for module in (old, tv):
                        self.clock = 100.0
                        with mock.patch.object(helpers, "target_vision", module):
                            c, bridge, _ = helpers.make_controller(field_mode=field)
                        c.begin_chassis_station("DISC_CATCH")
                        self.clock = c.chassis_station_deadline
                        self.tick(c)
                        started = self.clock
                        states = []
                        for step in range(1, 300):
                            self.clock = started + step / 10.0
                            # Alternate own-field/yellow pulses with brief target losses.
                            color = ("red" if field == tv.FieldMode.RED else "blue") if step % 30 < 15 else "yellow"
                            ball = None if no_ball or step % 10 < 4 else {"color": color, "center": (400, 300)}
                            with mock.patch.object(c, "_disc_catch_ball_visible", return_value=ball):
                                self.tick(c, fresh=True)
                            states.append((c.chassis_station_stage, c.chassis_station_done_reason, c.chassis_station_error_reason))
                        traces.append((bridge.timed_sent, states))
                    self.assertEqual(traces[0], traces[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
