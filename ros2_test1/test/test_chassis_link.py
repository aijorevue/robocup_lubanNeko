import unittest
from unittest import mock

from ros2_test1.chassis_link import ChassisArmLink
from ros2_test1.field_mode import FieldMode, policy_for


class RecordingLink(ChassisArmLink):
    def __init__(self):
        super().__init__(False, "auto", 115200, 2.0)
        self.sent = []
        self.ready_to_run = True

    def send_line(self, text):
        self.sent.append(text)
        return True


class FieldPolicyTests(unittest.TestCase):
    def test_mirrored_field_targets(self):
        red = policy_for(FieldMode.RED)
        blue = policy_for(FieldMode.BLUE)
        self.assertEqual(red.platform_color, "red")
        self.assertEqual(blue.platform_color, "blue")
        self.assertEqual(red.disc_colors, frozenset(("red", "yellow")))
        self.assertEqual(blue.disc_colors, frozenset(("blue", "yellow")))


class ChassisProtocolTests(unittest.TestCase):
    def test_prep_high_ack_is_sent_only_after_motion_completion(self):
        link = RecordingLink()
        link.ready_to_run = True
        link._handle_line("ARM,DISC_CATCH,PREP_HIGH,SEQ,19,FIELD,RED,ID1,500,ID2,600")

        self.assertFalse(any("PREP_HIGH_ACK" in line for line in link.sent))
        prep = link.consume_preps()[0]
        self.assertEqual(prep["sequence"], 19)
        link.complete_prep(prep, True)
        self.assertTrue(any("PREP_HIGH_ACK,SEQ,19" in line for line in link.sent))

    def test_plain_sync_does_not_stop_active_task(self):
        link = RecordingLink()
        link.active_task = "PLATFORM_PICK"
        link._handle_line("ARM,SYNC,FIELD,RED")
        self.assertEqual(link.active_task, "PLATFORM_PICK")
        self.assertEqual(link.pending_stops, [])

    def test_reset_clears_task_only_after_home_completion(self):
        link = RecordingLink()
        link.active_task = "PLATFORM_PICK"
        link.pending_starts.append("PLATFORM_PICK")
        link._handle_line("ARM,SYNC,RESET,FIELD,BLUE")
        self.assertTrue(link.reset_pending)
        self.assertEqual(link.active_task, "PLATFORM_PICK")
        self.assertTrue(link.consume_reset_request())
        self.assertTrue(link.complete_reset(True, "home ok"))
        self.assertIsNone(link.active_task)
        self.assertEqual(link.pending_starts, [])
        self.assertEqual(link.field_mode, FieldMode.BLUE)
        self.assertTrue(any("RK,ARM,RESET,DONE" in line for line in link.sent))

    def test_duplicate_reset_replays_done_without_new_motion(self):
        link = RecordingLink()
        link._handle_line("ARM,SYNC,RESET,FIELD,RED")
        self.assertTrue(link.consume_reset_request())
        self.assertTrue(link.complete_reset(True, "home ok"))
        link._handle_line("ARM,SYNC,RESET,FIELD,RED")
        self.assertFalse(link.reset_pending)
        self.assertFalse(link.consume_reset_request())

    def test_duplicate_start_is_queued_once(self):
        link = RecordingLink()
        line = "ARM,PLATFORM_PICK,START,SEQ,7,FIELD,RED"
        link._handle_line(line)
        link._handle_line(line)
        self.assertEqual(link.pending_starts, ["PLATFORM_PICK"])
        self.assertEqual(link.active_sequence, 7)

    def test_start_before_controller_ready_returns_busy(self):
        link = RecordingLink()
        link.ready_to_run = False
        link._handle_line("ARM,DISC_CATCH,START,SEQ,8,FIELD,RED")
        self.assertIsNone(link.active_task)
        self.assertEqual(link.pending_starts, [])
        self.assertTrue(any("RK,ARM,DISC_CATCH,BUSY,SEQ,8" in line for line in link.sent))

    def test_target_watch_restarts_after_ready_motion(self):
        link = RecordingLink()
        with mock.patch("ros2_test1.chassis_link.time.monotonic", return_value=15.0):
            link._handle_line("ARM,PLATFORM_PICK,START,SEQ,9,FIELD,RED")
        with mock.patch("ros2_test1.chassis_link.time.monotonic", return_value=18.5):
            self.assertTrue(link.restart_target_watch(0.5))
        self.assertEqual(link.station_started, 18.5)
        self.assertEqual(link.last_target_seen, 19.0)

    def test_completed_sequence_replays_without_restarting_motion(self):
        link = RecordingLink()
        command = "ARM,PLATFORM_PICK,START,SEQ,11,FIELD,RED"
        link._handle_line(command)
        link.pending_starts.clear()
        self.assertTrue(link.finish_active("GRASP_DONE"))

        link._handle_line(command)

        self.assertIsNone(link.active_task)
        self.assertEqual(link.pending_starts, [])
        self.assertTrue(
            any("RK,ARM,PLATFORM_PICK,DONE,SEQ,11" in line for line in link.sent)
        )

    def test_stale_completion_is_not_reused_for_next_platform_pick(self):
        link = RecordingLink()
        link._handle_line("ARM,PLATFORM_PICK,START,SEQ,20,FIELD,RED")
        link.pending_starts.clear()
        link.finish_active("GRASP_DONE")
        link.sent.clear()

        link._handle_line("ARM,PLATFORM_PICK,STATUS,SEQ,21")

        self.assertEqual(link.sent, ["RK,ARM,PLATFORM_PICK,IDLE,SEQ,21"])

    def test_stop_with_wrong_sequence_cannot_stop_active_task(self):
        link = RecordingLink()
        link._handle_line("ARM,COLUMN_CATCH,START,SEQ,30,FIELD,RED")
        link.pending_starts.clear()
        link.sent.clear()

        link._handle_line("ARM,COLUMN_CATCH,STOP,SEQ,31")

        self.assertEqual(link.pending_stops, [])
        self.assertEqual(link.active_sequence, 30)
        self.assertTrue(any("BUSY,SEQ,31" in line for line in link.sent))

    def test_remote_failure_is_replayed_for_same_sequence(self):
        link = RecordingLink()
        command = "ARM,DISC_CATCH,START,SEQ,40,FIELD,RED"
        link._handle_line(command)
        link.pending_starts.clear()
        link.fail_active("SERVO_OR_CONTROL_FAULT")
        link.sent.clear()

        link._handle_line("ARM,DISC_CATCH,STATUS,SEQ,40")

        self.assertTrue(
            any("RK,ARM,DISC_CATCH,ERR,SEQ,40" in line for line in link.sent)
        )

    def test_disconnect_clears_partial_line(self):
        link = RecordingLink()
        link.line_buffer = "ARM,PLATFORM"
        link._close_fd()
        self.assertEqual(link.line_buffer, "")

    def test_field_change_is_rejected_during_task(self):
        link = RecordingLink()
        link.active_task = "COLUMN_CATCH"
        link._handle_line("FIELD,BLUE")
        self.assertEqual(link.field_mode, FieldMode.RED)
        self.assertTrue(any("RK,FIELD,BUSY" in line for line in link.sent))


if __name__ == "__main__":
    unittest.main()
