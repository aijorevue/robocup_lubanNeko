import time

from ros2_test1.target_vision import RedSquareGraspController


class Bridge:
    enabled = True
    write_enabled = True

    def __init__(self):
        self.commands = []
        self.last_command_ok = True
        self.status = "OK"

    def handshake(self):
        self.commands.append(("handshake", {}))
        self.last_command_ok = True
        return True

    def command_arm_ready(self, timeout_s=7.0):
        self.commands.append(("armready", {"timeout_s": timeout_s}))
        self.last_command_ok = True
        return 550, 300, 510

    def command_arm_home(self, timeout_s=8.0):
        self.commands.append(("armhome", {"timeout_s": timeout_s}))
        self.last_command_ok = True
        return 450, 10, 510

    def command_claw(self, target):
        self.commands.append(("claw", {"target": int(target)}))
        self.last_command_ok = True
        self.status = f"ID4 confirmed target={target} actual={target}"
        return int(target)

    def send_targets(self, *args, **kwargs):
        self.commands.append(("targets", dict(kwargs)))
        self.last_command_ok = True
        self.status = "ACK targets"
        return self.status


class Preview:
    id6 = 510

    def set_targets(self, *args):
        pass

    def set_feedback(self, *args):
        pass

    def publish(self, *args, **kwargs):
        pass

    def publish_plan_marker(self, *args):
        pass


bridge = Bridge()
controller = RedSquareGraspController(
    enabled=True,
    servo_bridge=bridge,
    arm_preview=Preview(),
    id1_ready=550,
    id2_ready=300,
    id4_closed=1000,
    id4_open=1500,
    center_deadband_px=30,
    stable_frames=1,
    command_interval_s=0.20,
    id2_pixel_gain=0.15,
    id6_pixel_gain=0.18,
    id6_max_step_ticks=60,
    id1_pixel_gain_y=0.02,
    id2_distance_gain=0.25,
    camera_gripper_offset_mm=50.0,
    target_gripper_distance_mm=20.0,
    distance_deadband_mm=12.0,
    max_step_ticks=18,
    id1_limits=(0, 1000),
    id2_limits=(0, 1000),
    angle_gap_degrees=0.0,
    startup_sequence=True,
    one_shot=False,
    direct_grasp=False,
    camera_gripper_vertical_offset_mm=0.0,
    max_lateral_offset_mm=45.0,
    max_one_shot_ik_error_mm=15.0,
    post_center_retreat_mm=0.0,
    post_center_down_mm=280.0,
    post_center_ik_error_mm=18.0,
)

controller.last_command_time = 0.0
controller._update_startup(time.monotonic())
assert bridge.commands[:2] == [
    ("handshake", {}),
    ("armready", {"timeout_s": 7.0}),
]
assert (controller.id1, controller.id2, controller.id6) == (550, 300, 510)

controller.algorithm_stage = "open"
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1] == ("claw", {"target": 1500})
assert controller.algorithm_stage == "open_settle"

controller.claw_settle_deadline = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert controller.algorithm_stage == "descend"

controller.locked_plan = {
    "id1": 320,
    "id2": 260,
    "progress_ratio": 1.0,
    "ik_error_mm": 0.0,
}
controller.approach_attempts = 0
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1] == (
    "targets",
    {"id1": 320, "id2": 260, "id4": None, "id6": None,
     "synchronize_pair": True},
)

controller.feedback_pending = False
controller.id1, controller.id2 = 320, 260
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1] == ("claw", {"target": 1000})
assert controller.algorithm_stage == "close_settle"

controller.claw_settle_deadline = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert controller.algorithm_stage == "carry_lift"

controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1][1]["synchronize_pair"] is True
controller.feedback_pending = False
controller.id1, controller.id2 = 550, 300
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert controller.algorithm_stage == "carry_retract"

controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1][1]["id1"] is None
assert bridge.commands[-1][1]["id2"] == 10
controller.feedback_pending = False
controller.id1, controller.id2 = 550, 10
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert controller.algorithm_stage == "drop_deploy_id5"

controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1] == ("targets", {"id5": 950})
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1] == ("claw", {"target": 1500})

controller.claw_settle_deadline = 0.0
controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert bridge.commands[-1] == ("claw", {"target": 1000})
controller.claw_settle_deadline = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert controller.algorithm_stage == "ready_return"

controller.last_command_time = 0.0
controller._update_locked_grasp((600, 800, 3), time.monotonic(), False)
assert ("armready", {"timeout_s": 7.0}) in bridge.commands
assert bridge.commands[-1] == ("targets", {"id5": 700})
assert controller.algorithm_stage == "centering"
assert controller.cycle_search_enabled

print("Unified arm flow verified")
