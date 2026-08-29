#!/usr/bin/env python3
"""Isolated H7 task-two route with the production RK task-two implementation.

Only the TEST,TASK2 chassis transport and local preview are test-specific.
Detection, preselection, per-slot policy, calibration, arm callbacks and white
line replies are imported from the production package. Importing this module
does not open cameras, serial ports, ROS nodes, or the formal service.
"""
from __future__ import annotations

import argparse
from collections import deque
import errno
import fcntl
import os
from pathlib import Path
import select
import subprocess
import sys
import termios
import time

import cv2

ROOT = Path(__file__).resolve().parent
ABCD_DIR = Path(os.environ.get("ABCD_PACKAGE_DIR", ROOT / "ABCD_detector"))
BALLS_DIR = Path(os.environ.get("BALLS_PACKAGE_DIR", ROOT / "balls_detector"))
ROS_PACKAGE_DIR = ROOT / "ros2_test1"
MAIN_CAMERA = "/dev/v4l/by-path/platform-fc800000.usb-usb-0:1:1.0-video-index0"
SECONDARY_CAMERA = "/dev/v4l/by-path/platform-fc880000.usb-usb-0:1.3:1.0-video-index0"
H7_DEVICE = "/dev/h7_chassis"
ARM_DEVICE = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82109853-if00"


def load_formal():
    """Import definitions only: never invoke target_vision.main()."""
    for directory in (ABCD_DIR, BALLS_DIR, ROS_PACKAGE_DIR):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
    from ros2_test1 import target_vision, platform_task
    return target_vision, platform_task


def open_uart(path: str, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while True:
        fd = None
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            fcntl.ioctl(fd, termios.TIOCEXCL)
            attrs = termios.tcgetattr(fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[4] = attrs[5] = termios.B115200
            attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            return fd
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            if exc.errno not in {
                errno.ENOENT, errno.ENODEV, errno.ENXIO, errno.EIO, errno.EBUSY
            } or time.monotonic() >= deadline:
                raise
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def write_all(fd, payload):
    deadline = time.monotonic() + 1.0
    while payload:
        try:
            count = os.write(fd, payload)
            if count <= 0:
                raise OSError(errno.EIO, "serial write returned no bytes")
            payload = payload[count:]
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("H7 serial write timed out")
            select.select([], [fd], [], min(0.02, remaining))


class H7Link:
    """Keep the existing standalone TEST,TASK2 route, not the formal route."""
    def __init__(self, path, white_line_service=None):
        self.path = path
        self.fd = None
        self.rx = bytearray()
        self.lines = deque()
        self.white_line_service = white_line_service

    def open(self):
        self.fd = open_uart(self.path)

    def send(self, line):
        if self.fd is None:
            self.open()
        write_all(self.fd, (line.rstrip("\r\n") + "\r\n").encode("ascii"))
        print(f"TASK2 H7 TX {line}", flush=True)

    @staticmethod
    def _status_matches(fields, sequence, statuses, field):
        if (len(fields) < 6 or fields[:3] != ["H7", "TEST", "TASK2"]
                or fields[3] not in statuses or fields[4:6] != ["SEQ", str(sequence)]):
            return False
        if field is None:
            return True
        try:
            index = fields.index("FIELD")
        except ValueError:
            return False
        return index + 1 < len(fields) and fields[index + 1] == field

    def _read_lines(self, timeout):
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return
        data = os.read(self.fd, 4096)
        if not data:
            raise ConnectionError("H7 disconnected")
        self.rx.extend(data)
        if len(self.rx) > 65536:
            raise RuntimeError("H7_RX_OVERFLOW")
        while b"\n" in self.rx:
            raw, _, self.rx = self.rx.partition(b"\n")
            line = raw.decode("ascii", "replace").strip("\r")
            print(f"TASK2 H7 RX {line}", flush=True)
            fields = [item.strip() for item in line.split(",")]
            if fields[:3] == ["VISION", "WHITE_LINE", "QUERY"]:
                if self.white_line_service is not None:
                    self.white_line_service.enqueue(line)
            else:
                self.lines.append(fields)
        if self.white_line_service is not None:
            # The formal link retains only the latest query in this batch.
            self.white_line_service.process()

    def wait_status_any(self, sequence, statuses, timeout_s, field=None):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            while self.lines:
                fields = self.lines.popleft()
                if self._status_matches(fields, sequence, statuses, field):
                    return ",".join(fields)
                if self._status_matches(fields, sequence, {"ERR", "STOPPED"}, field):
                    raise RuntimeError("H7_TEST_FAILED " + ",".join(fields))
            self._read_lines(min(0.05, max(0.0, deadline - time.monotonic())))
        return None

    def wait_status(self, sequence, status, timeout_s, field=None):
        return self.wait_status_any(sequence, {status}, timeout_s, field)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Task2WhiteLineService:
    """Production task-two query parser/detector/reply; no extra 0.8s cache."""
    def __init__(self, vision, reader, send):
        self.reader = reader
        self.link = vision.ChassisArmLink(False, "auto", 115200, 3.0)
        self.link.send_line = send
        self.detector = vision.WhiteLineAlignmentDetector()
        self.last_query = None

    def enqueue(self, line):
        fields = [part.strip() for part in line.split(",")]
        # The isolated task-two route must not answer task-one/three requests.
        if "PHASE" in fields:
            index = fields.index("PHASE")
            if index + 1 < len(fields) and fields[index + 1] != "TASK2_AFTER_SECONDARY_SHIFT":
                self.link.send_line("RK,VISION,WHITE_LINE,ERR,REASON,INVALID_PHASE")
                return
        self.link._handle_line(line)

    def process(self):
        pending = self.link.consume_white_line_queries()
        if not pending:
            return
        query = (pending[-1], self.link.white_line_phase)
        if query != self.last_query:
            self.detector.reset_tracking()
            self.last_query = query
        ok, frame, _, error = self.reader.latest()
        measurement = None
        if ok and error is None and frame is not None and frame.size:
            if frame.shape[:2] != (600, 800):
                frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            measurement = self.detector.detect_task2(frame)
        for sequence in pending:
            # Includes RX and the same rounding as the formal controller.
            self.link.send_white_line_result(sequence, measurement)


class LocalPreview:
    """The formal arm callbacks need a preview sink, not a ROS publisher."""
    def __init__(self, id6):
        self.id6 = int(id6)

    def set_targets(self, id1, id2, id7, id6):
        self.id6 = int(id6)

    def publish(self, text):
        print(f"TASK2 ARM {text}", flush=True)


def build_controller(vision, platform, bridge, field):
    """Use the real controller's platform callbacks, limits and staged moves.

    Generic task-one/three tuning below is inert: this runner only ticks the
    controller's PlatformTask, never its general update/startup state machine.
    """
    return vision.TargetGraspController(
        enabled=True, servo_bridge=bridge, arm_preview=LocalPreview(platform.HIGH[2]),
        id1_ready=platform.HIGH[0], id2_ready=platform.HIGH[1],
        id7_closed=platform.PLATFORM_GRIPPER_CLOSED,
        id7_open=platform.PLATFORM_GRIPPER_OPEN,
        center_deadband_px=platform.CENTER_DEADBAND_PX, stable_frames=3,
        command_interval_s=0.18, retrigger_cooldown_s=1.2,
        id2_pixel_gain=0.13, id6_pixel_gain=0.11, id6_max_step_ticks=18,
        id1_pixel_gain_y=0, id2_distance_gain=0,
        camera_gripper_offset_mm=50, target_gripper_distance_mm=20,
        distance_deadband_mm=0, max_step_ticks=28,
        id1_limits=vision.ID1_SAFE_LIMITS, id2_limits=vision.ID2_SAFE_LIMITS,
        angle_gap_degrees=vision.MIN_ANGLE_GAP_DEG,
        startup_sequence=False, one_shot=False,
        camera_gripper_vertical_offset_mm=0, max_lateral_offset_mm=0,
        max_one_shot_ik_error_mm=0, post_center_retreat_mm=52,
        post_center_down_mm=150, post_center_ik_error_mm=0,
        field_mode=field,
    )


def preflight_camera(path):
    """Refuse to compete with the user's existing preview or formal service."""
    service = subprocess.run(
        ["systemctl", "is-active", "--quiet", "robocup-chassis-arm.service"],
        check=False, capture_output=True,
    )
    if service.returncode == 0:
        raise RuntimeError("FORMAL_SERVICE_ACTIVE; stop it yourself before isolated testing")
    owners = subprocess.run(["fuser", path], check=False, capture_output=True)
    if owners.returncode == 0:
        raise RuntimeError(f"CAMERA_BUSY {path}; leave the current camera owner untouched")
    if owners.returncode != 1:
        raise RuntimeError(f"CAMERA_OWNERSHIP_CHECK_FAILED {path}")


def open_reader(vision, path):
    preflight_camera(path)
    cap = vision.open_camera(path, 800, 600, 30, "TASK2_TEST")
    return vision.LatestFrameReader(cap)


def next_frame(reader, previous):
    ok, frame, _, error = reader.latest()
    if error is not None:
        raise RuntimeError(f"CAMERA_ERROR {error}")
    if not ok or frame is None or frame is previous:
        return None
    return frame


def show_frame(frame, text, headless=False):
    if headless:
        return
    view = frame.copy()
    cv2.putText(view, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    cv2.imshow("Task2 isolated (formal policy)", view)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        raise KeyboardInterrupt


def lock_pair(reader, detector, platform, timeout_s=None, headless=False):
    # A no-op pose callback keeps camera-only mode strictly free of arm I/O.
    task = platform.PlatformTask(lambda *args: 0.0, lambda *args: 0.0,
                                 lambda *args: 0.0)
    task.begin_preselect()
    if timeout_s is not None:
        task.timeout = task.clock() + timeout_s
    previous = None
    while True:
        frame = next_frame(reader, previous)
        if frame is not None:
            previous = frame
            task.preselect(detector.detect(frame), fresh=True)
            show_frame(frame, task.status, headless)
        task.tick()
        if task.error:
            raise RuntimeError(task.error)
        if len(task.selected) == 2:
            print(f"TASK2 LETTERS LOCKED pair={task.selected} source={task.preselect_lock_source}",
                  flush=True)
            return task.selected
        time.sleep(0.002)


def wait_high(task):
    while not task.high_ready:
        task.tick()
        if task.error:
            raise RuntimeError(task.error)
        time.sleep(0.002)


def run_slot(task, reader, detector, vision, field, slot,
             timeout_s=None, headless=False):
    task.begin_slot(field, slot)
    if timeout_s is not None and task.stage == "platform_detect":
        task.timeout = task.clock() + timeout_s
    previous = None
    while True:
        frame = next_frame(reader, previous)
        detections = ()
        if frame is not None:
            previous = frame
            if task.stage == "platform_detect":
                detections = detector.detect(
                    frame, mode=vision.DETECTION_MODE_PLATFORM_TARGETS,
                    field_name=field, target_letters=task.selected,
                )
        task.tick(detections, frame.shape if frame is not None else None,
                  fresh=frame is not None)
        if task.error:
            raise RuntimeError(task.error)
        if task.done:
            return task.done
        if frame is not None:
            show_frame(frame, f"SLOT {slot}: {task.status}", headless)
        time.sleep(0.002)


def stop_active_h7(h7, sequence, field, active):
    if h7 is None or sequence is None or not active:
        return
    try:
        h7.send(f"RK,TEST,TASK2,STOP,SEQ,{sequence},FIELD,{field.upper()}")
        result = h7.wait_status_any(sequence, {"STOPPED", "DONE"}, 4.0, field.upper())
        print(f"TASK2 STOP {'CONFIRMED' if result else 'TIMEOUT'}", flush=True)
    except Exception as exc:
        print(f"TASK2 STOP_ERROR {exc}", flush=True)


def run(args):
    if not args.headless:
        os.environ.setdefault("DISPLAY", ":0")
    secondary = main_reader = bridge = h7 = None
    sequence = None
    active = False
    try:
        vision, platform = load_formal()
        print(f"TASK2 FORMAL_SOURCE {vision.__file__} window={platform.TARGET_WINDOW_SIZE_PX} "
              f"fullscreen_slots={sorted(platform.PLATFORM_FULLSCREEN_SLOTS)}", flush=True)
        secondary = open_reader(vision, args.secondary_camera)
        pair = lock_pair(secondary, vision.SecondaryLetterDetector(), platform,
                         args.letter_timeout_s, args.headless)
        secondary.close()
        secondary = None
        if not args.execute_h7:
            print(f"TASK2 CAMERA_ONLY COMPLETE L1={pair[0]} L2={pair[1]}", flush=True)
            return 0
        # Check/open the main camera before sending any arm or chassis command.
        main_reader = open_reader(vision, args.main_camera)
        detector = vision.TargetDetector()
        # Do not charge cold model loading to the formal three-second slot budget.
        detector.letter_detector = vision.ABCDDetector()
        for device in (args.arm_uart, args.h7_device):
            owners = subprocess.run(["fuser", device], check=False, capture_output=True)
            if owners.returncode == 0:
                raise RuntimeError(f"SERIAL_BUSY {device}")
            if owners.returncode != 1:
                raise RuntimeError(f"SERIAL_OWNERSHIP_CHECK_FAILED {device}")
        bridge = vision.HiwonderSingleBusServoBridge(
            args.arm_uart, 115200, enabled=True, write_enabled=True,
            arm_time_ms=platform.PLATFORM_ARM_TIME_MS,
            aux_time_ms=platform.PLATFORM_AUX_TIME_MS,
            gripper_time_ms=platform.PLATFORM_GRIPPER_TIME_MS,
        )
        if bridge.arm_fd is None:
            raise RuntimeError(bridge.status)
        controller = build_controller(vision, platform, bridge, args.field)
        task = controller.platform_task
        task._lock_selected(pair, "ISOLATED_FORMAL_PRESELECT")
        wait_high(task)
        h7 = H7Link(args.h7_device)
        h7.white_line_service = Task2WhiteLineService(vision, main_reader, h7.send)
        sequence = int(time.time() * 1000) & 0xFFFFFFFF
        for step in range(8):
            command = (
                f"RK,TEST,TASK2,START,SEQ,{sequence},FIELD,{args.field.upper()},"
                f"L1,{pair[0]},L2,{pair[1]}"
                if step == 0 else
                f"RK,TEST,TASK2,NEXT,SEQ,{sequence},FIELD,{args.field.upper()},STEP,{step}"
            )
            # A transmitted command may execute even when its ACK is lost.
            active = True
            h7.send(command)
            for status, timeout in (("ACK", 4.0), ("RUNNING", 8.0),
                                    ("DONE", args.h7_timeout_s)):
                if not h7.wait_status(sequence, status, timeout, args.field.upper()):
                    raise RuntimeError(f"H7_{status}_TIMEOUT slot={step + 1}")
            result = run_slot(task, main_reader, detector, vision, args.field,
                              step + 1, args.main_timeout_s, args.headless)
            print(f"TASK2 SLOT_DONE slot={step + 1} result={result} "
                  f"letter_counts={task.letter_success_counts}", flush=True)
        active = False
        print(f"TASK2 TEST COMPLETE pair={pair} slots=8", flush=True)
        return 0
    except KeyboardInterrupt:
        stop_active_h7(h7, sequence, args.field, active)
        print("TASK2 ABORTED", flush=True)
        return 2
    except Exception as exc:
        stop_active_h7(h7, sequence, args.field, active)
        print(f"TASK2 ERROR {exc}", flush=True)
        return 1
    finally:
        for resource in (secondary, main_reader, bridge, h7):
            if resource is not None:
                resource.close()
        if not args.headless:
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="Isolated task-two route; production RK policy")
    parser.add_argument("--field", choices=("red", "blue"), required=True)
    parser.add_argument("--secondary-camera", default=SECONDARY_CAMERA)
    parser.add_argument("--main-camera", default=MAIN_CAMERA)
    parser.add_argument("--h7-device", default=H7_DEVICE)
    parser.add_argument("--arm-uart", default=ARM_DEVICE)
    parser.add_argument("--zp-uart", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--execute-h7", action="store_true",
                        help="explicitly enable arm commands and the H7 test route")
    parser.add_argument("--letter-timeout-s", type=float, default=None,
                        help="optional override; otherwise use formal preselection timeout")
    parser.add_argument("--main-timeout-s", type=float, default=None,
                        help="optional override; otherwise use formal slot timeout")
    parser.add_argument("--h7-timeout-s", type=float, default=120.0)
    parser.add_argument("--headless", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("letter_timeout_s", "main_timeout_s", "h7_timeout_s"):
        value = getattr(args, name)
        if value is not None and (not 0 < value < float("inf")):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
