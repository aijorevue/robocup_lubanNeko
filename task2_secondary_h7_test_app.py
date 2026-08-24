#!/usr/bin/env python3
"""Standalone task-two test: secondary letter preselection, H7 route, main target.

The app owns both camera sessions and the two direct servo-board UARTs.  It
does not start the formal ROS service.  H7 owns only the fixed chassis test
route and receives one explicit TEST,TASK2 command after the high arm pose is
sent.
"""

from __future__ import annotations

import argparse
import errno
import os
import select
import sys
import termios
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
ABCD_DIR = Path(os.environ.get("ABCD_PACKAGE_DIR", ROOT / "ABCD_detector"))
BALLS_DIR = Path(os.environ.get("BALLS_PACKAGE_DIR", ROOT / "balls_detector"))
ROS_PACKAGE_DIR = ROOT / "ros2_test1"
MAIN_CAMERA = "/dev/v4l/by-path/platform-fc800000.usb-usb-0:1:1.0-video-index0"
SECONDARY_CAMERA = "/dev/v4l/by-path/platform-fc880000.usb-usb-0:1.3:1.0-video-index0"
H7_DEVICE = "/dev/h7_chassis"
ARM_DEVICE = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82109853-if00"
ZP_DEVICE = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"

HIGH = {1: 650, 2: 600, 6: 350}
LETTER_WORK = {1: 500, 2: 350, 6: 600}
RING_AFTER_HIGH = {1: 535, 2: 330, 6: 120}
ZP_HIGH = {4: 1200, 5: 800, 7: 1300}
# These two SG90 outputs are fixed auxiliary channels on the ZL 24-channel
# board.  They must remain at their neutral test positions throughout the app.
AUX_ZP_HOLD = {12: 600, 23: 1000}
AUX_ZP_HOLD_TIME_MS = 800
GRIPPER_CLOSED = 1300
GRIPPER_OPEN = 1650
ARM_TIME_MS = 600
ZP_TIME_MS = 300
GRIPPER_TIME_MS = 300
RING_PLACE_TIME_MS = 800
CENTER_DEADBAND_PX = 45.0
CENTER_STEP_TICKS = 7
CENTER_ID6_STEP_TICKS = 5
CENTER_TIME_MS = 100
CENTER_TRACK_MAX_JUMP_PX = 320.0
CENTER_ID2_RANGE = (450, 700)
CENTER_ID6_RANGE = (0, 700)
LETTERS = {"A", "B", "C", "D"}
SECONDARY_LETTER_MIN_CONFIDENCE = 38.0
MAIN_LETTER_MIN_CONFIDENCE = 45.0
MAIN_TARGET_VOTE_WINDOW = 8
MAIN_TARGET_REQUIRED_VOTES = 3
MAIN_TARGET_VOTE_CENTER_TOL_PX = 140.0
TARGET_TRACK_MISSING_TIMEOUT_S = 4.0
MAIN_TARGET_TIMEOUT_S = 1.5
LETTER_LOCK_WARMUP_FRAMES = 20
LETTER_LOCK_HISTORY_FRAMES = 12
LETTER_LOCK_REQUIRED_FRAMES = 8
LETTER_SIZE_MM = 30.0
RING_SIZE_MM = 55.0
CAMERA_GRIPPER_OFFSET_MM = 50.0
TARGET_GRIPPER_DISTANCE_MM = 20.0
RING_DISTANCE_EXTRA_CM = 0.5
POST_OPEN_ID2_RETREAT_TICKS = 100
UART_OPEN_TIMEOUT_S = 10.0
UART_RETRY_INTERVAL_S = 0.2
UART_RETRY_ERRNOS = frozenset(
    (errno.ENOENT, errno.ENODEV, errno.ENXIO, errno.EIO, errno.EBUSY)
)
UART_COMMAND_TIMEOUT_S = 12.0
UART_COMMAND_RETRY_INTERVAL_S = 0.25
UART_COMMAND_MAX_ATTEMPTS = 6
TARGET_WINDOW_SIZE_PX = 300
TARGET_WINDOW_MIN_AREA_FRACTION = 0.80


def htd85_packet(servo_id: int, position: int, time_ms: int) -> bytes:
    body = bytes((servo_id, 7, 1, position & 0xFF, position >> 8,
                  time_ms & 0xFF, time_ms >> 8))
    return b"\x55\x55" + body + bytes(((~sum(body)) & 0xFF,))


def zp_packet(servo_id: int, position: int, time_ms: int) -> bytes:
    return f"#{servo_id:03d}P{position:04d}T{time_ms:04d}!".encode("ascii")


def write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "serial write returned no bytes")
        offset += written
    termios.tcdrain(fd)


def open_uart(path: str, role: str = "UART",
              timeout_s: float = UART_OPEN_TIMEOUT_S) -> int:
    """Open a USB UART across a short disconnect/re-enumeration window."""
    deadline = time.monotonic() + max(0.1, timeout_s)
    attempts = 0
    while True:
        fd = None
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            attrs = termios.tcgetattr(fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[4] = attrs[5] = termios.B115200
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            if attempts:
                print(
                    f"TASK2 UART READY role={role} path={path} "
                    f"attempts={attempts + 1}",
                    flush=True,
                )
            return fd
        except OSError as exc:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if exc.errno not in UART_RETRY_ERRNOS:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OSError(
                    exc.errno,
                    f"{exc.strerror}; {role} UART unavailable after "
                    f"{timeout_s:.1f}s: {path}",
                ) from exc
            if attempts == 0:
                print(
                    f"TASK2 UART WAIT role={role} path={path} "
                    f"reason=[Errno {exc.errno}] {exc.strerror}",
                    flush=True,
                )
            attempts += 1
            time.sleep(min(UART_RETRY_INTERVAL_S, remaining))


def send_uart_payload(path: str, payload: bytes, role: str) -> None:
    """Send one command with a fresh fd so USB re-enumeration is recoverable."""
    deadline = time.monotonic() + UART_COMMAND_TIMEOUT_S
    last_error = None
    for attempt in range(1, UART_COMMAND_MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        fd = None
        try:
            fd = open_uart(path, role, min(2.0, remaining))
            write_all(fd, payload)
            return
        except OSError as exc:
            last_error = exc
            if exc.errno not in UART_RETRY_ERRNOS:
                raise OSError(
                    exc.errno,
                    f"{role} command failed on {path}: {exc.strerror}",
                ) from exc
            if attempt >= UART_COMMAND_MAX_ATTEMPTS:
                break
            print(
                f"TASK2 UART RETRY role={role} attempt={attempt + 1} "
                f"path={path} reason=[Errno {exc.errno}] {exc.strerror}",
                flush=True,
            )
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        time.sleep(min(UART_COMMAND_RETRY_INTERVAL_S,
                       max(0.0, deadline - time.monotonic())))
    if last_error is None:
        raise OSError(errno.EIO, f"{role} command deadline expired on {path}")
    raise OSError(
        last_error.errno,
        f"{role} command failed after retries on {path}: "
        f"{last_error.strerror}",
    ) from last_error


class ServoBoards:
    def __init__(self, arm_path: str, zp_path: str):
        self.arm_path = arm_path
        self.zp_path = zp_path
        self.arm_fd = None
        self.zp_fd = None
        try:
            for role, path in (("ZP", zp_path), ("HTD85", arm_path)):
                fd = open_uart(path, role)
                os.close(fd)
            self.hold_aux_outputs()
        except Exception:
            self.close()
            raise

    def arm(self, targets: dict[int, int], motion_ms: int = ARM_TIME_MS) -> None:
        for servo_id, position in targets.items():
            packet = htd85_packet(servo_id, position, motion_ms)
            send_uart_payload(
                self.arm_path, packet, f"HTD85 ID{servo_id}"
            )
            print(f"TASK2 SERVO HTD85 ID{servo_id}={position} T={motion_ms}ms", flush=True)
            time.sleep(0.003)

    def zp(self, targets: dict[int, int], motion_ms: int = ZP_TIME_MS) -> None:
        for servo_id, position in targets.items():
            packet = zp_packet(servo_id, position, motion_ms)
            send_uart_payload(self.zp_path, packet, f"ZP ID{servo_id}")
            print(f"TASK2 SERVO ZP ID{servo_id}={position} T={motion_ms}ms", flush=True)
            time.sleep(0.003)

    def hold_aux_outputs(self) -> None:
        """Set the fixed ZL channels without touching task-arm ZP IDs."""
        self.zp(AUX_ZP_HOLD, AUX_ZP_HOLD_TIME_MS)
        print(
            "TASK2 AUX_ZP_HOLD S12=600 S23=1000 T=800ms",
            flush=True,
        )

    def pose_high(self) -> None:
        self.arm(HIGH)
        self.zp(ZP_HIGH)
        time.sleep(max(ARM_TIME_MS, ZP_TIME_MS) / 1000.0)

    def pulse_gripper(self) -> None:
        self.open_gripper()
        self.close_gripper()

    def open_gripper(self) -> None:
        self.zp({7: GRIPPER_OPEN}, GRIPPER_TIME_MS)
        time.sleep(GRIPPER_TIME_MS / 1000.0)

    def close_gripper(self) -> None:
        self.zp({7: GRIPPER_CLOSED}, GRIPPER_TIME_MS)
        time.sleep(GRIPPER_TIME_MS / 1000.0)

    def place_ring(self) -> None:
        """Move ring placement ID6/ID2 first, then ID1, each in 800 ms."""
        self.arm(
            {6: RING_AFTER_HIGH[6], 2: RING_AFTER_HIGH[2]},
            RING_PLACE_TIME_MS,
        )
        time.sleep(RING_PLACE_TIME_MS / 1000.0)
        self.arm({1: RING_AFTER_HIGH[1]}, RING_PLACE_TIME_MS)
        time.sleep(RING_PLACE_TIME_MS / 1000.0)

    def close(self) -> None:
        for name in ("arm_fd", "zp_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)


class H7Link:
    def __init__(self, path: str):
        self.path = path
        self.fd = None
        self.rx = bytearray()
        self.last_white_line_measurement = None
        self.last_white_line_time = 0.0

    def open(self) -> None:
        self.fd = open_uart(self.path, "H7")
        print(f"TASK2 H7 OPEN {self.path}", flush=True)

    def send(self, line: str) -> None:
        if self.fd is None:
            self.open()
        write_all(self.fd, (line.rstrip("\r\n") + "\r\n").encode("ascii"))
        print(f"TASK2 H7 TX {line.rstrip()}", flush=True)

    def wait_status(self, sequence: int, status: str, timeout_s: float,
                    field: str | None = None) -> str | None:
        return self.wait_status_any(sequence, {status}, timeout_s, field)

    @staticmethod
    def _status_matches(fields: list[str], sequence: int,
                        statuses: set[str], field: str | None) -> bool:
        if len(fields) < 6 or fields[:3] != ["H7", "TEST", "TASK2"]:
            return False
        if fields[3] not in statuses or fields[4] != "SEQ":
            return False
        if fields[5] != str(sequence):
            return False
        if field is None:
            return True
        try:
            field_index = fields.index("FIELD")
        except ValueError:
            return False
        return field_index + 1 < len(fields) and fields[field_index + 1] == field

    def wait_status_any(self, sequence: int, statuses: set[str],
                        timeout_s: float, field: str | None = None) -> str | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if not ready:
                continue
            self.rx.extend(os.read(self.fd, 4096))
            while b"\n" in self.rx:
                raw, _, self.rx = self.rx.partition(b"\n")
                line = raw.decode("ascii", "replace").strip("\r")
                print(f"TASK2 H7 RX {line}", flush=True)
                fields = [item.strip() for item in line.split(",")]
                if self._status_matches(fields, sequence, statuses, field):
                    return line
        return None

    def wait_status_with_white_line(
        self, sequence: int, status: str, timeout_s: float,
        camera: cv2.VideoCapture, white_line_detector,
        field: str | None = None,
    ) -> str | None:
        """Wait for H7 motion while servicing its main-camera line queries."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if not ready:
                continue
            self.rx.extend(os.read(self.fd, 4096))
            while b"\n" in self.rx:
                raw, _, self.rx = self.rx.partition(b"\n")
                line = raw.decode("ascii", "replace").strip("\r")
                print(f"TASK2 H7 RX {line}", flush=True)
                fields = [item.strip() for item in line.split(",")]
                if (fields[:3] == ["VISION", "WHITE_LINE", "QUERY"] and
                        len(fields) >= 5 and fields[3] == "SEQ"):
                    try:
                        query_sequence = int(fields[4])
                    except ValueError:
                        continue
                    # Camera motion and USB capture timing can make one read
                    # stale or blurred.  Try a short burst and keep the last
                    # valid geometric measurement for transient misses.
                    measurement = None
                    for _ in range(3):
                        ok, frame = camera.read()
                        if ok and frame is not None:
                            candidate = white_line_detector.detect(frame)
                            if self._valid_white_line(candidate):
                                measurement = candidate
                                break
                    now = time.monotonic()
                    held = False
                    if measurement is not None:
                        self.last_white_line_measurement = measurement
                        self.last_white_line_time = now
                    elif (
                        self.last_white_line_measurement is not None and
                        now - self.last_white_line_time <= 0.8
                    ):
                        measurement = self.last_white_line_measurement
                        held = True
                    if measurement is None:
                        self.send(
                            f"RK,VISION,WHITE_LINE,NOT_FOUND,SEQ,{query_sequence}"
                        )
                        print(
                            "TASK2 WHITE_LINE MISS no_previous_measurement "
                            f"seq={query_sequence}", flush=True
                        )
                    else:
                        self.send(
                            "RK,VISION,WHITE_LINE,FOUND,"
                            f"SEQ,{query_sequence},"
                            f"Y10,{int(round(measurement['y_at_center'] * 10.0))},"
                            f"A100,{int(round(measurement['angle_deg'] * 100.0))},"
                            f"W,{measurement['frame_width']},"
                            f"H,{measurement['frame_height']}"
                        )
                        print(
                            f"TASK2 WHITE_LINE {'HOLD' if held else 'FOUND'} "
                            f"seq={query_sequence} "
                            f"y10={int(round(measurement['y_at_center'] * 10.0))} "
                            f"a100={int(round(measurement['angle_deg'] * 100.0))}",
                            flush=True,
                        )
                    continue
                if self._status_matches(fields, sequence, {status}, field):
                    return line
        return None

    @staticmethod
    def _valid_white_line(measurement) -> bool:
        if measurement is None:
            return False
        height = int(measurement.get("frame_height", 0))
        width = int(measurement.get("frame_width", 0))
        y_at_center = float(measurement.get("y_at_center", -1.0))
        angle_deg = float(measurement.get("angle_deg", 90.0))
        bounds = measurement.get("bounds", (0, 0, 0, 0))
        return (
            width == 800
            and height == 600
            and 0.0 <= y_at_center < float(height)
            and abs(angle_deg) <= 25.0
            and int(bounds[2]) >= int(width * 0.25)
        )

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def open_camera(path: str) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"camera open failed: {path}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def flush_camera(camera: cv2.VideoCapture, frames: int = 8) -> None:
    """Discard buffered frames left over from the previous H7 station."""
    for _ in range(frames):
        camera.grab()


def import_main_detector():
    sys.path.insert(0, str(ABCD_DIR))
    try:
        from abcd_detector import ABCDDetector
    except ImportError:
        from abcd_detector.detector import ABCDDetector
    return ABCDDetector()


def import_secondary_detector():
    sys.path.insert(0, str(ABCD_DIR))
    from abcd_detector.secondary_detector import SecondaryLetterDetector
    return SecondaryLetterDetector()


def import_grasp_model():
    sys.path.insert(0, str(ROS_PACKAGE_DIR))
    from ros2_test1.grasp_calibration import calibrated_grasp_ticks
    return calibrated_grasp_ticks


def import_white_line_detector():
    sys.path.insert(0, str(ROS_PACKAGE_DIR))
    from ros2_test1.white_line_alignment import WhiteLineAlignmentDetector
    return WhiteLineAlignmentDetector()


def detect_rings(frame: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = []
    ranges = {"red": [((0, 80, 70), (10, 255, 255)),
                       ((165, 80, 70), (180, 255, 255))],
              "blue": [((95, 50, 50), (135, 255, 255))]}
    for color, limits in ranges.items():
        mask = None
        for low, high in limits:
            part = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            child_index = hierarchy[0][index][2]
            if child_index < 0:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
            (cx, cy), outer = cv2.minEnclosingCircle(contour)
            child = contours[child_index]
            inner_area = cv2.contourArea(child)
            (_, _), inner = cv2.minEnclosingCircle(child)
            if area < 300 or circularity < 0.55 or outer < 14 or inner < 5:
                continue
            ratio = inner / outer if outer else 0
            if not 0.20 <= ratio <= 0.75 or inner_area < 40:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
            area_percent = (max(float(w * h), float(area)) / frame_area) * 100.0
            distance_cm = (-1.6072186919749336 + RING_DISTANCE_EXTRA_CM
                           + 31.628878020276648 * RING_SIZE_MM / 42.67
                           / max(0.001, float(area_percent) ** 0.5))
            output.append({"kind": "ring", "color": color,
                           "center": (int(cx), int(cy)),
                           "radius": int(outer), "bbox": (x, y, w, h),
                           "score": circularity, "distance_cm": distance_cm,
                           "depth_cm": distance_cm, "depth_mm": distance_cm * 10.0,
                           "area_percent": area_percent})
    return output


def letter_detections(detector, frame: np.ndarray,
                      min_confidence: float = MAIN_LETTER_MIN_CONFIDENCE) -> list[dict]:
    return [
        item for item in detector.detect(frame)
        if str(item.get("letter", "")).upper() in LETTERS
        and float(item.get("confidence", 0.0)) >= min_confidence
    ]


def lock_pair(camera: cv2.VideoCapture, detector, timeout_s: float) -> tuple[str, str]:
    history: deque[tuple[str, str]] = deque(maxlen=LETTER_LOCK_HISTORY_FRAMES)
    started = time.monotonic()
    warmup_frames = 0
    while time.monotonic() - started < timeout_s:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue
        if warmup_frames < LETTER_LOCK_WARMUP_FRAMES:
            warmup_frames += 1
            continue
        detections = letter_detections(
            detector, frame, SECONDARY_LETTER_MIN_CONFIDENCE
        )
        # Track the two physical targets by x position. A transient A/B class
        # fluctuation contributes one vote instead of clearing all history.
        ordered = sorted(detections, key=lambda item: item["center"][0])
        if len(ordered) >= 2:
            history.append(
                (
                    str(ordered[0]["letter"]).upper(),
                    str(ordered[-1]["letter"]).upper(),
                )
            )
        if len(history) >= LETTER_LOCK_REQUIRED_FRAMES:
            winners = []
            winner_votes = []
            for position in (0, 1):
                counts = {
                    letter: sum(pair[position] == letter for pair in history)
                    for letter in LETTERS
                }
                winner = max(LETTERS, key=lambda letter: counts[letter])
                winners.append(winner)
                winner_votes.append(counts[winner])
            if (
                winners[0] != winners[1]
                and min(winner_votes) >= max(5, LETTER_LOCK_REQUIRED_FRAMES - 2)
            ):
                print(
                    f"TASK2 LETTERS LOCKED L1={winners[0]} L2={winners[1]} "
                    f"CAMERA=SECONDARY votes={winner_votes[0]}/{len(history)},"
                    f"{winner_votes[1]}/{len(history)}",
                    flush=True,
                )
                return winners[0], winners[1]
        view = frame.copy()
        for item in detections:
            x, y = map(int, item.get("center", (0, 0)))
            cv2.putText(view, str(item["letter"]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        cv2.putText(view, "SECONDARY: LOCK TWO LETTERS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Task2 secondary", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt
    raise RuntimeError("LETTER_TIMEOUT")


def identify_main_target(camera: cv2.VideoCapture, detector, field: str,
                         pair: tuple[str, str], timeout_s: float) -> dict:
    started = time.monotonic()
    history: deque[tuple[str, tuple[int, int] | None, dict | None]] = deque(
        maxlen=MAIN_TARGET_VOTE_WINDOW
    )
    last_candidate_log = 0.0
    while time.monotonic() - started < timeout_s:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue
        detected_letters = letter_detections(
            detector, frame, MAIN_LETTER_MIN_CONFIDENCE
        )
        allowed_letters = [
            item for item in detected_letters
            if (
                str(item.get("letter", "")).upper() in pair
                and _target_in_center_window(item, frame.shape)
            )
        ]
        letters = (
            [_nearest_letter_to_frame_center(allowed_letters, frame.shape)]
            if allowed_letters else []
        )
        rings = detect_rings(frame)
        own_rings = [
            d for d in rings
            if d["color"] == field.lower()
            and _target_in_center_window(d, frame.shape)
        ]
        candidates = letters + own_rings
        target = (
            min(candidates, key=lambda item: _target_center_distance(item, frame.shape))
            if candidates else None
        )
        target_key = _target_key(target)
        target_center = (
            tuple(map(int, target.get("center", (0, 0)))) if target else None
        )
        history.append((target_key, target_center, dict(target) if target else None))
        if target is not None:
            stable_observations = [
                item for key, center, item in history
                if (
                    key == target_key
                    and center is not None
                    and item is not None
                    and np.linalg.norm(
                        np.asarray(center, dtype=np.float32)
                        - np.asarray(target_center, dtype=np.float32)
                    ) <= MAIN_TARGET_VOTE_CENTER_TOL_PX
                )
            ]
            if len(stable_observations) >= MAIN_TARGET_REQUIRED_VOTES:
                confirmed = max(
                    stable_observations,
                    key=lambda item: float(
                        item.get("confidence", item.get("score", 0.0))
                    ),
                )
                confirmed["frame_shape"] = frame.shape
                print(
                    f"TASK2 MAIN TARGET {_target_display_name(confirmed)} "
                    f"FIELD={field.upper()} CAMERA=MAIN "
                    f"ALLOWED={pair[0]},{pair[1]} "
                    f"votes={len(stable_observations)}/{len(history)} "
                    f"depth_cm={confirmed.get('distance_cm', '-')}",
                    flush=True,
                )
                return confirmed
        now = time.monotonic()
        if now - last_candidate_log >= 0.5:
            letter_summary = ";".join(
                f"{str(item.get('letter', '?')).upper()}:{float(item.get('confidence', 0.0)):.1f}"
                f"@{tuple(map(int, item.get('center', (0, 0))))}"
                for item in sorted(
                    letters, key=lambda value: float(value.get("confidence", 0.0)),
                    reverse=True,
                )[:4]
            ) or "none"
            ring_summary = ";".join(
                f"{item.get('color', '?')}:{float(item.get('score', 0.0)):.2f}"
                f"@{tuple(map(int, item.get('center', (0, 0))))}"
                for item in rings[:4]
            ) or "none"
            print(
                f"TASK2 MAIN CANDIDATES letters={letter_summary} rings={ring_summary} "
                f"allowed={pair[0]},{pair[1]} "
                f"votes={[item[0] for item in history]}", flush=True,
            )
            last_candidate_log = now
        view = frame.copy()
        for item in letters:
            x, y = map(int, item.get("center", (0, 0)))
            cv2.putText(view, str(item["letter"]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        for item in rings:
            x, y = item["center"]
            cv2.circle(view, (x, y), item["radius"], (0, 255, 255), 2)
            cv2.putText(view, "RING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        cv2.putText(view, "MAIN: TARGET / SKIP", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Task2 main", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt
    raise RuntimeError("MAIN_TARGET_TIMEOUT")


def _target_matches(reference: dict, candidate: dict, pair: tuple[str, str]) -> bool:
    if reference.get("kind") == "letter":
        return (
            candidate.get("kind") == "letter"
            and str(candidate.get("letter", "")).upper()
            == str(reference.get("letter", "")).upper()
            and str(candidate.get("letter", "")).upper() in pair
        )
    if reference.get("kind") == "ring":
        return (
            candidate.get("kind") == "ring"
            and candidate.get("color") == reference.get("color")
        )
    return False


def _target_key(target: dict | None) -> str:
    if target is None:
        return "none"
    if target.get("kind") == "letter":
        return f"letter:{str(target.get('letter', '')).upper()}"
    if target.get("kind") == "ring":
        return f"ring:{str(target.get('color', '')).lower()}"
    return str(target.get("kind", "unknown")).lower()


def _target_display_name(target: dict) -> str:
    if target.get("kind") == "letter":
        return str(target.get("letter", "?")).upper()
    if target.get("kind") == "ring":
        return f"RING_{str(target.get('color', '?')).upper()}"
    return str(target.get("kind", "UNKNOWN")).upper()


def _target_center_distance(target: dict, frame_shape) -> float:
    height, width = frame_shape[:2]
    center = target.get("center", (width / 2.0, height / 2.0))
    return float(
        np.hypot(
            float(center[0]) - width / 2.0,
            float(center[1]) - height / 2.0,
        )
    )


def _target_in_center_window(target: dict, frame_shape) -> bool:
    """Require most of a target bbox to be inside the centered image window."""
    bbox = target.get("bbox")
    if bbox is None or len(bbox) != 4:
        return False
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    frame_height, frame_width = frame_shape[:2]
    window_size = min(
        float(TARGET_WINDOW_SIZE_PX), float(frame_width), float(frame_height)
    )
    window_left = (float(frame_width) - window_size) / 2.0
    window_top = (float(frame_height) - window_size) / 2.0
    intersection_width = max(
        0.0,
        min(x + width, window_left + window_size) - max(x, window_left),
    )
    intersection_height = max(
        0.0,
        min(y + height, window_top + window_size) - max(y, window_top),
    )
    inside_fraction = (intersection_width * intersection_height) / (width * height)
    return inside_fraction >= TARGET_WINDOW_MIN_AREA_FRACTION


def _nearest_letter_to_frame_center(letters: list[dict], frame_shape) -> dict:
    """Keep the letter block geometrically closest to the main-view center."""
    if not letters:
        raise ValueError("letters must not be empty")
    height, width = frame_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    return min(
        letters,
        key=lambda item: (
            (float(item.get("center", (center_x, center_y))[0]) - center_x) ** 2
            + (float(item.get("center", (center_x, center_y))[1]) - center_y) ** 2,
            -float(item.get("confidence", 0.0)),
        ),
    )


def _main_target_candidates(frame: np.ndarray, detector, field: str,
                             pair: tuple[str, str]) -> list[dict]:
    letters = [
        item for item in letter_detections(
            detector, frame, MAIN_LETTER_MIN_CONFIDENCE
        )
        if (
            str(item.get("letter", "")).upper() in pair
            and _target_in_center_window(item, frame.shape)
        )
    ]
    nearest_letter = (
        _nearest_letter_to_frame_center(letters, frame.shape)
        if letters else None
    )
    letters = [nearest_letter] if nearest_letter is not None else []
    rings = detect_rings(frame)
    return [
        item for item in letters + rings
        if (
            (
                item.get("kind") == "ring"
                and item.get("color") == field.lower()
            )
            or item.get("kind") == "letter"
        ) and _target_in_center_window(item, frame.shape)
    ]


def center_main_target(camera: cv2.VideoCapture, detector, field: str,
                       pair: tuple[str, str], reference: dict,
                       boards: ServoBoards) -> dict:
    """Track a confirmed target until centered using bounded servo corrections."""
    id2 = int(HIGH[2])
    id6 = int(HIGH[6])
    corrections = 0
    last_target = reference
    last_report = 0.0
    missing_started = None
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        candidates = [
            item for item in _main_target_candidates(frame, detector, field, pair)
            if _target_matches(reference, item, pair)
        ]
        if not candidates:
            now = time.monotonic()
            if missing_started is None:
                missing_started = now
            if now - last_report >= 1.0:
                print(
                    "TASK2 TARGET TRACK_WAIT target_temporarily_missing "
                    f"missing_s={now - missing_started:.1f}",
                    flush=True,
                )
                last_report = now
            if now - missing_started >= TARGET_TRACK_MISSING_TIMEOUT_S:
                print(
                    "TASK2 TARGET TRACK_TIMEOUT "
                    f"missing_s={now - missing_started:.1f}",
                    flush=True,
                )
                raise RuntimeError("TARGET_TRACK_TIMEOUT")
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise KeyboardInterrupt
            time.sleep(0.01)
            continue
        missing_started = None
        previous_center = np.asarray(last_target.get("center", ()), dtype=np.float32)
        if previous_center.shape == (2,):
            distances = [
                float(np.linalg.norm(
                    np.asarray(item.get("center", previous_center), dtype=np.float32)
                    - previous_center
                ))
                for item in candidates
            ]
            nearest_index = int(np.argmin(distances))
            target = (
                candidates[nearest_index]
                if distances[nearest_index] <= CENTER_TRACK_MAX_JUMP_PX
                else max(
                    candidates,
                    key=lambda item: float(
                        item.get("confidence", item.get("score", 0.0))
                    ),
                )
            )
        else:
            target = max(
                candidates,
                key=lambda item: float(item.get("confidence", item.get("score", 0.0))),
            )
        last_target = target
        height, width = frame.shape[:2]
        cx, cy = target.get("center", (width / 2.0, height / 2.0))
        error_x = float(cx) - width / 2.0
        error_y = float(cy) - height / 2.0
        if abs(error_x) <= CENTER_DEADBAND_PX and abs(error_y) <= CENTER_DEADBAND_PX:
            target["frame_shape"] = frame.shape
            target["center_id2"] = id2
            target["center_id6"] = id6
            print(
                f"TASK2 TARGET CENTERED dx={error_x:.0f} dy={error_y:.0f} "
                f"ID2={id2} ID6={id6} corrections={corrections}",
                flush=True,
            )
            return target
        next_id2 = id2
        next_id6 = id6
        if abs(error_x) > CENTER_DEADBAND_PX:
            next_id6 += (
                -CENTER_ID6_STEP_TICKS
                if error_x > 0.0 else CENTER_ID6_STEP_TICKS
            )
        if abs(error_y) > CENTER_DEADBAND_PX:
            next_id2 += -CENTER_STEP_TICKS if error_y > 0.0 else CENTER_STEP_TICKS
        next_id2 = max(CENTER_ID2_RANGE[0], min(CENTER_ID2_RANGE[1], next_id2))
        next_id6 = max(CENTER_ID6_RANGE[0], min(CENTER_ID6_RANGE[1], next_id6))
        if next_id2 == id2 and next_id6 == id6:
            target["frame_shape"] = frame.shape
            target["center_id2"] = id2
            target["center_id6"] = id6
            print(
                f"TASK2 TARGET CENTER_LIMIT proceed dx={error_x:.0f} dy={error_y:.0f} "
                f"ID2={id2} ID6={id6} corrections={corrections}",
                flush=True,
            )
            return target
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt
        id2, id6 = next_id2, next_id6
        boards.arm({2: id2, 6: id6}, CENTER_TIME_MS)
        corrections += 1
        print(
            f"TASK2 TARGET CENTER_STEP dx={error_x:.0f} dy={error_y:.0f} "
            f"ID2={id2} ID6={id6} dID6={CENTER_ID6_STEP_TICKS} "
            f"motion={CENTER_TIME_MS}ms",
            flush=True,
        )
        time.sleep(CENTER_TIME_MS / 1000.0)


def solve_descend_pose(target: dict, grasp_model):
    """Map fresh target depth through the measured 7-30 cm calibration."""
    distance_cm = target.get("distance_cm")
    if distance_cm is None:
        raise RuntimeError("TARGET_DEPTH_INVALID")
    try:
        id1, id2 = grasp_model(float(distance_cm))
    except ValueError as exc:
        raise RuntimeError(f"TARGET_DEPTH_INVALID {exc}") from exc
    return {
        "id1": int(id1),
        "id2": int(id2),
        "distance_cm": float(distance_cm),
        "model": "measured_7_30cm",
    }


def open_gripper_then_retreat_id2(boards: ServoBoards, current_id2: int) -> int:
    """Open ID7, then move only ID2 back before the final IK pose."""
    boards.open_gripper()
    retreat_id2 = max(CENTER_ID2_RANGE[0], int(current_id2) - POST_OPEN_ID2_RETREAT_TICKS)
    boards.arm({2: retreat_id2}, ARM_TIME_MS)
    print(
        f"TASK2 POST_OPEN_ID2_RETREAT ID2={current_id2}->{retreat_id2} "
        f"DELTA=-{POST_OPEN_ID2_RETREAT_TICKS} T={ARM_TIME_MS}ms",
        flush=True,
    )
    time.sleep(ARM_TIME_MS / 1000.0)
    return retreat_id2


def stop_active_h7(h7: H7Link | None, sequence: int | None,
                   field: str, active: bool) -> None:
    """Stop an acknowledged test command before returning control to the user."""
    if h7 is None or sequence is None or not active:
        return
    try:
        h7.send(
            f"RK,TEST,TASK2,STOP,SEQ,{sequence},FIELD,{field.upper()}"
        )
        reply = h7.wait_status_any(
            sequence, {"STOPPED", "DONE"}, 4.0, field.upper()
        )
        if reply is None:
            print(
                f"TASK2 H7 STOP_TIMEOUT seq={sequence} field={field.upper()}",
                flush=True,
            )
        else:
            print(
                f"TASK2 H7 STOP_CONFIRMED seq={sequence} field={field.upper()} "
                f"reply={reply}",
                flush=True,
            )
    except Exception as exc:
        print(f"TASK2 H7 STOP_ERROR {exc}", flush=True)


def run(args) -> int:
    if os.environ.get("DISPLAY") is None:
        os.environ["DISPLAY"] = ":0"
    secondary = main_camera = boards = h7 = None
    session_sequence = None
    h7_active = False
    try:
        main_detector = import_main_detector()
        secondary_detector = import_secondary_detector()
        white_line_detector = import_white_line_detector()
        grasp_model = import_grasp_model()
        secondary = open_camera(args.secondary_camera)
        pair = lock_pair(secondary, secondary_detector, args.letter_timeout_s)
        secondary.release()
        secondary = None
        if not args.execute_h7:
            print(f"TASK2 CAMERA_ONLY COMPLETE L1={pair[0]} L2={pair[1]}", flush=True)
            return 0
        boards = ServoBoards(args.arm_uart, args.zp_uart)
        boards.pose_high()
        print("TASK2 HIGH_POSE_READY ID1=650 ID2=600 ID6=350 ZP4=1200 ZP5=800 ZP7=1300", flush=True)
        h7 = H7Link(args.h7_device)
        session_sequence = (int(time.time() * 1000)) & 0xFFFFFFFF
        for slot in range(8):
            if slot == 0:
                command = (f"RK,TEST,TASK2,START,SEQ,{session_sequence},FIELD,{args.field.upper()},"
                           f"L1,{pair[0]},L2,{pair[1]}")
            else:
                command = (f"RK,TEST,TASK2,NEXT,SEQ,{session_sequence},FIELD,{args.field.upper()},"
                           f"STEP,{slot}")
            h7.send(command)
            if not h7.wait_status(
                    session_sequence, "ACK", 4.0, args.field.upper()):
                raise RuntimeError(f"H7_ACK_TIMEOUT slot={slot + 1}")
            h7_active = True
            if not h7.wait_status(
                    session_sequence, "RUNNING", 8.0, args.field.upper()):
                raise RuntimeError(f"H7_RUNNING_TIMEOUT slot={slot + 1}")
            if main_camera is None:
                main_camera = open_camera(args.main_camera)
                print(
                    f"TASK2 MAIN CAMERA_OPEN path={args.main_camera} "
                    "WHITE_LINE_REF_Y10=2000,TOL=100,ACCEL=0.10m/s2,"
                    "AFTER_CROSSED_FORWARD=170mm",
                    flush=True,
                )
            done = (
                h7.wait_status_with_white_line(
                    session_sequence, "DONE", args.h7_timeout_s,
                    main_camera, white_line_detector, args.field.upper(),
                )
                if slot == 0 else h7.wait_status(
                    session_sequence, "DONE", args.h7_timeout_s,
                    args.field.upper())
            )
            if not done:
                raise RuntimeError(f"H7_DONE_TIMEOUT slot={slot + 1}")
            flush_camera(main_camera, frames=8)
            print(
                f"TASK2 MAIN CAMERA_FRESH station={slot + 1} discarded=8",
                flush=True,
            )
            try:
                target = identify_main_target(
                    main_camera, main_detector, args.field, pair, args.main_timeout_s
                )
                if target.get("kind") in ("letter", "ring"):
                    target = center_main_target(
                        main_camera, main_detector, args.field, pair, target, boards
                    )
            except RuntimeError as exc:
                reason = str(exc)
                if reason.startswith((
                        "MAIN_TARGET_TIMEOUT",
                        "TARGET_DEPTH_INVALID",
                        "TARGET_TRACK_TIMEOUT",
                )):
                    if reason.startswith("TARGET_TRACK_TIMEOUT"):
                        boards.pose_high()
                        print(
                            f"TASK2 TARGET TRACK_RECOVER high_pose slot={slot + 1}",
                            flush=True,
                        )
                    print(
                        f"TASK2 SLOT_SKIP slot={slot + 1} reason={reason.split()[0]}",
                        flush=True,
                    )
                    target = {"kind": "skip", "label": reason.split()[0]}
                else:
                    raise
            if target.get("kind") == "ring":
                descend = solve_descend_pose(target, grasp_model)
                current_id2 = int(target.get("center_id2", HIGH[2]))
                open_gripper_then_retreat_id2(boards, current_id2)
                boards.arm({1: descend["id1"], 2: descend["id2"],
                            6: int(target.get("center_id6", HIGH[6]))})
                print(
                    f"TASK2 RING_DESCEND depth={descend['distance_cm']:.1f}cm "
                    f"ID1={descend['id1']} ID2={descend['id2']} "
                    f"model={descend['model']}",
                    flush=True,
                )
                time.sleep(ARM_TIME_MS / 1000.0)
                boards.close_gripper()
                boards.pose_high()
                boards.place_ring()
                boards.pulse_gripper()
                boards.pose_high()
                print("TASK2 RING_DONE after_high ID1=535 ID2=330 ID6=120", flush=True)
            elif (
                    target.get("kind") == "letter"
                    and str(target.get("letter", "")).upper() in pair
            ):
                descend = solve_descend_pose(target, grasp_model)
                current_id2 = int(target.get("center_id2", HIGH[2]))
                open_gripper_then_retreat_id2(boards, current_id2)
                boards.arm({1: descend["id1"], 2: descend["id2"],
                            6: int(target.get("center_id6", HIGH[6]))})
                print(
                    f"TASK2 LETTER_DESCEND letter={target['letter']} "
                    f"depth={descend['distance_cm']:.1f}cm "
                    f"ID1={descend['id1']} ID2={descend['id2']} "
                    f"model={descend['model']}",
                    flush=True,
                )
                time.sleep(ARM_TIME_MS / 1000.0)
                boards.close_gripper()
                boards.pose_high()
                boards.arm(LETTER_WORK)
                time.sleep(ARM_TIME_MS / 1000.0)
                boards.pulse_gripper()
                boards.pose_high()
                print(
                    f"TASK2 LETTER_DONE letter={target['letter']} "
                    "ID1=500 ID2=350 ID6=600", flush=True
                )
            else:
                print(
                    f"TASK2 SKIP target={target.get('kind', 'unknown')} "
                    f"field={args.field.upper()}", flush=True
                )
            print(
                f"TASK2 SLOT_DONE slot={slot + 1} seq={session_sequence} "
                f"target={target.get('letter', target.get('kind', 'unknown'))}",
                flush=True,
            )
        print(f"TASK2 TEST COMPLETE pair={pair[0]},{pair[1]} slots=8", flush=True)
        return 0
    except KeyboardInterrupt:
        stop_active_h7(h7, session_sequence, args.field, h7_active)
        print("TASK2 ABORTED", flush=True)
        return 2
    except Exception as exc:
        stop_active_h7(h7, session_sequence, args.field, h7_active)
        print(f"TASK2 ERROR {exc}", flush=True)
        return 1
    finally:
        if secondary is not None:
            secondary.release()
        if main_camera is not None:
            main_camera.release()
        if boards is not None:
            boards.close()
        if h7 is not None:
            h7.close()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="RoboCup task-two isolated test")
    parser.add_argument("--field", choices=("red", "blue"), required=True)
    parser.add_argument("--secondary-camera", default=SECONDARY_CAMERA)
    parser.add_argument("--main-camera", default=MAIN_CAMERA)
    parser.add_argument("--h7-device", default=H7_DEVICE)
    parser.add_argument("--arm-uart", default=ARM_DEVICE)
    parser.add_argument("--zp-uart", default=ZP_DEVICE)
    parser.add_argument("--execute-h7", action="store_true",
                        help="after letter lock, drive the arm and H7 test route")
    parser.add_argument("--letter-timeout-s", type=float, default=30.0)
    parser.add_argument("--main-timeout-s", type=float, default=MAIN_TARGET_TIMEOUT_S)
    parser.add_argument("--h7-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
