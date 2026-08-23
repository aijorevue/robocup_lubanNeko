#!/usr/bin/env python3
"""Standalone task-two test: secondary letter preselection, H7 route, main target.

The app owns both camera sessions and the two direct servo-board UARTs.  It
does not start the formal ROS service.  H7 owns only the fixed chassis test
route and receives one explicit TEST,TASK2 command after the high arm pose is
sent.
"""

from __future__ import annotations

import argparse
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

HIGH = {1: 600, 2: 600, 6: 640}
LETTER_WORK = {1: 500, 2: 350, 6: 900}
RING_AFTER_HIGH = {1: 460, 2: 315, 6: 400}
ZP_HIGH = {4: 1200, 5: 800, 7: 1300}
# These two SG90 outputs are fixed auxiliary channels on the ZL 24-channel
# board.  They must remain at their neutral test positions throughout the app.
AUX_ZP_HOLD = {12: 600, 23: 1000}
AUX_ZP_HOLD_TIME_MS = 800
GRIPPER_CLOSED = 1300
GRIPPER_OPEN = 1700
ARM_TIME_MS = 600
ZP_TIME_MS = 300
GRIPPER_TIME_MS = 300
CENTER_DEADBAND_PX = 45.0
CENTER_STEP_TICKS = 5
CENTER_TIME_MS = 100
CENTER_TRACK_MAX_JUMP_PX = 320.0
CENTER_ID2_RANGE = (450, 700)
CENTER_ID6_RANGE = (500, 800)
LETTERS = {"A", "B", "C", "D"}
SECONDARY_LETTER_MIN_CONFIDENCE = 38.0
MAIN_LETTER_MIN_CONFIDENCE = 45.0
MAIN_TARGET_VOTE_WINDOW = 8
MAIN_TARGET_REQUIRED_VOTES = 3
MAIN_TARGET_VOTE_CENTER_TOL_PX = 140.0
TARGET_TRACK_MISSING_TIMEOUT_S = 4.0
LETTER_LOCK_WARMUP_FRAMES = 20
LETTER_LOCK_HISTORY_FRAMES = 12
LETTER_LOCK_REQUIRED_FRAMES = 8
LETTER_SIZE_MM = 30.0
RING_SIZE_MM = 55.0
CAMERA_GRIPPER_OFFSET_MM = 50.0
TARGET_GRIPPER_DISTANCE_MM = 20.0
RING_DISTANCE_EXTRA_CM = 0.5


def htd85_packet(servo_id: int, position: int, time_ms: int) -> bytes:
    body = bytes((servo_id, 7, 1, position & 0xFF, position >> 8,
                  time_ms & 0xFF, time_ms >> 8))
    return b"\x55\x55" + body + bytes(((~sum(body)) & 0xFF,))


def zp_packet(servo_id: int, position: int, time_ms: int) -> bytes:
    return f"#{servo_id:03d}P{position:04d}T{time_ms:04d}!".encode("ascii")


def write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
    termios.tcdrain(fd)


def open_uart(path: str) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[4] = attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


class ServoBoards:
    def __init__(self, arm_path: str, zp_path: str):
        self.arm_path = arm_path
        self.zp_path = zp_path
        self.arm_fd = open_uart(arm_path)
        try:
            self.zp_fd = open_uart(zp_path)
        except Exception:
            os.close(self.arm_fd)
            raise
        self.hold_aux_outputs()

    def arm(self, targets: dict[int, int], motion_ms: int = ARM_TIME_MS) -> None:
        for servo_id, position in targets.items():
            packet = htd85_packet(servo_id, position, motion_ms)
            write_all(self.arm_fd, packet)
            print(f"TASK2 SERVO HTD85 ID{servo_id}={position} T={motion_ms}ms", flush=True)
            time.sleep(0.003)

    def zp(self, targets: dict[int, int], motion_ms: int = ZP_TIME_MS) -> None:
        for servo_id, position in targets.items():
            packet = zp_packet(servo_id, position, motion_ms)
            write_all(self.zp_fd, packet)
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

    def close(self) -> None:
        for fd in (getattr(self, "arm_fd", None), getattr(self, "zp_fd", None)):
            if fd is not None:
                os.close(fd)


class H7Link:
    def __init__(self, path: str):
        self.path = path
        self.fd = None
        self.rx = bytearray()
        self.last_white_line_measurement = None
        self.last_white_line_time = 0.0

    def open(self) -> None:
        if not os.path.exists(self.path):
            raise RuntimeError(f"H7 CDC missing: {self.path}")
        self.fd = open_uart(self.path)
        print(f"TASK2 H7 OPEN {self.path}", flush=True)

    def send(self, line: str) -> None:
        if self.fd is None:
            self.open()
        write_all(self.fd, (line.rstrip("\r\n") + "\r\n").encode("ascii"))
        print(f"TASK2 H7 TX {line.rstrip()}", flush=True)

    def wait_status(self, sequence: int, status: str, timeout_s: float) -> str | None:
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
                fields = line.split(",")
                if (len(fields) >= 8 and fields[:4] ==
                        ["H7", "TEST", "TASK2", status] and
                        fields[4] == "SEQ" and fields[5] == str(sequence)):
                    return line
        return None

    def wait_status_with_white_line(
        self, sequence: int, status: str, timeout_s: float,
        camera: cv2.VideoCapture, white_line_detector,
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
                if (len(fields) >= 6 and fields[:4] ==
                        ["H7", "TEST", "TASK2", status] and
                        fields[4] == "SEQ" and fields[5] == str(sequence)):
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
            if str(item.get("letter", "")).upper() in pair
        ]
        letters = (
            [_nearest_letter_to_frame_center(allowed_letters, frame.shape)]
            if allowed_letters else []
        )
        rings = detect_rings(frame)
        own_rings = [d for d in rings if d["color"] == field.lower()]
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
        if str(item.get("letter", "")).upper() in pair
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
            item.get("kind") == "ring" and item.get("color") == field.lower()
        ) or item.get("kind") == "letter"
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
            next_id6 += -CENTER_STEP_TICKS if error_x > 0.0 else CENTER_STEP_TICKS
        if abs(error_y) > CENTER_DEADBAND_PX:
            next_id2 += -CENTER_STEP_TICKS if error_y > 0.0 else CENTER_STEP_TICKS
        next_id2 = max(CENTER_ID2_RANGE[0], min(CENTER_ID2_RANGE[1], next_id2))
        next_id6 = max(CENTER_ID6_RANGE[0], min(CENTER_ID6_RANGE[1], next_id6))
        if next_id2 == id2 and next_id6 == id6:
            target["frame_shape"] = frame.shape
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
            f"ID2={id2} ID6={id6} step={CENTER_STEP_TICKS} "
            f"motion={CENTER_TIME_MS}ms",
            flush=True,
        )
        time.sleep(CENTER_TIME_MS / 1000.0)


def solve_descend_pose(target: dict, grasp_model):
    """Map fresh target depth through the measured 10-30 cm calibration."""
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
        "model": "measured_10_30cm",
    }


def run(args) -> int:
    if os.environ.get("DISPLAY") is None:
        os.environ["DISPLAY"] = ":0"
    secondary = main_camera = boards = h7 = None
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
        print("TASK2 HIGH_POSE_READY ID1=600 ID2=600 ID6=640 ZP4=1200 ZP5=800 ZP7=1300", flush=True)
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
            if not h7.wait_status(session_sequence, "ACK", 4.0):
                raise RuntimeError(f"H7_ACK_TIMEOUT slot={slot + 1}")
            if not h7.wait_status(session_sequence, "RUNNING", 8.0):
                raise RuntimeError(f"H7_RUNNING_TIMEOUT slot={slot + 1}")
            if main_camera is None:
                main_camera = open_camera(args.main_camera)
                print(
                    f"TASK2 MAIN CAMERA_OPEN path={args.main_camera} "
                    "WHITE_LINE_REF_Y10=2000,TOL=100,ACCEL=0.10m/s2,"
                    "AFTER_CROSSED_FORWARD=210mm",
                    flush=True,
                )
            done = (
                h7.wait_status_with_white_line(
                    session_sequence, "DONE", args.h7_timeout_s,
                    main_camera, white_line_detector,
                )
                if slot == 0 else
                h7.wait_status(session_sequence, "DONE", args.h7_timeout_s)
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
                boards.open_gripper()
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
                boards.arm(RING_AFTER_HIGH)
                time.sleep(ARM_TIME_MS / 1000.0)
                boards.pulse_gripper()
                boards.pose_high()
                print("TASK2 RING_DONE after_high ID1=460 ID2=315 ID6=400", flush=True)
            elif (
                    target.get("kind") == "letter"
                    and str(target.get("letter", "")).upper() in pair
            ):
                descend = solve_descend_pose(target, grasp_model)
                boards.open_gripper()
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
                    "ID1=500 ID2=350 ID6=900", flush=True
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
        print("TASK2 ABORTED", flush=True)
        return 2
    except Exception as exc:
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
    parser.add_argument("--main-timeout-s", type=float, default=4.0)
    parser.add_argument("--h7-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
