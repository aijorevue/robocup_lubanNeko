#!/usr/bin/env python3
"""Standalone task-three test: main-camera letter pickup while orbiting.

The H7 owns a closed-loop 400 mm radius, 360 degree orbit. RK owns the
main-camera ABCD detector and pauses the orbit only for a confirmed letter.
Arm motion reuses the task-two direct servo-board path and the measured IK
model from the deployed workspace.
"""

from __future__ import annotations

import argparse
import copy
import os
import select
import sys
import time
from collections import deque

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Keep one implementation of the tested servo framing and arm behavior.
from task2_secondary_h7_test_app import (  # noqa: E402
    ARM_TIME_MS,
    CENTER_DEADBAND_PX,
    CENTER_ID2_RANGE,
    CENTER_ID6_RANGE,
    CENTER_STEP_TICKS,
    CENTER_TIME_MS,
    CENTER_TRACK_MAX_JUMP_PX,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GRIPPER_TIME_MS,
    HIGH,
    LETTER_WORK,
    LETTERS,
    MAIN_LETTER_MIN_CONFIDENCE,
    TARGET_TRACK_MISSING_TIMEOUT_S,
    ZP_HIGH,
    ZP_TIME_MS,
    ServoBoards,
    import_main_detector,
    letter_detections,
    open_camera,
)


MAIN_CAMERA = "/dev/v4l/by-path/platform-fc800000.usb-usb-0:1:1.0-video-index0"
H7_DEVICE = "/dev/h7_chassis"
ARM_DEVICE = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82109853-if00"
ZP_DEVICE = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"

TARGET_VOTE_WINDOW = 10
# One valid observation inside the capture window is sufficient to stop H7.
# The orbit is already moving slowly near the center, and waiting for another
# frame can carry a target past the gripper axis.
TARGET_REQUIRED_VOTES = 1
# A moving target can shift substantially between two detector frames. Keep
# the main-camera detector strict, but do not reject a valid track because the
# chassis moved the target more than the task-two stationary tolerance.
TARGET_TRACK_MAX_JUMP_PX = max(320.0, CENTER_TRACK_MAX_JUMP_PX)
TARGET_RECHECK_DELAY_S = 0.15
ORBIT_RESCUE_INTERVAL_FRAMES = 3
ORBIT_RESCUE_MIN_CONFIDENCE = 38.0
TASK3_LETTERS = tuple(sorted(LETTERS))
TASK3_ORBIT_RADIUS_MM = 400
TASK3_ORBIT_ANGLE_DEG = 360
TASK3_LETTER_MIN_CONFIDENCE = max(40.0, MAIN_LETTER_MIN_CONFIDENCE - 5.0)
TASK3_HIGH = {1: 600, 2: 500, 6: 640}
# The pickup depth is relative to the final centered pose. Do not reuse the
# removed fixed task-two/task-three descent model here.
TASK3_DESCEND_OFFSET_TICKS = 100
# Candidates may be tracked while they cross the image, but a target is
# actionable only inside the center window below. This prevents a side target
# from pausing the orbit before it reaches the gripper axis.
TASK3_MAX_TRACK_CENTER_DISTANCE_PX = 360.0
TASK3_CAPTURE_CENTER_TOL_X_PX = 90.0
TASK3_CAPTURE_CENTER_TOL_Y_PX = 90.0
# Keep the single actionable observation inside the capture window; do not
# combine an old edge observation with a new centered observation.
TASK3_TARGET_VOTE_CENTER_TOL_PX = 90.0
# Keep the task-three-only rescue path from promoting weak background shapes to
# D. Task two and the shared detector thresholds remain unchanged.
TASK3_D_MIN_CONFIDENCE = 55.0

# Task three's station has a green surface, a white square, and a black
# Times New Roman glyph. These thresholds are task-three-only and leave the
# shared detector used by task two and the formal route untouched.
TASK3_GREEN_HUE_RANGE = (35, 105)
TASK3_GREEN_MIN_SATURATION = 40
TASK3_GREEN_MIN_VALUE = 25
TASK3_WHITE_MIN_VALUE = 115
TASK3_WHITE_MAX_SATURATION = 135
TASK3_WHITE_PATCH_MIN_AREA = 700.0
TASK3_WHITE_PATCH_MAX_AREA_RATIO = 0.12
TASK3_WHITE_PATCH_MIN_SIDE = 28.0
TASK3_WHITE_PATCH_MAX_SIDE_RATIO = 0.45
TASK3_WHITE_PATCH_MIN_RECTANGULARITY = 0.55
TASK3_WHITE_PATCH_MIN_SOLIDITY = 0.72
TASK3_WHITE_PATCH_MIN_GREEN_SUPPORT = 0.42
TASK3_GREEN_LETTER_MIN_CONFIDENCE = 0.42


class Task3H7Link:
    """Sequence-aware line transport for the isolated TASK3 protocol."""

    def __init__(self, path: str):
        self.path = path
        self.fd = None
        self.rx = bytearray()

    def open(self) -> None:
        if not os.path.exists(self.path):
            raise RuntimeError(f"H7 CDC missing: {self.path}")
        # Importing the tested UART setup avoids pyserial ownership changes.
        from task2_secondary_h7_test_app import open_uart

        self.fd = open_uart(self.path)
        print(f"TASK3 H7 OPEN {self.path}", flush=True)

    def send(self, line: str) -> None:
        if self.fd is None:
            self.open()
        payload = (line.rstrip("\r\n") + "\r\n").encode("ascii")
        offset = 0
        while offset < len(payload):
            offset += os.write(self.fd, payload[offset:])
        termios_drain(self.fd)
        print(f"TASK3 H7 TX {line.rstrip()}", flush=True)

    def _read_lines(self, timeout_s: float) -> list[str]:
        if self.fd is None:
            return []
        ready, _, _ = select.select([self.fd], [], [], timeout_s)
        if not ready:
            return []
        self.rx.extend(os.read(self.fd, 4096))
        lines = []
        while b"\n" in self.rx:
            raw, _, self.rx = self.rx.partition(b"\n")
            line = raw.decode("ascii", "replace").strip("\r")
            if line:
                print(f"TASK3 H7 RX {line}", flush=True)
                lines.append(line)
        return lines

    def poll(self) -> list[str]:
        """Read already available H7 lines without delaying camera capture."""
        return self._read_lines(0.0)

    @staticmethod
    def _matches(line: str, status: str, sequence: int) -> bool:
        fields = [item.strip() for item in line.split(",")]
        return (
            len(fields) >= 6
            and fields[:4] == ["H7", "TEST", "TASK3", status]
            and fields[4] == "SEQ"
            and fields[5] == str(sequence)
        )

    def wait_status(self, sequence: int, status: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for line in self._read_lines(min(0.1, max(0.0, deadline - time.monotonic()))):
                if self._matches(line, status, sequence):
                    return line
                if self._matches(line, "ERR", sequence):
                    raise RuntimeError(f"H7_{status}_ERR {line}")
        raise RuntimeError(f"H7_{status}_TIMEOUT")

    def pause_until_confirmed(
        self, sequence: int, field: str, timeout_s: float
    ) -> str:
        """Retry PAUSE until H7 confirms that the motors are stopped.

        The ACK is useful diagnostics, but PAUSED is the safety point at which
        RK may move the arm. Retrying also covers a CDC packet lost while H7
        was transmitting an unrelated status line.
        """
        deadline = time.monotonic() + timeout_s
        next_send = 0.0
        command = f"RK,TEST,TASK3,PAUSE,SEQ,{sequence},FIELD,{field}"
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                self.send(command)
                next_send = now + 0.25
            for line in self._read_lines(
                min(0.05, max(0.0, deadline - time.monotonic()))
            ):
                if self._matches(line, "PAUSED", sequence):
                    return line
                if self._matches(line, "ERR", sequence):
                    raise RuntimeError(f"H7_PAUSE_ERR {line}")
        raise RuntimeError(f"H7_PAUSED_TIMEOUT retries=PAUSE")

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def termios_drain(fd: int) -> None:
    """Drain a raw UART without importing termios at module import time."""
    import termios

    termios.tcdrain(fd)


def _motion_enhanced_frame(frame: np.ndarray) -> np.ndarray:
    """Recover local black-glyph contrast from short motion-blurred frames."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        lightness
    )
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0.0)


def _configure_orbit_rescue_detector(detector):
    """Relax only the orbit rescue pass; task-two main detection stays strict."""
    for name, limit in (
        ("min_confidence", 0.38),
        ("dark_letter_min_confidence", 0.38),
    ):
        if hasattr(detector, name):
            try:
                setattr(detector, name, min(float(getattr(detector, name)), limit))
            except (TypeError, ValueError):
                pass
    return detector


def _classify_task3_white_patch(
    frame: np.ndarray, contour, detector, bbox: tuple[int, int, int, int]
) -> tuple[str | None, float, float]:
    """Classify the black glyph inside a white station patch.

    The generic detector searches the whole frame and can lose this target when
    the green station merges with the camera background. The task-three path
    first isolates the white patch, then gives the tight patch to the detector's
    black-glyph classifier. A rectified retry handles a rotated patch.
    """
    x, y, width, height = bbox
    candidates: list[tuple[str | None, float, float]] = []
    tight = frame[y : y + height, x : x + width]
    classify_dark = getattr(detector, "_classify_dark_letter", None)
    if callable(classify_dark) and tight.size:
        try:
            candidates.append(classify_dark(tight))
        except (cv2.error, TypeError, ValueError):
            pass

    rectify = getattr(detector, "_rectify", None)
    ordered_box = getattr(detector, "_ordered_box", None)
    if callable(rectify) and callable(ordered_box):
        try:
            rect = cv2.minAreaRect(contour)
            box = ordered_box(cv2.boxPoints(rect))
            rectified = rectify(frame, box, 160)
            if callable(classify_dark):
                candidates.append(classify_dark(rectified))
            classify_card = getattr(detector, "_classify", None)
            if callable(classify_card):
                candidates.append(classify_card(rectified))
        except (cv2.error, TypeError, ValueError):
            pass

    valid = [
        item for item in candidates
        if item[0] in TASK3_LETTERS
        and float(item[1]) >= TASK3_GREEN_LETTER_MIN_CONFIDENCE
        and 0.025 <= float(item[2]) <= 0.58
    ]
    if not valid:
        return None, 0.0, 0.0
    return max(valid, key=lambda item: float(item[1]))


def _task3_green_white_letter_detections(
    frame: np.ndarray, detector
) -> list[dict]:
    """Find letters on white patches supported by the green station surface."""
    if frame is None or frame.ndim != 3:
        return []
    height, width = frame.shape[:2]
    frame_area = float(max(1, height * width))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array(
            (
                TASK3_GREEN_HUE_RANGE[0],
                TASK3_GREEN_MIN_SATURATION,
                TASK3_GREEN_MIN_VALUE,
            ),
            dtype=np.uint8,
        ),
        np.array((TASK3_GREEN_HUE_RANGE[1], 255, 255), dtype=np.uint8),
    )
    white = cv2.inRange(
        hsv,
        np.array((0, 0, TASK3_WHITE_MIN_VALUE), dtype=np.uint8),
        np.array((180, TASK3_WHITE_MAX_SATURATION, 255), dtype=np.uint8),
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    detections: list[dict] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < TASK3_WHITE_PATCH_MIN_AREA:
            continue
        if area > frame_area * TASK3_WHITE_PATCH_MAX_AREA_RATIO:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if min(box_width, box_height) < TASK3_WHITE_PATCH_MIN_SIDE:
            continue
        if max(box_width, box_height) > min(width, height) * TASK3_WHITE_PATCH_MAX_SIDE_RATIO:
            continue
        aspect = box_width / float(max(1, box_height))
        if not 0.55 <= aspect <= 1.80:
            continue
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        rect_area = max(1.0, float(rect_width * rect_height))
        rectangularity = area / rect_area
        if rectangularity < TASK3_WHITE_PATCH_MIN_RECTANGULARITY:
            continue
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        if area / hull_area < TASK3_WHITE_PATCH_MIN_SOLIDITY:
            continue

        pad = max(10, int(round(max(box_width, box_height) * 0.18)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + box_width + pad), min(height, y + box_height + pad)
        expanded_green = green[y0:y1, x0:x1]
        green_support = cv2.countNonZero(expanded_green) / float(
            max(1, expanded_green.size)
        )
        if green_support < TASK3_WHITE_PATCH_MIN_GREEN_SUPPORT:
            continue

        letter, confidence, occupancy = _classify_task3_white_patch(
            frame, contour, detector, (x, y, box_width, box_height)
        )
        if letter is None:
            continue
        detections.append(
            {
                "kind": "letter",
                "letter": letter,
                "color": "green_station_white_patch",
                "source": "task3_green_white_black_letter",
                "confidence": round(float(confidence) * 100.0, 1),
                "glyph_occupancy": round(float(occupancy), 4),
                "center": (
                    int(round(x + box_width / 2.0)),
                    int(round(y + box_height / 2.0)),
                ),
                "box": contour.astype(np.int32),
                "bbox": (int(x), int(y), int(box_width), int(box_height)),
                "projected_area": max(area, rect_area),
                "fully_visible": (
                    x > 2 and y > 2 and x + box_width < width - 2
                    and y + box_height < height - 2
                ),
                "angle": round(float(rect[2]), 1),
            }
        )
    return detections


def _merge_letter_candidates(candidates: list[dict]) -> list[dict]:
    """Keep the strongest observation of each nearby letter candidate."""
    merged = []
    for candidate in sorted(
        candidates,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    ):
        label = str(candidate.get("letter", "")).upper()
        center = np.asarray(candidate.get("center", (0, 0)), dtype=np.float32)
        duplicate = any(
            label == str(existing.get("letter", "")).upper()
            and np.linalg.norm(
                center - np.asarray(existing.get("center", (0, 0)), dtype=np.float32)
            ) <= 0.35 * max(
                float(candidate.get("bbox", (0, 0, 1, 1))[2]),
                float(candidate.get("bbox", (0, 0, 1, 1))[3]),
                float(existing.get("bbox", (0, 0, 1, 1))[2]),
                float(existing.get("bbox", (0, 0, 1, 1))[3]),
            )
            for existing in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


def _letter_candidates(
    frame: np.ndarray,
    detector,
    grabbed: set[str],
    min_confidence: float,
    rescue_detector=None,
    frame_index: int = 0,
) -> list[dict]:
    # The task-three station geometry is authoritative: a real target is the
    # black glyph inside a white patch on the green surface. When that pass
    # finds a patch, do not fuse whole-frame candidates back in, because a
    # background edge can otherwise win nearest-to-center selection as a
    # different letter. The generic detector remains the fallback for motion
    # blur or frames where the patch is temporarily not segmented.
    station_detections = _task3_green_white_letter_detections(frame, detector)
    detections = (
        station_detections
        if station_detections
        else letter_detections(detector, frame, min_confidence)
    )
    rescue = []
    # A moving target may be blurred without making the strict pass empty:
    # run a bounded rescue pass periodically and fuse it with the strict one.
    if rescue_detector is not None and (
        not detections or frame_index % ORBIT_RESCUE_INTERVAL_FRAMES == 0
    ):
        rescue = letter_detections(
            rescue_detector,
            _motion_enhanced_frame(frame),
            ORBIT_RESCUE_MIN_CONFIDENCE,
        )
    detections = _merge_letter_candidates(detections + rescue)
    height, width = frame.shape[:2]
    frame_center = np.asarray((width / 2.0, height / 2.0), dtype=np.float32)

    def task3_actionable(item: dict) -> bool:
        center = np.asarray(item.get("center", frame_center), dtype=np.float32)
        center_distance = float(np.linalg.norm(center - frame_center))
        if center_distance > TASK3_MAX_TRACK_CENTER_DISTANCE_PX:
            return False
        label = str(item.get("letter", "")).upper()
        if label == "D" and float(item.get("confidence", 0.0)) < TASK3_D_MIN_CONFIDENCE:
            return False
        return True

    return [
        item for item in detections
        if str(item.get("letter", "")).upper() in TASK3_LETTERS
        and str(item.get("letter", "")).upper() not in grabbed
        and task3_actionable(item)
    ]


def _nearest_target(candidates: list[dict], frame_shape) -> dict | None:
    if not candidates:
        return None
    height, width = frame_shape[:2]
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float32)
    return min(
        candidates,
        key=lambda item: float(
            np.linalg.norm(
                np.asarray(item.get("center", center), dtype=np.float32) - center
            )
        ),
    )


def _draw_view(frame: np.ndarray, candidates: list[dict], grabbed: set[str], state: str):
    view = frame.copy()
    for item in candidates:
        x, y = map(int, item.get("center", (0, 0)))
        label = str(item.get("letter", "?")).upper()
        cv2.circle(view, (x, y), 8, (0, 255, 0), 2)
        cv2.putText(
            view, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2,
        )
    cv2.putText(
        view, f"TASK3 MAIN / {state} / GRABBED={','.join(sorted(grabbed)) or '-'}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
    )
    return view


def _stable_target(
    candidates: list[dict],
    frame_shape,
    history: deque,
) -> dict | None:
    target = _nearest_target(candidates, frame_shape)
    if target is None:
        # An edge target that disappears must not leave enough stale votes to
        # pause and grab it later. The orbit keeps running and starts a fresh
        # confirmation when another target reaches the center window.
        history.clear()
        return None

    key = str(target.get("letter", "")).upper()
    center = tuple(map(int, target.get("center", (0, 0))))
    height, width = frame_shape[:2]
    error_x = float(center[0]) - width / 2.0
    error_y = float(center[1]) - height / 2.0
    if (
        abs(error_x) > TASK3_CAPTURE_CENTER_TOL_X_PX
        or abs(error_y) > TASK3_CAPTURE_CENTER_TOL_Y_PX
    ):
        # Recognition while moving is only a track hint. Do not pause until
        # the same letter is actually in the gripper's central capture area.
        history.clear()
        return None
    # Require genuinely consecutive observations of the same target. A
    # filtered vote over an old deque could otherwise confirm a target after
    # another letter had already crossed the capture window.
    if history:
        previous_key, previous_center, _ = history[-1]
        if (
            previous_key != key
            or previous_center is None
            or np.linalg.norm(
                np.asarray(previous_center, dtype=np.float32)
                - np.asarray(center, dtype=np.float32)
            ) > TASK3_TARGET_VOTE_CENTER_TOL_PX
        ):
            history.clear()
    history.append((key, center, dict(target)))
    if len(history) < TARGET_REQUIRED_VOTES:
        return None
    return max((item for item_key, item_center, item in history),
               key=lambda item: float(item.get("confidence", 0.0)))


def _letter_grasp(boards: ServoBoards, target: dict) -> None:
    """Execute the task-three letter grasp sequence at a paused orbit station."""
    boards.open_gripper()
    centered_id1 = int(target.get("center_id1", TASK3_HIGH[1]))
    centered_id2 = int(target.get("center_id2", TASK3_HIGH[2]))
    descend = {
        "id1": max(0, centered_id1 - TASK3_DESCEND_OFFSET_TICKS),
        "id2": max(0, centered_id2 - TASK3_DESCEND_OFFSET_TICKS),
    }
    boards.arm(
        {
            1: descend["id1"],
            2: descend["id2"],
            6: int(target.get("center_id6", TASK3_HIGH[6])),
        }
    )
    print(
        f"TASK3 LETTER_DESCEND letter={target['letter']} "
        f"CENTER_ID1={centered_id1} CENTER_ID2={centered_id2} "
        f"ID1={descend['id1']} ID2={descend['id2']} "
        f"OFFSET={TASK3_DESCEND_OFFSET_TICKS}",
        flush=True,
    )
    time.sleep(ARM_TIME_MS / 1000.0)
    boards.close_gripper()
    _task3_pose_high(boards)
    boards.arm(LETTER_WORK)
    time.sleep(ARM_TIME_MS / 1000.0)
    boards.open_gripper()
    boards.close_gripper()
    _task3_pose_high(boards)
    print(
        "TASK3 LETTER_DONE "
        f"letter={target['letter']} ID1={LETTER_WORK[1]} "
        f"ID2={LETTER_WORK[2]} ID6={LETTER_WORK[6]} "
        f"GRIPPER={GRIPPER_OPEN}->{GRIPPER_CLOSED}",
        flush=True,
    )


def _task3_pose_high(boards: ServoBoards) -> None:
    """Use task-three's independent high pose without changing task two."""
    boards.arm(TASK3_HIGH)
    boards.zp(ZP_HIGH)
    time.sleep(max(ARM_TIME_MS, ZP_TIME_MS) / 1000.0)


def _center_task3_target(
    camera: cv2.VideoCapture,
    detector,
    rescue_detector,
    reference: dict,
    boards: ServoBoards,
) -> dict:
    """Re-center a paused target with the same rescue path used in orbit."""
    id1 = int(TASK3_HIGH[1])
    id2 = int(TASK3_HIGH[2])
    id6 = int(TASK3_HIGH[6])
    corrections = 0
    frame_index = 0
    last_target = reference
    missing_started = None
    reference_label = str(reference.get("letter", "")).upper()

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        candidates = [
            item for item in _letter_candidates(
                frame,
                detector,
                set(),
                TASK3_LETTER_MIN_CONFIDENCE,
                rescue_detector,
                frame_index,
            )
            if str(item.get("letter", "")).upper() == reference_label
        ]
        frame_index += 1
        if not candidates:
            now = time.monotonic()
            if missing_started is None:
                missing_started = now
            if now - missing_started >= TARGET_TRACK_MISSING_TIMEOUT_S:
                print(
                    "TASK3 TARGET TRACK_TIMEOUT "
                    f"missing_s={now - missing_started:.1f}",
                    flush=True,
                )
                raise RuntimeError("TARGET_TRACK_TIMEOUT")
            continue

        missing_started = None
        previous_center = np.asarray(
            last_target.get("center", ()), dtype=np.float32
        )
        if previous_center.shape == (2,):
            distances = [
                float(
                    np.linalg.norm(
                        np.asarray(item.get("center", previous_center), dtype=np.float32)
                        - previous_center
                    )
                )
                for item in candidates
            ]
            nearest_index = int(np.argmin(distances))
            target = (
                candidates[nearest_index]
                if distances[nearest_index] <= TARGET_TRACK_MAX_JUMP_PX
                else max(
                    candidates,
                    key=lambda item: float(item.get("confidence", 0.0)),
                )
            )
        else:
            target = max(
                candidates,
                key=lambda item: float(item.get("confidence", 0.0)),
            )
        last_target = target

        height, width = frame.shape[:2]
        cx, cy = target.get("center", (width / 2.0, height / 2.0))
        error_x = float(cx) - width / 2.0
        error_y = float(cy) - height / 2.0
        if abs(error_x) <= CENTER_DEADBAND_PX and abs(error_y) <= CENTER_DEADBAND_PX:
            target["frame_shape"] = frame.shape
            target["center_id1"] = id1
            target["center_id2"] = id2
            target["center_id6"] = id6
            print(
                f"TASK3 TARGET CENTERED dx={error_x:.0f} dy={error_y:.0f} "
                f"ID1={id1} ID2={id2} ID6={id6} corrections={corrections}",
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
            target["center_id1"] = id1
            target["center_id2"] = id2
            target["center_id6"] = id6
            print(
                f"TASK3 TARGET CENTER_LIMIT dx={error_x:.0f} dy={error_y:.0f} "
                f"ID1={id1} ID2={id2} ID6={id6} corrections={corrections}",
                flush=True,
            )
            return target

        id2, id6 = next_id2, next_id6
        boards.arm({2: id2, 6: id6}, CENTER_TIME_MS)
        corrections += 1
        print(
            f"TASK3 TARGET CENTER_STEP dx={error_x:.0f} dy={error_y:.0f} "
            f"ID2={id2} ID6={id6} step={CENTER_STEP_TICKS} "
            f"motion={CENTER_TIME_MS}ms",
            flush=True,
        )
        time.sleep(CENTER_TIME_MS / 1000.0)


def run(args) -> int:
    camera = boards = h7 = None
    started = False
    completed = False
    sequence = (int(time.time() * 1000.0)) & 0xFFFFFFFF
    grabbed: set[str] = set()
    history: deque = deque(maxlen=TARGET_VOTE_WINDOW)
    try:
        detector = import_main_detector()
        orbit_rescue_detector = _configure_orbit_rescue_detector(
            copy.copy(detector)
        )
        camera = open_camera(args.main_camera)
        print(
            f"TASK3 MAIN CAMERA_OPEN path={args.main_camera} "
            "ROLE=MAIN ONLY",
            flush=True,
        )
        if not args.execute_h7:
            print("TASK3 CAMERA_ONLY READY detector=ABCD", flush=True)
            deadline = time.monotonic() + args.camera_timeout_s
            while time.monotonic() < deadline:
                ok, frame = camera.read()
                if not ok or frame is None:
                    continue
                candidates = _letter_candidates(frame, detector, set(), MAIN_LETTER_MIN_CONFIDENCE)
                cv2.imshow(
                    "Task3 main",
                    _draw_view(frame, candidates, set(), "CAMERA_ONLY"),
                )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            return 0

        boards = ServoBoards(args.arm_uart, args.zp_uart)
        _task3_pose_high(boards)
        print(
            "TASK3 ARM_HIGH ID1=600 ID2=500 ID6=640 "
            "ZP4=1200 ZP5=800 ZP7=1300",
            flush=True,
        )
        h7 = Task3H7Link(args.h7_device)
        h7.open()
        field = args.field.upper()
        h7.send(
            f"RK,TEST,TASK3,START,SEQ,{sequence},FIELD,{field},"
            f"RADIUS_MM,{TASK3_ORBIT_RADIUS_MM},"
            f"ANGLE_DEG,{TASK3_ORBIT_ANGLE_DEG}"
        )
        h7.wait_status(sequence, "ACK", args.ack_timeout_s)
        h7.wait_status(sequence, "RUNNING", args.running_timeout_s)
        started = True
        print(
            f"TASK3 ORBIT_STARTED radius={TASK3_ORBIT_RADIUS_MM}mm "
            f"angle={TASK3_ORBIT_ANGLE_DEG}deg field={field}",
            flush=True,
        )

        orbit_frame_index = 0
        while True:
            for line in h7.poll():
                if h7._matches(line, "DONE", sequence):
                    completed = True
                    break
                if h7._matches(line, "ERR", sequence):
                    raise RuntimeError(f"H7_ORBIT_ERR {line}")
            if completed:
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            candidates = _letter_candidates(
                frame,
                detector,
                grabbed,
                TASK3_LETTER_MIN_CONFIDENCE,
                orbit_rescue_detector,
                orbit_frame_index,
            )
            orbit_frame_index += 1
            target = _stable_target(candidates, frame.shape, history)
            cv2.imshow(
                "Task3 main",
                _draw_view(frame, candidates, grabbed, "ORBIT"),
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise KeyboardInterrupt
            if target is None:
                continue

            label = str(target.get("letter", "")).upper()
            history.clear()
            print(
                f"TASK3 TARGET_CONFIRMED letter={label} "
                f"center={target.get('center')} depth_cm={target.get('distance_cm', '-')}",
                flush=True,
            )
            h7.pause_until_confirmed(sequence, field, args.pause_timeout_s)
            print(f"TASK3 ORBIT_PAUSED letter={label}", flush=True)

            try:
                centered = _center_task3_target(
                    camera, detector, orbit_rescue_detector, target, boards
                )
            except RuntimeError as exc:
                if not str(exc).startswith("TARGET_TRACK_TIMEOUT"):
                    raise
                # The orbit is already safely paused. If the confirmed target
                # disappears during the bounded re-centering window, release
                # the pause and keep scanning instead of leaving the chassis
                # stopped until the outer error handler sends STOP.
                print(
                    "TASK3 TARGET_LOST_AFTER_PAUSE "
                    f"missing_timeout={TARGET_TRACK_MISSING_TIMEOUT_S:.1f}s "
                    "action=RESUME_ORBIT",
                    flush=True,
                )
                h7.send(
                    f"RK,TEST,TASK3,RESUME,SEQ,{sequence},FIELD,{field}"
                )
                h7.wait_status(sequence, "RESUMED", args.resume_timeout_s)
                history.clear()
                print("TASK3 ORBIT_RESUMED reason=TARGET_TRACK_TIMEOUT", flush=True)
                continue
            _letter_grasp(boards, centered)
            grabbed.add(label)
            history.clear()
            time.sleep(TARGET_RECHECK_DELAY_S)

            h7.send(
                f"RK,TEST,TASK3,RESUME,SEQ,{sequence},FIELD,{field}"
            )
            h7.wait_status(sequence, "RESUMED", args.resume_timeout_s)
            print(
                f"TASK3 ORBIT_RESUMED grabbed={','.join(sorted(grabbed))}",
                flush=True,
            )

        print(
            f"TASK3 TEST COMPLETE orbit={TASK3_ORBIT_ANGLE_DEG}deg "
            f"radius={TASK3_ORBIT_RADIUS_MM}mm "
            f"grabbed={','.join(sorted(grabbed)) or 'none'}",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        print("TASK3 ABORTED", flush=True)
        return 2
    except Exception as exc:
        print(f"TASK3 ERROR {exc}", flush=True)
        return 1
    finally:
        if h7 is not None and started and not completed:
            try:
                h7.send(
                    f"RK,TEST,TASK3,STOP,SEQ,{sequence},FIELD,{args.field.upper()}"
                )
                h7.wait_status(sequence, "STOPPED", 2.0)
            except Exception as exc:
                print(f"TASK3 STOP_RESULT {exc}", flush=True)
        if camera is not None:
            camera.release()
        if boards is not None:
            boards.close()
        if h7 is not None:
            h7.close()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoboCup task-three standalone orbit and ABCD test"
    )
    parser.add_argument("--field", choices=("red", "blue"), required=True)
    parser.add_argument("--main-camera", default=MAIN_CAMERA)
    parser.add_argument("--h7-device", default=H7_DEVICE)
    parser.add_argument("--arm-uart", default=ARM_DEVICE)
    parser.add_argument("--zp-uart", default=ZP_DEVICE)
    parser.add_argument("--execute-h7", action="store_true")
    parser.add_argument("--camera-timeout-s", type=float, default=30.0)
    parser.add_argument("--ack-timeout-s", type=float, default=4.0)
    parser.add_argument("--running-timeout-s", type=float, default=8.0)
    parser.add_argument("--pause-timeout-s", type=float, default=5.0)
    parser.add_argument("--resume-timeout-s", type=float, default=3.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
