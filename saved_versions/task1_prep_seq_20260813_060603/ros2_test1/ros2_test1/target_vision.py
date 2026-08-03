#!/usr/bin/env python3
"""Detect the selected colored targets and draw them on the camera view."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import getpass
import http.server
import json
import math
import multiprocessing
import os
import re
import select
import signal
import socket
import subprocess
import termios
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from .arm_kinematics import (
    BASE_YAW_CENTER_TICK,
    GRIPPER_CLOSED_TICK,
    GRIPPER_OPEN_TICK,
    HOME_ID1_TICK,
    HOME_ID2_TICK,
    ID1_SAFE_LIMITS,
    ID2_SAFE_LIMITS,
    JOINT_NAMES,
    MIN_ANGLE_GAP_DEG,
    READY_ID1_TICK,
    READY_ID2_TICK,
    angle_gap_degrees,
    enforce_angle_gap,
    gripper_position_mm,
    id1_degrees,
    id2_degrees,
    id2_tick_from_degrees,
    joint_positions,
    solve_gripper_position,
)
from .detectors import (
    BALL_COLORS,
    BALL_DETECTORS,
    DRAW_COLORS,
    MASK_BUILDERS,
    SHAPE_COLORS,
    WHITE_DETECTOR,
    blue,
    red,
    yellow,
)
from .detectors.common import circularity as contour_circularity
from .detectors.common import external_contours

try:
    from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode
except ImportError:  # Keep color detection usable if the QR decoder is missing.
    ZBarSymbol = None
    zbar_decode = None


WINDOW_NAME = "Ros2_test1 Target Vision"

BALL_DISTANCE_OFFSET_CM = -1.6072186919749336
BALL_DISTANCE_SCALE_CM = 31.628878020276648
GOLF_BALL_DIAMETER_MM = 42.67
RED_CUBE_SIDE_MM = 30.0
RED_RING_OUTER_DIAMETER_MM = 55.0
RING_DISTANCE_EXTRA_CM = 6.0
RING_EXTRA_DESCEND_MM = 30.0
RING_DISTANCE_DESCEND_COMPENSATION = (
    (20.0, 0.0),
    (22.0, 10.0),
    (24.0, 30.0),
)
RING_CENTER_DEADBAND_PX = 24.0
RING_LOCK_DISTANCE_JUMP_CM = 1.8
RING_LOCK_DISTANCE_SPREAD_CM = 2.2
QR_EXTRA_DESCEND_MM = 40.0
GRASP_ID1_CALIBRATION_TICKS = -5
GRASP_DESCEND_ID2_BIAS_TICKS = 15
POST_CENTER_REFERENCE_DISTANCE_MM = 235.0
POST_CENTER_MIN_DOWN_MM = 105.0
POST_CENTER_MAX_DOWN_MM = 210.0
SINGLE_ID2_DEADBAND_PX = 45.0
SINGLE_ID2_MIN_STEP_TICKS = 8
SINGLE_ID2_MAX_STEP_TICKS = 40
SINGLE_ID6_MIN_STEP_TICKS = 1
ID6_SAFE_LIMITS = (300, 700)
CENTERING_SLOW_COMMAND_INTERVAL_S = 0.40
CENTERING_SLOW_ID2_GAIN = 0.08
CENTERING_SLOW_ID6_GAIN = 0.06
CENTERING_SLOW_ID2_MAX_STEP_TICKS = 16
CENTERING_SLOW_ID6_MAX_STEP_TICKS = 8
SPLITTER_YELLOW_TICK = 800
SPLITTER_OTHER_BALL_TICK = 1600
CATCHER_HOME_TICK = 800
CATCHER_RELEASE_READY_TICK = 1100
POST_GRAB_ID2_RETREAT_TICK = 100
DISC_CATCH_FIXED_ID1_TICK = 500
DISC_CATCH_FIXED_ID2_TICK = 600
GRASP_DISTANCE_CALIBRATION = (
    (10.0, (551, 460), (470, 461)),
    (15.0, (551, 460), (413, 466)),
    (30.0, (551, 460), (242, 481)),
)
GRASP_MIN_DISTANCE_CM = 10.0
GRASP_MAX_DISTANCE_CM = 30.0
SQUARE_DISTANCE_OFFSET_CM = BALL_DISTANCE_OFFSET_CM
SQUARE_DISTANCE_SCALE_CM = (
    BALL_DISTANCE_SCALE_CM
    * 2.0
    * RED_CUBE_SIDE_MM
    / (GOLF_BALL_DIAMETER_MM * np.sqrt(np.pi))
    * 1.20
)
RING_DISTANCE_OFFSET_CM = BALL_DISTANCE_OFFSET_CM + RING_DISTANCE_EXTRA_CM
RING_DISTANCE_SCALE_CM = (
    BALL_DISTANCE_SCALE_CM
    * RED_RING_OUTER_DIAMETER_MM
    / GOLF_BALL_DIAMETER_MM
)

POSITION_REPORT_RE = re.compile(
    r"ID1=(-?\d+)\s+ID2=(-?\d+)(?:\s+ID6=(-?\d+|ERR))?"
)
ARM_READY_REPORT_RE = re.compile(r"OK\s+ARMREADY\s+ID1=(-?\d+)\s+ID2=(-?\d+)")
DIRECT_85KG_IDS = {1, 2, 6}
DIRECT_ZP_IDS = {4, 5, 7}
DIRECT_SERVO_BROADCAST_ID = 0xFE
DIRECT_CMD_MOVE_TIME_WRITE = 0x01
DIRECT_CMD_MOVE_TIME_WAIT_WRITE = 0x07
DIRECT_CMD_MOVE_START = 0x0B


def direct_servo_checksum(body):
    return (~(sum(body) & 0xFF)) & 0xFF


def direct_servo_packet(servo_id, command, params=b""):
    body = bytes([servo_id, len(params) + 3, command]) + bytes(params)
    return b"\x55\x55" + body + bytes([direct_servo_checksum(body)])


def direct_85kg_move_packet(servo_id, position, time_ms, wait=False):
    position = max(0, min(1000, int(position)))
    time_ms = max(0, min(30000, int(time_ms)))
    command = DIRECT_CMD_MOVE_TIME_WAIT_WRITE if wait else DIRECT_CMD_MOVE_TIME_WRITE
    params = bytes(
        [
            position & 0xFF,
            (position >> 8) & 0xFF,
            time_ms & 0xFF,
            (time_ms >> 8) & 0xFF,
        ]
    )
    return direct_servo_packet(servo_id, command, params)


def direct_85kg_start_packet(servo_id=DIRECT_SERVO_BROADCAST_ID):
    return direct_servo_packet(servo_id, DIRECT_CMD_MOVE_START)


def direct_zp_move_packet(servo_id, position, time_ms):
    position = max(500, min(2500, int(position)))
    time_ms = max(0, min(9999, int(time_ms)))
    return f"#{servo_id:03d}P{position:04d}T{time_ms:04d}!".encode("ascii")


def load_last_id4_target(default):
    try:
        state = json.loads((Path.home() / ".servo_zp_state.json").read_text())
        return max(500, min(2500, int(state.get("zp4_target", default))))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return int(default)


def calibrated_grasp_ticks(distance_cm, phase):
    target_index = 1 if phase == "overhead" else 2
    distance_cm = float(distance_cm)
    if distance_cm <= GRASP_DISTANCE_CALIBRATION[0][0]:
        return GRASP_DISTANCE_CALIBRATION[0][target_index]
    if distance_cm >= GRASP_DISTANCE_CALIBRATION[-1][0]:
        return GRASP_DISTANCE_CALIBRATION[-1][target_index]

    for lower, upper in zip(
        GRASP_DISTANCE_CALIBRATION,
        GRASP_DISTANCE_CALIBRATION[1:],
    ):
        if lower[0] <= distance_cm <= upper[0]:
            ratio = (distance_cm - lower[0]) / (upper[0] - lower[0])
            lower_ticks = lower[target_index]
            upper_ticks = upper[target_index]
            return tuple(
                int(math.floor(start + (end - start) * ratio + 0.5))
                for start, end in zip(lower_ticks, upper_ticks)
            )
    return GRASP_DISTANCE_CALIBRATION[-1][target_index]


def ring_extra_descend_mm(distance_cm):
    base_extra_mm = float(RING_EXTRA_DESCEND_MM)
    if distance_cm is None:
        return base_extra_mm

    distance_cm = float(distance_cm)
    points = RING_DISTANCE_DESCEND_COMPENSATION
    if distance_cm <= points[0][0]:
        return base_extra_mm + points[0][1]
    if distance_cm >= points[-1][0]:
        return base_extra_mm + points[-1][1]

    for lower, upper in zip(points, points[1:]):
        lower_cm, lower_extra_mm = lower
        upper_cm, upper_extra_mm = upper
        if lower_cm <= distance_cm <= upper_cm:
            ratio = (distance_cm - lower_cm) / (upper_cm - lower_cm)
            return base_extra_mm + lower_extra_mm + (
                upper_extra_mm - lower_extra_mm
            ) * ratio
    return base_extra_mm


def matches_trigger_target(det, color, kind):
    if det.get("color") != color:
        return False
    det_kind = det.get("kind")
    if kind == "any":
        return det_kind in {"square", "ring", "qr"}
    return det_kind == kind


def scale_detections(detections, scale_x, scale_y):
    radius_scale = (scale_x + scale_y) * 0.5
    for detection in detections:
        if "center" in detection:
            center_x, center_y = detection["center"]
            detection["center"] = (
                int(round(center_x * scale_x)),
                int(round(center_y * scale_y)),
            )
        if "bbox" in detection:
            x, y, width, height = detection["bbox"]
            detection["bbox"] = (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
                int(round(width * scale_x)),
                int(round(height * scale_y)),
            )
        for key in ("radius", "outer_radius", "inner_radius"):
            if key in detection:
                detection[key] = int(round(detection[key] * radius_scale))
        for key in ("box", "points"):
            if key in detection:
                points = np.asarray(detection[key], dtype=np.float32).copy()
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                detection[key] = points.astype(np.int32)
        if "projected_area" in detection:
            detection["projected_area"] *= scale_x * scale_y
    return detections


class DetectionSmoother:
    def __init__(self, alpha=0.32, max_match_px=90.0, keep_missing_frames=6):
        self.alpha = max(0.05, min(1.0, float(alpha)))
        self.max_match_px = max(8.0, float(max_match_px))
        self.keep_missing_frames = max(1, int(keep_missing_frames))
        self.tracks = []

    @staticmethod
    def _center(det):
        if "center" in det:
            return np.asarray(det["center"], dtype=np.float32)
        if "bbox" in det:
            x, y, w, h = det["bbox"]
            return np.asarray((x + w * 0.5, y + h * 0.5), dtype=np.float32)
        return None

    @staticmethod
    def _copy_detection(det):
        copied = {}
        for key, value in det.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def _new_track(self, det):
        track = {
            "color": det.get("color"),
            "kind": det.get("kind"),
            "missing": 0,
            "det": self._copy_detection(det),
        }
        self._init_float_state(track, det)
        return track

    @staticmethod
    def _init_float_state(track, det):
        for key in ("center", "bbox"):
            if key in det:
                track[key] = np.asarray(det[key], dtype=np.float32)
        for key in (
            "radius",
            "outer_radius",
            "inner_radius",
            "projected_area",
            "area_ratio",
            "area_percent",
            "diameter_ratio",
            "distance_cm",
        ):
            if key in det and det[key] is not None:
                track[key] = float(det[key])

    def _match_track(self, det, used):
        center = self._center(det)
        if center is None:
            return None
        best_index = None
        best_distance = self.max_match_px
        for index, track in enumerate(self.tracks):
            if index in used:
                continue
            if track.get("color") != det.get("color") or track.get("kind") != det.get("kind"):
                continue
            track_center = track.get("center")
            if track_center is None:
                track_center = self._center(track["det"])
            if track_center is None:
                continue
            distance = float(np.linalg.norm(center - track_center))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _smooth_track(self, track, det):
        alpha = self.alpha
        track["missing"] = 0
        track["det"] = self._copy_detection(det)
        for key in ("center", "bbox"):
            if key not in det:
                continue
            value = np.asarray(det[key], dtype=np.float32)
            if key in track and np.asarray(track[key]).shape == value.shape:
                track[key] = (1.0 - alpha) * track[key] + alpha * value
            else:
                track[key] = value
        for key in (
            "radius",
            "outer_radius",
            "inner_radius",
            "projected_area",
            "area_ratio",
            "area_percent",
            "diameter_ratio",
            "distance_cm",
        ):
            if key not in det or det[key] is None:
                continue
            value = float(det[key])
            track[key] = (1.0 - alpha) * track[key] + alpha * value if key in track else value

    def _track_to_detection(self, track):
        det = self._copy_detection(track["det"])
        if "center" in track:
            det["center"] = tuple(int(round(v)) for v in track["center"])
        if "bbox" in track:
            det["bbox"] = tuple(int(round(v)) for v in track["bbox"])
        for key in ("radius", "outer_radius", "inner_radius"):
            if key in track:
                det[key] = int(round(track[key]))
        for key in (
            "projected_area",
            "area_ratio",
            "area_percent",
            "diameter_ratio",
            "distance_cm",
        ):
            if key in track:
                det[key] = float(track[key])
        return det

    def update(self, detections):
        used = set()
        smoothed = []
        for det in detections:
            index = self._match_track(det, used)
            if index is None:
                track = self._new_track(det)
                self.tracks.append(track)
                index = len(self.tracks) - 1
            else:
                track = self.tracks[index]
                self._smooth_track(track, det)
            used.add(index)
            smoothed.append(self._track_to_detection(track))

        kept_tracks = []
        for index, track in enumerate(self.tracks):
            if index in used:
                kept_tracks.append(track)
                continue
            track["missing"] = track.get("missing", 0) + 1
            if track["missing"] <= self.keep_missing_frames:
                kept_tracks.append(track)
        self.tracks = kept_tracks
        return smoothed


_PROCESS_DETECTOR = None


def detection_process_worker(source_frame, detection_scale, qr_template_every=1):
    global _PROCESS_DETECTOR
    if _PROCESS_DETECTOR is None:
        _PROCESS_DETECTOR = TargetDetector(qr_template_every=qr_template_every)
    detect_started = time.perf_counter()
    if detection_scale < 0.999:
        source_height, source_width = source_frame.shape[:2]
        detect_width = max(1, int(round(source_width * detection_scale)))
        detect_height = max(1, int(round(source_height * detection_scale)))
        detect_frame = cv2.resize(
            source_frame,
            (detect_width, detect_height),
            interpolation=cv2.INTER_AREA,
        )
        result = _PROCESS_DETECTOR.detect(detect_frame)
        result = scale_detections(
            result,
            source_width / float(detect_width),
            source_height / float(detect_height),
        )
    else:
        result = _PROCESS_DETECTOR.detect(source_frame)
    return result, time.perf_counter() - detect_started

QR_COLOR_ALIASES = {
    "r": "red",
    "red": "red",
    "red square": "red",
    "red_square": "red",
    "hong": "red",
    "hongse": "red",
    "b": "blue",
    "blue": "blue",
    "blue square": "blue",
    "blue_square": "blue",
    "lan": "blue",
    "lanse": "blue",
}

class FrameState:
    def __init__(self):
        self.frame = None
        self.info = "starting"
        self.running = True
        self.lock = threading.Lock()

    def update(self, frame, info):
        with self.lock:
            self.frame = frame.copy()
            self.info = info

    def snapshot(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, self.info


class WindowCloseWatcher:
    def __init__(self, title):
        self.title = title
        self.closed = threading.Event()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=0.5)

    def _run(self):
        window_id = None
        while not self.stopped.wait(0.2):
            result = subprocess.run(
                ["xwininfo", "-name", self.title],
                capture_output=True,
                text=True,
                check=False,
            )
            match = re.search(r"Window id:\s+(0x[0-9a-fA-F]+)", result.stdout)
            if match is not None:
                window_id = match.group(1)
                break
        if window_id is None:
            return
        while not self.stopped.wait(0.2):
            result = subprocess.run(
                ["xprop", "-id", window_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                self.closed.set()
                return


class StreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        state = self.server.state
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><head><meta charset='utf-8'><title>Target Vision</title></head>"
                b"<body style='margin:0;background:#111;color:#eee;text-align:center'>"
                b"<img src='/video' style='max-width:100%;height:auto'>"
                b"<pre id='info' style='font-size:18px'></pre>"
                b"<script>setInterval(()=>fetch('/status').then(r=>r.text()).then(t=>info.textContent=t),300)</script>"
                b"</body></html>"
            )
            return

        if self.path == "/status":
            _, info = state.snapshot()
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(info.encode("utf-8", "replace"))
            return

        if self.path == "/video":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while state.running:
                    frame, _ = state.snapshot()
                    if frame is not None:
                        ok, jpg = cv2.imencode(
                            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                        )
                        if ok:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                + jpg.tobytes()
                                + b"\r\n"
                            )
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404)

    def log_message(self, *args):
        pass


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, addr, handler, state):
        self.state = state
        super().__init__(addr, handler)


class SerialTrigger:
    def __init__(
        self,
        device,
        baudrate,
        command,
        color,
        kind,
        stable_frames,
        reset_frames,
        cooldown_s,
        enabled=True,
    ):
        self.device = device
        self.baudrate = baudrate
        self.command = (command.strip() + "\n").encode("ascii", "ignore")
        self.color = color
        self.kind = kind
        self.stable_frames = max(1, stable_frames)
        self.reset_frames = max(1, reset_frames)
        self.cooldown_s = max(0.0, cooldown_s)
        self.enabled = enabled
        self.fd = None
        self.seen_frames = 0
        self.missing_frames = 0
        self.armed = True
        self.last_trigger = 0.0
        self.status = "servo trigger disabled" if not enabled else "servo trigger ready"

        if self.enabled:
            self._open()

    def _open(self):
        try:
            subprocess.run(
                [
                    "stty",
                    "-F",
                    self.device,
                    str(self.baudrate),
                    "cs8",
                    "-cstopb",
                    "-parenb",
                    "-ixon",
                    "-ixoff",
                    "-crtscts",
                    "raw",
                    "-echo",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            self.status = f"servo trigger on {self.device}"
            print(self.status)
        except OSError as exc:
            self.enabled = False
            self.status = f"servo uart open failed: {exc}"
            print(self.status)
        except subprocess.CalledProcessError as exc:
            self.enabled = False
            self.status = f"servo uart setup failed: {exc}"
            print(self.status)

    def update(self, detections):
        if not self.enabled:
            return self.status

        matched = any(
            det.get("color") == self.color and det.get("kind") == self.kind
            for det in detections
        )
        if matched:
            self.seen_frames += 1
            self.missing_frames = 0
        else:
            self.seen_frames = 0
            self.missing_frames += 1
            if self.missing_frames >= self.reset_frames:
                self.armed = True
            return self.status

        now = time.monotonic()
        if not self.armed:
            return f"{self.status}, waiting target leave"
        if self.seen_frames < self.stable_frames:
            return f"{self.status}, confirming {self.color} {self.kind}"
        if (now - self.last_trigger) < self.cooldown_s:
            return f"{self.status}, cooldown"

        try:
            os.write(self.fd, self.command)
            self.last_trigger = now
            self.armed = False
            command_text = self.command.decode("ascii", "replace").strip()
            self.status = f"sent {command_text} for {self.color} {self.kind}"
            print(self.status)
        except OSError as exc:
            self.status = f"servo uart write failed: {exc}"
            print(self.status)
        return self.status

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class AbsoluteServoBridge:
    def __init__(self, device, baudrate, enabled=True, write_enabled=True):
        self.device = device
        self.baudrate = baudrate
        self.enabled = enabled
        self.write_enabled = enabled and write_enabled
        self.fd = None
        self.last_feedback = None
        self.last_command_ok = False
        self.last_ack = {}
        self.status = "servo bridge disabled"
        if self.enabled:
            self._open()

    def _open(self):
        try:
            subprocess.run(
                [
                    "stty",
                    "-F",
                    self.device,
                    str(self.baudrate),
                    "cs8",
                    "-cstopb",
                    "-parenb",
                    "-ixon",
                    "-ixoff",
                    "-crtscts",
                    "raw",
                    "-echo",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            self.status = f"servo bridge on {self.device}"
            print(self.status)
        except (OSError, subprocess.CalledProcessError) as exc:
            self.enabled = False
            self.status = f"servo bridge open failed: {exc}"
            print(self.status)

    def _read_text(self, timeout_s):
        if self.fd is None:
            return ""
        deadline = time.monotonic() + timeout_s
        chunks = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            readable, _, _ = select.select([self.fd], [], [], remaining)
            if not readable:
                break
            try:
                data = os.read(self.fd, 512)
            except BlockingIOError:
                continue
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                time.sleep(0.01)
        return b"".join(chunks).decode("ascii", "replace")

    def send_targets(
        self,
        id1=None,
        id2=None,
        id4=None,
        id6=None,
        id5=None,
        splitter_id4=None,
    ):
        if not self.enabled or self.fd is None:
            return self.status
        if not self.write_enabled:
            return "servo bridge read-only; targets not sent"

        commands = []
        if id1 is not None and id2 is not None:
            value1 = int(id1)
            value2 = int(id2)
            commands.append(
                (f"armpair={value1},{value2}", f"OK ARMPAIR {value1} {value2}", 2, 0.25, 0.25, 1)
            )
        elif id1 is not None:
            value = int(id1)
            commands.append((f"id1={value}", f"OK ID1 TARGET {value}", 2, 0.25, 0.25, 1))
        elif id2 is not None:
            value = int(id2)
            commands.append((f"id2={value}", f"OK ID2 TARGET {value}", 2, 0.25, 0.25, 1))
        if id6 is not None:
            value = int(id6)
            commands.append((f"id6={value}", f"OK ID6 TARGET {value}", 2, 0.25, 0.25, 1))
        if id4 is not None:
            value = int(id4)
            commands.append((f"zp4:{value}", f"OK ZP4 {value}", 4, 0.45, 0.45, 2))
        if id5 is not None:
            value = int(id5)
            commands.append((f"zp5:{value}", f"OK ZP5 {value}", 4, 0.45, 0.45, 2))
        if splitter_id4 is not None:
            value = int(splitter_id4)
            commands.append((f"zp4:{value}", f"OK ZP4 {value}", 4, 0.45, 0.45, 2))
        if not commands:
            return self.status

        self.last_command_ok = False
        self.last_ack = {}
        try:
            self._read_text(0.02)
            for command, expected_ack, attempts, read_timeout, retry_delay, required_acks in commands:
                response = ""
                ack_count = 0
                for attempt in range(attempts):
                    os.write(self.fd, (command + "\n").encode("ascii", "ignore"))
                    response = self._read_text(read_timeout).strip()
                    normalized = response.replace("\r", " ").replace("\n", " ")
                    print(
                        f"SERVO UART TX={command} attempt={attempt + 1} "
                        f"RX={normalized or 'timeout'}",
                        flush=True,
                    )
                    if expected_ack in normalized and "ERR" not in normalized:
                        ack_count += 1
                        self.last_ack[command] = normalized
                        if ack_count >= required_acks:
                            break
                    if attempt < attempts - 1:
                        time.sleep(retry_delay)
                if ack_count < required_acks:
                    detail = response.replace("\r", " ").replace("\n", " ").strip()
                    self.status = f"ACK failed {command}: {detail or 'timeout'}"
                    return self.status
            self.last_command_ok = True
            self.status = "ACK " + ",".join(command for command, *_ in commands)
        except OSError as exc:
            self.status = f"servo bridge write failed: {exc}"
            print(self.status)
        return self.status

    def wait_ready(self, timeout_s=0.2):
        if not self.enabled or self.fd is None:
            return False
        text = self._read_text(timeout_s)
        if "ARM UART READY" in text:
            self.status = "RCT6 ARM UART READY"
            return True
        return False

    def command_arm_ready(self, timeout_s=4.0):
        if not self.enabled or self.fd is None or not self.write_enabled:
            return None
        self.last_command_ok = False
        try:
            self._read_text(0.05)
            os.write(self.fd, b"armready\n")
            response = self._read_text(timeout_s)
        except OSError as exc:
            self.status = f"ARMREADY write failed: {exc}"
            return None
        normalized = response.replace("\r", " ").replace("\n", " ").strip()
        print(f"SERVO UART TX=armready RX={normalized or 'timeout'}", flush=True)
        match = ARM_READY_REPORT_RE.search(response)
        if match is None or "ERR ARMREADY" in response:
            self.status = f"ARMREADY failed: {normalized or 'timeout'}"
            return None
        self.last_feedback = (int(match.group(1)), int(match.group(2)))
        self.last_command_ok = True
        self.status = (
            f"ARMREADY complete ID1={self.last_feedback[0]} "
            f"ID2={self.last_feedback[1]}"
        )
        return self.last_feedback

    def query_positions(self, timeout_s=0.90):
        if not self.enabled or self.fd is None:
            return None
        try:
            self._read_text(0.02)
            os.write(self.fd, b"pos\n")
            response = self._read_text(timeout_s)
        except OSError as exc:
            self.status = f"servo feedback failed: {exc}"
            return None
        match = POSITION_REPORT_RE.search(response)
        if match is None:
            self.status = f"servo feedback invalid: {response.strip() or 'timeout'}"
            return None
        id6_text = match.group(3)
        id6 = None if id6_text in (None, "ERR") else int(id6_text)
        self.last_feedback = (int(match.group(1)), int(match.group(2)), id6)
        id6_status = "" if id6 is None else f" ID6={id6}"
        self.status = (
            f"feedback ID1={self.last_feedback[0]} ID2={self.last_feedback[1]}"
            f"{id6_status}"
        )
        return self.last_feedback

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class DirectBusServoBridge:
    def __init__(
        self,
        device,
        baudrate,
        enabled=True,
        write_enabled=True,
        arm_device=None,
        zp_device=None,
        arm_time_ms=1200,
        zp_time_ms=1000,
        gripper_time_ms=None,
        repeat=1,
    ):
        self.device = device
        self.arm_device = arm_device or device
        self.zp_device = zp_device or device
        self.baudrate = baudrate
        self.enabled = enabled
        self.write_enabled = enabled and write_enabled
        self.arm_time_ms = int(arm_time_ms)
        self.zp_time_ms = int(zp_time_ms)
        self.gripper_time_ms = (
            self.zp_time_ms if gripper_time_ms is None else int(gripper_time_ms)
        )
        self.repeat = max(1, min(8, int(repeat)))
        self.arm_fd = None
        self.zp_fd = None
        self.last_feedback = None
        self.last_command_ok = False
        self.last_ack = {}
        self.assumed_feedback = True
        self.status = "direct bus servo bridge disabled"
        if self.enabled:
            self._open()

    def _open_device(self, device):
        subprocess.run(
            [
                "stty",
                "-F",
                device,
                str(self.baudrate),
                "cs8",
                "-cstopb",
                "-parenb",
                "-ixon",
                "-ixoff",
                "-crtscts",
                "raw",
                "-echo",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)

    def _open(self):
        try:
            self.arm_fd = self._open_device(self.arm_device)
            if self.zp_device == self.arm_device:
                self.zp_fd = self.arm_fd
            else:
                self.zp_fd = self._open_device(self.zp_device)
            self.status = (
                f"direct bus servo bridge arm={self.arm_device} "
                f"zp={self.zp_device}"
            )
            print(self.status)
        except (OSError, subprocess.CalledProcessError) as exc:
            self.close()
            self.enabled = False
            self.write_enabled = False
            self.status = f"direct bus servo open failed: {exc}"
            print(self.status)

    def _write_payload(self, fd, payload):
        for attempt in range(self.repeat):
            os.write(fd, payload)
            if attempt < self.repeat - 1:
                time.sleep(0.08)

    def _format_payload(self, payload):
        try:
            text = payload.decode("ascii")
            if text.startswith("#"):
                return text
        except UnicodeDecodeError:
            pass
        return " ".join(f"{byte:02X}" for byte in payload)

    def send_targets(
        self,
        id1=None,
        id2=None,
        id4=None,
        id6=None,
        id5=None,
        splitter_id4=None,
    ):
        if not self.enabled or self.arm_fd is None or self.zp_fd is None:
            return self.status
        if not self.write_enabled:
            return "direct bus servo read-only; targets not sent"

        payloads = []
        targets = {}
        arm_targets = []
        if id1 is not None:
            arm_targets.append((1, int(id1)))
        if id2 is not None:
            arm_targets.append((2, int(id2)))
        if id6 is not None:
            arm_targets.append((6, int(id6)))

        if len(arm_targets) > 1:
            for servo_id, value in arm_targets:
                payloads.append(
                    (
                        self.arm_fd,
                        self.arm_device,
                        direct_85kg_move_packet(
                            servo_id,
                            value,
                            self.arm_time_ms,
                            wait=True,
                        ),
                    )
                )
                targets[servo_id] = value
            payloads.append((self.arm_fd, self.arm_device, direct_85kg_start_packet()))
        elif id1 is not None:
            payloads.append((self.arm_fd, self.arm_device, direct_85kg_move_packet(1, id1, self.arm_time_ms)))
            targets[1] = int(id1)
        elif id2 is not None:
            payloads.append((self.arm_fd, self.arm_device, direct_85kg_move_packet(2, id2, self.arm_time_ms)))
            targets[2] = int(id2)
        elif id6 is not None:
            payloads.append((self.arm_fd, self.arm_device, direct_85kg_move_packet(6, id6, self.arm_time_ms)))
            targets[6] = int(id6)
        if id4 is not None:
            payloads.append((self.zp_fd, self.zp_device, direct_zp_move_packet(7, id4, self.gripper_time_ms)))
            targets[7] = int(id4)
        if splitter_id4 is not None:
            payloads.append((self.zp_fd, self.zp_device, direct_zp_move_packet(4, splitter_id4, self.zp_time_ms)))
            targets[4] = int(splitter_id4)
        if id5 is not None:
            payloads.append((self.zp_fd, self.zp_device, direct_zp_move_packet(5, id5, self.zp_time_ms)))
            targets[5] = int(id5)
        if not payloads:
            return self.status

        self.last_command_ok = False
        try:
            for fd, device, payload in payloads:
                self._write_payload(fd, payload)
                print(
                    f"DIRECT SERVO TX {device}={self._format_payload(payload)}",
                    flush=True,
                )
                time.sleep(0.02)
            previous = self.last_feedback or (READY_ID1_TICK, READY_ID2_TICK, None)
            self.last_feedback = (
                targets.get(1, previous[0]),
                targets.get(2, previous[1]),
                targets.get(6, previous[2] if len(previous) > 2 else None),
            )
            self.last_command_ok = True
            self.status = "direct TX " + ",".join(f"ID{k}={v}" for k, v in targets.items())
        except OSError as exc:
            self.status = f"direct bus servo write failed: {exc}"
            print(self.status)
        return self.status

    def wait_ready(self, timeout_s=0.2):
        self.status = "direct bus servo ready without RCT6 feedback"
        return self.enabled and self.arm_fd is not None and self.zp_fd is not None

    def command_arm_ready(self, timeout_s=4.0):
        status = self.send_targets(id1=READY_ID1_TICK, id2=READY_ID2_TICK)
        if not self.last_command_ok:
            self.status = f"direct ARMREADY failed: {status}"
            return None
        return self.last_feedback

    def query_positions(self, timeout_s=0.90):
        if not self.enabled or self.arm_fd is None or self.zp_fd is None:
            return None
        if self.last_feedback is None:
            self.last_feedback = (READY_ID1_TICK, READY_ID2_TICK, None)
        self.status = (
            f"direct assumed feedback ID1={self.last_feedback[0]} "
            f"ID2={self.last_feedback[1]}"
        )
        return self.last_feedback

    def close(self):
        if self.zp_fd is not None and self.zp_fd != self.arm_fd:
            os.close(self.zp_fd)
        self.zp_fd = None
        if self.arm_fd is not None:
            os.close(self.arm_fd)
        self.arm_fd = None


class ChassisLink:
    def __init__(self, device="auto", baudrate=115200, enabled=False):
        self.device = device
        self.baudrate = int(baudrate)
        self.enabled = bool(enabled)
        self.fd = None
        self.active_device = None
        self.rx_buffer = ""
        self.last_open_attempt = 0.0
        self.status = "chassis link disabled"

    def _candidate_devices(self):
        if self.device and self.device != "auto":
            return [self.device]
        candidates = []
        for pattern in (
            "/dev/serial/by-id/*",
            "/dev/ttyACM*",
            "/dev/ttyUSB*",
        ):
            candidates.extend(sorted(glob.glob(pattern)))
        unique = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return unique

    def _configure_tty(self, fd):
        try:
            attrs = termios.tcgetattr(fd)
            speed = getattr(termios, f"B{self.baudrate}", termios.B115200)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = attrs[2] | termios.CLOCAL | termios.CREAD
            attrs[2] = attrs[2] & ~termios.PARENB & ~termios.CSTOPB & ~termios.CSIZE
            attrs[2] = attrs[2] | termios.CS8
            attrs[3] = 0
            attrs[4] = speed
            attrs[5] = speed
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except termios.error:
            pass

    def _open(self):
        if not self.enabled or self.fd is not None:
            return self.fd is not None
        now = time.monotonic()
        if now - self.last_open_attempt < 1.0:
            return False
        self.last_open_attempt = now
        for candidate in self._candidate_devices():
            try:
                fd = os.open(candidate, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                self._configure_tty(fd)
                self.fd = fd
                self.active_device = candidate
                self.status = f"chassis link open {candidate}"
                print(self.status, flush=True)
                return True
            except OSError as exc:
                self.status = f"chassis link open failed {candidate}: {exc}"
        return False

    def read_lines(self):
        if not self.enabled or not self._open():
            return []
        lines = []
        try:
            while True:
                chunk = os.read(self.fd, 256)
                if not chunk:
                    break
                self.rx_buffer += chunk.decode("ascii", errors="ignore")
        except BlockingIOError:
            pass
        except OSError as exc:
            print(f"CHASSIS LINK read failed: {exc}", flush=True)
            self.close()
            return []
        while "\n" in self.rx_buffer or "\r" in self.rx_buffer:
            split_at = min(
                index
                for index in (
                    self.rx_buffer.find("\n"),
                    self.rx_buffer.find("\r"),
                )
                if index >= 0
            )
            line = self.rx_buffer[:split_at].strip()
            self.rx_buffer = self.rx_buffer[split_at + 1 :]
            self.rx_buffer = self.rx_buffer.lstrip("\r\n")
            if line:
                print(f"CHASSIS RX {line}", flush=True)
                lines.append(line)
        return lines

    def send(self, line):
        if not self.enabled or not self._open():
            return False
        payload = (line.rstrip("\r\n") + "\r\n").encode("ascii", errors="ignore")
        try:
            os.write(self.fd, payload)
            print(f"CHASSIS TX {line.rstrip()}", flush=True)
            return True
        except OSError as exc:
            print(f"CHASSIS LINK write failed: {exc}", flush=True)
            self.close()
            return False

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None
        self.active_device = None


class RedSquareGraspController:
    def __init__(
        self,
        enabled,
        servo_bridge,
        arm_preview,
        id1_ready,
        id2_ready,
        id4_closed,
        id4_open,
        center_deadband_px,
        stable_frames,
        command_interval_s,
        retrigger_cooldown_s,
        id2_pixel_gain,
        id6_pixel_gain,
        id6_max_step_ticks,
        id1_pixel_gain_y,
        id2_distance_gain,
        camera_gripper_offset_mm,
        target_gripper_distance_mm,
        distance_deadband_mm,
        max_step_ticks,
        id1_limits,
        id2_limits,
        angle_gap_degrees,
        startup_sequence,
        one_shot,
        camera_gripper_vertical_offset_mm,
        max_lateral_offset_mm,
        max_one_shot_ik_error_mm,
        post_center_retreat_mm,
        post_center_down_mm,
        post_center_ik_error_mm,
        min_target_area_percent=0.4,
        min_target_distance_cm=GRASP_MIN_DISTANCE_CM,
        max_target_distance_cm=GRASP_MAX_DISTANCE_CM,
        use_calibrated_grasp_table=False,
        use_post_center_tick_bias=False,
        post_center_direct_descend=False,
        simple_vertical_grasp=False,
        vertical_grasp_id1=HOME_ID1_TICK,
        vertical_grasp_id2=READY_ID2_TICK,
        initial_id4=None,
        id2_center_min=None,
        id2_center_max=None,
    ):
        self.enabled = enabled
        self.servo_bridge = servo_bridge
        self.arm_preview = arm_preview
        self.id1 = int(id1_ready)
        self.id2 = int(id2_ready)
        self.active_station_task = None
        self.id4 = int(id4_closed if initial_id4 is None else initial_id4)
        self.id6 = int(arm_preview.id6)
        self.splitter_id4 = SPLITTER_YELLOW_TICK
        self.id5 = CATCHER_HOME_TICK
        self.id4_closed = int(id4_closed)
        self.id4_open = int(id4_open)
        self.center_deadband_px = center_deadband_px
        self.stable_frames_required = max(1, stable_frames)
        self.command_interval_s = command_interval_s
        self.retrigger_cooldown_s = max(0.0, float(retrigger_cooldown_s))
        self.ignore_new_targets_until = 0.0
        self.id2_pixel_gain = id2_pixel_gain
        self.id6_pixel_gain = max(0.0, float(id6_pixel_gain))
        self.id6_max_step_ticks = max(1, int(id6_max_step_ticks))
        self.id1_pixel_gain_y = id1_pixel_gain_y
        self.id2_distance_gain = id2_distance_gain
        self.camera_gripper_offset_mm = camera_gripper_offset_mm
        self.camera_gripper_vertical_offset_mm = camera_gripper_vertical_offset_mm
        self.target_gripper_distance_mm = target_gripper_distance_mm
        self.max_lateral_offset_mm = max_lateral_offset_mm
        self.max_one_shot_ik_error_mm = max_one_shot_ik_error_mm
        self.post_center_retreat_mm = float(post_center_retreat_mm)
        self.post_center_down_mm = float(post_center_down_mm)
        self.post_center_ik_error_mm = float(post_center_ik_error_mm)
        self.min_target_area_percent = max(0.0, float(min_target_area_percent))
        self.min_target_distance_cm = float(min_target_distance_cm)
        self.max_target_distance_cm = float(max_target_distance_cm)
        self.use_calibrated_grasp_table = bool(use_calibrated_grasp_table)
        self.use_post_center_tick_bias = bool(use_post_center_tick_bias)
        self.post_center_direct_descend = bool(post_center_direct_descend)
        self.simple_vertical_grasp = bool(simple_vertical_grasp)
        self.vertical_grasp_id1 = int(vertical_grasp_id1)
        self.vertical_grasp_id2 = int(vertical_grasp_id2)
        self.id2_center_min = None if id2_center_min is None else int(id2_center_min)
        self.id2_center_max = None if id2_center_max is None else int(id2_center_max)
        self.distance_deadband_mm = distance_deadband_mm
        self.max_step_ticks = max(1, int(max_step_ticks))
        self.id1_limits = id1_limits
        self.id2_limits = id2_limits
        self.preview_id1_limits = (
            max(id1_limits[0], self.id1 - 90),
            min(id1_limits[1], self.id1 + 90),
        )
        self.preview_id2_limits = (
            max(id2_limits[0], self.id2 - 90),
            min(id2_limits[1], self.id2 + 90),
        )
        self.angle_gap_degrees = max(0.0, float(angle_gap_degrees))
        self.centered_frames = 0
        self.horizontal_correction_done = False
        self.centering_correction_count = 0
        self.ring_lock_distance_samples = []
        self.last_command_time = 0.0
        self.last_aux_command_time = 0.0
        self.aux_command_interval_s = 0.25
        self.last_preview_step_time = 0.0
        self.preview_step_interval_s = 0.12
        self.last_feedback_attempt = 0.0
        self.feedback_due = 0.0
        self.feedback_pending = False
        self.feedback_failures = 0
        self.startup_sequence = bool(startup_sequence)
        self.startup_stage = "complete"
        self.startup_deadline = 0.0
        self.startup_verify_deadline = 0.0
        self.startup_next_feedback = 0.0
        self.startup_feedback_attempts = 0
        self.startup_ready_resends = 0
        self.startup_ready_timeout = time.monotonic() + 120.0
        self.startup_position_tolerance = 10
        self.one_shot = one_shot
        self.one_shot_target = None
        self.one_shot_approach_sent = False
        self.one_shot_complete = False
        self.post_center_move_complete = False
        self.locked_target = None
        self.locked_plan = None
        self.last_visual_target = None
        self.visual_confirm_frames = 0
        self.visual_lost_frames = 0
        self.visual_descend_start = None
        self.visual_descend_step_index = 0
        self.visual_descend_steps = 4
        self.target_jump_reset_px = max(180.0, float(center_deadband_px) * 5.0)
        self.centering_jump_reset_px = max(120.0, float(center_deadband_px) * 4.0)
        self.abort_after_return = False
        self.approach_attempts = 0
        self.return_attempts = 0
        self.approach_feedback_tolerance = 10
        self.stage_deadline = 0.0
        self.next_stage_after_open = None
        self.motion_stage_after_wait = None
        self.algorithm_stage = "centering"
        self.completed_cycles = 0
        self.last_cycle_status = None
        self.read_only_sync = enabled and servo_bridge.enabled and not servo_bridge.write_enabled
        self.read_only_feedback_lock = threading.Lock()
        self.read_only_feedback_thread = None
        self.read_only_feedback_ready = False
        self.read_only_feedback = None
        self.synchronized = (
            not (enabled and servo_bridge.enabled)
            or getattr(servo_bridge, "assumed_feedback", False)
        )
        self.state = "disabled" if not enabled else "searching"
        self.status = "red-square grasp disabled" if not enabled else "red-square grasp ready"

        self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        if self.enabled and self.servo_bridge.write_enabled:
            if self.startup_sequence:
                self.startup_stage = "wait_ready"
                self.state = "startup_wait_ready"
                self.status = "waiting for RCT6 ARM UART READY"
            else:
                self.state = "waiting servo feedback"
                self.status = "waiting for RCT6 position synchronization"
        elif self.read_only_sync:
            self.state = "read_only_sync"
            self.status = "waiting for read-only RCT6 position synchronization"

    @staticmethod
    def _clamp(value, limits):
        lower, upper = limits
        return max(lower, min(upper, int(round(value))))

    def _limited_delta(self, delta):
        return max(-self.max_step_ticks, min(self.max_step_ticks, int(round(delta))))

    def _arm_settle_s(self):
        return max(0.15, getattr(self.servo_bridge, "arm_time_ms", 700) / 1000.0 + 0.12)

    def _zp_settle_s(self):
        return max(0.12, getattr(self.servo_bridge, "zp_time_ms", 450) / 1000.0 + 0.08)

    def _clamp_center_id2(self, value):
        value = self._clamp(value, self.id2_limits)
        if self.id2_center_min is not None:
            value = max(value, self.id2_center_min)
        if self.id2_center_max is not None:
            value = min(value, self.id2_center_max)
        return value

    def _id2_center_limit_blocks(self, delta_id2, target_id2):
        if delta_id2 == 0 or target_id2 != self.id2:
            return False
        if delta_id2 > 0 and self.id2_center_max is not None:
            return self.id2 >= self.id2_center_max
        if delta_id2 < 0 and self.id2_center_min is not None:
            return self.id2 <= self.id2_center_min
        return False

    @staticmethod
    def _target_center(target):
        if target is None or "center" not in target:
            return None
        return target["center"]

    @staticmethod
    def _target_error(target, frame_shape):
        height, width = frame_shape[:2]
        cx, cy = target["center"]
        return cx - width / 2.0, cy - height / 2.0

    def _accept_live_target(self, live_target):
        center = self._target_center(live_target)
        if center is None:
            self.visual_lost_frames += 1
            return None, "target lost"
        if not live_target.get("fully_visible", True):
            self.visual_lost_frames += 1
            return None, "target at edge"
        reference = self.last_visual_target or self.locked_target
        reference_center = self._target_center(reference)
        if reference_center is not None:
            jump_px = math.hypot(
                float(center[0]) - float(reference_center[0]),
                float(center[1]) - float(reference_center[1]),
            )
            if jump_px > self.target_jump_reset_px:
                self.visual_lost_frames += 1
                return None, f"target jump {jump_px:.0f}px"
        self.visual_lost_frames = 0
        self.last_visual_target = self._copy_target(live_target)
        return live_target, "target live"

    def _centering_target_hold_reason(self, target, frame_shape):
        center = self._target_center(target)
        if center is None:
            self.visual_lost_frames += 1
            if self.visual_lost_frames > 6:
                self.last_visual_target = None
            return "target lost before centering"
        error_x, error_y = self._target_error(target, frame_shape)
        if not target.get("fully_visible", True):
            self.visual_lost_frames += 1
            if self.visual_lost_frames > 6:
                self.last_visual_target = None
            return (
                f"target at edge; hold centering dx={error_x:.0f} dy={error_y:.0f}"
            )
        reference_center = self._target_center(self.last_visual_target)
        if reference_center is not None:
            jump_px = math.hypot(
                float(center[0]) - float(reference_center[0]),
                float(center[1]) - float(reference_center[1]),
            )
            if jump_px > self.centering_jump_reset_px:
                self.centered_frames = 0
                self.visual_lost_frames += 1
                if self.visual_lost_frames >= 3:
                    self.last_visual_target = self._copy_target(target)
                    self.visual_lost_frames = 0
                    return f"target jump {jump_px:.0f}px; rebase and hold"
                return f"target jump {jump_px:.0f}px; hold centering"
        self.visual_lost_frames = 0
        self.last_visual_target = self._copy_target(target)
        return None

    def _abort_to_standby(self, reason):
        self.locked_target = None
        self.locked_plan = None
        self.last_visual_target = None
        self.visual_confirm_frames = 0
        self.visual_lost_frames = 0
        self.visual_descend_start = None
        self.visual_descend_step_index = 0
        self.post_center_move_complete = False
        self.approach_attempts = 0
        self.return_attempts = 0
        self.abort_after_return = True
        self.algorithm_stage = "return"
        self.state = "abort to standby"
        self.status = f"{reason}; aborting to standby"
        self.arm_preview.publish(self.status)
        return self.status

    def _visual_center_step(self, target, frame_shape, now, can_preview_step, label):
        error_x, error_y = self._target_error(target, frame_shape)
        centering_profile = self._centering_profile()
        center_deadband_px = self._center_deadband_for_target(target)
        delta_id6 = self._id6_centering_delta(
            error_x,
            centering_profile["id6_gain"],
            centering_profile["id6_max_step_ticks"],
            center_deadband_px,
        )
        delta_id2 = self._centering_delta(
            -error_y,
            centering_profile["id2_gain"],
            centering_profile["id2_max_step_ticks"],
            center_deadband_px,
        )
        target_id2 = self._clamp_center_id2(self.id2 + delta_id2)
        id2_center_limited = self._id2_center_limit_blocks(delta_id2, target_id2)
        if id2_center_limited:
            delta_id2 = 0
            if delta_id6 == 0:
                return True, (
                    f"{label} centered at ID2 limit dx={error_x:.0f} dy={error_y:.0f} "
                    f"ID2={self.id2} ID6={self.id6}"
                )
        delta_id2, delta_id6, axis_note = self._vector_centering_deltas(
            delta_id2,
            delta_id6,
            id2_center_limited,
            error_x,
            error_y,
            centering_profile["id2_max_step_ticks"],
            centering_profile["id6_max_step_ticks"],
        )
        target_id2 = self._clamp_center_id2(self.id2 + delta_id2)
        if delta_id2 == 0 and delta_id6 == 0:
            return True, f"{label} centered dx={error_x:.0f} dy={error_y:.0f}"

        target_id6 = self._clamp(self.id6 + delta_id6, ID6_SAFE_LIMITS)
        target_id1 = self.id1
        target_id1, target_id2 = enforce_angle_gap(
            target_id1,
            target_id2,
            self.id2_limits,
            self.angle_gap_degrees,
        )
        if target_id2 == self.id2 and target_id6 == self.id6:
            return False, (
                f"{label} blocked by limits dx={error_x:.0f} dy={error_y:.0f} "
                f"ID2={self.id2} ID6={self.id6}"
            )

        can_command = now - self.last_command_time >= centering_profile["command_interval_s"]
        if not self.servo_bridge.write_enabled:
            if can_preview_step:
                self.id2 = target_id2
                self.id6 = target_id6
                self.last_preview_step_time = now
            self.arm_preview.set_targets(target_id1, target_id2, self.id4, target_id6)
            return False, (
                f"preview {label} dx={error_x:.0f} dy={error_y:.0f} "
                f"ID2={target_id2} ID6={target_id6}"
            )
        if not can_command:
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return False, f"waiting {label} dx={error_x:.0f} dy={error_y:.0f}"

        previous_id2 = self.id2
        previous_id6 = self.id6
        self.id2 = target_id2
        self.id6 = target_id6
        send_id2 = self.id2 != previous_id2
        send_id6 = self.id6 != previous_id6
        return False, self._send_center_correction(
            f"{label} dx={error_x:.0f} dy={error_y:.0f} "
            f"axis={axis_note} dID2={self.id2 - previous_id2} "
            f"dID6={self.id6 - previous_id6} {centering_profile['note']}",
            send_id2=send_id2,
            send_id6=send_id6,
        )

    def _center_deadband_for_target(self, target):
        if target is not None and target.get("kind") == "ring":
            return min(self.center_deadband_px, RING_CENTER_DEADBAND_PX)
        return self.center_deadband_px

    def _reset_ring_lock_guard(self):
        self.ring_lock_distance_samples = []

    def _ring_lock_hold_reason(self, target, error_x, error_y, id2_center_limited):
        if target.get("kind") != "ring":
            self._reset_ring_lock_guard()
            return None

        ring_deadband = self._center_deadband_for_target(target)
        if id2_center_limited and abs(error_y) > ring_deadband:
            self._reset_ring_lock_guard()
            return (
                f"ring y not centered at ID2 limit dx={error_x:.0f} "
                f"dy={error_y:.0f} ID2={self.id2} ID6={self.id6}"
            )

        distance_cm = target.get("distance_cm")
        if distance_cm is None:
            self._reset_ring_lock_guard()
            return "ring distance unknown; hold lock"

        samples = self.ring_lock_distance_samples
        distance_cm = float(distance_cm)
        if samples and abs(distance_cm - samples[-1]) > RING_LOCK_DISTANCE_JUMP_CM:
            self.ring_lock_distance_samples = [distance_cm]
            self.centered_frames = 0
            return (
                f"ring distance jump {samples[-1]:.1f}->{distance_cm:.1f}cm; "
                "hold lock"
            )

        samples.append(distance_cm)
        keep = max(2, self.stable_frames_required)
        if len(samples) > keep:
            del samples[:-keep]

        if len(samples) >= keep:
            spread = max(samples) - min(samples)
            if spread > RING_LOCK_DISTANCE_SPREAD_CM:
                self.ring_lock_distance_samples = samples[-2:]
                self.centered_frames = min(self.centered_frames, 1)
                return f"ring distance unstable spread={spread:.1f}cm; hold lock"
        return None

    def _centering_profile(self):
        correction_index = max(0, int(self.centering_correction_count))
        decay = 0.5 ** (correction_index / 3.0)
        return {
            "index": correction_index + 1,
            "decay": decay,
            "command_interval_s": min(
                CENTERING_SLOW_COMMAND_INTERVAL_S,
                self.command_interval_s / max(decay, 0.001),
            ),
            "id2_gain": max(CENTERING_SLOW_ID2_GAIN, self.id2_pixel_gain * decay),
            "id6_gain": max(CENTERING_SLOW_ID6_GAIN, self.id6_pixel_gain * decay),
            "id2_max_step_ticks": max(
                CENTERING_SLOW_ID2_MAX_STEP_TICKS,
                int(round(self.max_step_ticks * decay)),
            ),
            "id6_max_step_ticks": max(
                CENTERING_SLOW_ID6_MAX_STEP_TICKS,
                int(round(self.id6_max_step_ticks * decay)),
            ),
            "note": (
                f"centerStep={correction_index + 1} decay={decay:.2f}"
            ),
        }

    def _centering_delta(self, error_px, gain, max_step_ticks=None, deadband_px=None):
        if deadband_px is None:
            deadband_px = self.center_deadband_px
        if abs(error_px) <= deadband_px:
            return 0
        if max_step_ticks is None:
            max_step_ticks = self.max_step_ticks
        delta = int(round(error_px * gain))
        delta = max(-max_step_ticks, min(max_step_ticks, delta))
        minimum_step = min(4, max_step_ticks)
        if abs(delta) < minimum_step:
            return minimum_step if error_px > 0 else -minimum_step
        return delta

    def _id6_centering_delta(self, error_x_px, gain=None, max_step_ticks=None, deadband_px=None):
        if deadband_px is None:
            deadband_px = self.center_deadband_px
        if abs(error_x_px) <= deadband_px:
            return 0
        if gain is None:
            gain = self.id6_pixel_gain
        if max_step_ticks is None:
            max_step_ticks = self.id6_max_step_ticks
        delta = int(round(-error_x_px * gain))
        delta = max(-max_step_ticks, min(max_step_ticks, delta))
        minimum_step = min(SINGLE_ID6_MIN_STEP_TICKS, max_step_ticks)
        if abs(delta) < minimum_step:
            return -minimum_step if error_x_px > 0 else minimum_step
        return delta

    def _vector_centering_deltas(
        self,
        delta_id2,
        delta_id6,
        id2_center_limited,
        error_x_px=None,
        error_y_px=None,
        id2_max_step_ticks=None,
        id6_max_step_ticks=None,
    ):
        if id2_max_step_ticks is None:
            id2_max_step_ticks = self.max_step_ticks
        if id6_max_step_ticks is None:
            id6_max_step_ticks = self.id6_max_step_ticks
        if id2_center_limited:
            delta_id2 = 0
        if delta_id2 == 0 and delta_id6 == 0:
            return 0, 0, "none"

        distance_px = None
        if error_x_px is not None and error_y_px is not None:
            distance_px = math.hypot(float(error_x_px), float(error_y_px))

        if distance_px and delta_id2 != 0 and delta_id6 != 0:
            biggest_component = max(abs(float(error_x_px)), abs(float(error_y_px)), 1.0)
            diagonal_boost = min(1.35, max(1.0, distance_px / biggest_component))
            delta_id2 = int(round(delta_id2 * diagonal_boost))
            delta_id6 = int(round(delta_id6 * diagonal_boost))
            delta_id2 = max(-id2_max_step_ticks, min(id2_max_step_ticks, delta_id2))
            delta_id6 = max(
                -id6_max_step_ticks,
                min(id6_max_step_ticks, delta_id6),
            )
            return delta_id2, delta_id6, f"ID2+ID6 vector r={distance_px:.0f}px"

        if delta_id2 != 0:
            return delta_id2, 0, "ID2"
        axis = "ID6 after ID2 limit" if id2_center_limited else "ID6"
        return 0, delta_id6, axis

    def _single_id2_correction(self, error_px):
        if abs(error_px) <= SINGLE_ID2_DEADBAND_PX:
            return 0
        delta = int(round(error_px * self.id2_pixel_gain))
        delta = max(-SINGLE_ID2_MAX_STEP_TICKS, min(SINGLE_ID2_MAX_STEP_TICKS, delta))
        if abs(delta) < SINGLE_ID2_MIN_STEP_TICKS:
            return SINGLE_ID2_MIN_STEP_TICKS if error_px > 0 else -SINGLE_ID2_MIN_STEP_TICKS
        return delta

    @staticmethod
    def _id1_degrees(id1_tick):
        return id1_degrees(id1_tick)

    @staticmethod
    def _id2_degrees(id2_tick):
        return id2_degrees(id2_tick)

    @staticmethod
    def _id2_tick_from_degrees(degrees):
        return id2_tick_from_degrees(degrees)

    def _enforce_angle_gap(self):
        if self.active_station_task == "DISC_CATCH":
            self.id1 = DISC_CATCH_FIXED_ID1_TICK
            self.id2 = DISC_CATCH_FIXED_ID2_TICK
            return angle_gap_degrees(self.id1, self.id2)
        self.id1, self.id2 = enforce_angle_gap(
            self.id1,
            self.id2,
            self.id2_limits,
            self.angle_gap_degrees,
        )
        return angle_gap_degrees(self.id1, self.id2)

    def _apply_feedback(self, feedback):
        if not (self.id1_limits[0] <= feedback[0] <= self.id1_limits[1]):
            self.state = "fault"
            self.status = f"ID1 feedback outside safe range: {feedback[0]}"
            return False
        if not (self.id2_limits[0] <= feedback[1] <= self.id2_limits[1]):
            self.state = "fault"
            self.status = f"ID2 feedback outside safe range: {feedback[1]}"
            return False
        if self.active_station_task == "DISC_CATCH":
            self.id1 = DISC_CATCH_FIXED_ID1_TICK
            self.id2 = DISC_CATCH_FIXED_ID2_TICK
        else:
            self.id1 = int(feedback[0])
            self.id2 = int(feedback[1])
        id6 = feedback[2] if len(feedback) > 2 else None
        if id6 is not None:
            self.id6 = int(id6)
        gap = angle_gap_degrees(self.id1, self.id2)
        if self.angle_gap_degrees > 0.0 and gap <= self.angle_gap_degrees:
            self.state = "fault"
            self.status = f"unsafe measured angle gap {gap:.1f}deg"
            return False
        self.synchronized = True
        self.feedback_pending = False
        self.feedback_failures = 0
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish("measured servo feedback")
        return True

    def _read_only_feedback_worker(self):
        feedback = self.servo_bridge.query_positions(timeout_s=0.45)
        with self.read_only_feedback_lock:
            self.read_only_feedback = feedback
            self.read_only_feedback_ready = True

    def _start_read_only_feedback(self):
        with self.read_only_feedback_lock:
            if (
                self.read_only_feedback_thread is not None
                and self.read_only_feedback_thread.is_alive()
            ):
                return
            self.read_only_feedback_ready = False
            self.read_only_feedback_thread = threading.Thread(
                target=self._read_only_feedback_worker,
                daemon=True,
            )
            self.read_only_feedback_thread.start()

    def _send(self, reason, require_feedback=True):
        gap = self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish(reason)
        print(
            f"GRASP COMMAND reason={reason} ID1={self.id1} "
            f"ID2={self.id2} ID6={self.id6} ID7={self.id4} gap={gap:.1f}deg",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(
            id1=DISC_CATCH_FIXED_ID1_TICK
            if self.active_station_task == "DISC_CATCH"
            else self.id1,
            id2=DISC_CATCH_FIXED_ID2_TICK
            if self.active_station_task == "DISC_CATCH"
            else self.id2,
            id4=self.id4,
            id6=self.id6,
        )
        self.last_command_time = time.monotonic()
        if self.servo_bridge.write_enabled and not self.servo_bridge.last_command_ok:
            self.state = "fault"
            self.algorithm_stage = "fault"
            self.feedback_pending = False
            self.status = f"automatic motion stopped: {self.servo_bridge.status}"
            self.arm_preview.publish(self.status)
            return self.status
        if self.servo_bridge.write_enabled and require_feedback:
            if getattr(self.servo_bridge, "assumed_feedback", False):
                self.feedback_pending = False
            else:
                self.feedback_pending = True
                self.feedback_due = self.last_command_time + max(1.35, self.command_interval_s)
        gap_text = "" if gap is None else f" gap={gap:.1f}deg"
        return f"{reason}: id1={self.id1} id2={self.id2} id6={self.id6} id7={self.id4}{gap_text} | {self.status}"

    def _send_retreat_with_catcher_open(self, reason, require_feedback=True):
        gap = self._enforce_angle_gap()
        self.id5 = CATCHER_RELEASE_READY_TICK
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish(reason)
        print(
            f"GRASP COMMAND reason={reason} ID1={self.id1} "
            f"ID2={self.id2} ID6={self.id6} ID5={self.id5} "
            f"ID7={self.id4} gap={gap:.1f}deg",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(
            id1=DISC_CATCH_FIXED_ID1_TICK
            if self.active_station_task == "DISC_CATCH"
            else self.id1,
            id2=DISC_CATCH_FIXED_ID2_TICK
            if self.active_station_task == "DISC_CATCH"
            else self.id2,
            id6=self.id6,
            id5=self.id5,
        )
        self.last_command_time = time.monotonic()
        if self.servo_bridge.write_enabled and not self.servo_bridge.last_command_ok:
            self.state = "fault"
            self.algorithm_stage = "fault"
            self.feedback_pending = False
            self.status = f"automatic motion stopped: {self.servo_bridge.status}"
            self.arm_preview.publish(self.status)
            return self.status
        if self.servo_bridge.write_enabled and require_feedback:
            if getattr(self.servo_bridge, "assumed_feedback", False):
                self.feedback_pending = False
            else:
                self.feedback_pending = True
                self.feedback_due = self.last_command_time + max(1.35, self.command_interval_s)
        gap_text = "" if gap is None else f" gap={gap:.1f}deg"
        return (
            f"{reason}: id1={self.id1} id2={self.id2} id5={self.id5} "
            f"id6={self.id6} id7={self.id4}{gap_text} | {self.status}"
        )

    def _send_center_correction(self, reason, send_id2, send_id6, require_feedback=False):
        if self.active_station_task == "DISC_CATCH":
            self.id1 = DISC_CATCH_FIXED_ID1_TICK
            self.id2 = DISC_CATCH_FIXED_ID2_TICK
            send_id2 = False
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish(reason)
        print(
            f"CENTER COMMAND reason={reason} ID2={self.id2} ID6={self.id6}",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(
            id2=self.id2 if send_id2 else None,
            id6=self.id6 if send_id6 else None,
        )
        self.last_command_time = time.monotonic()
        if self.servo_bridge.write_enabled and not self.servo_bridge.last_command_ok:
            self.state = "fault"
            self.algorithm_stage = "fault"
            self.feedback_pending = False
            self.status = f"automatic motion stopped: {self.servo_bridge.status}"
            self.arm_preview.publish(self.status)
            return self.status
        if send_id2 or send_id6:
            self.centering_correction_count += 1
        if self.servo_bridge.write_enabled and require_feedback:
            if getattr(self.servo_bridge, "assumed_feedback", False):
                self.feedback_pending = False
            else:
                self.feedback_pending = True
                self.feedback_due = self.last_command_time + max(1.35, self.command_interval_s)
        return f"{reason}: id2={self.id2} id6={self.id6} | {self.status}"

    def _send_id4(self, target, reason):
        previous_id4 = self.id4
        target = int(target)
        self.arm_preview.set_targets(self.id1, self.id2, target, self.id6)
        self.arm_preview.publish(reason)
        print(
            f"GRASP COMMAND reason={reason} ID1={self.id1} "
            f"ID2={self.id2} ID6={self.id6} ID7={previous_id4}->{target}",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(id4=target)
        self.last_command_time = time.monotonic()
        if self.servo_bridge.write_enabled and not self.servo_bridge.last_command_ok:
            self.id4 = previous_id4
            self.state = "fault"
            self.algorithm_stage = "fault"
            self.feedback_pending = False
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            self.status = f"automatic motion stopped: {self.servo_bridge.status}"
            self.arm_preview.publish(self.status)
            return self.status
        self.id4 = target
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        return f"{reason}: id7={self.id4} | {self.status}"

    def _send_id5(self, target, reason, critical=True):
        previous_id5 = self.id5
        target = int(target)
        print(
            f"AUX COMMAND reason={reason} ID5={previous_id5}->{target}",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(id5=target)
        self.last_command_time = time.monotonic()
        if self.servo_bridge.write_enabled and not self.servo_bridge.last_command_ok:
            self.id5 = previous_id5
            if critical:
                self.state = "fault"
                self.algorithm_stage = "fault"
                self.feedback_pending = False
                self.status = f"automatic motion stopped: {self.servo_bridge.status}"
                self.arm_preview.publish(self.status)
            return self.status
        self.id5 = target
        return f"{reason}: id5={self.id5} | {self.status}"

    def _send_splitter_id4(self, target, reason):
        previous = self.splitter_id4
        target = int(target)
        print(
            f"AUX COMMAND reason={reason} ID4={previous}->{target}",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(splitter_id4=target)
        self.last_aux_command_time = time.monotonic()
        if self.servo_bridge.write_enabled and self.servo_bridge.last_command_ok:
            self.splitter_id4 = target
        return f"{reason}: id4={self.splitter_id4} | {self.status}"

    def _disc_catch_splitter_target_for_locked_ball(self):
        target = self.locked_target
        if target is None or target.get("kind") != "ball":
            return None
        if target.get("color") == "yellow":
            return SPLITTER_YELLOW_TICK
        return SPLITTER_OTHER_BALL_TICK

    def _send_disc_catch_open_claw_and_splitter(self):
        target = self._disc_catch_splitter_target_for_locked_ball()
        if target is None:
            return self._send_id4(self.id4_open, "locked target; open claw")

        previous_id7 = self.id4
        previous_splitter = self.splitter_id4
        color = self.locked_target.get("color", "ball")
        self.arm_preview.set_targets(self.id1, self.id2, self.id4_open, self.id6)
        self.arm_preview.publish(
            f"disc catch {color} ball; open ID7 and route ID4 together"
        )
        print(
            "GRASP COMMAND reason=disc catch ball pulse "
            f"color={color} ID7={previous_id7}->{self.id4_open} "
            f"ID4={previous_splitter}->{target}",
            flush=True,
        )
        self.status = self.servo_bridge.send_targets(
            id4=self.id4_open,
            splitter_id4=target,
        )
        now = time.monotonic()
        self.last_command_time = now
        self.last_aux_command_time = now
        if self.servo_bridge.write_enabled and not self.servo_bridge.last_command_ok:
            self.state = "fault"
            self.algorithm_stage = "fault"
            self.feedback_pending = False
            self.arm_preview.set_targets(self.id1, self.id2, previous_id7, self.id6)
            self.status = f"disc catch ID7/ID4 sync command failed: {self.servo_bridge.status}"
            self.arm_preview.publish(self.status)
            return self.status
        self.id4 = self.id4_open
        self.splitter_id4 = target
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        return (
            f"disc catch {color} ball: id7={self.id4} id4={self.splitter_id4} "
            f"| {self.status}"
        )

    def update_auxiliary(self, detections):
        if not self.enabled or not self.servo_bridge.write_enabled:
            return ""
        now = time.monotonic()
        if now - self.last_aux_command_time < self.aux_command_interval_s:
            return ""
        balls = [det for det in detections if det.get("kind") == "ball"]
        if any(det.get("color") == "yellow" for det in balls):
            target = SPLITTER_YELLOW_TICK
            reason = "yellow ball detected; splitter retract"
        elif balls:
            target = SPLITTER_OTHER_BALL_TICK
            reason = "non-yellow ball detected; splitter extend"
        else:
            return ""
        if target == self.splitter_id4:
            return ""
        return self._send_splitter_id4(target, reason)

    def _reset_cycle_for_search(
        self,
        status,
        start_retrigger_cooldown=False,
        cycle_complete=False,
    ):
        self.locked_target = None
        self.locked_plan = None
        self.last_visual_target = None
        self.visual_confirm_frames = 0
        self.visual_lost_frames = 0
        self.visual_descend_start = None
        self.visual_descend_step_index = 0
        self.centered_frames = 0
        self.horizontal_correction_done = False
        self.centering_correction_count = 0
        self.post_center_move_complete = False
        self.approach_attempts = 0
        self.return_attempts = 0
        self.next_stage_after_open = None
        self.motion_stage_after_wait = None
        self.abort_after_return = False
        self.algorithm_stage = "centering"
        if start_retrigger_cooldown and self.retrigger_cooldown_s > 0.0:
            self.ignore_new_targets_until = (
                time.monotonic() + self.retrigger_cooldown_s
            )
        self.state = "searching"
        self.status = status
        if cycle_complete:
            self.completed_cycles += 1
            self.last_cycle_status = status
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish(status)
        return status

    def begin_station_task(self, task_name):
        if not self.enabled:
            return "station task ignored; grasp disabled"
        self._reset_cycle_for_search(
            f"station {task_name} starting",
            start_retrigger_cooldown=False,
        )
        self.active_station_task = task_name
        if task_name == "DISC_CATCH":
            self.id1 = DISC_CATCH_FIXED_ID1_TICK
            self.id2 = DISC_CATCH_FIXED_ID2_TICK
        else:
            self.id1 = READY_ID1_TICK
            self.id2 = READY_ID2_TICK
        self.id6 = BASE_YAW_CENTER_TICK
        self.id4 = self.id4_closed
        self.id5 = CATCHER_HOME_TICK
        self.splitter_id4 = SPLITTER_YELLOW_TICK
        self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        if not self.servo_bridge.write_enabled:
            self.status = f"station {task_name} preview ready"
            return self.status
        result = self._send(f"station {task_name}; expand arm", require_feedback=False)
        self.status = result
        return result

    def prepare_disc_catch_high(self):
        """Raise the arm while H7 is still driving into the disc station."""
        if not self.enabled:
            return "DISC_CATCH prep ignored; grasp disabled"

        self.active_station_task = "DISC_CATCH"
        self.id1 = DISC_CATCH_FIXED_ID1_TICK
        self.id2 = DISC_CATCH_FIXED_ID2_TICK
        self.id6 = 600
        self.id4 = self.id4_closed
        self.id5 = CATCHER_HOME_TICK
        self.splitter_id4 = SPLITTER_YELLOW_TICK
        self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish("DISC_CATCH prep high")

        if not self.servo_bridge.write_enabled:
            self.status = (
                f"DISC_CATCH prep preview ID1={self.id1} ID2={self.id2}"
            )
            return self.status

        self.status = self.servo_bridge.send_targets(
            id1=self.id1,
            id2=self.id2,
            id4=self.id4,
            id5=self.id5,
            id6=self.id6,
            splitter_id4=self.splitter_id4,
        )
        self.last_command_time = time.monotonic()
        if not self.servo_bridge.last_command_ok:
            self.status = f"DISC_CATCH prep failed: {self.servo_bridge.status}"
            self.arm_preview.publish(self.status)
            return self.status
        self.status = (
            f"DISC_CATCH prep high ID1={self.id1} ID2={self.id2} "
            f"ID5={self.id5} ID6={self.id6} ID7={self.id4}"
        )
        print(self.status, flush=True)
        return self.status

    def hold_home(self, reason):
        if not self.enabled:
            return "home hold skipped; grasp disabled"
        self.active_station_task = None
        target = (
            HOME_ID1_TICK,
            HOME_ID2_TICK,
            BASE_YAW_CENTER_TICK,
            self.id4_closed,
            CATCHER_HOME_TICK,
            SPLITTER_YELLOW_TICK,
        )
        already_home = (
            self.id1,
            self.id2,
            self.id6,
            self.id4,
            self.id5,
            self.splitter_id4,
        ) == target
        self.locked_target = None
        self.locked_plan = None
        self.last_visual_target = None
        self.visual_confirm_frames = 0
        self.visual_lost_frames = 0
        self.centered_frames = 0
        self.centering_correction_count = 0
        self.algorithm_stage = "centering"
        self.state = "station_home"
        if already_home:
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            self.status = f"{reason}; already home"
            return self.status
        self.id1, self.id2, self.id6, self.id4, self.id5, self.splitter_id4 = target
        self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        if not self.servo_bridge.write_enabled:
            self.status = f"{reason}; preview home"
            return self.status
        result = self.servo_bridge.send_targets(
            id1=self.id1,
            id2=self.id2,
            id4=self.id4,
            id6=self.id6,
            id5=self.id5,
            splitter_id4=self.splitter_id4,
        )
        self.last_command_time = time.monotonic()
        self.status = f"{reason}; home command {result}"
        print(f"STATION HOME {self.status}", flush=True)
        return self.status

    def shutdown_contract(self):
        if not self.enabled or not self.servo_bridge.write_enabled:
            return "shutdown contract skipped; servo writes disabled"
        self.active_station_task = None
        self.id1 = HOME_ID1_TICK
        self.id2 = HOME_ID2_TICK
        self.id6 = BASE_YAW_CENTER_TICK
        self.id4 = self.id4_closed
        self.id5 = CATCHER_HOME_TICK
        self.splitter_id4 = SPLITTER_YELLOW_TICK
        self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish("shutdown contract arm")
        print(
            "SHUTDOWN CONTRACT "
            f"ID1={self.id1} ID2={self.id2} ID4={self.splitter_id4} "
            f"ID5={self.id5} ID6={self.id6} ID7={self.id4}",
            flush=True,
        )
        status = self.servo_bridge.send_targets(
            id1=self.id1,
            id2=self.id2,
            id4=self.id4,
            id6=self.id6,
            id5=self.id5,
            splitter_id4=self.splitter_id4,
        )
        self.last_command_time = time.monotonic()
        if not self.servo_bridge.last_command_ok:
            self.status = f"shutdown contract failed: {status}"
            self.arm_preview.publish(self.status)
            print(self.status, flush=True)
            return self.status
        self.status = f"shutdown contracted: {status}"
        self.arm_preview.publish(self.status)
        time.sleep(max(0.15, min(1.2, self._arm_settle_s())))
        return self.status

    def _startup_fault(self, detail):
        self.startup_stage = "fault"
        self.state = "fault"
        self.feedback_pending = False
        self.status = f"startup fault: {detail}"
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        return self.status

    def _startup_send_targets(self, id1, id2, id4, label):
        self.id1 = int(id1)
        self.id2 = int(id2)
        self.id4 = int(id4)
        self.id6 = BASE_YAW_CENTER_TICK
        self.splitter_id4 = SPLITTER_YELLOW_TICK
        self.id5 = CATCHER_HOME_TICK
        self._enforce_angle_gap()
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        status = self.servo_bridge.send_targets(
            id1=self.id1,
            id2=self.id2,
            id4=self.id4,
            id6=self.id6,
            id5=self.id5,
            splitter_id4=self.splitter_id4,
        )
        self.last_command_time = time.monotonic()
        if not self.servo_bridge.last_command_ok:
            return self._startup_fault(f"{label} {status}")
        self.status = (
            f"{label} ACK ID1={self.id1} ID2={self.id2} "
            f"ID4={self.splitter_id4} ID5={self.id5} "
            f"ID6={self.id6} ID7={self.id4}"
        )
        return self.status

    def _startup_positions_ready(self, feedback):
        if feedback is None:
            return False
        return (
            abs(feedback[0] - READY_ID1_TICK) <= self.startup_position_tolerance
            and abs(feedback[1] - READY_ID2_TICK) <= self.startup_position_tolerance
        )

    def _update_startup(self, now):
        if self.startup_stage == "fault":
            return self.status

        if self.startup_stage == "wait_ready":
            self.state = "startup_wait_ready"
            if not self.servo_bridge.wait_ready(0.12):
                if now >= self.startup_ready_timeout:
                    return self._startup_fault("RCT6 ARM UART READY timeout")
                return "startup_wait_ready: waiting for RCT6"
            feedback = self.servo_bridge.command_arm_ready()
            if feedback is None:
                return self._startup_fault(self.servo_bridge.status)
            self.id4 = self.id4_closed
            self.id6 = BASE_YAW_CENTER_TICK
            self.splitter_id4 = SPLITTER_YELLOW_TICK
            self.id5 = CATCHER_HOME_TICK
            yaw_status = self.servo_bridge.send_targets(id6=self.id6)
            if not self.servo_bridge.last_command_ok:
                return self._startup_fault(f"startup ID6 center {yaw_status}")
            aux_status = self.servo_bridge.send_targets(
                splitter_id4=self.splitter_id4,
                id5=self.id5,
            )
            if not self.servo_bridge.last_command_ok:
                return self._startup_fault(f"startup auxiliaries home {aux_status}")
            claw_status = self.servo_bridge.send_targets(id4=self.id4_closed)
            if not self.servo_bridge.last_command_ok:
                return self._startup_fault(f"startup claw close {claw_status}")
            if not self._apply_feedback(feedback):
                return self._startup_fault(self.status)
            if getattr(self.servo_bridge, "assumed_feedback", False):
                self.startup_stage = "direct_ready_settle"
                self.startup_deadline = time.monotonic() + max(
                    self._arm_settle_s(),
                    self._zp_settle_s(),
                )
                self.state = "startup_direct_ready_settle"
                self.status = (
                    f"startup_ready commands sent; settling ID1={self.id1} "
                    f"ID2={self.id2} ID4={self.splitter_id4} ID5={self.id5} "
                    f"ID6={self.id6} ID7={self.id4}"
                )
                return self.status
            self.startup_stage = "complete"
            self.state = "startup_ready"
            self.status = (
                f"startup_ready atomic ID1={self.id1} "
                f"ID2={self.id2} ID4={self.splitter_id4} ID5={self.id5} "
                f"ID6={self.id6} ID7={self.id4}"
            )
            return self.status

        if self.startup_stage == "direct_ready_settle":
            self.state = "startup_direct_ready_settle"
            if now < self.startup_deadline:
                return f"startup_direct_ready_settle {self.startup_deadline - now:.1f}s"
            self.startup_stage = "complete"
            self.state = "startup_ready"
            self.status = (
                f"startup_ready settled ID1={self.id1} ID2={self.id2} "
                f"ID4={self.splitter_id4} ID5={self.id5} "
                f"ID6={self.id6} ID7={self.id4}"
            )
            return self.status

        if self.startup_stage == "send_home":
            self.state = "startup_home"
            result = self._startup_send_targets(
                HOME_ID1_TICK,
                HOME_ID2_TICK,
                self.id4_closed,
                "startup_home",
            )
            if self.startup_stage == "fault":
                return result
            self.startup_deadline = now + 1.5
            self.startup_stage = "home_settle"
            return result

        if self.startup_stage == "home_settle":
            self.state = "startup_home"
            if now < self.startup_deadline:
                return f"startup_home settling {self.startup_deadline - now:.1f}s"
            self.startup_deadline = now + 5.0
            self.startup_stage = "home_hold"

        if self.startup_stage == "home_hold":
            self.state = "startup_home_hold"
            if now < self.startup_deadline:
                return f"startup_home_hold {self.startup_deadline - now:.1f}s"
            self.startup_stage = "send_ready"

        if self.startup_stage == "send_ready":
            self.state = "startup_extend"
            result = self._startup_send_targets(
                READY_ID1_TICK,
                READY_ID2_TICK,
                self.id4_closed,
                "startup_extend",
            )
            if self.startup_stage == "fault":
                return result
            self.startup_deadline = now + 0.8
            self.startup_verify_deadline = now + 6.0
            self.startup_next_feedback = self.startup_deadline
            self.startup_ready_resends = 0
            self.startup_stage = "ready_settle"
            return result

        if self.startup_stage == "ready_settle":
            self.state = "startup_extend"
            if now < self.startup_deadline:
                return f"startup_extend settling {self.startup_deadline - now:.1f}s"
            self.startup_stage = "verify_ready"
            self.startup_feedback_attempts = 0

        if self.startup_stage == "verify_ready":
            self.state = "startup_verify"
            if now < self.startup_next_feedback:
                return f"startup_verify waiting {self.startup_next_feedback - now:.1f}s"
            feedback = self.servo_bridge.query_positions()
            self.startup_feedback_attempts += 1
            if self._startup_positions_ready(feedback):
                self.id4 = self.id4_closed
                if not self._apply_feedback(feedback):
                    return self._startup_fault(self.status)
                self.startup_stage = "complete"
                self.state = "startup_ready"
                self.status = (
                    f"startup_ready ID1={self.id1} ID2={self.id2} "
                    f"ID4={self.splitter_id4} ID5={self.id5} "
                    f"ID6={self.id6} ID7={self.id4}"
                )
                return self.status
            if now >= self.startup_verify_deadline:
                return self._startup_fault(
                    f"ready feedback outside +/-{self.startup_position_tolerance}: "
                    f"{feedback or self.servo_bridge.status}"
                )
            if self.startup_ready_resends < 2:
                resend_status = self.servo_bridge.send_targets(
                    id1=READY_ID1_TICK,
                    id2=READY_ID2_TICK,
                    id6=BASE_YAW_CENTER_TICK,
                )
                if not self.servo_bridge.last_command_ok:
                    return self._startup_fault(f"ready resend {resend_status}")
                self.startup_ready_resends += 1
            self.startup_next_feedback = time.monotonic() + 1.0
            return (
                f"startup_verify moving feedback={feedback} "
                f"attempt={self.startup_feedback_attempts} "
                f"resend={self.startup_ready_resends}/2"
            )

        return self.status

    @staticmethod
    def _copy_target(target):
        if target is None:
            return None
        copied = dict(target)
        if "center" in copied:
            copied["center"] = tuple(copied["center"])
        if "bbox" in copied:
            copied["bbox"] = tuple(copied["bbox"])
        return copied

    def _target_ready(self, target):
        label = "target" if target is None else f"{target.get('color', '')} {target.get('kind', 'target')}".strip()
        if target is None:
            return False, "grasp searching target"
        if not target.get("fully_visible", True):
            return False, f"{label} touches frame edge; move it fully into view"
        if target.get("area_percent", 0.0) < self.min_target_area_percent:
            return (
                False,
                f"{label} area below {self.min_target_area_percent:.2f}%; grasp disabled",
            )
        return True, f"{label} locked"

    @staticmethod
    def _target_size_mm(target):
        if target.get("kind") == "ball":
            return GOLF_BALL_DIAMETER_MM
        if target.get("kind") == "ring":
            return RED_RING_OUTER_DIAMETER_MM
        return RED_CUBE_SIDE_MM

    def _apply_ik_calibration(self, id1, id2, target_x_mm, target_z_mm):
        id1 = self._clamp(
            id1 + GRASP_ID1_CALIBRATION_TICKS,
            self.id1_limits,
        )
        id1, id2 = enforce_angle_gap(
            id1,
            id2,
            self.id2_limits,
            self.angle_gap_degrees,
        )
        actual_x_mm, actual_z_mm = gripper_position_mm(id1, id2)
        error_mm = math.hypot(
            actual_x_mm - target_x_mm,
            actual_z_mm - target_z_mm,
        )
        return id1, id2, error_mm

    @staticmethod
    def _estimate_target_offsets_mm(target, frame_shape):
        distance_cm = target.get("distance_cm")
        if distance_cm is None:
            return None
        distance_mm = distance_cm * 10.0
        height, width = frame_shape[:2]
        cx, cy = target.get("center", (width / 2.0, height / 2.0))
        _, _, bbox_w, bbox_h = target.get("bbox", (0, 0, 0, 0))
        apparent_side_px = max(1.0, float(max(bbox_w, bbox_h)))
        target_size_mm = RedSquareGraspController._target_size_mm(target)
        focal_px = apparent_side_px * distance_mm / target_size_mm
        lateral_mm = (float(cx) - (width / 2.0)) * distance_mm / focal_px
        vertical_mm = (float(cy) - (height / 2.0)) * distance_mm / focal_px
        return {
            "distance_mm": distance_mm,
            "focal_px": focal_px,
            "lateral_mm": lateral_mm,
            "vertical_mm": vertical_mm,
            "target_kind": target.get("kind", "square"),
            "target_source": target.get("source"),
            "target_size_mm": target_size_mm,
        }

    def _one_shot_plan(self, target, frame_shape):
        offsets = self._estimate_target_offsets_mm(target, frame_shape)
        if offsets is None:
            return None, "one-shot locked; waiting distance estimate"

        lateral_mm = offsets["lateral_mm"]
        if abs(lateral_mm) > self.max_lateral_offset_mm:
            return None, (
                f"target lateral offset {lateral_mm:.0f}mm exceeds "
                f"{self.max_lateral_offset_mm:.0f}mm; center with base first"
            )

        gripper_forward_mm = (
            offsets["distance_mm"]
            - self.camera_gripper_offset_mm
            - self.target_gripper_distance_mm
        )
        gripper_vertical_mm = (
            self.camera_gripper_vertical_offset_mm
            - offsets["vertical_mm"]
        )
        current_x_mm, current_z_mm = gripper_position_mm(self.id1, self.id2)
        solved_id1, solved_id2, ik_error_mm = solve_gripper_position(
            current_x_mm + gripper_forward_mm,
            current_z_mm + gripper_vertical_mm,
            self.id1,
            self.id2,
            self.id1_limits,
            self.id2_limits,
            self.angle_gap_degrees,
        )
        solved_id1, solved_id2, ik_error_mm = self._apply_ik_calibration(
            solved_id1,
            solved_id2,
            current_x_mm + gripper_forward_mm,
            current_z_mm + gripper_vertical_mm,
        )
        plan = {
            **offsets,
            "forward_mm": gripper_forward_mm,
            "vertical_target_mm": gripper_vertical_mm,
            "target_distance_mm": self.target_gripper_distance_mm,
            "current_x_mm": current_x_mm,
            "current_z_mm": current_z_mm,
            "id1": solved_id1,
            "id2": solved_id2,
            "ik_error_mm": ik_error_mm,
        }
        return plan, (
            f"plan d={offsets['distance_mm']:.0f}mm "
            f"lat={lateral_mm:.0f}mm dz={gripper_vertical_mm:.0f}mm "
            f"fwd={gripper_forward_mm:.0f}mm -> "
            f"ID1={solved_id1} ID2={solved_id2} err={ik_error_mm:.1f}mm"
        )

    def _post_center_plan(self, target=None, frame_shape=None, phase="descend"):
        offsets = None
        if target is not None and frame_shape is not None:
            offsets = self._estimate_target_offsets_mm(target, frame_shape)
        current_x_mm, current_z_mm = gripper_position_mm(self.id1, self.id2)

        if (
            self.use_calibrated_grasp_table
            and offsets is not None
            and phase in ("overhead", "descend")
        ):
            solved_id1, solved_id2 = calibrated_grasp_ticks(
                offsets["distance_mm"] / 10.0,
                phase,
            )
            solved_id1 = self._clamp(solved_id1, self.id1_limits)
            solved_id2 = self._clamp(solved_id2, self.id2_limits)
            solved_id1, solved_id2 = enforce_angle_gap(
                solved_id1,
                solved_id2,
                self.id2_limits,
                self.angle_gap_degrees,
            )
            target_x_mm, target_z_mm = gripper_position_mm(solved_id1, solved_id2)
            plan = {
                **offsets,
                "current_x_mm": current_x_mm,
                "current_z_mm": current_z_mm,
                "forward_mm": target_x_mm - current_x_mm,
                "vertical_target_mm": target_z_mm - current_z_mm,
                "requested_forward_mm": target_x_mm - current_x_mm,
                "requested_vertical_mm": target_z_mm - current_z_mm,
                "progress_ratio": 1.0,
                "target_distance_mm": 0.0,
                "lateral_mm": 0.0,
                "id1": solved_id1,
                "id2": solved_id2,
                "ik_error_mm": 0.0,
                "phase": phase,
                "calibrated": True,
            }
            return plan, (
                f"calibrated distance={offsets['distance_mm'] / 10.0:.1f}cm "
                f"phase={phase} ID1={solved_id1} ID2={solved_id2}"
            )

        if offsets is not None:
            camera_to_target_mm = offsets["distance_mm"]
            camera_to_gripper_mm = max(0.0, self.camera_gripper_offset_mm)
            extra_down_mm = 0.0
            if offsets.get("target_kind") == "ring":
                extra_down_mm = ring_extra_descend_mm(camera_to_target_mm / 10.0)
            elif offsets.get("target_source") == "qr":
                extra_down_mm = QR_EXTRA_DESCEND_MM
            # The camera is 50 mm behind the gripper. Once the target is
            # centered, camera-target and camera-gripper are perpendicular
            # components: move back by the fixed offset and down by the
            # measured camera distance, leaving a small gripping clearance.
            gripper_to_target_mm = math.hypot(
                camera_to_target_mm,
                camera_to_gripper_mm,
            )
            if phase == "overhead":
                move_x_mm = -camera_to_gripper_mm
                move_z_mm = 0.0
            else:
                distance_scale = camera_to_target_mm / POST_CENTER_REFERENCE_DISTANCE_MM
                scaled_down_mm = self.post_center_down_mm * distance_scale
                scaled_down_mm = max(
                    POST_CENTER_MIN_DOWN_MM,
                    min(POST_CENTER_MAX_DOWN_MM, scaled_down_mm),
                )
                move_x_mm = -self.post_center_retreat_mm
                move_z_mm = -(scaled_down_mm + extra_down_mm)
            target_kind = offsets.get("target_kind", "square")
            target_source = offsets.get("target_source")
            target_size_mm = offsets.get("target_size_mm", RED_CUBE_SIDE_MM)
        else:
            camera_to_target_mm = None
            camera_to_gripper_mm = self.camera_gripper_offset_mm
            gripper_to_target_mm = None
            extra_down_mm = 0.0
            move_x_mm = -self.post_center_retreat_mm
            move_z_mm = -self.post_center_down_mm
            target_kind = "square"
            target_source = None
            target_size_mm = RED_CUBE_SIDE_MM

        target_x_mm = current_x_mm + move_x_mm
        target_z_mm = current_z_mm - self.post_center_down_mm
        if offsets is not None:
            target_z_mm = current_z_mm + move_z_mm
        solved_id1, solved_id2, ik_error_mm = solve_gripper_position(
            target_x_mm,
            target_z_mm,
            self.id1,
            self.id2,
            self.id1_limits,
            self.id2_limits,
            self.angle_gap_degrees,
        )
        progress_ratio = 1.0
        if ik_error_mm > self.post_center_ik_error_mm:
            for candidate_ratio in (0.85, 0.70, 0.55, 0.40, 0.25, 0.10):
                candidate_x_mm = current_x_mm + move_x_mm * candidate_ratio
                candidate_z_mm = current_z_mm + move_z_mm * candidate_ratio
                candidate_id1, candidate_id2, candidate_error_mm = solve_gripper_position(
                    candidate_x_mm,
                    candidate_z_mm,
                    self.id1,
                    self.id2,
                    self.id1_limits,
                    self.id2_limits,
                    self.angle_gap_degrees,
                )
                if candidate_error_mm <= self.post_center_ik_error_mm:
                    progress_ratio = candidate_ratio
                    target_x_mm = candidate_x_mm
                    target_z_mm = candidate_z_mm
                    solved_id1 = candidate_id1
                    solved_id2 = candidate_id2
                    ik_error_mm = candidate_error_mm
                    break
        if self.use_post_center_tick_bias:
            solved_id1, solved_id2, ik_error_mm = self._apply_ik_calibration(
                solved_id1,
                solved_id2,
                target_x_mm,
                target_z_mm,
            )
        if phase == "descend" and self.use_post_center_tick_bias:
            solved_id2 = self._clamp(
                solved_id2 + GRASP_DESCEND_ID2_BIAS_TICKS,
                self.id2_limits,
            )
            solved_id1, solved_id2 = enforce_angle_gap(
                solved_id1,
                solved_id2,
                self.id2_limits,
                self.angle_gap_degrees,
            )
            actual_x_mm, actual_z_mm = gripper_position_mm(solved_id1, solved_id2)
            ik_error_mm = math.hypot(
                actual_x_mm - target_x_mm,
                actual_z_mm - target_z_mm,
            )
        planned_move_x_mm = move_x_mm * progress_ratio
        planned_move_z_mm = move_z_mm * progress_ratio
        plan = {
            "current_x_mm": current_x_mm,
            "current_z_mm": current_z_mm,
            "forward_mm": planned_move_x_mm,
            "vertical_target_mm": planned_move_z_mm,
            "requested_forward_mm": move_x_mm,
            "requested_vertical_mm": move_z_mm,
            "progress_ratio": progress_ratio,
            "target_distance_mm": 0.0,
            "distance_mm": camera_to_target_mm if camera_to_target_mm is not None else abs(move_x_mm),
            "lateral_mm": 0.0,
            "target_kind": target_kind,
            "target_source": target_source,
            "target_size_mm": target_size_mm,
            "distance_scaled_down_mm": abs(move_z_mm),
            "extra_down_mm": extra_down_mm,
            "id1": solved_id1,
            "id2": solved_id2,
            "ik_error_mm": ik_error_mm,
            "phase": phase,
        }
        if offsets is not None:
            plan.update(offsets)
        return plan, (
            f"triangle Dcam-target={camera_to_target_mm:.0f}mm "
            f"Dcam-gripper={camera_to_gripper_mm:.0f}mm "
            f"Dgripper-target={gripper_to_target_mm:.0f}mm "
            if camera_to_target_mm is not None and gripper_to_target_mm is not None
            else f"post-center fixed move x={move_x_mm:.0f}mm "
        ) + (
            f"phase={phase} move x={planned_move_x_mm:.0f}/{move_x_mm:.0f}mm "
            f"z={planned_move_z_mm:.0f}/{move_z_mm:.0f}mm "
            f"extra_down={extra_down_mm:.0f}mm "
            f"progress={progress_ratio * 100:.0f}%: "
            f"ID1 {self.id1}->{solved_id1} "
            f"ID2 {self.id2}->{solved_id2} err={ik_error_mm:.1f}mm"
        )

    def _update_locked_grasp(self, frame_shape, now, can_preview_step, live_target=None):
        can_command = now - self.last_command_time >= self.command_interval_s

        if self.algorithm_stage == "open":
            self.state = "locked target open claw"
            if not self.servo_bridge.write_enabled:
                next_id4 = min(self.id4_open, self.id4 + 20)
                if can_preview_step:
                    self.id4 = next_id4
                    self.last_preview_step_time = now
                if self.id4 >= self.id4_open:
                    self.id4 = self.id4_open
                    if self.simple_vertical_grasp:
                        self.algorithm_stage = "vertical_wait_open"
                        self.stage_deadline = time.monotonic() + self._zp_settle_s()
                    elif self.post_center_direct_descend:
                        self.next_stage_after_open = "descend"
                        self.algorithm_stage = "open_wait"
                        self.stage_deadline = time.monotonic() + self._zp_settle_s()
                    else:
                        self.next_stage_after_open = "post_lock_visual_confirm"
                        self.algorithm_stage = "open_wait"
                        self.stage_deadline = time.monotonic() + self._zp_settle_s()
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return (
                    f"preview locked target; opening claw "
                    f"ID7={self.id4}->{self.id4_open}"
                )
            if can_command:
                self.state = "claw_open_ack"
                result = self._send_disc_catch_open_claw_and_splitter()
                if self.servo_bridge.last_command_ok:
                    if self.simple_vertical_grasp:
                        self.algorithm_stage = "vertical_wait_open"
                        self.stage_deadline = time.monotonic() + self._zp_settle_s()
                    elif self.post_center_direct_descend:
                        self.next_stage_after_open = "descend"
                        self.algorithm_stage = "open_wait"
                        self.stage_deadline = time.monotonic() + self._zp_settle_s()
                    else:
                        self.next_stage_after_open = "post_lock_visual_confirm"
                        self.algorithm_stage = "open_wait"
                        self.stage_deadline = time.monotonic() + self._zp_settle_s()
                return result
            return "locked target; waiting to open claw"

        if self.algorithm_stage == "open_wait":
            self.state = "locked target waiting open claw"
            if now < self.stage_deadline:
                return f"locked target; waiting open claw {self.stage_deadline - now:.1f}s"
            if self.next_stage_after_open is not None:
                next_stage = self.next_stage_after_open
            elif self.simple_vertical_grasp:
                next_stage = "vertical_descend"
            elif self.post_center_direct_descend:
                next_stage = "descend"
            else:
                next_stage = "post_lock_visual_confirm"
            print(
                "GRASP STAGE open_wait complete "
                f"next={next_stage} direct_descend={self.post_center_direct_descend} "
                f"locked={self.locked_target is not None}",
                flush=True,
            )
            self.algorithm_stage = next_stage
            self.next_stage_after_open = None

        if self.algorithm_stage == "post_lock_visual_confirm":
            self.state = "post-lock visual confirm"
            accepted_target, reason = self._accept_live_target(live_target)
            if accepted_target is None:
                self.visual_confirm_frames = 0
                if self.visual_lost_frames <= max(3, self.stable_frames_required):
                    self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                    return f"post-lock visual confirm waiting: {reason}"
                return self._abort_to_standby(f"post-lock visual confirm failed: {reason}")
            centered, message = self._visual_center_step(
                accepted_target,
                frame_shape,
                now,
                can_preview_step,
                "post-lock visual confirm",
            )
            if not centered:
                self.visual_confirm_frames = 0
                if "blocked by limits" in message:
                    return self._abort_to_standby(message)
                return message
            self.visual_confirm_frames += 1
            if self.visual_confirm_frames < max(2, self.stable_frames_required):
                return (
                    f"{message} confirm "
                    f"{self.visual_confirm_frames}/{max(2, self.stable_frames_required)}"
                )
            plan, plan_text = self._post_center_plan(
                accepted_target,
                frame_shape,
                "descend",
            )
            self.locked_plan = plan
            self.arm_preview.publish_plan_marker(plan, plan_text)
            print(
                "GRASP VISUAL LOCK "
                f"target={json.dumps(self._copy_target(accepted_target), ensure_ascii=True, default=str)} "
                f"plan={plan_text}",
                flush=True,
            )
            if plan["ik_error_mm"] > self.post_center_ik_error_mm:
                self.state = "post-center unreachable"
                self.algorithm_stage = "fault"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return f"visual locked target IK failed: {plan_text}"
            self.visual_descend_start = (self.id1, self.id2)
            self.visual_descend_step_index = 0
            self.visual_lost_frames = 0
            self.algorithm_stage = "visual_descend"

        if self.algorithm_stage == "visual_step_wait":
            self.state = "visual descend waiting arm"
            if now < self.stage_deadline:
                return f"visual descend; waiting arm {self.stage_deadline - now:.1f}s"
            self.visual_descend_step_index += 1
            if self.visual_descend_step_index >= self.visual_descend_steps:
                self.algorithm_stage = "final_grab"
            else:
                self.algorithm_stage = "visual_descend"

        if self.algorithm_stage == "visual_descend":
            self.state = "visual descend"
            if self.locked_plan is None or self.visual_descend_start is None:
                self.algorithm_stage = "post_lock_visual_confirm"
                return "visual descend waiting for plan"
            step_index = self.visual_descend_step_index
            step_total = max(1, self.visual_descend_steps)
            ratio = min(1.0, float(step_index + 1) / float(step_total))
            start_id1, start_id2 = self.visual_descend_start
            goal_id1 = self._clamp(self.locked_plan["id1"], self.id1_limits)
            goal_id2 = self._clamp(self.locked_plan["id2"], self.id2_limits)
            target_id1 = self._clamp(
                start_id1 + (goal_id1 - start_id1) * ratio,
                self.id1_limits,
            )
            target_id2 = self._clamp(
                start_id2 + (goal_id2 - start_id2) * ratio,
                self.id2_limits,
            )
            target_id6 = self.id6
            visual_note = "no live visual correction"
            accepted_target, reason = self._accept_live_target(live_target)
            if accepted_target is not None:
                error_x, error_y = self._target_error(accepted_target, frame_shape)
                delta_id6 = self._id6_centering_delta(error_x)
                delta_id2 = self._centering_delta(-error_y, self.id2_pixel_gain)
                target_id2 = self._clamp(target_id2 + delta_id2, self.id2_limits)
                target_id6 = self._clamp(self.id6 + delta_id6, ID6_SAFE_LIMITS)
                visual_note = (
                    f"live dx={error_x:.0f} dy={error_y:.0f} "
                    f"dID2={delta_id2} dID6={delta_id6}"
                )
            else:
                if step_index == 0:
                    return self._abort_to_standby(f"visual descend failed before first step: {reason}")
                if self.visual_lost_frames > 4 and step_index < step_total - 1:
                    return self._abort_to_standby(f"visual descend lost target: {reason}")

            target_id1, target_id2 = enforce_angle_gap(
                target_id1,
                target_id2,
                self.id2_limits,
                self.angle_gap_degrees,
            )
            if not self.servo_bridge.write_enabled:
                if can_preview_step:
                    self.id1 = target_id1
                    self.id2 = target_id2
                    self.id6 = target_id6
                    self.visual_descend_step_index += 1
                    self.last_preview_step_time = now
                    if self.visual_descend_step_index >= step_total:
                        self.algorithm_stage = "final_grab"
                self.arm_preview.set_targets(target_id1, target_id2, self.id4, target_id6)
                return (
                    f"preview visual descend step={step_index + 1}/{step_total} "
                    f"ID1={target_id1} ID2={target_id2} ID6={target_id6} | {visual_note}"
                )
            if not can_command:
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return (
                    f"waiting visual descend step={step_index + 1}/{step_total} | "
                    f"{visual_note}"
                )
            self.id1 = target_id1
            self.id2 = target_id2
            self.id6 = target_id6
            result = self._send(
                f"visual descend step={step_index + 1}/{step_total} | {visual_note}",
                require_feedback=True,
            )
            if self.servo_bridge.last_command_ok:
                self.algorithm_stage = "visual_step_wait"
                self.stage_deadline = time.monotonic() + self._arm_settle_s()
            return result

        if self.algorithm_stage == "final_grab":
            self.state = "final grab"
            self.post_center_move_complete = True
            self.algorithm_stage = "close"

        if self.algorithm_stage == "vertical_wait_open":
            self.state = "vertical grasp waiting open claw"
            if now < self.stage_deadline:
                return f"vertical grasp; waiting open claw {self.stage_deadline - now:.1f}s"
            self.algorithm_stage = "vertical_descend"

        if self.algorithm_stage == "vertical_descend":
            self.state = "vertical grasp descend"
            target_id1 = self._clamp(self.vertical_grasp_id1, self.id1_limits)
            target_id2 = self._clamp(self.vertical_grasp_id2, self.id2_limits)
            target_id1, target_id2 = enforce_angle_gap(
                target_id1,
                target_id2,
                self.id2_limits,
                self.angle_gap_degrees,
            )
            if not self.servo_bridge.write_enabled:
                if can_preview_step:
                    self.id1 = target_id1
                    self.id2 = target_id2
                    self.last_preview_step_time = now
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                self.algorithm_stage = "close"
                return f"preview vertical descend ID1={self.id1} ID2={self.id2}"
            if can_command:
                self.id1 = target_id1
                self.id2 = target_id2
                result = self._send(
                    f"vertical descend only ID1={self.id1} ID2={self.id2}",
                    require_feedback=False,
                )
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "vertical_wait_descend"
                    self.stage_deadline = time.monotonic() + self._arm_settle_s()
                return result
            return "vertical grasp; waiting to descend"

        if self.algorithm_stage == "vertical_wait_descend":
            self.state = "vertical grasp waiting descend"
            if now < self.stage_deadline:
                return f"vertical grasp; waiting descend {self.stage_deadline - now:.1f}s"
            self.algorithm_stage = "close"

        if self.algorithm_stage == "motion_wait":
            motion_stage = self.motion_stage_after_wait or "descend"
            self.state = f"locked target waiting {motion_stage}"
            if now < self.stage_deadline:
                return (
                    f"locked target; waiting {motion_stage} "
                    f"{self.stage_deadline - now:.1f}s"
                )
            if motion_stage == "overhead":
                self.algorithm_stage = "descend"
                self.locked_plan = None
                self.approach_attempts = 0
                self.state = "overhead_reached"
            else:
                self.post_center_move_complete = True
                self.algorithm_stage = "close"
                self.state = "descend_reached"
            self.motion_stage_after_wait = None

        if self.algorithm_stage in ("overhead", "descend") and self.locked_plan is None:
            plan, plan_text = self._post_center_plan(
                self.locked_target,
                frame_shape,
                self.algorithm_stage,
            )
            self.locked_plan = plan
            self.arm_preview.publish_plan_marker(plan, plan_text)
            print(
                "GRASP LOCK "
                f"target={json.dumps(self.locked_target, ensure_ascii=True, default=str)} "
                f"plan={plan_text}",
                flush=True,
            )
            if plan["ik_error_mm"] > self.post_center_ik_error_mm:
                self.state = "post-center unreachable"
                self.algorithm_stage = "fault"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return f"claw opened; locked target IK failed: {plan_text}"
        elif self.algorithm_stage in ("overhead", "descend"):
            plan = self.locked_plan
            plan_text = (
                f"locked IK ID1={plan['id1']} ID2={plan['id2']} "
                f"progress={plan.get('progress_ratio', 1.0) * 100:.0f}% "
                f"error={plan['ik_error_mm']:.1f}mm"
            )

        if self.algorithm_stage in ("overhead", "descend"):
            motion_stage = self.algorithm_stage
            self.state = f"locked target {motion_stage}"
            if not self.servo_bridge.write_enabled:
                next_id1 = self._clamp(
                    self.id1 + self._limited_delta(plan["id1"] - self.id1),
                    self.id1_limits,
                )
                next_id2 = self._clamp(
                    self.id2 + self._limited_delta(plan["id2"] - self.id2),
                    self.id2_limits,
                )
                next_id1, next_id2 = enforce_angle_gap(
                    next_id1,
                    next_id2,
                    self.id2_limits,
                    self.angle_gap_degrees,
                )
                if can_preview_step:
                    self.id1, self.id2 = next_id1, next_id2
                    self.last_preview_step_time = now
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                if self.id1 == plan["id1"] and self.id2 == plan["id2"]:
                    if motion_stage == "overhead":
                        self.algorithm_stage = "descend"
                        self.locked_plan = None
                        self.approach_attempts = 0
                    else:
                        self.post_center_move_complete = True
                        self.algorithm_stage = "close"
                return (
                    f"preview algorithm {motion_stage} ID1={self.id1}->{plan['id1']} "
                    f"ID2={self.id2}->{plan['id2']} | {plan_text}"
                )
            target_id1 = self._clamp(plan["id1"], self.id1_limits)
            target_id2 = self._clamp(plan["id2"], self.id2_limits)
            if self.approach_attempts > 0:
                error_id1 = abs(self.id1 - target_id1)
                error_id2 = abs(self.id2 - target_id2)
                if (
                    error_id1 <= self.approach_feedback_tolerance
                    and error_id2 <= self.approach_feedback_tolerance
                ):
                    if motion_stage == "overhead":
                        self.algorithm_stage = "descend"
                        self.locked_plan = None
                        self.approach_attempts = 0
                        self.state = "overhead_reached"
                    else:
                        self.post_center_move_complete = True
                        self.algorithm_stage = "close"
                        self.state = "descend_reached"
                    return (
                        f"{motion_stage} reached ID1={self.id1}/{target_id1} "
                        f"ID2={self.id2}/{target_id2}"
                    )
                if self.approach_attempts >= 3:
                    self.algorithm_stage = "fault"
                    self.state = "fault"
                    self.status = (
                        f"{motion_stage} feedback outside +/-{self.approach_feedback_tolerance}: "
                        f"ID1={self.id1}/{target_id1} ID2={self.id2}/{target_id2}"
                    )
                    return self.status
            if can_command:
                self.id1 = target_id1
                self.id2 = target_id2
                self.approach_attempts += 1
                self.state = "approach_feedback"
                result = self._send(
                    f"locked target algorithm {motion_stage} attempt={self.approach_attempts}/3 | {plan_text}",
                    require_feedback=True,
                )
                if (
                    self.servo_bridge.last_command_ok
                    and getattr(self.servo_bridge, "assumed_feedback", False)
                ):
                    self.motion_stage_after_wait = motion_stage
                    self.algorithm_stage = "motion_wait"
                    self.stage_deadline = time.monotonic() + self._arm_settle_s()
                return result
            return f"locked target; waiting algorithm {motion_stage}"

        if self.algorithm_stage == "close":
            self.state = "locked target close claw"
            if not self.servo_bridge.write_enabled:
                self.id4 = self.id4_closed
                if self.post_center_direct_descend and not self.abort_after_return:
                    self.algorithm_stage = "post_grab_id2_retreat"
                else:
                    self.algorithm_stage = "return"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return "preview claw closed"
            if can_command:
                self.state = "claw_close_ack"
                result = self._send_id4(self.id4_closed, "locked target; close claw")
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "close_wait"
                    self.stage_deadline = time.monotonic() + self._zp_settle_s()
                return result
            return "locked target; waiting to close claw"

        if self.algorithm_stage == "close_wait":
            self.state = "locked target waiting close claw"
            if now < self.stage_deadline:
                return f"locked target; waiting close claw {self.stage_deadline - now:.1f}s"
            if self.post_center_direct_descend and not self.abort_after_return:
                self.algorithm_stage = "post_grab_id2_retreat"
            else:
                self.algorithm_stage = "return"
            self.return_attempts = 0

        if self.algorithm_stage == "post_grab_id2_retreat":
            self.state = "post-grab retreat"
            target_id1 = self._clamp(READY_ID1_TICK, self.id1_limits)
            target_id2 = self._clamp(POST_GRAB_ID2_RETREAT_TICK, self.id2_limits)
            target_id6 = BASE_YAW_CENTER_TICK
            target_id1, target_id2 = enforce_angle_gap(
                target_id1,
                target_id2,
                self.id2_limits,
                self.angle_gap_degrees,
            )
            if not self.servo_bridge.write_enabled:
                self.id1 = target_id1
                self.id2 = target_id2
                self.id6 = target_id6
                self.id5 = CATCHER_RELEASE_READY_TICK
                self.algorithm_stage = "release"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return (
                    f"preview post-grab retreat ID1={self.id1} "
                    f"ID2={self.id2} ID5={self.id5} ID6={self.id6}"
                )
            if can_command:
                self.id1 = target_id1
                self.id2 = target_id2
                self.id6 = target_id6
                result = self._send_retreat_with_catcher_open(
                    f"post-grab retreat/open catcher ID1={self.id1} "
                    f"ID2={self.id2} ID5={CATCHER_RELEASE_READY_TICK} ID6={self.id6}",
                    require_feedback=True,
                )
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "post_grab_id2_retreat_wait"
                    self.stage_deadline = time.monotonic() + self._arm_settle_s()
                return result
            return "post-grab retreat; waiting to move ID2"

        if self.algorithm_stage == "post_grab_id2_retreat_wait":
            self.state = "post-grab ID2 retreat wait"
            if now < self.stage_deadline:
                return f"post-grab retreat; waiting arm {self.stage_deadline - now:.1f}s"
            self.algorithm_stage = "release"

        if self.algorithm_stage == "return":
            self.state = "returning to standby"
            target_id1 = READY_ID1_TICK
            target_id2 = READY_ID2_TICK
            target_id6 = BASE_YAW_CENTER_TICK
            if not self.servo_bridge.write_enabled:
                self.id1 = target_id1
                self.id2 = target_id2
                self.id6 = target_id6
                self.id4 = self.id4_closed
                if self.abort_after_return:
                    return self._reset_cycle_for_search("preview abort returned to standby")
                self.algorithm_stage = "complete"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return "preview grasp complete; returned to standby"
            if self.return_attempts > 0:
                error_id1 = abs(self.id1 - target_id1)
                error_id2 = abs(self.id2 - target_id2)
                if (
                    error_id1 <= self.approach_feedback_tolerance
                    and error_id2 <= self.approach_feedback_tolerance
                ):
                    if self.abort_after_return:
                        return self._reset_cycle_for_search("abort returned to standby")
                    self.algorithm_stage = "catcher"
                    self.state = "standby prepare catcher"
                    self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                    return (
                        f"standby reached; preparing catcher ID1={self.id1} "
                        f"ID2={self.id2} ID6={self.id6} ID7={self.id4}"
                    )
                if self.return_attempts >= 3:
                    self.algorithm_stage = "fault"
                    self.state = "fault"
                    self.status = (
                        f"standby feedback outside +/-{self.approach_feedback_tolerance}: "
                        f"ID1={self.id1}/{target_id1} ID2={self.id2}/{target_id2}"
                    )
                    return self.status
            if can_command:
                self.id1 = target_id1
                self.id2 = target_id2
                self.id6 = target_id6
                self.id4 = self.id4_closed
                self.return_attempts += 1
                result = self._send(
                    f"return standby attempt={self.return_attempts}/3",
                    require_feedback=True,
                )
                if (
                    self.servo_bridge.last_command_ok
                    and getattr(self.servo_bridge, "assumed_feedback", False)
                ):
                    self.algorithm_stage = "return_wait"
                    self.stage_deadline = time.monotonic() + self._arm_settle_s()
                return result
            return "grasp complete; waiting to return standby"

        if self.algorithm_stage == "return_wait":
            self.state = "returning to standby"
            if now < self.stage_deadline:
                return f"return standby; waiting arm {self.stage_deadline - now:.1f}s"
            if self.abort_after_return:
                return self._reset_cycle_for_search("abort returned to standby")
            self.algorithm_stage = "catcher"
            self.state = "standby prepare catcher"
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return (
                f"standby reached; preparing catcher ID1={self.id1} "
                f"ID2={self.id2} ID6={self.id6} ID7={self.id4}"
            )

        if self.algorithm_stage == "catcher":
            self.state = "extend catcher before release"
            if not self.servo_bridge.write_enabled:
                self.id5 = CATCHER_RELEASE_READY_TICK
                self.algorithm_stage = "release"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return "preview catcher ready; releasing next"
            if can_command:
                result = self._send_id5(
                    CATCHER_RELEASE_READY_TICK,
                    "extend catcher before release",
                    critical=True,
                )
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "release"
                return result
            return "standby; waiting to extend catcher"

        if self.algorithm_stage == "release":
            release_reason = (
                "release target after ID2 retreat"
                if self.post_center_direct_descend and not self.abort_after_return
                else "release target at standby"
            )
            self.state = release_reason
            if not self.servo_bridge.write_enabled:
                self.id4 = self.id4_open
                self.algorithm_stage = "close_catcher_after_release"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return "preview release target; closing catcher next"
            if can_command:
                result = self._send_id4(self.id4_open, release_reason)
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "release_wait"
                    self.stage_deadline = time.monotonic() + self._zp_settle_s() + 0.30
                return result
            return "standby; waiting to release target"

        if self.algorithm_stage == "release_wait":
            self.state = "release target at standby"
            if now < self.stage_deadline:
                return f"standby; waiting release {self.stage_deadline - now:.1f}s"
            self.algorithm_stage = "close_catcher_after_release"

        if self.algorithm_stage == "close_catcher_after_release":
            self.state = "close catcher after release"
            if not self.servo_bridge.write_enabled:
                self.id5 = CATCHER_HOME_TICK
                self.algorithm_stage = "reclose"
                return "preview catcher closed after release"
            if self.id5 == CATCHER_HOME_TICK:
                self.algorithm_stage = "reclose"
            elif can_command:
                result = self._send_id5(
                    CATCHER_HOME_TICK,
                    "close catcher 0.3s after ID7 release",
                    critical=False,
                )
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "reclose"
                return result
            else:
                return "release complete; waiting to close catcher"

        if self.algorithm_stage == "reclose":
            self.state = "close claw for next search"
            if not self.servo_bridge.write_enabled:
                self.id4 = self.id4_closed
                self.id5 = CATCHER_HOME_TICK
                return self._reset_cycle_for_search(
                    "preview ready for next target",
                    start_retrigger_cooldown=True,
                    cycle_complete=True,
                )
            if can_command:
                result = self._send_id4(self.id4_closed, "close claw for next search")
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "reclose_wait"
                    self.stage_deadline = time.monotonic() + self._zp_settle_s()
                return result
            return "release complete; waiting to close claw for next search"

        if self.algorithm_stage == "reclose_wait":
            self.state = "close claw for next search"
            if now < self.stage_deadline:
                return f"waiting close claw for next search {self.stage_deadline - now:.1f}s"
            if self.post_center_direct_descend and not self.abort_after_return:
                self.algorithm_stage = "final_return"
            else:
                if self.id5 != CATCHER_HOME_TICK:
                    home_result = self._send_id5(
                        CATCHER_HOME_TICK,
                        "catcher home for next search",
                        critical=False,
                    )
                    if not self.servo_bridge.last_command_ok:
                        return home_result
                return self._reset_cycle_for_search(
                    f"ready for next target ID1={self.id1} ID2={self.id2} "
                    f"ID5={self.id5} ID6={self.id6} ID7={self.id4}",
                    start_retrigger_cooldown=True,
                    cycle_complete=True,
                )

        if self.algorithm_stage == "final_return":
            self.state = "final return to standby"
            target_id1 = READY_ID1_TICK
            target_id2 = READY_ID2_TICK
            target_id6 = BASE_YAW_CENTER_TICK
            if not self.servo_bridge.write_enabled:
                self.id1 = target_id1
                self.id2 = target_id2
                self.id6 = target_id6
                return self._reset_cycle_for_search(
                    "preview final return to standby",
                    start_retrigger_cooldown=True,
                    cycle_complete=True,
                )
            if can_command:
                self.id1 = target_id1
                self.id2 = target_id2
                self.id6 = target_id6
                result = self._send(
                    "final return standby after release/reclose",
                    require_feedback=True,
                )
                if self.servo_bridge.last_command_ok:
                    self.algorithm_stage = "final_return_wait"
                    self.stage_deadline = time.monotonic() + self._arm_settle_s()
                return result
            return "final return; waiting to move standby"

        if self.algorithm_stage == "final_return_wait":
            self.state = "final return waiting arm"
            if now < self.stage_deadline:
                return f"final return; waiting arm {self.stage_deadline - now:.1f}s"
            if self.id5 != CATCHER_HOME_TICK:
                home_result = self._send_id5(
                    CATCHER_HOME_TICK,
                    "catcher home for next search",
                    critical=False,
                )
                if not self.servo_bridge.last_command_ok:
                    return home_result
            return self._reset_cycle_for_search(
                f"ready for next target ID1={self.id1} ID2={self.id2} "
                f"ID5={self.id5} ID6={self.id6} ID7={self.id4}",
                start_retrigger_cooldown=True,
                cycle_complete=True,
            )

        if self.algorithm_stage == "fault":
            return self.status

        self.state = "algorithm grasp complete"
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        return (
            f"algorithm grasp complete ID1={self.id1} "
            f"ID2={self.id2} ID6={self.id6} ID7={self.id4}; camera ignored"
        )

    def _update_one_shot(self, target, frame_shape):
        if self.one_shot_complete:
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return "one-shot grasp complete; servos stopped"

        if self.one_shot_target is None:
            ready, reason = self._target_ready(target)
            if not ready:
                self.arm_preview.set_targets(READY_ID1_TICK, READY_ID2_TICK, self.id4, self.id6)
                return reason
            self.one_shot_target = self._copy_target(target)
            self.centered_frames = self.stable_frames_required

        target = self.one_shot_target
        plan, plan_text = self._one_shot_plan(target, frame_shape)

        if not self.servo_bridge.write_enabled:
            if plan is None:
                self.arm_preview.set_targets(READY_ID1_TICK, READY_ID2_TICK, self.id4, self.id6)
                return plan_text
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            self.arm_preview.publish_plan_marker(plan, plan_text)
            return f"preview one-shot {plan_text}"

        now = time.monotonic()
        can_command = now - self.last_command_time >= self.command_interval_s
        if not can_command:
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return "one-shot waiting command interval"

        if self.id4 != self.id4_open:
            self.state = "one-shot open claw"
            return self._send_id4(self.id4_open, "one-shot open claw")

        if not self.one_shot_approach_sent:
            if plan is None:
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return plan_text
            if plan["ik_error_mm"] > self.max_one_shot_ik_error_mm:
                self.state = "unreachable"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                self.arm_preview.publish_plan_marker(plan, plan_text)
                return (
                    f"IK simulation failed: {plan_text}; "
                    "real servos not commanded"
                )
            self.id1 = self._clamp(plan["id1"], self.id1_limits)
            self.id2 = self._clamp(plan["id2"], self.id2_limits)
            self.one_shot_approach_sent = True
            self.state = "one-shot approach"
            self.arm_preview.publish_plan_marker(plan, plan_text)
            return self._send(
                f"one-shot simulated approach {plan_text}",
                require_feedback=True,
            )

        if self.id4 != self.id4_closed:
            self.state = "one-shot close claw"
            return self._send_id4(self.id4_closed, "one-shot close claw")

        self.one_shot_complete = True
        self.state = "one-shot complete"
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        self.arm_preview.publish("one-shot grasp complete; servos stopped")
        if plan is None:
            return "one-shot grasp complete; servos stopped"
        return f"one-shot grasp complete {plan_text}; servos stopped"

    def update(self, target, frame_shape):
        if not self.enabled:
            return self.status
        if self.active_station_task == "DISC_CATCH":
            self.id1 = DISC_CATCH_FIXED_ID1_TICK
            self.id2 = DISC_CATCH_FIXED_ID2_TICK
        now = time.monotonic()
        can_preview_step = (
            now - self.last_preview_step_time >= self.preview_step_interval_s
        )
        if self.read_only_sync:
            feedback_ready = False
            feedback = None
            with self.read_only_feedback_lock:
                if self.read_only_feedback_ready:
                    feedback_ready = True
                    feedback = self.read_only_feedback
                    self.read_only_feedback_ready = False
            if feedback_ready:
                if feedback is not None and self._apply_feedback(feedback):
                    self.state = "read_only_sync"
                    self.status = (
                        f"read-only synchronized ID1={self.id1} ID2={self.id2} "
                        f"ID4={self.id4} ID6={self.arm_preview.id6}"
                    )
                elif feedback is None:
                    self.status = f"read-only sync waiting | {self.servo_bridge.status}"
            if now - self.last_feedback_attempt >= 1.0:
                self.last_feedback_attempt = now
                self._start_read_only_feedback()
            self.arm_preview.publish(self.status, state="READ_ONLY_SYNC")
            return self.status
        if self.servo_bridge.write_enabled and self.startup_stage != "complete":
            return self._update_startup(now)
        if self.servo_bridge.write_enabled and not self.synchronized:
            if now - self.last_feedback_attempt < 0.8:
                return self.status
            self.last_feedback_attempt = now
            feedback = self.servo_bridge.query_positions()
            if feedback is None:
                self.status = "waiting for RCT6 position synchronization"
                return f"{self.status} | {self.servo_bridge.status}"
            if not self._apply_feedback(feedback):
                return self.status
            self.state = "searching"
            self.status = f"synchronized ID1={self.id1} ID2={self.id2}"
            return self.status

        if self.feedback_pending:
            if now < self.feedback_due:
                return f"moving; feedback in {self.feedback_due - now:.1f}s"
            feedback = self.servo_bridge.query_positions()
            if feedback is None:
                self.feedback_failures += 1
                if self.feedback_failures >= 3:
                    self.state = "fault"
                    self.feedback_pending = False
                    self.status = "servo feedback timeout; automatic motion stopped"
                    return self.status
                self.feedback_due = now + 0.5
                return f"servo feedback retry {self.feedback_failures}/3"
            if not self._apply_feedback(feedback):
                return self.status

        if self.state == "fault":
            return self.status
        if self.one_shot:
            return self._update_one_shot(target, frame_shape)
        live_target = target
        if self.locked_target is not None:
            return self._update_locked_grasp(
                frame_shape,
                now,
                can_preview_step,
                live_target=live_target,
            )
        cooldown_remaining = self.ignore_new_targets_until - now
        ignoring_new_target = target is not None and cooldown_remaining > 0.0
        if ignoring_new_target:
            target = None
            self.centered_frames = 0
            self.last_visual_target = None
        if target is None:
            self.centered_frames = 0
            self.centering_correction_count = 0
            if not ignoring_new_target:
                self.visual_lost_frames += 1
                if self.visual_lost_frames > 6:
                    self.last_visual_target = None
            self.state = "post-grasp cooldown" if ignoring_new_target else "searching"
            can_command = now - self.last_command_time >= self.command_interval_s
            expand_id1 = self._clamp(
                self.id1 + self._limited_delta(READY_ID1_TICK - self.id1),
                self.id1_limits,
            )
            expand_id2 = self._clamp(
                self.id2 + self._limited_delta(READY_ID2_TICK - self.id2),
                self.id2_limits,
            )
            expand_id6 = self._clamp(
                self.id6 + self._limited_delta(BASE_YAW_CENTER_TICK - self.id6),
                ID6_SAFE_LIMITS,
            )
            expand_id1, expand_id2 = enforce_angle_gap(
                expand_id1,
                expand_id2,
                self.id2_limits,
                self.angle_gap_degrees,
            )
            if not self.servo_bridge.write_enabled:
                if can_preview_step:
                    self.id1, self.id2, self.id6 = expand_id1, expand_id2, expand_id6
                    self.last_preview_step_time = now
                self.arm_preview.set_targets(expand_id1, expand_id2, self.id4, expand_id6)
                return (
                    f"preview {self.state}; slow expand "
                    f"ID1={expand_id1} ID2={expand_id2} ID6={expand_id6}"
                )
            if can_command and (
                expand_id1 != self.id1
                or expand_id2 != self.id2
                or expand_id6 != self.id6
            ):
                self.id1, self.id2, self.id6 = expand_id1, expand_id2, expand_id6
                reason = (
                    "post-grasp cooldown; slow expand"
                    if ignoring_new_target
                    else "searching red target; slow expand"
                )
                return self._send(reason, require_feedback=False)
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            if ignoring_new_target:
                return (
                    f"post-grasp cooldown {max(0.0, cooldown_remaining):.1f}s; "
                    "new target ignored"
                )
            return "grasp searching red target"
        edge_note = ""
        hold_reason = self._centering_target_hold_reason(target, frame_shape)
        if hold_reason is not None:
            self.centered_frames = 0
            self._reset_ring_lock_guard()
            self.state = "target unstable"
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return hold_reason
        if target.get("area_percent", 0.0) < self.min_target_area_percent:
            self.centered_frames = 0
            self.state = "target too small"
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return (
                f"red square area below {self.min_target_area_percent:.2f}%; "
                "grasp disabled"
            )
        distance_cm = target.get("distance_cm")
        if (
            distance_cm is None
            or not self.min_target_distance_cm
            <= distance_cm
            <= self.max_target_distance_cm
        ):
            self.centered_frames = 0
            self.state = "target outside calibrated range"
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            distance_text = "unknown" if distance_cm is None else f"{distance_cm:.1f}cm"
            return (
                f"target distance {distance_text} outside calibrated "
                f"{self.min_target_distance_cm:.0f}-"
                f"{self.max_target_distance_cm:.0f}cm range"
            )

        height, width = frame_shape[:2]
        cx, cy = target["center"]
        error_x = cx - width / 2.0
        error_y = cy - height / 2.0
        square_distance_mm = None if distance_cm is None else distance_cm * 10.0
        gripper_distance_mm = None
        if square_distance_mm is not None:
            gripper_distance_mm = square_distance_mm - self.camera_gripper_offset_mm

        can_command = now - self.last_command_time >= self.command_interval_s
        if self.locked_target is None:
            centering_profile = self._centering_profile()
            center_deadband_px = self._center_deadband_for_target(target)
            delta_id6 = self._id6_centering_delta(
                error_x,
                centering_profile["id6_gain"],
                centering_profile["id6_max_step_ticks"],
                center_deadband_px,
            )
            delta_id2 = self._centering_delta(
                -error_y,
                centering_profile["id2_gain"],
                centering_profile["id2_max_step_ticks"],
                center_deadband_px,
            )
            target_id2 = self._clamp_center_id2(self.id2 + delta_id2)
            id2_center_limited = self._id2_center_limit_blocks(delta_id2, target_id2)
            if id2_center_limited:
                delta_id2 = 0
                if delta_id6 == 0:
                    self.state = "centered at ID2 limit"
            delta_id2, delta_id6, axis_note = self._vector_centering_deltas(
                delta_id2,
                delta_id6,
                id2_center_limited,
                error_x,
                error_y,
                centering_profile["id2_max_step_ticks"],
                centering_profile["id6_max_step_ticks"],
            )
            target_id2 = self._clamp_center_id2(self.id2 + delta_id2)
            if delta_id2 != 0 or delta_id6 != 0:
                self.centered_frames = 0
                self._reset_ring_lock_guard()
                self.horizontal_correction_done = False
                target_id6 = self._clamp(self.id6 + delta_id6, ID6_SAFE_LIMITS)
                target_id1 = self.id1
                target_id1, target_id2 = enforce_angle_gap(
                    target_id1,
                    target_id2,
                    self.id2_limits,
                    self.angle_gap_degrees,
                )
                self.state = "centering target"
                if not self.servo_bridge.write_enabled:
                    if can_preview_step:
                        self.id2 = target_id2
                        self.id6 = target_id6
                        self.last_preview_step_time = now
                    self.arm_preview.set_targets(target_id1, target_id2, self.id4, target_id6)
                    return (
                        f"preview centering dx={error_x:.0f} dy={error_y:.0f} "
                        f"ID2={target_id2} ID6={target_id6}"
                    )
                can_command = now - self.last_command_time >= centering_profile["command_interval_s"]
                if can_command:
                    previous_id2 = self.id2
                    previous_id6 = self.id6
                    self.id2 = target_id2
                    self.id6 = target_id6
                    send_id2 = self.id2 != previous_id2
                    send_id6 = self.id6 != previous_id6
                    if not send_id2 and not send_id6:
                        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                        return (
                            f"centering blocked by servo limits dx={error_x:.0f} dy={error_y:.0f} "
                            f"ID2={self.id2} ID6={self.id6}"
                        )
                    return self._send_center_correction(
                        f"centering target{edge_note} dx={error_x:.0f} dy={error_y:.0f} "
                        f"axis={axis_note} dID2={self.id2 - previous_id2} "
                        f"dID6={self.id6 - previous_id6} {centering_profile['note']}",
                        send_id2=send_id2,
                        send_id6=send_id6,
                    )
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return f"waiting centering dx={error_x:.0f} dy={error_y:.0f}"

            ring_hold_reason = self._ring_lock_hold_reason(
                target,
                error_x,
                error_y,
                id2_center_limited,
            )
            if ring_hold_reason is not None:
                self.state = "ring lock guard"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return ring_hold_reason

            self.centered_frames += 1
            self.horizontal_correction_done = True
            if self.centered_frames < self.stable_frames_required:
                self.state = "confirming center"
                self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
                return (
                    f"center confirm {self.centered_frames}/{self.stable_frames_required} "
                    f"dx={error_x:.0f} dy={error_y:.0f}"
                )
            self._reset_ring_lock_guard()
            self.locked_target = self._copy_target(target)
            self.last_visual_target = self._copy_target(target)
            self.visual_confirm_frames = 0
            self.visual_lost_frames = 0
            self.visual_descend_start = None
            self.visual_descend_step_index = 0
            self.locked_plan = None
            self.approach_attempts = 0
            self.return_attempts = 0
            self.abort_after_return = False
            self.algorithm_stage = "open"
            self.state = "target locked"

        return self._update_locked_grasp(
            frame_shape,
            now,
            can_preview_step,
            live_target=live_target,
        )


def arm_joint_positions(id1_tick, id2_tick, id4_tick, id6_tick=BASE_YAW_CENTER_TICK):
    return joint_positions(id1_tick, id2_tick, id4_tick, id6_tick)


class ArmPreviewPublisher:
    def __init__(self, enabled, id1, id2, id4, id6=BASE_YAW_CENTER_TICK):
        self.enabled = enabled
        self.id1 = id1
        self.id2 = id2
        self.id4 = id4
        self.id6 = id6
        self.rclpy = None
        self.String = None
        self.JointState = None
        self.Marker = None
        self.node = None
        self.joint_pub = None
        self.status_pub = None
        self.target_pub = None
        self.marker_pub = None
        self.last_joint_publish_time = 0.0
        self.last_joint_ticks = None
        self.joint_publish_interval_s = 0.12

        if self.enabled:
            self._open()

    def set_targets(self, id1, id2, id4, id6=None):
        self.id1 = int(id1)
        self.id2 = int(id2)
        self.id4 = int(id4)
        if id6 is not None:
            self.id6 = int(id6)

    def _open(self):
        try:
            import rclpy
            from sensor_msgs.msg import JointState
            from std_msgs.msg import String
            from visualization_msgs.msg import Marker
        except ImportError as exc:
            self.enabled = False
            print(f"arm preview disabled: ROS import failed: {exc}")
            return

        self.rclpy = rclpy
        self.JointState = JointState
        self.String = String
        self.Marker = Marker
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("target_vision_arm_preview")
        self.joint_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.status_pub = self.node.create_publisher(String, "/arm/grasp_status", 10)
        self.target_pub = self.node.create_publisher(String, "/arm/servo_targets", 10)
        self.marker_pub = self.node.create_publisher(Marker, "/arm/vision_target_marker", 10)
        self.default_marker_pub = self.node.create_publisher(Marker, "/visualization_marker", 10)
        print("arm preview publishing /joint_states, /arm/grasp_status, and target markers")

    def _publish_marker(self, marker):
        self.marker_pub.publish(marker)
        self.default_marker_pub.publish(marker)

    def publish_plan_marker(self, plan, detail="vision target model"):
        if not self.enabled or plan is None or self.marker_pub is None:
            return

        try:
            now = self.node.get_clock().now().to_msg()
            target_x_m = (plan["current_x_mm"] + plan["forward_mm"] + plan.get("target_distance_mm", 0.0)) / 1000.0
            target_y_m = plan["lateral_mm"] / 1000.0
            target_z_m = (plan["current_z_mm"] + plan["vertical_target_mm"]) / 1000.0
            gripper_x_m = (plan["current_x_mm"] + plan["forward_mm"]) / 1000.0
            gripper_z_m = (plan["current_z_mm"] + plan["vertical_target_mm"]) / 1000.0

            target = self.Marker()
            target.header.frame_id = "base_link"
            target.header.stamp = now
            target.ns = "vision_target"
            target.id = 1
            # Debug visualization is intentionally larger than the real object
            # so it remains visible on the small RK screen.
            real_target_size_m = plan.get("target_size_mm", RED_CUBE_SIDE_MM) / 1000.0
            target_size_m = max(0.08, real_target_size_m * 1.8)
            target.type = (
                self.Marker.SPHERE
                if plan.get("target_kind") == "ball"
                else self.Marker.CUBE
            )
            target.action = self.Marker.ADD
            target.pose.position.x = target_x_m
            target.pose.position.y = target_y_m
            target.pose.position.z = target_z_m
            target.pose.orientation.w = 1.0
            target.scale.x = target_size_m
            target.scale.y = target_size_m
            target.scale.z = target_size_m
            target.color.r = 1.0
            target.color.g = 0.05
            target.color.b = 0.02
            target.color.a = 0.9
            self._publish_marker(target)

            goal = self.Marker()
            goal.header.frame_id = "base_link"
            goal.header.stamp = now
            goal.ns = "vision_target"
            goal.id = 2
            goal.type = self.Marker.SPHERE
            goal.action = self.Marker.ADD
            goal.pose.position.x = gripper_x_m
            goal.pose.position.y = 0.0
            goal.pose.position.z = gripper_z_m
            goal.pose.orientation.w = 1.0
            goal.scale.x = 0.055
            goal.scale.y = 0.055
            goal.scale.z = 0.055
            goal.color.r = 0.0
            goal.color.g = 1.0
            goal.color.b = 0.15
            goal.color.a = 0.95
            self._publish_marker(goal)

            line = self.Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = now
            line.ns = "vision_target"
            line.id = 3
            line.type = self.Marker.LINE_STRIP
            line.action = self.Marker.ADD
            line.scale.x = 0.012
            line.color.r = 0.1
            line.color.g = 0.65
            line.color.b = 1.0
            line.color.a = 0.9
            point_type = type(line.points[0]) if line.points else None
            if point_type is None:
                from geometry_msgs.msg import Point
                point_type = Point
            p1 = point_type()
            p1.x = gripper_x_m
            p1.y = 0.0
            p1.z = gripper_z_m
            p2 = point_type()
            p2.x = target_x_m
            p2.y = target_y_m
            p2.z = target_z_m
            line.points = [p1, p2]
            self._publish_marker(line)

            label = self.Marker()
            label.header.frame_id = "base_link"
            label.header.stamp = now
            label.ns = "vision_target"
            label.id = 4
            label.type = self.Marker.TEXT_VIEW_FACING
            label.action = self.Marker.ADD
            label.pose.position.x = target_x_m
            label.pose.position.y = target_y_m
            label.pose.position.z = target_z_m + 0.09
            label.pose.orientation.w = 1.0
            label.scale.z = 0.045
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 0.15
            label.color.a = 1.0
            label.text = (
                f"d={plan['distance_mm']:.0f}mm "
                f"lat={plan['lateral_mm']:.0f} "
                f"dz={plan['vertical_target_mm']:.0f}\n"
                f"ID1={plan['id1']} ID2={plan['id2']} "
                f"err={plan['ik_error_mm']:.1f}mm"
            )
            self._publish_marker(label)
        except Exception as exc:
            print(f"target marker publish failed: {exc}")

    def publish(self, detail, target=None, state=None):
        if not self.enabled:
            return

        try:
            now = time.monotonic()
            joint_ticks = (self.id1, self.id2, self.id4, self.id6)
            should_publish_joint = joint_ticks != self.last_joint_ticks
            if should_publish_joint:
                msg = self.JointState()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.header.frame_id = "base_mount"
                msg.name = JOINT_NAMES
                msg.position = arm_joint_positions(self.id1, self.id2, self.id4, self.id6)
                self.joint_pub.publish(msg)
                self.last_joint_ticks = joint_ticks
                self.last_joint_publish_time = now

            payload = {
                "state": state or "VISION_PREVIEW",
                "detail": detail,
                "id1": self.id1,
                "id2": self.id2,
                "id4": self.id4,
                "id6": self.id6,
            }
            if target is not None:
                payload.update(
                    {
                        "color": target.get("color"),
                        "kind": target.get("kind"),
                        "center": target.get("center"),
                        "bbox": target.get("bbox"),
                        "area_percent": round(target.get("area_percent", 0.0), 2),
                        "distance_cm": None
                        if target.get("distance_cm") is None
                        else round(target.get("distance_cm"), 1),
                    }
                )
            status = self.String(data=json.dumps(payload, ensure_ascii=False))
            self.status_pub.publish(status)
            self.target_pub.publish(status)
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            self.enabled = False
            if "context is invalid" not in str(exc):
                print(f"arm preview stopped: {exc}")

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.rclpy is not None and self.rclpy.ok():
            self.rclpy.shutdown()


class TargetDetector:
    def __init__(self, qr_template_every=1):
        self.qr = cv2.QRCodeDetector()
        self.templates = self._load_templates()
        self.qr_feature_detector = None
        self.qr_template_features = {}
        self.qr_template_every = max(1, int(qr_template_every))
        self._qr_fallback_counter = 0
        self.qr_matcher = cv2.BFMatcher(cv2.NORM_L2)
        # cv2.QRCodeDetector cannot decode without quirc; those paths only
        # spam "Library QUIRC is not linked" and never return decoded text.
        self._cv_qr_usable = bool(
            re.search(r"QUIRC:\s*YES", cv2.getBuildInformation())
        )
        if hasattr(cv2, "SIFT_create"):
            self.qr_feature_detector = cv2.SIFT_create(
                nfeatures=2000,
                contrastThreshold=0.02,
            )
            for label, template in self.templates.items():
                keypoints, descriptors = self.qr_feature_detector.detectAndCompute(
                    template,
                    None,
                )
                if descriptors is not None and len(keypoints) >= 8:
                    self.qr_template_features[label] = (
                        template.shape,
                        keypoints,
                        descriptors,
                    )

    def _load_templates(self):
        asset_dir = Path(__file__).resolve().parent / "assets"
        templates = {}
        for label in ("red", "blue"):
            path = asset_dir / f"{label}.png"
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[label] = img
        return templates

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rings = self._detect_rings(hsv)
        colored_balls = []
        for detector in BALL_DETECTORS:
            colored_balls.extend(detector.detect(hsv, rings))
        squares = self._detect_squares(hsv)
        qr_detections = self._detect_qr(frame)
        colored_balls = self._suppress_qr_overlap(
            colored_balls,
            qr_detections,
            colors={"blue"},
            kinds={"ball"},
            min_overlap=0.34,
        )
        squares = self._suppress_qr_overlap(
            squares,
            qr_detections,
            colors={"red", "blue"},
            kinds={"square"},
            min_overlap=0.40,
        )
        non_white_mask = cv2.bitwise_or(
            yellow.mask(hsv),
            cv2.bitwise_or(red.mask(hsv), blue.mask(hsv)),
        )
        strong_color_mask = cv2.inRange(
            hsv,
            np.array((0, 90, 50)),
            np.array((180, 255, 255)),
        )
        non_white_mask = cv2.bitwise_and(non_white_mask, strong_color_mask)
        white_balls = WHITE_DETECTOR.detect(
            hsv,
            [*colored_balls, *rings, *squares, *qr_detections],
            non_white_mask,
        )
        white_balls = self._suppress_qr_overlap(
            white_balls,
            qr_detections,
            colors={"white"},
            kinds={"ball"},
            min_overlap=0.24,
        )
        if colored_balls and len(white_balls) > 1:
            white_balls = []
        detections = []
        detections.extend(colored_balls)
        detections.extend(squares)
        detections.extend(rings)
        detections.extend(qr_detections)
        detections.extend(white_balls)
        self._add_frame_ratios(detections, frame.shape)
        return detections

    @staticmethod
    def _add_frame_ratios(detections, frame_shape):
        height, width = frame_shape[:2]
        frame_area = max(1, width * height)
        short_side = max(1, min(width, height))

        for det in detections:
            if det.get("kind") == "ball":
                radius = float(det.get("radius", 0))
                circle_area = np.pi * radius * radius
                det["area_ratio"] = circle_area / frame_area
                det["area_percent"] = det["area_ratio"] * 100.0
                det["diameter_ratio"] = (2.0 * radius) / short_side
                det["distance_cm"] = TargetDetector._estimate_distance_cm(
                    det["area_percent"],
                    RING_DISTANCE_OFFSET_CM,
                    RING_DISTANCE_SCALE_CM,
                )
            elif det.get("kind") == "square":
                x, y, w, h = det.get("bbox", (0, 0, 0, 0))
                box_area = max(0.0, float(det.get("projected_area", 0.0)))
                if box_area == 0.0:
                    box_area = max(0, w * h)
                det["area_ratio"] = box_area / frame_area
                det["area_percent"] = det["area_ratio"] * 100.0
                det["diameter_ratio"] = max(w, h) / short_side
                det["distance_cm"] = TargetDetector._estimate_distance_cm(
                    det["area_percent"],
                    SQUARE_DISTANCE_OFFSET_CM,
                    SQUARE_DISTANCE_SCALE_CM,
                )
            elif det.get("kind") == "ring":
                radius = float(det.get("outer_radius", 0))
                circle_area = np.pi * radius * radius
                det["area_ratio"] = circle_area / frame_area
                det["area_percent"] = det["area_ratio"] * 100.0
                det["diameter_ratio"] = (2.0 * radius) / short_side
                det["distance_cm"] = TargetDetector._estimate_distance_cm(
                    det["area_percent"],
                    BALL_DISTANCE_OFFSET_CM,
                    BALL_DISTANCE_SCALE_CM,
                )

    @staticmethod
    def _estimate_distance_cm(area_percent, offset_cm, scale_cm):
        if area_percent <= 0.0:
            return None
        return offset_cm + scale_cm / np.sqrt(area_percent)

    def _suppress_qr_overlap(
        self,
        detections,
        qr_detections,
        colors=None,
        kinds=None,
        min_overlap=0.35,
    ):
        if not detections or not qr_detections:
            return detections

        colors = None if colors is None else set(colors)
        kinds = None if kinds is None else set(kinds)
        filtered = []
        for det in detections:
            if colors is not None and det.get("color") not in colors:
                filtered.append(det)
                continue
            if kinds is not None and det.get("kind") not in kinds:
                filtered.append(det)
                continue

            det_bbox = det.get("bbox")
            if det_bbox is None:
                filtered.append(det)
                continue

            blocked = any(
                self._bbox_overlap_ratio(det_bbox, qr.get("bbox")) >= min_overlap
                for qr in qr_detections
            )
            if not blocked:
                filtered.append(det)
        return filtered

    @staticmethod
    def _bbox_overlap_ratio(bbox_a, bbox_b):
        if bbox_a is None or bbox_b is None:
            return 0.0

        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        ix0 = max(ax, bx)
        iy0 = max(ay, by)
        ix1 = min(ax + aw, bx + bw)
        iy1 = min(ay + ah, by + bh)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        intersection = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(1, aw * ah)
        area_b = max(1, bw * bh)
        return intersection / min(area_a, area_b)

    def draw(self, frame, detections):
        out = frame.copy()
        height, width = out.shape[:2]
        center_x = width // 2
        center_y = height // 2
        cv2.line(out, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.line(out, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.circle(out, (center_x, center_y), 10, (0, 255, 255), 2)
        for det in detections:
            color = det["color"]
            bgr = DRAW_COLORS.get(color, (0, 255, 0))
            label = f"{color} {det['kind']}"

            if det["kind"] == "ball":
                x, y, w, h = det["bbox"]
                cx, cy = det["center"]
                radius = det["radius"]
                area_percent = det.get("area_percent", 0.0)
                distance_cm = det.get("distance_cm")
                if distance_cm is None:
                    label = f"{label} fill {area_percent:.2f}%"
                else:
                    label = f"{label} fill {area_percent:.2f}% d {distance_cm:.1f}cm"
                cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
                cv2.circle(out, (cx, cy), radius, bgr, 2)
                self._label(out, label, x, y, bgr)
            elif det["kind"] == "square":
                cv2.drawContours(out, [det["box"]], 0, bgr, 2)
                x, y, w, h = det["bbox"]
                cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 1)
                area_percent = det.get("area_percent", 0.0)
                distance_cm = det.get("distance_cm")
                if det.get("source") == "qr":
                    confidence = det.get("confidence", 0)
                    label = f"{color} QR conf {confidence:.0f}%"
                    if distance_cm is None:
                        label = f"{label} fill {area_percent:.2f}%"
                    else:
                        label = (
                            f"{label} fill {area_percent:.2f}% "
                            f"depth {distance_cm:.1f}cm"
                        )
                elif distance_cm is None:
                    label = f"{label} fill {area_percent:.2f}%"
                else:
                    label = f"{label} fill {area_percent:.2f}% d {distance_cm:.1f}cm"
                if not det.get("fully_visible", True):
                    label = f"{label} EDGE"
                self._label(out, label, x, y, bgr)
            elif det["kind"] == "ring":
                cx, cy = det["center"]
                cv2.circle(out, (cx, cy), det["outer_radius"], bgr, 2)
                cv2.circle(out, (cx, cy), det["inner_radius"], bgr, 2)
                x, y, w, h = det["bbox"]
                cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 1)
                area_percent = det.get("area_percent", 0.0)
                distance_cm = det.get("distance_cm")
                if distance_cm is None:
                    label = f"{label} fill {area_percent:.2f}%"
                else:
                    label = (
                        f"{label} fill {area_percent:.2f}% "
                        f"depth {distance_cm:.1f}cm"
                    )
                self._label(out, label, x, y, bgr)
            elif det["kind"] == "qr":
                pts = det["points"].astype(np.int32)
                cv2.polylines(out, [pts], True, bgr, 3)
                x, y, _, _ = cv2.boundingRect(pts)
                self._label(out, f"{color} QR", x, y, bgr)

        info = self.summary(detections)
        cv2.putText(out, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        return out, info

    @staticmethod
    def _label(img, text, x, y, bgr):
        y = max(22, y)
        cv2.putText(img, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

    @staticmethod
    def summary(detections):
        order = [
            ("yellow", "ball"),
            ("red", "ball"),
            ("blue", "ball"),
            ("white", "ball"),
            ("red", "square"),
            ("blue", "square"),
            ("red", "ring"),
            ("blue", "ring"),
            ("red", "qr"),
            ("blue", "qr"),
        ]
        parts = []
        for color, kind in order:
            matching = [
                det
                for det in detections
                if det["color"] == color and det["kind"] == kind
            ]
            count = len(matching)
            if count:
                if kind == "ball":
                    measurements = ",".join(
                        TargetDetector._format_area_measurement(det)
                        for det in matching
                    )
                    parts.append(f"{color}-{kind}:{count} {measurements}")
                elif kind == "square":
                    measurements = ",".join(
                        TargetDetector._format_area_measurement(det)
                        for det in matching
                    )
                    parts.append(f"{color}-{kind}:{count} {measurements}")
                else:
                    parts.append(f"{color}-{kind}:{count}")
        return " | ".join(parts) if parts else "searching selected targets..."

    @staticmethod
    def _format_area_measurement(det):
        area_percent = det.get("area_percent", 0.0)
        distance_cm = det.get("distance_cm")
        if distance_cm is None:
            return f"fill={area_percent:.2f}%"
        return f"fill={area_percent:.2f}% dist={distance_cm:.1f}cm"

    def _mask(self, hsv, color):
        return MASK_BUILDERS[color](hsv)

    def _detect_balls(self, hsv, rings=None):
        results = []
        rings = rings or []
        for detector in BALL_DETECTORS:
            results.extend(detector.detect(hsv, rings))
        return results

    def _detect_white_balls(self, hsv, occupied_detections=None, non_white_mask=None):
        occupied_detections = occupied_detections or []
        return WHITE_DETECTOR.detect(hsv, occupied_detections, non_white_mask)

    def _detect_squares(self, hsv):
        results = []
        frame_height, frame_width = hsv.shape[:2]
        edge_margin = max(12, int(min(frame_width, frame_height) * 0.015))
        for color in SHAPE_COLORS:
            contours = self._external_contours(self._mask(hsv, color))
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 450:
                    continue
                contour_circularity = self._circularity(contour)
                peri = cv2.arcLength(contour, True)
                poly = cv2.approxPolyDP(contour, 0.035 * peri, True)
                if len(poly) < 4 or len(poly) > 6:
                    continue
                if contour_circularity > 0.88:
                    continue
                rect = cv2.minAreaRect(contour)
                (cx, cy), (w, h), angle = rect
                if min(w, h) < 16:
                    continue
                fill = area / (w * h) if w * h > 0 else 0
                aspect = max(w, h) / min(w, h) if min(w, h) else 999
                if fill < 0.62 or aspect > 1.8:
                    continue
                box = cv2.boxPoints(rect).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(box)
                fully_visible = (
                    x > edge_margin
                    and y > edge_margin
                    and x + bw < frame_width - edge_margin
                    and y + bh < frame_height - edge_margin
                )
                results.append(
                    {
                        "kind": "square",
                        "color": color,
                        "center": (int(cx), int(cy)),
                        "box": box,
                        "bbox": (x, y, bw, bh),
                        "projected_area": float(w * h),
                        "fully_visible": fully_visible,
                        "angle": round(angle, 1),
                    }
                )
        return results

    def _detect_rings(self, hsv):
        results = []
        for color in SHAPE_COLORS:
            mask = self._mask(hsv, color)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            hierarchy = hierarchy[0]
            for idx, contour in enumerate(contours):
                child_idx = hierarchy[idx][2]
                if child_idx < 0:
                    continue
                area = cv2.contourArea(contour)
                outer_circularity = self._circularity(contour)
                if area < 300 or outer_circularity < 0.60:
                    continue
                (cx, cy), outer_radius = cv2.minEnclosingCircle(contour)
                child = contours[child_idx]
                child_area = cv2.contourArea(child)
                (inner_center, inner_radius) = cv2.minEnclosingCircle(child)
                inner_circularity = self._circularity(child)
                if outer_radius < 14 or inner_radius < 5 or child_area < 40:
                    continue
                hole_ratio = inner_radius / outer_radius if outer_radius > 0 else 0
                center_offset = np.hypot(inner_center[0] - cx, inner_center[1] - cy)
                if not 0.24 <= hole_ratio <= 0.72:
                    continue
                if center_offset > outer_radius * 0.35 or inner_circularity < 0.42:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                results.append(
                    {
                        "kind": "ring",
                        "color": color,
                        "center": (int(cx), int(cy)),
                        "outer_radius": int(outer_radius),
                        "inner_radius": int(inner_radius),
                        "bbox": (x, y, w, h),
                        "score": round(outer_circularity, 2),
                    }
                )
        return results

    @staticmethod
    def _center_fill(mask, cx, cy, radius):
        radius = max(3, int(radius * 0.28))
        center_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(center_mask, (int(cx), int(cy)), radius, 255, -1)
        center_area = np.count_nonzero(center_mask)
        if center_area == 0:
            return 0.0
        colored_area = np.count_nonzero(cv2.bitwise_and(mask, center_mask))
        return colored_area / center_area

    @staticmethod
    def _mask_ratio_in_circle(mask, cx, cy, radius, scale=0.82):
        if mask is None:
            return 0.0
        radius = max(3, int(radius * scale))
        circle_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (int(cx), int(cy)), radius, 255, -1)
        circle_area = np.count_nonzero(circle_mask)
        if circle_area == 0:
            return 0.0
        masked_area = np.count_nonzero(cv2.bitwise_and(mask, circle_mask))
        return masked_area / circle_area

    @staticmethod
    def _circle_hsv_mean(hsv, cx, cy, radius, scale=0.55):
        radius = max(3, int(radius * scale))
        circle_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, (int(cx), int(cy)), radius, 255, -1)
        sat_mean = cv2.mean(hsv[:, :, 1], mask=circle_mask)[0]
        val_mean = cv2.mean(hsv[:, :, 2], mask=circle_mask)[0]
        return sat_mean, val_mean

    @staticmethod
    def _overlaps_existing_ring(bbox, color, rings):
        x, y, w, h = bbox
        area = max(1, w * h)
        for ring in rings:
            if ring.get("color") != color:
                continue
            rx, ry, rw, rh = ring["bbox"]
            ix0 = max(x, rx)
            iy0 = max(y, ry)
            ix1 = min(x + w, rx + rw)
            iy1 = min(y + h, ry + rh)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            overlap = (ix1 - ix0) * (iy1 - iy0) / area
            if overlap > 0.45:
                return True
        return False

    @staticmethod
    def _overlaps_existing_detection(bbox, detections, min_overlap=0.38):
        x, y, w, h = bbox
        area = max(1, w * h)
        for det in detections:
            dx, dy, dw, dh = det.get("bbox", (0, 0, 0, 0))
            ix0 = max(x, dx)
            iy0 = max(y, dy)
            ix1 = min(x + w, dx + dw)
            iy1 = min(y + h, dy + dh)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            overlap = (ix1 - ix0) * (iy1 - iy0) / area
            if overlap > min_overlap:
                return True
        return False

    def _detect_qr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []

        def add_result(candidate):
            for index, existing in enumerate(results):
                overlap = self._bbox_overlap_ratio(
                    candidate.get("bbox"),
                    existing.get("bbox"),
                )
                if candidate.get("color") == existing.get("color") or overlap >= 0.45:
                    if candidate.get("confidence", 0.0) > existing.get("confidence", 0.0):
                        results[index] = candidate
                    return
            results.append(candidate)

        if zbar_decode is not None:
            symbols = [ZBarSymbol.QRCODE] if ZBarSymbol is not None else None
            for barcode in zbar_decode(gray, symbols=symbols):
                decoded = barcode.data.decode("utf-8", "replace")
                color = self._color_from_qr_text(decoded)
                if color not in ("red", "blue"):
                    continue
                points = self._barcode_points(barcode)
                confidence = self._barcode_confidence(barcode, gray)
                add_result(self._qr_as_square(color, points, decoded, confidence))
            if {item["color"] for item in results} == {"red", "blue"}:
                return results

        if self._cv_qr_usable and hasattr(self.qr, "detectAndDecodeMulti"):
            try:
                ok, decoded_info, points, _ = self.qr.detectAndDecodeMulti(gray)
                if ok and points is not None:
                    for idx, qr_points in enumerate(points):
                        decoded = decoded_info[idx] if idx < len(decoded_info) else ""
                        color = self._color_from_qr_text(decoded)
                        if color in ("red", "blue"):
                            add_result(self._qr_as_square(color, qr_points, decoded, 100.0))
            except cv2.error:
                pass

        if self._cv_qr_usable and {item["color"] for item in results} != {"red", "blue"}:
            try:
                decoded, points, _ = self.qr.detectAndDecode(gray)
                if points is not None:
                    color = self._color_from_qr_text(decoded)
                    if color in ("red", "blue"):
                        add_result(self._qr_as_square(color, points, decoded, 100.0))
            except cv2.error:
                pass

        if {item["color"] for item in results} != {"red", "blue"}:
            self._qr_fallback_counter += 1
            if self._qr_fallback_counter >= self.qr_template_every:
                self._qr_fallback_counter = 0
                for candidate in self._detect_qr_templates(gray):
                    add_result(candidate)

        return results

    def _detect_qr_templates(self, gray):
        if self.qr_feature_detector is None or not self.qr_template_features:
            return []
        scene_keypoints, scene_descriptors = self.qr_feature_detector.detectAndCompute(
            gray,
            None,
        )
        if scene_descriptors is None or len(scene_keypoints) < 8:
            return []

        matcher = self.qr_matcher
        frame_height, frame_width = gray.shape[:2]
        frame_area = frame_height * frame_width
        results = []
        for label, (template_shape, template_keypoints, template_descriptors) in self.qr_template_features.items():
            try:
                matches = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)
            except cv2.error:
                continue
            good = [
                first
                for pair in matches
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < 0.72 * second.distance
            ]
            if len(good) < 8:
                continue
            source_points = np.float32(
                [template_keypoints[item.queryIdx].pt for item in good]
            )
            target_points = np.float32(
                [scene_keypoints[item.trainIdx].pt for item in good]
            )
            matrix, inlier_mask = cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                4.0,
            )
            if matrix is None or inlier_mask is None:
                continue
            inliers = int(inlier_mask.sum())
            inlier_ratio = inliers / max(1, len(good))
            if inliers < 8 or inlier_ratio < 0.35:
                continue

            template_height, template_width = template_shape[:2]
            corners = np.float32(
                [[
                    [0, 0],
                    [template_width - 1, 0],
                    [template_width - 1, template_height - 1],
                    [0, template_height - 1],
                ]]
            )
            try:
                projected = cv2.perspectiveTransform(corners, matrix)[0]
            except cv2.error:
                continue
            if not np.isfinite(projected).all():
                continue
            contour = np.rint(projected).astype(np.int32)
            area = abs(cv2.contourArea(contour))
            if area < 400.0 or area > frame_area * 0.35:
                continue
            if not cv2.isContourConvex(contour):
                continue
            if (
                projected[:, 0].min() < -12
                or projected[:, 1].min() < -12
                or projected[:, 0].max() > frame_width + 12
                or projected[:, 1].max() > frame_height + 12
            ):
                continue
            side_lengths = [
                float(np.linalg.norm(projected[(index + 1) % 4] - projected[index]))
                for index in range(4)
            ]
            if min(side_lengths) < 12.0 or max(side_lengths) / min(side_lengths) > 2.0:
                continue
            confidence = 100.0 * (
                0.55 * min(1.0, inliers / 30.0)
                + 0.45 * min(1.0, inlier_ratio)
            )
            results.append(
                self._qr_as_square(
                    label,
                    projected,
                    "R" if label == "red" else "B",
                    confidence,
                )
            )
        return results

    @staticmethod
    def _color_from_qr_text(text):
        normalized = (text or "").strip().lower()
        return QR_COLOR_ALIASES.get(normalized)

    @staticmethod
    def _barcode_points(barcode):
        polygon = getattr(barcode, "polygon", None) or []
        if len(polygon) >= 4:
            pts = np.array([(p.x, p.y) for p in polygon], dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            return cv2.boxPoints(rect)
        rect = barcode.rect
        x, y, w, h = rect.left, rect.top, rect.width, rect.height
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    @staticmethod
    def _barcode_confidence(barcode, gray):
        quality = getattr(barcode, "quality", None)
        if quality is not None:
            return max(0.0, min(100.0, float(quality)))

        rect = barcode.rect
        height, width = gray.shape[:2]
        x0 = max(0, rect.left)
        y0 = max(0, rect.top)
        x1 = min(width, rect.left + rect.width)
        y1 = min(height, rect.top + rect.height)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return 60.0

        sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()
        sharp_score = min(1.0, sharpness / 900.0)
        size_score = min(1.0, max(rect.width, rect.height) / 180.0)
        return 60.0 + 40.0 * (0.65 * sharp_score + 0.35 * size_score)

    @staticmethod
    def _qr_as_square(color, points, decoded, confidence):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) != 4:
            points = cv2.boxPoints(cv2.minAreaRect(points))
        box = points.astype(np.int32)
        x, y, w, h = cv2.boundingRect(box)
        center = (int(x + w / 2), int(y + h / 2))
        contour_area = abs(float(cv2.contourArea(points)))
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(points)
        rect_area = max(0.0, float(rect_w) * float(rect_h))
        bbox_area = max(0.0, float(w) * float(h))
        projected_area = max(contour_area, rect_area, bbox_area * 0.55)
        return {
            "kind": "square",
            "color": color,
            "source": "qr",
            "decoded": decoded,
            "confidence": float(confidence),
            "center": center,
            "box": box,
            "bbox": (x, y, w, h),
            "projected_area": projected_area,
            "angle": 0.0,
        }

    def _match_qr_template(self, gray, points):
        if not self.templates:
            return None, 0.0

        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        side = 240
        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        try:
            matrix = cv2.getPerspectiveTransform(points, dst)
            roi = cv2.warpPerspective(gray, matrix, (side, side))
        except cv2.error:
            return None, 0.0

        if roi.size == 0:
            return None, 0.0
        scores = {}
        for label, template in self.templates.items():
            resized = cv2.resize(template, (side, side))
            score = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED).max()
            scores[label] = float(score)
        best = max(scores, key=scores.get)
        confidence = max(0.0, min(100.0, scores[best] * 100.0))
        return (best, confidence) if scores[best] > 0.35 else (None, 0.0)

    @staticmethod
    def _external_contours(mask):
        return external_contours(mask)

    @staticmethod
    def _circularity(contour):
        return contour_circularity(contour)


def parse_device(value):
    return int(value) if str(value).isdigit() else value


def camera_candidates(device):
    if device != "auto":
        return [device]

    candidates = []
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        for item in sorted(by_id.iterdir()):
            if "video-index0" in item.name or "camera" in item.name.lower():
                try:
                    candidates.append(str(item.resolve()))
                except OSError:
                    pass

    candidates.extend(["/dev/video20", "/dev/video21"])
    for item in sorted(Path("/dev").glob("video*"), key=lambda p: p.name):
        if item.name[5:].isdigit():
            candidates.append(str(item))

    deduped = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def open_camera(device, width, height, fps):
    errors = []
    read_timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    open_timeout_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    buffer_size_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    for candidate in camera_candidates(device):
        candidate_errors = []
        for attempt in range(1, 4):
            cap = cv2.VideoCapture(parse_device(candidate), cv2.CAP_V4L2)
            if open_timeout_prop is not None:
                cap.set(open_timeout_prop, 1800)
            if read_timeout_prop is not None:
                cap.set(read_timeout_prop, 1800)
            if buffer_size_prop is not None:
                cap.set(buffer_size_prop, 1)
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    str(candidate),
                    "--set-ctrl=exposure_dynamic_framerate=0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if not cap.isOpened():
                candidate_errors.append(f"attempt {attempt}: open failed")
                cap.release()
                time.sleep(0.15)
                continue
            for frame_attempt in range(8):
                ok, _ = cap.read()
                if ok:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    print(
                        f"Camera: {candidate} {actual_width}x{actual_height} "
                        f"reported_fps={actual_fps:.1f} attempt={attempt}"
                    )
                    return cap
                time.sleep(0.10)
            candidate_errors.append(f"attempt {attempt}: no frames")
            cap.release()
            time.sleep(0.25)
        errors.append(f"{candidate}: " + ", ".join(candidate_errors))
    raise RuntimeError("Cannot open camera. Tried: " + "; ".join(errors))

def set_pipewire(active):
    command = "start" if active else "stop"
    services = ["pipewire.socket", "pipewire.service", "pipewire-media-session.service"]
    subprocess.run(
        ["systemctl", "--user", command, *services],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def stop_old_camera_viewers():
    user = getpass.getuser()
    for name in ("cheese", "guvcview", "gst-launch-1.0"):
        subprocess.run(
            ["pkill", "-u", user, "-x", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    subprocess.run(
        ["pkill", "-u", user, "-f", "/home/cat/bin/camera-preview"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def ensure_display_env():
    uid = os.getuid()
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
    os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    os.environ.setdefault("DISPLAY", ":0")


def local_ip_hint():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("CAMERA_DEVICE", "auto"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detect-every-n-frames", type=int, default=1)
    parser.add_argument("--detection-scale", type=float, default=1.0)
    parser.add_argument(
        "--qr-template-every",
        type=int,
        default=3,
        help="run the SIFT QR-template fallback only every N detection passes "
        "(1 restores the old always-on behavior); zbar QR decoding still runs "
        "on every detection pass",
    )
    parser.add_argument("--detection-smoothing-alpha", type=float, default=0.32)
    parser.add_argument(
        "--sync-detection-for-grasp",
        action="store_true",
        help="run detection on the current frame while auto grasp is active",
    )
    parser.add_argument(
        "--fresh-detection-on-lock",
        action="store_true",
        help="run one current-frame detection before the final grasp lock",
    )
    parser.add_argument("--detection-smoothing-match-px", type=float, default=90.0)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--keep-pipewire", action="store_true")
    parser.add_argument("--keep-camera-users", action="store_true")
    parser.add_argument("--disable-servo-trigger", action="store_true")
    parser.add_argument("--servo-uart", default=os.environ.get("SERVO_UART", "/dev/ttyS0"))
    parser.add_argument("--servo-baud", type=int, default=int(os.environ.get("SERVO_BAUD", "115200")))
    parser.add_argument(
        "--chassis-link",
        action="store_true",
        default=os.environ.get("CHASSIS_LINK", "").lower() in {"1", "true", "yes"},
        help="wait for H7 USB CDC ARM,<task>,START commands before grasping",
    )
    parser.add_argument(
        "--chassis-uart",
        default=os.environ.get("CHASSIS_UART", "auto"),
        help="H7 USB CDC device, usually /dev/ttyACM0; auto scans ttyACM/ttyUSB",
    )
    parser.add_argument(
        "--chassis-baud",
        type=int,
        default=int(os.environ.get("CHASSIS_BAUD", "115200")),
    )
    parser.add_argument(
        "--station-no-target-timeout",
        type=float,
        default=float(os.environ.get("STATION_NO_TARGET_TIMEOUT", "5.0")),
        help="seconds without a visible target before RK homes the arm and lets H7 continue",
    )
    parser.add_argument(
        "--direct-servo-bus",
        action="store_true",
        default=os.environ.get("DIRECT_SERVO_BUS", "").lower() in {"1", "true", "yes"},
        help="bypass RCT6 and transmit servo bus frames directly from RK UART TX",
    )
    parser.add_argument("--direct-arm-time-ms", type=int, default=int(os.environ.get("DIRECT_ARM_TIME_MS", "1200")))
    parser.add_argument("--direct-zp-time-ms", type=int, default=int(os.environ.get("DIRECT_ZP_TIME_MS", "1000")))
    parser.add_argument(
        "--direct-gripper-time-ms",
        type=int,
        default=int(os.environ.get("DIRECT_GRIPPER_TIME_MS", "0")),
        help="ID7 gripper move time in ms; 0 uses --direct-zp-time-ms",
    )
    parser.add_argument("--direct-repeat", type=int, default=int(os.environ.get("DIRECT_SERVO_REPEAT", "1")))
    parser.add_argument("--direct-arm-uart", default=os.environ.get("DIRECT_ARM_UART", os.environ.get("SERVO_ARM_UART", "/dev/ttyS9")))
    parser.add_argument("--direct-zp-uart", default=os.environ.get("DIRECT_ZP_UART", os.environ.get("SERVO_ZP_UART", "/dev/ttyS0")))
    parser.add_argument("--trigger-command", default=os.environ.get("SERVO_TRIGGER_COMMAND", "linkstart"))
    parser.add_argument("--trigger-color", default="red", choices=BALL_COLORS)
    parser.add_argument("--trigger-kind", default="square", choices=("ball", "square", "ring", "qr", "any"))
    parser.add_argument("--trigger-stable-frames", type=int, default=3)
    parser.add_argument("--trigger-reset-frames", type=int, default=8)
    parser.add_argument("--trigger-cooldown", type=float, default=5.0)
    parser.add_argument("--enable-arm-preview", action="store_true")
    parser.add_argument("--preview-id1", type=int, default=READY_ID1_TICK)
    parser.add_argument("--preview-id2", type=int, default=READY_ID2_TICK)
    parser.add_argument("--preview-id4", type=int, default=GRIPPER_CLOSED_TICK)
    parser.add_argument("--preview-id6", type=int, default=BASE_YAW_CENTER_TICK)
    parser.add_argument("--enable-red-square-grasp", action="store_true")
    parser.add_argument(
        "--execute-red-square-grasp",
        action="store_true",
        default=os.environ.get("RED_SQUARE_EXECUTE", "").lower() in {"1", "true", "yes"},
    )
    parser.add_argument("--skip-grasp-startup-sequence", action="store_true")
    parser.add_argument("--grasp-id1-ready", type=int, default=READY_ID1_TICK)
    parser.add_argument("--grasp-id2-ready", type=int, default=READY_ID2_TICK)
    parser.add_argument("--grasp-id4-closed", type=int, default=GRIPPER_CLOSED_TICK)
    parser.add_argument("--grasp-id4-open", type=int, default=GRIPPER_OPEN_TICK)
    parser.add_argument("--grasp-center-deadband-px", type=float, default=28.0)
    parser.add_argument("--grasp-stable-frames", type=int, default=5)
    parser.add_argument("--grasp-command-interval", type=float, default=1.40)
    parser.add_argument("--grasp-retrigger-cooldown", type=float, default=0.0)
    parser.add_argument("--grasp-id2-pixel-gain", type=float, default=0.12)
    parser.add_argument("--grasp-id6-pixel-gain", type=float, default=0.18)
    parser.add_argument("--grasp-id6-max-step-ticks", type=int, default=60)
    parser.add_argument("--grasp-id1-pixel-gain-y", type=float, default=0.0)
    parser.add_argument("--grasp-id2-distance-gain", type=float, default=0.25)
    parser.add_argument("--camera-gripper-offset-mm", type=float, default=50.0)
    parser.add_argument("--target-gripper-distance-mm", type=float, default=20.0)
    parser.add_argument("--grasp-distance-deadband-mm", type=float, default=12.0)
    parser.add_argument("--grasp-max-step-ticks", type=int, default=18)
    parser.add_argument("--servo-angle-gap-deg", type=float, default=MIN_ANGLE_GAP_DEG)
    parser.add_argument("--grasp-id1-min", type=int, default=ID1_SAFE_LIMITS[0])
    parser.add_argument("--grasp-id1-max", type=int, default=ID1_SAFE_LIMITS[1])
    parser.add_argument("--grasp-id2-min", type=int, default=ID2_SAFE_LIMITS[0])
    parser.add_argument("--grasp-id2-max", type=int, default=ID2_SAFE_LIMITS[1])
    parser.add_argument("--grasp-id2-center-min", type=int, default=None)
    parser.add_argument("--grasp-id2-center-max", type=int, default=None)
    parser.add_argument("--grasp-one-shot", action="store_true")
    parser.add_argument("--camera-gripper-vertical-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-lateral-offset-mm", type=float, default=45.0)
    parser.add_argument("--max-one-shot-ik-error-mm", type=float, default=15.0)
    parser.add_argument("--post-center-retreat-mm", type=float, default=50.0)
    parser.add_argument("--post-center-down-mm", type=float, default=30.0)
    parser.add_argument("--post-center-ik-error-mm", type=float, default=18.0)
    parser.add_argument(
        "--use-calibrated-grasp-table",
        action="store_true",
        help="use the legacy distance-to-tick table instead of corrected IK",
    )
    parser.add_argument(
        "--use-post-center-tick-bias",
        action="store_true",
        help="apply the legacy ID1/ID2 tick bias after post-center IK",
    )
    parser.add_argument(
        "--post-center-direct-descend",
        action="store_true",
        help="after opening ID7, skip overhead and descend using corrected post-center IK",
    )
    parser.add_argument(
        "--simple-vertical-grasp",
        action="store_true",
        default=os.environ.get("SIMPLE_VERTICAL_GRASP", "").lower()
        in {"1", "true", "yes"},
        help="after image centering, open ID7 and descend with ID1 only; keep old IK path disabled",
    )
    parser.add_argument(
        "--vertical-grasp-id1",
        type=int,
        default=int(os.environ.get("VERTICAL_GRASP_ID1", str(HOME_ID1_TICK))),
        help="ID1 tick used for the simple vertical descend",
    )
    parser.add_argument(
        "--vertical-grasp-id2",
        type=int,
        default=int(os.environ.get("VERTICAL_GRASP_ID2", str(READY_ID2_TICK))),
        help="ID2 tick held during the simple vertical descend",
    )
    parser.add_argument("--window-x", type=int, default=None)
    parser.add_argument("--window-y", type=int, default=None)
    parser.add_argument("--window-width", type=int, default=None)
    parser.add_argument("--window-height", type=int, default=None)
    return parser


def main(argv=None):
    args, _ = build_arg_parser().parse_known_args(argv)
    ensure_display_env()

    state = FrameState()
    detector = TargetDetector()
    detection_smoother = DetectionSmoother(
        alpha=args.detection_smoothing_alpha,
        max_match_px=args.detection_smoothing_match_px,
    )
    auto_grasp_enabled = args.enable_red_square_grasp
    execute_auto_grasp = auto_grasp_enabled and args.execute_red_square_grasp
    preview_id4 = (
        args.preview_id4
        if execute_auto_grasp
        else load_last_id4_target(args.preview_id4)
    )
    servo_trigger = SerialTrigger(
        args.servo_uart,
        args.servo_baud,
        args.trigger_command,
        args.trigger_color,
        args.trigger_kind,
        args.trigger_stable_frames,
        args.trigger_reset_frames,
        args.trigger_cooldown,
        enabled=not args.disable_servo_trigger and not auto_grasp_enabled,
    )
    arm_preview = ArmPreviewPublisher(
        args.enable_arm_preview or auto_grasp_enabled,
        args.preview_id1,
        args.preview_id2,
        preview_id4,
        args.preview_id6,
    )
    servo_bridge_class = DirectBusServoBridge if args.direct_servo_bus else AbsoluteServoBridge
    if args.direct_servo_bus:
        servo_bridge = servo_bridge_class(
            args.servo_uart,
            args.servo_baud,
            enabled=auto_grasp_enabled,
            write_enabled=execute_auto_grasp,
            arm_device=args.direct_arm_uart,
            zp_device=args.direct_zp_uart,
            arm_time_ms=args.direct_arm_time_ms,
            zp_time_ms=args.direct_zp_time_ms,
            gripper_time_ms=(
                None if args.direct_gripper_time_ms <= 0 else args.direct_gripper_time_ms
            ),
            repeat=args.direct_repeat,
        )
    else:
        servo_bridge = servo_bridge_class(
            args.servo_uart,
            args.servo_baud,
            enabled=auto_grasp_enabled,
            write_enabled=execute_auto_grasp,
        )
    grasp_controller = RedSquareGraspController(
        auto_grasp_enabled,
        servo_bridge,
        arm_preview,
        args.grasp_id1_ready,
        args.grasp_id2_ready,
        args.grasp_id4_closed,
        args.grasp_id4_open,
        args.grasp_center_deadband_px,
        args.grasp_stable_frames,
        args.grasp_command_interval,
        args.grasp_retrigger_cooldown,
        args.grasp_id2_pixel_gain,
        args.grasp_id6_pixel_gain,
        args.grasp_id6_max_step_ticks,
        args.grasp_id1_pixel_gain_y,
        args.grasp_id2_distance_gain,
        args.camera_gripper_offset_mm,
        args.target_gripper_distance_mm,
        args.grasp_distance_deadband_mm,
        args.grasp_max_step_ticks,
        (args.grasp_id1_min, args.grasp_id1_max),
        (args.grasp_id2_min, args.grasp_id2_max),
        args.servo_angle_gap_deg,
        startup_sequence=not args.skip_grasp_startup_sequence,
        one_shot=args.grasp_one_shot,
        camera_gripper_vertical_offset_mm=args.camera_gripper_vertical_offset_mm,
        max_lateral_offset_mm=args.max_lateral_offset_mm,
        max_one_shot_ik_error_mm=args.max_one_shot_ik_error_mm,
        post_center_retreat_mm=args.post_center_retreat_mm,
        post_center_down_mm=args.post_center_down_mm,
        post_center_ik_error_mm=args.post_center_ik_error_mm,
        use_calibrated_grasp_table=args.use_calibrated_grasp_table,
        use_post_center_tick_bias=args.use_post_center_tick_bias,
        post_center_direct_descend=args.post_center_direct_descend,
        simple_vertical_grasp=args.simple_vertical_grasp,
        vertical_grasp_id1=args.vertical_grasp_id1,
        vertical_grasp_id2=args.vertical_grasp_id2,
        initial_id4=preview_id4,
        id2_center_min=args.grasp_id2_center_min,
        id2_center_max=args.grasp_id2_center_max,
    )
    chassis_link = ChassisLink(
        args.chassis_uart,
        args.chassis_baud,
        enabled=args.chassis_link and execute_auto_grasp,
    )
    station = {
        "active": not chassis_link.enabled,
        "finishing": False,
        "task": "MANUAL",
        "outcome": None,
        "started_at": time.monotonic(),
        "last_seen_at": time.monotonic(),
        "finish_deadline": 0.0,
        "cycle_start": grasp_controller.completed_cycles,
    }
    if chassis_link.enabled:
        grasp_controller.startup_stage = "complete"
        grasp_controller.hold_home("waiting H7 station")
    server = None
    if not args.no_web:
        server = ThreadedHTTPServer(("0.0.0.0", args.web_port), StreamHandler, state)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Web view: http://{local_ip_hint()}:{args.web_port}")

    if not args.keep_camera_users:
        stop_old_camera_viewers()
    if not args.keep_pipewire:
        set_pipewire(False)
    cap = open_camera(args.device, args.width, args.height, args.fps)
    window_enabled = not args.no_window
    window_close_watcher = None
    if window_enabled:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            args.window_width or min(args.width, 1120),
            args.window_height or min(args.height, 630),
        )
        if args.window_x is not None and args.window_y is not None:
            cv2.moveWindow(WINDOW_NAME, args.window_x, args.window_y)
        window_close_watcher = WindowCloseWatcher(WINDOW_NAME)
        window_close_watcher.start()

    shutdown_signal = {"received": None}

    def request_shutdown(signum, _frame):
        shutdown_signal["received"] = signum
        raise KeyboardInterrupt

    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(handled_signal, request_shutdown)

    print("Press q or Esc in the preview window to quit.")
    fps_started = time.monotonic()
    fps_last_log = fps_started
    fps_frames = 0
    display_fps = 0.0
    detection_frame_index = 0
    detections = []
    detection_interval = max(1, args.detect_every_n_frames)
    detection_scale = max(0.4, min(1.0, float(args.detection_scale)))
    sync_detection_for_grasp = bool(args.sync_detection_for_grasp and auto_grasp_enabled)
    detection_executor = None
    if not sync_detection_for_grasp:
        detection_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
    detection_future = None

    perf_started = fps_started
    perf_frames = 0
    perf_detect_frames = 0
    perf_read_s = 0.0
    perf_detect_s = 0.0
    perf_post_s = 0.0
    perf_display_s = 0.0
    try:
        while True:
            read_started = time.perf_counter()
            ok, frame = cap.read()
            read_elapsed = time.perf_counter() - read_started
            if not ok:
                time.sleep(0.02)
                continue
            detect_elapsed = 0.0
            if sync_detection_for_grasp:
                if detection_frame_index % detection_interval == 0:
                    try:
                        detections, detect_elapsed = detection_process_worker(
                            frame.copy(),
                            detection_scale,
                            args.qr_template_every,
                        )
                        detections = detection_smoother.update(detections)
                        perf_detect_frames += 1
                    except Exception as exc:
                        print(f"detection worker failed: {exc}", flush=True)
            else:
                if detection_future is not None and detection_future.done():
                    try:
                        detections, detect_elapsed = detection_future.result()
                        detections = detection_smoother.update(detections)
                        perf_detect_frames += 1
                    except Exception as exc:
                        print(f"detection worker failed: {exc}", flush=True)
                    detection_future = None
                if (
                    detection_frame_index % detection_interval == 0
                    and detection_future is None
                ):
                    detection_future = detection_executor.submit(
                        detection_process_worker,
                        frame.copy(),
                        detection_scale,
                        args.qr_template_every,
                    )
            detection_frame_index += 1
            post_started = time.perf_counter()
            now = time.monotonic()
            if chassis_link.enabled:
                for chassis_line in chassis_link.read_lines():
                    upper_line = chassis_line.strip().upper()
                    parts = [part.strip() for part in upper_line.split(",")]
                    if upper_line == "PING":
                        chassis_link.send("RK,PONG")
                    elif (
                        len(parts) >= 3
                        and parts[0] == "ARM"
                        and parts[1] == "DISC_CATCH"
                        and parts[2] == "PREP_HIGH"
                    ):
                        prep_status = grasp_controller.prepare_disc_catch_high()
                        print(
                            f"CHASSIS STATION PREP_HIGH {prep_status}",
                            flush=True,
                        )
                        chassis_link.send(
                            "RK,ARM,DISC_CATCH,PREP_HIGH_ACK"
                        )
                    elif len(parts) >= 3 and parts[0] == "ARM" and parts[2] == "START":
                        task_name = parts[1] or "UNKNOWN"
                        station.update(
                            {
                                "active": True,
                                "finishing": False,
                                "task": task_name,
                                "outcome": None,
                                "started_at": now,
                                "last_seen_at": now,
                                "finish_deadline": 0.0,
                                "cycle_start": grasp_controller.completed_cycles,
                            }
                        )
                        begin_status = grasp_controller.begin_station_task(task_name)
                        print(
                            f"CHASSIS STATION START task={task_name} | {begin_status}",
                            flush=True,
                        )
                        chassis_link.send(f"RK,ARM,{task_name},ACK")
                    elif upper_line == "STOP":
                        station["active"] = False
                        station["finishing"] = False
                        grasp_controller.hold_home("H7 stop requested")
                        chassis_link.send("RK,STOPPED")
                if station["finishing"] and now >= station["finish_deadline"]:
                    chassis_link.send(
                        f"RK,ARM,{station['task']},DONE,{station['outcome']}"
                    )
                    station["finishing"] = False
                    station["active"] = False
            selected_targets = [
                det for det in detections
                if matches_trigger_target(det, args.trigger_color, args.trigger_kind)
            ]
            preview_target = max(
                selected_targets,
                key=lambda det: det.get("bbox", (0, 0, 0, 0))[2]
                * det.get("bbox", (0, 0, 0, 0))[3],
                default=None,
            )
            if (
                auto_grasp_enabled
                and args.fresh_detection_on_lock
                and preview_target is not None
                and grasp_controller.locked_target is None
                and grasp_controller.centered_frames
                >= max(0, grasp_controller.stable_frames_required - 1)
            ):
                try:
                    fresh_detections, fresh_elapsed = detection_process_worker(
                        frame.copy(),
                        detection_scale,
                        args.qr_template_every,
                    )
                    detections = detection_smoother.update(fresh_detections)
                    perf_detect_frames += 1
                    detect_elapsed += fresh_elapsed
                    selected_targets = [
                        det for det in detections
                        if matches_trigger_target(det, args.trigger_color, args.trigger_kind)
                    ]
                    preview_target = max(
                        selected_targets,
                        key=lambda det: det.get("bbox", (0, 0, 0, 0))[2]
                        * det.get("bbox", (0, 0, 0, 0))[3],
                        default=None,
                    )
                except Exception as exc:
                    print(f"fresh lock detection failed: {exc}", flush=True)
            display_detections = [
                det for det in detections
                if not (
                    matches_trigger_target(det, args.trigger_color, args.trigger_kind)
                )
            ]
            if preview_target is not None:
                display_detections.append(preview_target)
            trigger_info = servo_trigger.update(display_detections)
            output, info = detector.draw(frame, display_detections)
            aux_info = ""
            if auto_grasp_enabled:
                target_for_grasp = preview_target
                station_info = ""
                if chassis_link.enabled:
                    if station["active"] and preview_target is not None:
                        station["last_seen_at"] = now
                    if (
                        station["active"]
                        and grasp_controller.locked_target is None
                        and now - station["last_seen_at"]
                        >= max(0.5, args.station_no_target_timeout)
                    ):
                        station["active"] = False
                        station["finishing"] = True
                        station["outcome"] = "NO_TARGET"
                        station["finish_deadline"] = (
                            now + max(0.2, grasp_controller._arm_settle_s())
                        )
                        station_info = grasp_controller.hold_home(
                            f"station {station['task']} no target timeout"
                        )
                        print(
                            f"CHASSIS STATION TIMEOUT task={station['task']} "
                            f"after={args.station_no_target_timeout:.1f}s",
                            flush=True,
                        )
                    elif (
                        station["active"]
                        and grasp_controller.completed_cycles
                        > station["cycle_start"]
                    ):
                        station["active"] = False
                        station["finishing"] = True
                        station["outcome"] = "OK"
                        station["finish_deadline"] = (
                            now + max(0.2, grasp_controller._arm_settle_s())
                        )
                        station_info = grasp_controller.hold_home(
                            f"station {station['task']} complete"
                        )
                        print(
                            f"CHASSIS STATION COMPLETE task={station['task']}",
                            flush=True,
                        )
                    if not station["active"]:
                        target_for_grasp = None
                if (
                    station["active"]
                    and target_for_grasp is None
                    and grasp_controller.locked_target is None
                ):
                    aux_info = grasp_controller.update_auxiliary(detections)
                if chassis_link.enabled and not station["active"]:
                    grasp_info = station_info or grasp_controller.hold_home(
                        "waiting H7 station"
                        if not station["finishing"]
                        else f"station {station['task']} finishing"
                    )
                else:
                    grasp_info = grasp_controller.update(target_for_grasp, frame.shape)
                arm_preview.publish(
                    grasp_info,
                    preview_target,
                    grasp_controller.state,
                )
                info = f"{info} | {grasp_info}"
            elif preview_target is not None:
                arm_preview.publish(
                    f"{args.trigger_color} {args.trigger_kind} detected",
                    preview_target,
                )
            else:
                arm_preview.publish(
                    f"searching {args.trigger_color} {args.trigger_kind}"
                )
            if trigger_info:
                info = f"{info} | {trigger_info}"
            if aux_info:
                info = f"{info} | {aux_info}"
            fps_frames += 1
            fps_elapsed = time.monotonic() - fps_started
            if fps_elapsed >= 1.0:
                display_fps = fps_frames / fps_elapsed
                fps_started = time.monotonic()
                fps_frames = 0
            if time.monotonic() - fps_last_log >= 5.0:
                print(f"VISION FPS={display_fps:.1f}", flush=True)
                fps_last_log = time.monotonic()
            info = f"{info} | FPS {display_fps:.1f}"
            state.update(output, info)
            post_elapsed = time.perf_counter() - post_started
            display_started = time.perf_counter()
            if window_enabled:
                cv2.imshow(WINDOW_NAME, output)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
                        break
                except cv2.error:
                    pass
                if window_close_watcher.closed.is_set():
                    break
            display_elapsed = time.perf_counter() - display_started
            perf_frames += 1
            perf_read_s += read_elapsed
            perf_detect_s += detect_elapsed
            perf_post_s += post_elapsed
            perf_display_s += display_elapsed
            perf_elapsed = time.monotonic() - perf_started
            if perf_elapsed >= 5.0 and perf_frames:
                detect_divisor = max(1, perf_detect_frames)
                print(
                    "VISION PERF "
                    f"read={perf_read_s * 1000.0 / perf_frames:.1f}ms "
                    f"detect={perf_detect_s * 1000.0 / detect_divisor:.1f}ms/detect "
                    f"post={perf_post_s * 1000.0 / perf_frames:.1f}ms "
                    f"display={perf_display_s * 1000.0 / perf_frames:.1f}ms",
                    flush=True,
                )
                perf_started = time.monotonic()
                perf_frames = 0
                perf_detect_frames = 0
                perf_read_s = 0.0
                perf_detect_s = 0.0
                perf_post_s = 0.0
                perf_display_s = 0.0
    except KeyboardInterrupt:
        detail = shutdown_signal["received"]
        if detail is None:
            print("Shutdown requested from terminal.", flush=True)
        else:
            print(f"Shutdown signal {detail} received.", flush=True)
    finally:
        state.running = False
        if detection_executor is not None:
            detection_executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        if auto_grasp_enabled and execute_auto_grasp:
            try:
                grasp_controller.shutdown_contract()
            except Exception as exc:
                print(f"shutdown contract error: {exc}", flush=True)
        servo_trigger.close()
        chassis_link.close()
        servo_bridge.close()
        arm_preview.close()
        if server is not None:
            server.shutdown()
        if window_enabled:
            cv2.destroyAllWindows()
        if window_close_watcher is not None:
            window_close_watcher.stop()
        if not args.keep_pipewire:
            set_pipewire(True)


if __name__ == "__main__":
    main()
