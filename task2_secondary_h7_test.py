#!/usr/bin/env python3
"""Temporary task-two secondary-camera and H7 link smoke test.

Default mode is camera-only. H7 motion requires --execute-h7 and a firmware
parser that implements the dedicated RK,TEST,TASK2 protocol.
"""
from __future__ import annotations

import argparse
import glob
import os
import select
import sys
import time
from collections import defaultdict, deque

import cv2


LETTERS = {"A", "B", "C", "D"}
SECONDARY_DEFAULT = (
    "/dev/v4l/by-path/platform-fc880000.usb-usb-0:1.3:1.0-video-index0"
)


def detector_from_package():
    package_dir = os.environ.get("ABCD_PACKAGE_DIR", "")
    if package_dir and package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    try:
        from abcd_detector import ABCDDetector
    except ImportError:
        # Allow execution from the deployed workspace without embedding a
        # machine-specific path in the detector package itself.
        candidates = [
            os.path.join(os.path.dirname(__file__), "ABCD_detector"),
            os.path.join(os.path.dirname(__file__), "ros2_test1", "ABCD_detector"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        from abcd_detector import ABCDDetector
    return ABCDDetector()


def choose_h7_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.path.exists("/dev/h7_chassis"):
        return "/dev/h7_chassis"
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    for device in candidates:
        real = os.path.realpath(device)
        if "1a86" not in real and "USB_Serial" not in device:
            return device
    return candidates[0] if candidates else ""


class H7Link:
    def __init__(self, device: str, baud: int):
        self.device = choose_h7_device(device)
        self.baud = baud
        self.fd = None
        self.rx = bytearray()

    def open(self):
        if not self.device:
            raise RuntimeError("H7 USB CDC device not found")
        import termios

        self.fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        speed = getattr(termios, f"B{self.baud}", termios.B115200)
        attrs[4] = attrs[5] = speed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        print(f"TASK2_TEST H7_OPEN device={self.device}", flush=True)

    def send(self, line: str):
        if self.fd is None:
            self.open()
        payload = (line.rstrip("\r\n") + "\r\n").encode("ascii")
        offset = 0
        deadline = time.monotonic() + 0.5
        while offset < len(payload):
            try:
                offset += os.write(self.fd, payload[offset:])
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("H7 CDC write timeout")
                select.select([], [self.fd], [], 0.02)
        print(f"TASK2_TEST H7_TX {line.rstrip()}", flush=True)

    def wait_for(self, sequence: int, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.fd], [], [], 0.1)
            if not readable:
                continue
            data = os.read(self.fd, 4096)
            if data:
                self.rx.extend(data)
            while b"\n" in self.rx:
                raw, _, self.rx = self.rx.partition(b"\n")
                line = raw.decode("ascii", "replace").strip("\r")
                print(f"TASK2_TEST H7_RX {line}", flush=True)
                parts = line.split(",")
                if "SEQ" in parts and str(sequence) in parts:
                    return line
        return None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def extract_letters(detections):
    valid = []
    for detection in detections:
        letter = str(detection.get("letter", "")).upper()
        if letter not in LETTERS:
            continue
        center = detection.get("center", (0, 0))
        confidence = float(detection.get("confidence", detection.get("score", 0)))
        valid.append((letter, int(center[0]), confidence, detection))
    return sorted(valid, key=lambda item: item[1])


def stable_pair(history, min_frames):
    if len(history) < min_frames:
        return None
    recent = list(history)[-min_frames:]
    pairs = []
    for frame in recent:
        letters = sorted({item[0] for item in frame})
        if len(letters) >= 2:
            pairs.append(tuple(letters[:2]))
    if len(pairs) < min_frames:
        return None
    counts = defaultdict(int)
    for pair in pairs:
        counts[pair] += 1
    pair, count = max(counts.items(), key=lambda item: item[1])
    return pair if count >= min_frames else None


def draw(frame, detections, state):
    view = frame.copy()
    for detection in detections:
        letter = str(detection.get("letter", "?")).upper()
        x, y = map(int, detection.get("center", (0, 0)))
        color = (0, 220, 0) if letter in LETTERS else (0, 0, 255)
        cv2.circle(view, (x, y), 10, color, 2)
        cv2.putText(view, letter, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
    cv2.putText(view, state, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def main():
    parser = argparse.ArgumentParser(description="Task-two secondary camera H7 test")
    parser.add_argument("--field", choices=("red", "blue"), required=True)
    parser.add_argument("--camera", choices=("secondary",), default="secondary")
    parser.add_argument("--camera-device", default=SECONDARY_DEFAULT)
    parser.add_argument("--h7-device", default="auto")
    parser.add_argument("--h7-baud", type=int, default=115200)
    parser.add_argument("--execute-h7", action="store_true",
                        help="send the dedicated test command after pair lock")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--stable-frames", type=int, default=5)
    args = parser.parse_args()

    detector = detector_from_package()
    camera = cv2.VideoCapture(args.camera_device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise SystemExit(f"SECONDARY_CAMERA_ERROR cannot open {args.camera_device}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    history = deque(maxlen=max(5, args.stable_frames * 2))
    started = time.monotonic()
    locked = None
    link = H7Link(args.h7_device, args.h7_baud) if args.execute_h7 else None
    print(f"TASK2_TEST CAMERA_OPEN role=secondary path={args.camera_device}", flush=True)
    try:
        while time.monotonic() - started < args.timeout_s:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("TASK2_TEST SECONDARY_CAMERA_FRAME_TIMEOUT", flush=True)
                continue
            detections = detector.detect(frame)
            letters = extract_letters(detections)
            history.append(letters)
            locked = stable_pair(history, args.stable_frames)
            state = "PRESELECTING " + ",".join(item[0] for item in letters)
            if locked:
                print(f"TASK2_TEST LETTERS_LOCKED field={args.field.upper()} "
                      f"L1={locked[0]} L2={locked[1]} camera=secondary",
                      flush=True)
                state = f"LOCKED {locked[0]}+{locked[1]}"
            cv2.imshow("Task 2 Secondary H7 Test", draw(frame, detections, state))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("TASK2_TEST ABORTED", flush=True)
                return 2
            if key == ord("r"):
                history.clear()
                locked = None
                print("TASK2_TEST RESTART_RECOGNITION", flush=True)
            if locked:
                break
        if not locked:
            print("TASK2_TEST LETTER_TIMEOUT", flush=True)
            return 3
        if not args.execute_h7:
            print("TASK2_TEST CAMERA_ONLY no H7 command sent", flush=True)
            return 0
        sequence = int(time.time() * 1000) & 0xFFFFFFFF
        line = (f"RK,TEST,TASK2,START,SEQ,{sequence},FIELD,{args.field.upper()},"
                f"L1,{locked[0]},L2,{locked[1]}")
        link.send(line)
        reply = link.wait_for(sequence, 3.0)
        if not reply:
            print(f"TASK2_TEST H7_REJECTED seq={sequence} reason=ACK_TIMEOUT", flush=True)
            return 4
        if "ACK" not in reply.split(","):
            print(f"TASK2_TEST H7_REJECTED seq={sequence} reason=NO_ACK", flush=True)
            return 4
        print(f"TASK2_TEST H7_ACKED seq={sequence}; waiting for DONE", flush=True)
        return 0 if link.wait_for(sequence, 120.0) else 5
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if link:
            link.close()


if __name__ == "__main__":
    raise SystemExit(main())
