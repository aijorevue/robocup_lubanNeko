#!/usr/bin/env python3
"""Standalone RoboCup task-one ball tester for the RK desktop.

The app owns the main camera, the Hiwonder HTD-85 USB bus and the ZP20S bus.
It deliberately does not open the H7 CDC device.
"""

from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import serial


APP_TITLE = "识别小球任务分红/蓝场"
SCRIPT_DIR = Path(__file__).resolve().parent
BALLS_PACKAGE_DIR = SCRIPT_DIR / "balls_detector"
DEFAULT_CAMERA = "/dev/v4l/by-path/platform-fc800000.usb-usb-0:1:1.0-video-index0"
DEFAULT_ARM_UART = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82109853-if00"
DEFAULT_ZP_UART = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

FIELD_RED = "RED"
FIELD_BLUE = "BLUE"
FIELD_ALLOWED = {
    FIELD_RED: {"red", "yellow"},
    FIELD_BLUE: {"blue", "yellow"},
}
FIELD_LABEL = {FIELD_RED: "红场：红球 / 黄球", FIELD_BLUE: "蓝场：蓝球 / 黄球"}
DRAW_COLORS = {"red": (0, 0, 255), "blue": (255, 0, 0), "yellow": (0, 255, 255)}

ZP_SPLITTER_ID = 4
ZP_CATCHER_ID = 5
ZP_GRIPPER_ID = 7
LOW_ID1 = 470
LOW_ID2 = 550
LOW_ID6 = 670
LOW_SPLITTER = 1300
LOW_CATCHER = 1110
SPLITTER_OWN_COLOR = 1100
SPLITTER_YELLOW = 1600
GRIPPER_CLOSED = 1300
GRIPPER_OPEN = 1710
ARM_TIME_MS = 600
ZP_TIME_MS = 300
GRIPPER_TIME_MS = 210
SPLITTER_TIME_MS = 300
GRIPPER_HOLD_AFTER_OPEN_MS = GRIPPER_TIME_MS + 100
YELLOW_POST_CLOSE_COOLDOWN_MS = 500
YELLOW_SPLITTER_RESET_TIME_MS = 300
ZP_BATCH_GAP_S = 0.003
LOW_POSE_SETTLE_MARGIN_S = 0.15
ARM_POSITION_TOLERANCE = 20
GRIPPER_CLOSE_CONFIRM_DELAY_S = 0.05
ACTION_JOIN_TIMEOUT_S = 2.0

HTD85_HEADER = b"\x55\x55"
HTD85_MOVE_COMMAND = 0x01
HTD85_POSITION_READ_COMMAND = 0x1C


def htd85_checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def htd85_move_packet(servo_id: int, position: int, time_ms: int) -> bytes:
    position = max(0, min(1000, int(position)))
    time_ms = max(0, min(30000, int(time_ms)))
    body = bytes(
        (
            int(servo_id),
            7,
            HTD85_MOVE_COMMAND,
            position & 0xFF,
            (position >> 8) & 0xFF,
            time_ms & 0xFF,
            (time_ms >> 8) & 0xFF,
        )
    )
    return HTD85_HEADER + body + bytes((htd85_checksum(body),))


def htd85_position_read_packet(servo_id: int) -> bytes:
    body = bytes((int(servo_id), 3, HTD85_POSITION_READ_COMMAND))
    return HTD85_HEADER + body + bytes((htd85_checksum(body),))


def zp_move_packet(servo_id: int, position: int, time_ms: int) -> bytes:
    position = max(500, min(2500, int(position)))
    time_ms = max(0, min(9999, int(time_ms)))
    return f"#{servo_id:03d}P{position:04d}T{time_ms:04d}!".encode("ascii")


class ZpBus:
    def __init__(self, device: str, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.fd = None
        self.lock = threading.Lock()
        self.status = "ZP 总线未打开"
        if enabled:
            self.open()

    def open(self) -> None:
        if self.fd is not None:
            return
        if not os.path.exists(self.device):
            raise RuntimeError(f"ZP 串口不存在: {self.device}")
        result = subprocess.run(
            [
                "stty", "-F", self.device, "115200", "cs8", "-cstopb",
                "-parenb", "-ixon", "-ixoff", "-crtscts", "raw", "-echo",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "stty 配置失败")
        import os as _os
        import fcntl
        import termios

        self.fd = _os.open(self.device, _os.O_RDWR | _os.O_NOCTTY | _os.O_NONBLOCK)
        if hasattr(termios, "TIOCEXCL"):
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
        self.status = f"ZP 总线已连接 {self.device}，物理 ID4/ID5/ID7"

    def send(self, servo_id: int, position: int, time_ms: int) -> str:
        if not self.enabled:
            return "模拟模式"
        if self.fd is None:
            raise RuntimeError("ZP 串口未打开")
        packet = zp_move_packet(servo_id, position, time_ms)
        with self.lock:
            offset = 0
            deadline = time.monotonic() + 0.3
            while offset < len(packet):
                try:
                    written = os.write(self.fd, packet[offset:])
                    if written <= 0:
                        raise OSError("串口写入返回 0")
                    offset += written
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("ZP 串口写入超时")
                    _, writable, _ = select.select([], [self.fd], [], min(0.02, remaining))
        text = packet.decode("ascii")
        print(f"BALL APP ZP TX {text}", flush=True)
        return text

    def send_group(self, targets) -> str:
        if not self.enabled:
            return "simulation"
        if self.fd is None:
            raise RuntimeError("ZP bus is not open")
        packets = [
            zp_move_packet(servo_id, position, time_ms)
            for servo_id, position, time_ms in targets
        ]
        with self.lock:
            for index, packet in enumerate(packets):
                offset = 0
                deadline = time.monotonic() + 0.3
                while offset < len(packet):
                    try:
                        written = os.write(self.fd, packet[offset:])
                        if written <= 0:
                            raise OSError("ZP write returned 0")
                        offset += written
                    except BlockingIOError:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("ZP batch write timed out")
                        _, writable, _ = select.select(
                            [], [self.fd], [], min(0.02, remaining)
                        )
                if index + 1 < len(packets):
                    time.sleep(ZP_BATCH_GAP_S)
        text = " ".join(packet.decode("ascii") for packet in packets)
        print(f"BALL APP ZP BATCH TX {text}", flush=True)
        return text

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.status = "ZP 总线已关闭"


class Htd85Bus:
    """Hiwonder USB transport for the HTD-85 binary servo board."""

    def __init__(self, device: str, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.port = None
        self.lock = threading.Lock()
        self.status = "HTD-85 USB bus 未打开"

    def open(self) -> None:
        if not self.enabled or self.port is not None:
            return
        if not os.path.exists(self.device):
            raise RuntimeError(f"幻尔 HTD-85 串口不存在: {self.device}")
        self.port = serial.Serial(
            self.device,
            115200,
            timeout=0.05,
            write_timeout=1.0,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            rtscts=False,
            dsrdtr=False,
        )
        self.status = f"HTD-85 USB bus 已连接 {self.device}"

    def send(self, servo_id: int, position: int, time_ms: int) -> str:
        if not self.enabled:
            return "模拟模式"
        self.open()
        packet = htd85_move_packet(servo_id, position, time_ms)
        with self.lock:
            self.port.write(packet)
            self.port.flush()
        text = packet.hex(" ")
        print(
            f"BALL APP HTD85 TX ID{servo_id} target={position} "
            f"time={time_ms}ms frame={text}",
            flush=True,
        )
        return text

    def read_position(self, servo_id: int, timeout_s: float = 0.35):
        if not self.enabled:
            return None
        self.open()
        request = htd85_position_read_packet(servo_id)
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        received = bytearray()
        with self.lock:
            self.port.reset_input_buffer()
            self.port.write(request)
            self.port.flush()
            while time.monotonic() < deadline:
                waiting = self.port.in_waiting
                if waiting:
                    received.extend(self.port.read(waiting))
                else:
                    time.sleep(0.01)
                while True:
                    header = received.find(HTD85_HEADER)
                    if header < 0:
                        received.clear()
                        break
                    if header:
                        del received[:header]
                    if len(received) < 4:
                        break
                    frame_size = int(received[3]) + 3
                    if frame_size < 6 or frame_size > 64:
                        del received[0]
                        continue
                    if len(received) < frame_size:
                        break
                    frame = bytes(received[:frame_size])
                    del received[:frame_size]
                    body = frame[2:-1]
                    if (
                        frame[2] == int(servo_id)
                        and frame[4] == HTD85_POSITION_READ_COMMAND
                        and frame[-1] == htd85_checksum(body)
                    ):
                        position = frame[5] | (frame[6] << 8)
                        print(
                            f"BALL APP HTD85 RX ID{servo_id} position={position}",
                            flush=True,
                        )
                        return position
        return None

    def close(self) -> None:
        if self.port is not None:
            self.port.close()
            self.port = None
        self.status = "HTD-85 USB bus 已关闭"


class BallTaskApp:
    def __init__(self, camera: str, arm_uart: str, zp_uart: str, execute: bool):
        self.camera_path = camera
        self.field = FIELD_RED
        self.execute = execute
        self.running = True
        self.camera = None
        self.zp = None
        self.last_frame = None
        self.last_detections = []
        self.last_action = "尚未触发动作"
        self.last_error = ""
        self.target_latched = False
        self.latched_color = None
        self.gripper_commanded = GRIPPER_CLOSED
        self.task_ready = False
        self.preparing = False
        self.action_lock = threading.Lock()
        self.action_thread = None
        self.frame_lock = threading.Lock()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._open_camera()
        if execute:
            self.zp = ZpBus(zp_uart, enabled=True)
            self.arm = Htd85Bus(arm_uart, enabled=True)
        else:
            self.zp = ZpBus(zp_uart, enabled=False)
            self.arm = Htd85Bus(arm_uart, enabled=False)

    def _open_camera(self):
        self.camera = cv2.VideoCapture(self.camera_path, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError(f"无法打开主摄像头: {self.camera_path}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    @staticmethod
    def _detect(frame):
        if str(BALLS_PACKAGE_DIR) not in sys.path:
            sys.path.insert(0, str(BALLS_PACKAGE_DIR))
        from balls_detector import BALL_DETECTORS

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        rings = []
        for detector in BALL_DETECTORS:
            detections.extend(detector.detect(hsv, rings))
        return detections

    def _draw(self, frame, detections):
        view = frame.copy()
        for det in detections:
            x, y = det["center"]
            radius = max(4, int(det["radius"]))
            color = DRAW_COLORS.get(det["color"], (255, 255, 255))
            cv2.circle(view, (x, y), radius, color, 3)
            cv2.putText(view, f'{det["color"]} {det["score"]:.2f}', (x - radius, y - radius - 8), self.font, 0.6, color, 2)
        allowed = FIELD_ALLOWED[self.field]
        cv2.rectangle(view, (0, 0), (view.shape[1], 78), (25, 25, 25), -1)
        cv2.rectangle(view, (12, 10), (170, 48), (55, 90, 180) if self.field == FIELD_RED else (55, 55, 55), -1)
        cv2.rectangle(view, (180, 10), (338, 48), (55, 90, 180) if self.field == FIELD_BLUE else (55, 55, 55), -1)
        cv2.putText(view, "RED FIELD", (28, 36), self.font, 0.55, (255, 255, 255), 2)
        cv2.putText(view, "BLUE FIELD", (192, 36), self.font, 0.55, (255, 255, 255), 2)
        cv2.putText(view, FIELD_LABEL[self.field], (360, 62), self.font, 0.52, (255, 255, 255), 1)
        cv2.putText(view, "R/B START FIELD; Q QUIT", (360, 34), self.font, 0.52, (190, 190, 190), 1)
        allowed_seen = [d for d in detections if d["color"] in allowed]
        if self.preparing:
            state = "PREPARING LOW POSE"
        elif self.task_ready:
            state = "TASK ACTIVE"
        else:
            state = "SELECT RED/BLUE TO START"
        status = f"{state}  allowed={','.join(sorted(allowed))}  seen={len(allowed_seen)}"
        cv2.putText(view, status, (18, view.shape[0] - 42), self.font, 0.58, (255, 255, 255), 2)
        cv2.putText(view, f"动作: {self.last_action[:100]}", (18, view.shape[0] - 15), self.font, 0.48, (0, 255, 0) if not self.last_error else (0, 0, 255), 1)
        return view, allowed_seen

    def _trigger_if_needed(self, allowed):
        if not self.task_ready or self.preparing:
            return
        # The preparation thread sets task_ready immediately before its final
        # cleanup. Do not latch a one-frame target while that thread is alive.
        if self.action_thread and self.action_thread.is_alive():
            return
        if not allowed:
            return
        if self.target_latched:
            return
        det = max(allowed, key=lambda item: item.get("radius", 0))
        self.target_latched = True
        self.latched_color = det["color"]
        print(
            f"BALL APP TARGET TRIGGER color={det['color']} "
            f"gripper={self.gripper_commanded}",
            flush=True,
        )
        self.action_thread = threading.Thread(target=self._run_pulse, args=(self.field, det["color"]), daemon=True)
        self.action_thread.start()

    def _run_pulse(self, field, color):
        with self.action_lock:
            splitter = SPLITTER_YELLOW if color == "yellow" else SPLITTER_OWN_COLOR
            open_sent = False
            close_sent = False
            close_confirmed = False
            try:
                self.last_error = ""
                if self.gripper_commanded != GRIPPER_CLOSED:
                    self.zp.send(ZP_GRIPPER_ID, GRIPPER_CLOSED, GRIPPER_TIME_MS)
                    self.gripper_commanded = GRIPPER_CLOSED
                    time.sleep(GRIPPER_TIME_MS / 1000.0)
                self.last_action = f"检测 {color}，ID4={splitter} + ID7={GRIPPER_OPEN}"
                self.zp.send_group(
                    (
                        (ZP_SPLITTER_ID, splitter, SPLITTER_TIME_MS),
                        (ZP_GRIPPER_ID, GRIPPER_OPEN, GRIPPER_TIME_MS),
                    )
                )
                open_sent = True
                self.gripper_commanded = GRIPPER_OPEN
                print(
                    f"BALL APP GRIPPER OPEN_SENT ID7={GRIPPER_CLOSED}->{GRIPPER_OPEN} "
                    f"motion={GRIPPER_TIME_MS}ms",
                    flush=True,
                )
                time.sleep(GRIPPER_HOLD_AFTER_OPEN_MS / 1000.0)
                self.zp.send(ZP_GRIPPER_ID, GRIPPER_CLOSED, GRIPPER_TIME_MS)
                close_sent = True
                self.gripper_commanded = GRIPPER_CLOSED
                print(
                    f"BALL APP GRIPPER CLOSE_SENT ID7={GRIPPER_OPEN}->{GRIPPER_CLOSED} "
                    f"motion={GRIPPER_TIME_MS}ms",
                    flush=True,
                )
                # There is no ZP position feedback. Repeat the close frame
                # after a short bus gap so a transient UART/device miss does
                # not leave the gripper open.
                time.sleep(GRIPPER_CLOSE_CONFIRM_DELAY_S)
                self.zp.send(ZP_GRIPPER_ID, GRIPPER_CLOSED, GRIPPER_TIME_MS)
                self.gripper_commanded = GRIPPER_CLOSED
                print(
                    f"BALL APP GRIPPER CLOSE_CONFIRM ID7={GRIPPER_CLOSED} "
                    f"motion={GRIPPER_TIME_MS}ms",
                    flush=True,
                )
                time.sleep(GRIPPER_TIME_MS / 1000.0)
                close_confirmed = True
                self.last_action = (
                    f"完成 {color}：ID4={splitter}，ID7 "
                    f"{GRIPPER_OPEN}->{GRIPPER_CLOSED}，动作各 "
                    f"{GRIPPER_TIME_MS}ms，开后保持 "
                    f"{GRIPPER_HOLD_AFTER_OPEN_MS}ms"
                )
            except Exception as exc:
                self.last_error = str(exc)
                self.last_action = f"动作失败: {exc}"
            finally:
                if open_sent and not close_confirmed:
                    try:
                        time.sleep(GRIPPER_CLOSE_CONFIRM_DELAY_S)
                        self.zp.send(ZP_GRIPPER_ID, GRIPPER_CLOSED, GRIPPER_TIME_MS)
                        self.gripper_commanded = GRIPPER_CLOSED
                        print(
                            f"BALL APP GRIPPER CLOSE_RETRY ID7={GRIPPER_CLOSED} "
                            f"motion={GRIPPER_TIME_MS}ms",
                            flush=True,
                        )
                        time.sleep(GRIPPER_TIME_MS / 1000.0)
                        close_confirmed = True
                    except Exception as retry_exc:
                        self.last_error = f"闭合重试失败: {retry_exc}"
                        self.last_action = f"动作失败，ID7未确认闭合: {retry_exc}"
                if close_confirmed and color == "yellow":
                    print(
                        f"BALL APP {field} FIELD YELLOW ID4 RESET target={LOW_SPLITTER} "
                        f"motion={YELLOW_SPLITTER_RESET_TIME_MS}ms; "
                        f"ID7 HOLD; cooldown={YELLOW_POST_CLOSE_COOLDOWN_MS}ms",
                        flush=True,
                    )
                    self.zp.send(
                        ZP_SPLITTER_ID,
                        LOW_SPLITTER,
                        YELLOW_SPLITTER_RESET_TIME_MS,
                    )
                    time.sleep(YELLOW_POST_CLOSE_COOLDOWN_MS / 1000.0)
                if not open_sent or close_confirmed:
                    self.target_latched = False
                    self.latched_color = None
                    print(
                        f"BALL APP NEXT TARGET READY gripper={self.gripper_commanded}",
                        flush=True,
                    )
                else:
                    print(
                        "BALL APP NEXT TARGET BLOCKED gripper close incomplete",
                        flush=True,
                    )

    def _start_field(self, field):
        if self.preparing:
            return
        self.field = field
        self.task_ready = False
        self.target_latched = False
        self.latched_color = None
        self.last_error = ""
        self.preparing = True
        self.action_thread = threading.Thread(
            target=self._prepare_low_pose,
            args=(field,),
            daemon=True,
        )
        self.action_thread.start()

    def _prepare_low_pose(self, field):
        with self.action_lock:
            try:
                self.last_action = (
                    f"{field} 低位准备: ID1={LOW_ID1} ID2={LOW_ID2} "
                    f"ID6={LOW_ID6} ID4={LOW_SPLITTER} "
                    f"ID5={LOW_CATCHER} ID7={GRIPPER_CLOSED}"
                )
                self.arm.send(1, LOW_ID1, ARM_TIME_MS)
                time.sleep(0.003)
                self.arm.send(2, LOW_ID2, ARM_TIME_MS)
                time.sleep(0.003)
                self.arm.send(6, LOW_ID6, ARM_TIME_MS)
                self.zp.send_group(
                    (
                        (ZP_SPLITTER_ID, LOW_SPLITTER, SPLITTER_TIME_MS),
                        (ZP_CATCHER_ID, LOW_CATCHER, ZP_TIME_MS),
                        (ZP_GRIPPER_ID, GRIPPER_CLOSED, GRIPPER_TIME_MS),
                    )
                )
                self.gripper_commanded = GRIPPER_CLOSED
                time.sleep(
                    max(ARM_TIME_MS, ZP_TIME_MS, GRIPPER_TIME_MS) / 1000.0
                    + LOW_POSE_SETTLE_MARGIN_S
                )

                if self.execute:
                    actual = {
                        servo_id: self.arm.read_position(servo_id)
                        for servo_id in (1, 2, 6)
                    }
                    expected = {1: LOW_ID1, 2: LOW_ID2, 6: LOW_ID6}
                    mismatched = [
                        f"ID{servo_id}={actual[servo_id]}"
                        for servo_id in expected
                        if actual[servo_id] is not None
                        and abs(actual[servo_id] - expected[servo_id])
                        > ARM_POSITION_TOLERANCE
                    ]
                    if mismatched:
                        raise RuntimeError(
                            "低位回读未到位: " + ", ".join(mismatched)
                        )
                    available = {
                        servo_id: position
                        for servo_id, position in actual.items()
                        if position is not None
                    }
                    if available:
                        readback = " ".join(
                            f"ID{servo_id}={available.get(servo_id, 'ERR')}"
                            for servo_id in (1, 2, 6)
                        )
                    else:
                        readback = "feedback=OPTIONAL (no servo reply)"
                else:
                    readback = "模拟模式"

                self.target_latched = False
                self.latched_color = None
                self.task_ready = True
                self.last_error = ""
                self.last_action = f"{field} 任务开始，低位已到位 {readback}"
                print(
                    f"BALL APP TASK READY field={field} low_pose "
                    f"ID1={LOW_ID1} ID2={LOW_ID2} ID6={LOW_ID6} "
                    f"ID4={LOW_SPLITTER} ID5={LOW_CATCHER} "
                    f"ID7={GRIPPER_CLOSED} readback={readback}",
                    flush=True,
                )
            except Exception as exc:
                self.task_ready = False
                self.last_error = str(exc)
                self.last_action = f"低位准备失败，任务未启动: {exc}"
                print(f"BALL APP PREP FAILED {exc}", flush=True)
            finally:
                self.preparing = False

    def mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        if y < 60 and x < 175:
            self._start_field(FIELD_RED)
        elif y < 60 and x < 345:
            self._start_field(FIELD_BLUE)

    def run(self):
        cv2.namedWindow(APP_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.setMouseCallback(APP_TITLE, self.mouse)
        while self.running:
            ok, frame = self.camera.read()
            if not ok:
                self.last_error = "摄像头取帧失败"
                time.sleep(0.05)
                continue
            try:
                detections = self._detect(frame)
            except Exception as exc:
                detections = []
                self.last_error = f"检测失败: {exc}"
            self.last_detections = detections
            view, allowed = self._draw(frame, detections)
            self._trigger_if_needed(allowed)
            cv2.imshow(APP_TITLE, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                self._start_field(FIELD_RED)
            elif key in (ord("b"), ord("B")):
                self._start_field(FIELD_BLUE)
        self.close()

    def close(self):
        self.running = False
        if (
            self.action_thread
            and self.action_thread.is_alive()
            and threading.current_thread() is not self.action_thread
        ):
            self.action_thread.join(ACTION_JOIN_TIMEOUT_S)
        if self.camera is not None:
            self.camera.release()
        if self.zp is not None:
            self.zp.close()
        if self.arm is not None:
            self.arm.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--arm-uart", default=DEFAULT_ARM_UART)
    parser.add_argument("--zp-uart", default=DEFAULT_ZP_UART)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="只看画面和识别，不驱动 ID1/2/4/5/6/7",
    )
    parser.add_argument(
        "--start-field",
        choices=("red", "blue"),
        help="启动后立即准备并开始指定场地；不指定时等待按 R/B",
    )
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
    # Refuse accidental coexistence with the integrated owner.
    owner = subprocess.run(
        ["systemctl", "--user", "is-active", "robocup-chassis-arm.service"],
        capture_output=True, text=True,
    ).stdout.strip()
    if owner == "active":
        raise SystemExit("自动联动服务正在运行，请先执行 systemctl --user stop robocup-chassis-arm.service")
    try:
        app = BallTaskApp(
            args.camera,
            args.arm_uart,
            args.zp_uart,
            execute=not args.no_execute,
        )

        def request_shutdown(_signum, _frame):
            # Let run() call close(), which waits for an in-flight pulse
            # before releasing the ZP bus.
            app.running = False

        import signal
        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        if args.start_field:
            app._start_field(FIELD_RED if args.start_field == "red" else FIELD_BLUE)
        app.run()
    except Exception as exc:
        raise SystemExit(f"App 启动失败: {exc}")


if __name__ == "__main__":
    main()
