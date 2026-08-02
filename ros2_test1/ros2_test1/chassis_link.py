"""Non-blocking H7/RK serial link and its small wire protocol."""

from __future__ import annotations

import glob
import os
import select
import time

try:
    import termios
except ImportError:  # Enables protocol unit tests on Windows.
    termios = None

from .field_mode import FieldMode, parse_field


class ChassisArmLink:
    VALID_TASKS = {"DISC_CATCH", "PLATFORM_PICK", "COLUMN_CATCH"}

    def __init__(self, enabled, device, baudrate, no_target_timeout_s):
        self.enabled = bool(enabled)
        self.device_arg = device
        self.baudrate = int(baudrate)
        self.no_target_timeout_s = max(0.5, float(no_target_timeout_s))
        self.fd = None
        self.device = None
        self.active_task = None
        self.field_mode = FieldMode.RED
        self.station_started = 0.0
        self.last_target_seen = 0.0
        self.line_buffer = ""
        self.status = "chassis link disabled"
        self.pending_starts = []
        self.pending_stops = []
        self.reset_pending = False
        self.reset_in_progress = False
        self.last_reset_completed = 0.0
        self.next_retry = 0.0
        self.last_ready_sent = 0.0
        self.ready_to_run = False
        self.last_completed_task = None
        self.last_completed_reason = None
        self.last_backpressure_log = 0.0
        self.last_open_failure_log = 0.0
        if self.enabled:
            self._open()

    def _candidate_devices(self):
        if self.device_arg and self.device_arg != "auto":
            return [self.device_arg]
        return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))

    def _open(self):
        now = time.monotonic()
        if now < self.next_retry:
            return False
        self.next_retry = now + 1.0
        if termios is None:
            self.status = "CHASSIS LINK requires a POSIX serial device"
            return False
        for device in self._candidate_devices():
            fd = None
            try:
                fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                attrs = termios.tcgetattr(fd)
                attrs[0] = 0
                attrs[1] = 0
                attrs[2] = (
                    (attrs[2] & ~(termios.CSIZE | termios.PARENB | termios.CSTOPB))
                    | termios.CS8
                    | termios.CLOCAL
                    | termios.CREAD
                )
                if hasattr(termios, "CRTSCTS"):
                    attrs[2] &= ~termios.CRTSCTS
                attrs[3] = 0
                baud_flag = getattr(termios, f"B{self.baudrate}", termios.B115200)
                attrs[4] = baud_flag
                attrs[5] = baud_flag
                attrs[6][termios.VMIN] = 0
                attrs[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
                self.fd = fd
                self.device = device
                self.status = f"CHASSIS LINK open {device}"
                print(self.status, flush=True)
                self._announce_ready()
                return True
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                if isinstance(exc, FileNotFoundError):
                    self.status = f"CHASSIS LINK waiting for {device}"
                else:
                    self.status = f"CHASSIS LINK open failed {device}: {exc}"
                if now - self.last_open_failure_log >= 10.0:
                    print(self.status, flush=True)
                    self.last_open_failure_log = now
        if self.device_arg == "auto":
            self.status = "CHASSIS LINK waiting for /dev/ttyACM* or /dev/ttyUSB*"
        return False

    def _close_fd(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None
        self.device = None

    def _write_payload(self, payload):
        """Drain partial non-blocking writes instead of silently truncating a line."""

        offset = 0
        deadline = time.monotonic() + 0.2
        while offset < len(payload):
            try:
                written = os.write(self.fd, payload[offset:])
                if written <= 0:
                    return False
                offset += written
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                _, writable, _ = select.select([], [self.fd], [], min(0.02, remaining))
                if not writable:
                    return False
        return True

    def send_line(self, text):
        if not self.enabled:
            return False
        if self.fd is None and not self._open():
            return False
        payload = (text.rstrip() + "\r\n").encode("ascii", "replace")
        try:
            if not self._write_payload(payload):
                raise BlockingIOError
            print(f"CHASSIS TX {text.rstrip()}", flush=True)
            return True
        except BlockingIOError:
            now = time.monotonic()
            self.status = "CHASSIS LINK TX backpressure"
            if now - self.last_backpressure_log >= 10.0:
                print(self.status, flush=True)
                self.last_backpressure_log = now
            return False
        except OSError as exc:
            self.status = f"CHASSIS LINK write failed: {exc}"
            print(self.status, flush=True)
            self._close_fd()
            return False

    @staticmethod
    def _field_from_parts(parts):
        for index, item in enumerate(parts[:-1]):
            if item in {"FIELD", "FIELD="}:
                value = parts[index + 1]
                if value in {"RED", "BLUE"}:
                    return FieldMode(value.lower())
        for item in parts:
            if item in {"RED", "BLUE"}:
                return FieldMode(item.lower())
        return None

    def _set_field_from_wire(self, field_mode, source):
        if field_mode is None:
            return False
        if self.active_task is not None and field_mode != self.field_mode:
            self.send_line(
                f"RK,FIELD,BUSY,{self.field_mode.wire_name},{self.active_task}"
            )
            print(
                f"CHASSIS FIELD change rejected during {self.active_task}: "
                f"{field_mode.wire_name}",
                flush=True,
            )
            return False
        changed = field_mode != self.field_mode
        self.field_mode = field_mode
        if changed:
            self.status = f"CHASSIS FIELD {field_mode.wire_name} from {source}"
            print(self.status, flush=True)
        self.send_line(f"RK,FIELD,ACK,{self.field_mode.wire_name}")
        return True

    def _force_field_from_reset(self, field_mode):
        if field_mode is None:
            return
        changed = field_mode != self.field_mode
        self.field_mode = field_mode
        if changed:
            self.status = f"CHASSIS FIELD {field_mode.wire_name} from ARM,SYNC,RESET"
            print(self.status, flush=True)

    def _announce_ready(self):
        if (
            not self.ready_to_run
            or self.active_task is not None
            or self.reset_pending
            or self.reset_in_progress
        ):
            return False
        self.last_ready_sent = time.monotonic()
        return self.send_line(
            f"RK,ARM,READY,FIELD,{self.field_mode.wire_name}"
        )

    def set_ready(self, ready):
        became_ready = bool(ready) and not self.ready_to_run
        self.ready_to_run = bool(ready)
        if became_ready:
            self._announce_ready()

    def _handle_line(self, line):
        normalized = line.strip().upper()
        if not normalized:
            return
        print(f"CHASSIS RX {line.strip()}", flush=True)
        parts = [part.strip() for part in normalized.split(",")]

        if parts[0] == "FIELD":
            requested = self._field_from_parts(parts[1:])
            if requested is None:
                self.send_line("RK,FIELD,ERR,UNKNOWN_FIELD")
            else:
                self._set_field_from_wire(requested, "FIELD")
            return

        if parts[:2] == ["ARM", "SYNC"]:
            requested = self._field_from_parts(parts[2:])
            if "RESET" in parts[2:]:
                self._force_field_from_reset(requested)
                now = time.monotonic()
                if (
                    not self.reset_pending
                    and not self.reset_in_progress
                    and self.active_task is None
                    and now - self.last_reset_completed < 2.0
                ):
                    self.send_line(
                        f"RK,ARM,RESET,DONE,FIELD,{self.field_mode.wire_name}"
                    )
                    return
                if not self.reset_pending and not self.reset_in_progress:
                    self.reset_pending = True
                    self.pending_starts.clear()
                    self.pending_stops.clear()
                    self.status = (
                        f"CHASSIS reset requested field={self.field_mode.wire_name}"
                    )
                    print(self.status, flush=True)
                self.send_line(
                    f"RK,ARM,RESET,ACK,FIELD,{self.field_mode.wire_name}"
                )
                return
            if requested is not None:
                self._set_field_from_wire(requested, "ARM,SYNC")
            if self.active_task is not None:
                self.send_line(
                    f"RK,ARM,{self.active_task},SYNC_WAIT,FIELD,"
                    f"{self.field_mode.wire_name}"
                )
            else:
                self._announce_ready()
            return

        if len(parts) >= 3 and parts[0] == "ARM" and parts[2] == "STATUS":
            task = parts[1]
            if self.active_task == task:
                self.send_line(
                    f"RK,ARM,{task},ACK,FIELD,{self.field_mode.wire_name}"
                )
            elif self.last_completed_task == task:
                self.send_line(
                    f"RK,ARM,{task},DONE,{self.last_completed_reason},"
                    f"FIELD,{self.field_mode.wire_name}"
                )
            else:
                self.send_line(f"RK,ARM,{task},IDLE")
            return

        if len(parts) >= 3 and parts[0] == "ARM" and parts[2] == "START":
            task = parts[1]
            if task not in self.VALID_TASKS:
                self.send_line(f"RK,ARM,{task},ERR,UNKNOWN_TASK")
                return
            if self.reset_pending or self.reset_in_progress:
                self.send_line(f"RK,ARM,{task},BUSY,RESET")
                return
            requested = self._field_from_parts(parts[3:])
            if self.active_task is None:
                if requested is not None:
                    self._set_field_from_wire(requested, "ARM,START")
                now = time.monotonic()
                self.active_task = task
                self.station_started = now
                self.last_target_seen = now
                self.pending_starts.append(task)
                self.status = (
                    f"CHASSIS station {task} active field={self.field_mode.wire_name}"
                )
                print(self.status, flush=True)
            elif self.active_task != task:
                self.send_line(f"RK,ARM,{task},BUSY,{self.active_task}")
                return
            self.send_line(
                f"RK,ARM,{task},ACK,FIELD,{self.field_mode.wire_name}"
            )
            return

        if len(parts) >= 3 and parts[0] == "ARM" and parts[2] == "STOP":
            task = parts[1]
            if self.active_task != task:
                if self.last_completed_task == task:
                    self.send_line(
                        f"RK,ARM,{task},DONE,{self.last_completed_reason}"
                    )
                    return
                self.send_line(f"RK,ARM,{task},ERR,NO_ACTIVE_TASK")
                return
            if task not in self.pending_stops:
                self.pending_stops.append(task)
            self.status = f"CHASSIS station {task} stop requested"
            print(self.status, flush=True)
            self.send_line(f"RK,ARM,{task},STOP_ACK")
            return

        if normalized in {"RUN", "START"}:
            self._announce_ready()

    def update(self):
        if not self.enabled:
            return []
        if self.fd is None:
            self._open()
            return []
        now = time.monotonic()
        if self.active_task is None and now - self.last_ready_sent >= 1.0:
            self._announce_ready()
            if self.fd is None:
                return []
        try:
            while True:
                readable, _, _ = select.select([self.fd], [], [], 0.0)
                if not readable:
                    break
                data = os.read(self.fd, 256)
                if not data:
                    self.status = "CHASSIS LINK disconnected"
                    print(self.status, flush=True)
                    self._close_fd()
                    return []
                self.line_buffer += data.decode("ascii", "replace")
                while "\n" in self.line_buffer or "\r" in self.line_buffer:
                    line, sep, rest = self.line_buffer.partition("\n")
                    if not sep:
                        line, _, rest = self.line_buffer.partition("\r")
                    self.line_buffer = rest
                    self._handle_line(line)
        except OSError as exc:
            self.status = f"CHASSIS LINK read failed: {exc}"
            print(self.status, flush=True)
            self._close_fd()
        starts = self.pending_starts
        self.pending_starts = []
        return starts

    def consume_stops(self):
        stops = self.pending_stops
        self.pending_stops = []
        return stops

    def consume_reset_request(self):
        if not self.reset_pending or self.reset_in_progress:
            return False
        self.reset_pending = False
        self.reset_in_progress = True
        return True

    def complete_reset(self, success, reason=""):
        if not self.reset_in_progress:
            return False
        self.reset_in_progress = False
        self.pending_starts.clear()
        self.pending_stops.clear()
        if not success:
            self.status = f"CHASSIS reset failed: {reason}"
            self.send_line(
                f"RK,ARM,RESET,ERR,HOME_FAILED,FIELD,{self.field_mode.wire_name}"
            )
            print(self.status, flush=True)
            return False
        self.active_task = None
        self.last_completed_task = None
        self.last_completed_reason = None
        self.station_started = 0.0
        self.last_target_seen = 0.0
        self.last_reset_completed = time.monotonic()
        self.status = f"CHASSIS reset complete: {reason}"
        self.send_line(
            f"RK,ARM,RESET,DONE,FIELD,{self.field_mode.wire_name}"
        )
        print(self.status, flush=True)
        self._announce_ready()
        return True

    def note_target(self, target):
        if self.active_task is not None and target is not None:
            self.last_target_seen = time.monotonic()

    def no_target_timed_out(self, now, searching_for_target):
        if self.active_task is None or not searching_for_target:
            return False
        if self.active_task in {"DISC_CATCH", "COLUMN_CATCH"}:
            return False
        return (now - self.last_target_seen) >= self.no_target_timeout_s

    def finish_active(self, reason):
        if self.active_task is None:
            return False
        task = self.active_task
        self.send_line(
            f"RK,ARM,{task},DONE,{reason},FIELD,{self.field_mode.wire_name}"
        )
        self.status = f"CHASSIS station {task} done: {reason}"
        print(self.status, flush=True)
        self.active_task = None
        self.last_completed_task = task
        self.last_completed_reason = reason
        self.station_started = 0.0
        self.last_target_seen = 0.0
        return True

    def close(self):
        self._close_fd()
