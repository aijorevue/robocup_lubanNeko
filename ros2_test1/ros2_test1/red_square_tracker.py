#!/usr/bin/env python3
"""Continuously track the largest red square with servo ID2 and ID6."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import threading
import time

import cv2

from .arm_kinematics import ID2_SAFE_LIMITS
from .target_vision import (
    CENTERING_DISTANCE_SCALE_LIMITS,
    CENTERING_REFERENCE_DISTANCE_CM,
    GRASP_MAX_DISTANCE_CM,
    GRASP_MIN_DISTANCE_CM,
    ID6_SAFE_LIMITS,
    AbsoluteServoBridge,
    TargetDetector,
    WindowCloseWatcher,
    detection_process_worker,
    ensure_display_env,
    open_camera,
    set_pipewire,
    stop_old_camera_viewers,
)


WINDOW_NAME = "Red Square Tracking Test"
TRACK_CENTER_DEADBAND_PX = 0.0
TRACK_MIN_STEP_TICKS = 1


class RedSquareTracker:
    def __init__(
        self,
        servo_bridge,
        execute,
        id2_pixel_gain=0.38,
        id6_pixel_gain=0.35,
        id2_max_step_ticks=100,
        id6_max_step_ticks=100,
        command_interval_s=1.20,
    ):
        self.servo_bridge = servo_bridge
        self.execute = bool(execute)
        self.id2_pixel_gain = float(id2_pixel_gain)
        self.id6_pixel_gain = float(id6_pixel_gain)
        self.id2_max_step_ticks = max(1, int(id2_max_step_ticks))
        self.id6_max_step_ticks = max(1, int(id6_max_step_ticks))
        self.command_interval_s = max(0.5, float(command_interval_s))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.target = None
        self.frame_shape = None
        self.target_time = 0.0
        self.id2 = None
        self.id6 = None
        self.delta_id2 = 0
        self.delta_id6 = 0
        self.error_x = 0.0
        self.error_y = 0.0
        self.distance_cm = None
        self.distance_scale = 1.0
        self.status = "waiting for servo position"
        self.last_command_time = 0.0
        self.feedback_due = 0.0
        self.feedback_pending = False
        self.one_shot_sent = False
        self.one_shot_complete = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _clamp(value, limits):
        return max(limits[0], min(limits[1], int(round(value))))

    def update_target(self, target, frame_shape):
        copied = None
        if target is not None:
            copied = dict(target)
            if "center" in copied:
                copied["center"] = tuple(copied["center"])
        with self.lock:
            self.target = copied
            self.frame_shape = tuple(frame_shape)
            self.target_time = time.monotonic()

    def snapshot(self):
        with self.lock:
            return {
                "execute": self.execute,
                "id2": self.id2,
                "id6": self.id6,
                "delta_id2": self.delta_id2,
                "delta_id6": self.delta_id6,
                "error_x": self.error_x,
                "error_y": self.error_y,
                "distance_cm": self.distance_cm,
                "distance_scale": self.distance_scale,
                "status": self.status,
            }

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def rearm(self):
        with self.lock:
            if self.feedback_pending:
                return
            self.one_shot_sent = False
            self.one_shot_complete = False
            self.status = "one-shot rearmed; waiting for red square"

    def _set_status(self, status, **values):
        with self.lock:
            self.status = status
            for key, value in values.items():
                setattr(self, key, value)

    def _sync_positions(self):
        feedback = self.servo_bridge.query_positions(timeout_s=0.90)
        if feedback is None:
            self._set_status(f"position read failed: {self.servo_bridge.status}")
            return False
        id6 = feedback[2] if len(feedback) > 2 else None
        if id6 is None:
            self._set_status("ID6 position unavailable; tracking stopped")
            return False
        self._set_status(
            "servo positions synchronized",
            id2=int(feedback[1]),
            id6=int(id6),
        )
        return True

    def _calculate_delta(self, target, frame_shape):
        height, width = frame_shape[:2]
        center_x, center_y = target["center"]
        error_x = center_x - width / 2.0
        error_y = center_y - height / 2.0
        distance_cm = target.get("distance_cm")
        if distance_cm is None:
            return None, "target distance unavailable"
        if not GRASP_MIN_DISTANCE_CM <= distance_cm <= GRASP_MAX_DISTANCE_CM:
            return None, f"distance {distance_cm:.1f}cm outside 10-30cm"
        if target.get("area_percent", 0.0) < 0.4:
            return None, "red square too small"
        if not target.get("fully_visible", True):
            return None, "red square touches image edge"

        distance_scale = max(
            CENTERING_DISTANCE_SCALE_LIMITS[0],
            min(
                CENTERING_DISTANCE_SCALE_LIMITS[1],
                distance_cm / CENTERING_REFERENCE_DISTANCE_CM,
            ),
        )
        delta_id2 = 0
        if abs(error_y) > TRACK_CENTER_DEADBAND_PX:
            delta_id2 = int(round(-error_y * self.id2_pixel_gain * distance_scale))
            delta_id2 = max(
                -self.id2_max_step_ticks,
                min(self.id2_max_step_ticks, delta_id2),
            )
            if abs(delta_id2) < TRACK_MIN_STEP_TICKS:
                delta_id2 = TRACK_MIN_STEP_TICKS if error_y < 0 else -TRACK_MIN_STEP_TICKS

        delta_id6 = 0
        if abs(error_x) > TRACK_CENTER_DEADBAND_PX:
            delta_id6 = int(round(-error_x * self.id6_pixel_gain * distance_scale))
            delta_id6 = max(
                -self.id6_max_step_ticks,
                min(self.id6_max_step_ticks, delta_id6),
            )
            if abs(delta_id6) < TRACK_MIN_STEP_TICKS:
                delta_id6 = TRACK_MIN_STEP_TICKS if error_x < 0 else -TRACK_MIN_STEP_TICKS

        return {
            "error_x": error_x,
            "error_y": error_y,
            "distance_cm": distance_cm,
            "distance_scale": distance_scale,
            "delta_id2": delta_id2,
            "delta_id6": delta_id6,
        }, None

    def _run(self):
        synchronized = False
        while not self.stop_event.wait(0.05):
            now = time.monotonic()
            if not synchronized:
                synchronized = self._sync_positions()
                if not synchronized:
                    self.stop_event.wait(0.8)
                continue

            if self.feedback_pending:
                if now < self.feedback_due:
                    continue
                if not self._sync_positions():
                    synchronized = False
                    self.feedback_pending = False
                    continue
                self.feedback_pending = False

            with self.lock:
                target = None if self.target is None else dict(self.target)
                frame_shape = self.frame_shape
                target_age = now - self.target_time
                current_id2 = self.id2
                current_id6 = self.id6

            if target is None or frame_shape is None or target_age > 0.7:
                self._set_status("searching largest red square", delta_id2=0, delta_id6=0)
                continue

            correction, error = self._calculate_delta(target, frame_shape)
            if correction is None:
                self._set_status(error, delta_id2=0, delta_id6=0)
                continue

            self._set_status("calculating tracking correction", **correction)
            if self.one_shot_complete:
                self._set_status(
                    "ONE-SHOT COMPLETE - press R to retry",
                    **correction,
                )
                continue
            delta_id2 = correction["delta_id2"]
            delta_id6 = correction["delta_id6"]
            if delta_id2 == 0 and delta_id6 == 0:
                self._set_status("CENTERED - holding position", **correction)
                continue
            if self.one_shot_sent:
                self.one_shot_complete = True
                self._set_status(
                    "ONE-SHOT COMPLETE - press R to retry",
                    **correction,
                )
                continue
            if now - self.last_command_time < self.command_interval_s:
                self._set_status("waiting for next tracking step", **correction)
                continue

            target_id2 = self._clamp(current_id2 + delta_id2, ID2_SAFE_LIMITS)
            target_id6 = self._clamp(current_id6 + delta_id6, ID6_SAFE_LIMITS)
            send_id2 = target_id2 != current_id2
            send_id6 = target_id6 != current_id6
            if not send_id2 and not send_id6:
                self._set_status("servo limit reached", **correction)
                continue

            if not self.execute:
                self._set_status(
                    f"PREVIEW ONE-SHOT ID2={target_id2} ID6={target_id6}",
                    **correction,
                )
                self.last_command_time = now
                self.one_shot_sent = True
                self.one_shot_complete = True
                continue

            status = self.servo_bridge.send_targets(
                id2=target_id2 if send_id2 else None,
                id6=target_id6 if send_id6 else None,
            )
            print(
                f"TRACK STEP dx={correction['error_x']:.0f} "
                f"dy={correction['error_y']:.0f} "
                f"distance={correction['distance_cm']:.1f}cm "
                f"scale={correction['distance_scale']:.2f} "
                f"dID2={delta_id2:+d} dID6={delta_id6:+d} "
                f"target2={target_id2} target6={target_id6}",
                flush=True,
            )
            self.last_command_time = time.monotonic()
            if not self.servo_bridge.last_command_ok:
                self._set_status(f"command failed: {status}", **correction)
                synchronized = False
                continue
            self.one_shot_sent = True
            self._set_status(
                f"ONE-SHOT MOVING ID2={target_id2} ID6={target_id6}",
                id2=target_id2,
                id6=target_id6,
                **correction,
            )
            self.feedback_pending = True
            self.feedback_due = self.last_command_time + self.command_interval_s


def largest_red_square(detections):
    candidates = [
        detection
        for detection in detections
        if detection.get("color") == "red"
        and detection.get("kind") == "square"
        and detection.get("source") != "qr"
    ]
    return max(
        candidates,
        key=lambda detection: detection.get("projected_area", 0.0),
        default=None,
    )


def draw_tracking_status(frame, target, state, fps):
    output = frame
    height, width = output.shape[:2]
    center_x = width // 2
    center_y = height // 2
    deadband = int(TRACK_CENTER_DEADBAND_PX)
    cv2.rectangle(
        output,
        (center_x - deadband, center_y - deadband),
        (center_x + deadband, center_y + deadband),
        (0, 255, 0),
        2,
    )
    if target is not None:
        target_x, target_y = target["center"]
        cv2.line(
            output,
            (center_x, center_y),
            (int(target_x), int(target_y)),
            (0, 255, 255),
            2,
        )
    distance = state["distance_cm"]
    distance_text = "--" if distance is None else f"{distance:.1f}cm"
    lines = [
        f"ONE-SHOT TRACK {'REAL' if state['execute'] else 'PREVIEW'}  FPS {fps:.1f}  R=retry",
        f"dx={state['error_x']:.0f}px dy={state['error_y']:.0f}px distance={distance_text} scale={state['distance_scale']:.2f}",
        f"ID2={state['id2']} dID2={state['delta_id2']:+d}  ID6={state['id6']} dID6={state['delta_id6']:+d}",
        state["status"],
    ]
    for index, text in enumerate(lines):
        y = height - 82 + index * 21
        cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("CAMERA_DEVICE", "auto"))
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detect-every-n-frames", type=int, default=2)
    parser.add_argument("--detection-scale", type=float, default=0.75)
    parser.add_argument("--servo-uart", default=os.environ.get("SERVO_UART", "/dev/ttyS0"))
    parser.add_argument("--servo-baud", type=int, default=int(os.environ.get("SERVO_BAUD", "115200")))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--window-width", type=int, default=800)
    parser.add_argument("--window-height", type=int, default=600)
    parser.add_argument("--window-x", type=int, default=80)
    parser.add_argument("--window-y", type=int, default=60)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ensure_display_env()
    stop_old_camera_viewers()
    set_pipewire(False)
    detector = TargetDetector()
    bridge = AbsoluteServoBridge(
        args.servo_uart,
        args.servo_baud,
        enabled=True,
        write_enabled=args.execute,
    )
    tracker = RedSquareTracker(bridge, args.execute)
    cap = open_camera(args.device, args.width, args.height, args.fps)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)
    cv2.moveWindow(WINDOW_NAME, args.window_x, args.window_y)
    close_watcher = WindowCloseWatcher(WINDOW_NAME)
    close_watcher.start()
    detection_executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    )
    detection_future = None
    detections = []
    target = None
    frame_index = 0
    fps_started = time.monotonic()
    fps_frames = 0
    display_fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if detection_future is not None and detection_future.done():
                try:
                    detections, _ = detection_future.result()
                    target = largest_red_square(detections)
                    tracker.update_target(target, frame.shape)
                except Exception as exc:
                    print(f"tracking detection failed: {exc}", flush=True)
                detection_future = None
            if frame_index % max(1, args.detect_every_n_frames) == 0 and detection_future is None:
                detection_future = detection_executor.submit(
                    detection_process_worker,
                    frame.copy(),
                    max(0.4, min(1.0, args.detection_scale)),
                )
            frame_index += 1
            output, _ = detector.draw(frame, [] if target is None else [target])
            fps_frames += 1
            elapsed = time.monotonic() - fps_started
            if elapsed >= 1.0:
                display_fps = fps_frames / elapsed
                fps_started = time.monotonic()
                fps_frames = 0
            output = draw_tracking_status(output, target, tracker.snapshot(), display_fps)
            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("r"), ord("R")):
                tracker.rearm()
            if key in (27, ord("q")) or close_watcher.closed.is_set():
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
                    break
            except cv2.error:
                break
    finally:
        tracker.stop()
        bridge.close()
        detection_executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        close_watcher.stop()
        cv2.destroyAllWindows()
        set_pipewire(True)


if __name__ == "__main__":
    main()
