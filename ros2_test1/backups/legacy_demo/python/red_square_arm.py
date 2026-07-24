#!/usr/bin/env python3
"""Bridge red-square detection, RViz preview, and RCT6 servo commands.

Motion is preview-only by default.  Set execute:=true only after the image
directions and limits in the calibration JSON have been measured.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .target_vision import TargetDetector, open_camera

try:
    import serial
except ImportError:
    serial = None


JOINT_NAMES = [
    "servo_blue",
    "blue_chain_passive",
    "servo_orange",
    "orange_chain_passive",
    "extension_level",
    "gripper_left",
    "gripper_right",
]


def clamp(value, lower, upper):
    return max(lower, min(upper, int(round(value))))


def arm_joint_positions(id1_tick, id2_tick, id4_tick):
    orange_degrees = 180.0 + (id1_tick - 470) * ((270.0 - 180.0) / (830 - 470))
    blue_degrees = 90.0 + (id2_tick - 510) * ((180.0 - 90.0) / (122 - 510))
    orange = orange_degrees * 3.141592653589793 / 180.0
    blue = blue_degrees * 3.141592653589793 / 180.0
    gripper_ratio = max(0.0, min(1.0, (id4_tick - 500) / 400.0))
    return [
        blue,
        orange - blue,
        orange,
        blue - orange,
        -orange,
        -0.05 - 0.75 * gripper_ratio,
        0.05 + 0.75 * gripper_ratio,
    ]


class ServoBridge:
    def __init__(self, device, baudrate):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self.serial = serial.Serial(device, baudrate, timeout=0.25)

    def move_absolute(self, id1_tick, id2_tick, id4_tick):
        commands = (
            f"id1={id1_tick}",
            f"id2={id2_tick}",
            f"zp4:{id4_tick}",
        )
        responses = []
        for command in commands:
            self.serial.reset_input_buffer()
            self.serial.write((command + "\n").encode("ascii"))
            self.serial.flush()
            responses.append(self.serial.readline().decode("utf-8", "replace").strip())
        return responses

    def close(self):
        self.serial.close()


class RedSquareArm(Node):
    def __init__(self):
        super().__init__("red_square_arm")
        self.declare_parameter("camera_device", "auto")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("execute", False)
        self.declare_parameter("serial_device", "/dev/ttyS0")
        self.declare_parameter("serial_baud", 115200)
        self.declare_parameter("calibration_file", "")

        self.config = self.load_config(self.get_parameter("calibration_file").value)
        self.execute = self.get_parameter("execute").value
        self.detector = TargetDetector()
        self.joint_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.status_publisher = self.create_publisher(String, "/arm/grasp_status", 10)
        self.target_publisher = self.create_publisher(String, "/arm/servo_targets", 10)
        self.id1 = self.config["initial"]["id1"]
        self.id2 = self.config["initial"]["id2"]
        self.id4 = self.config["initial"]["id4"]
        self.stable_frames = 0
        self.last_command_time = 0.0
        self.state = "SEARCH"
        self.bridge = None
        self.capture = open_camera(
            self.get_parameter("camera_device").value,
            self.get_parameter("camera_width").value,
            self.get_parameter("camera_height").value,
            30,
        )

        if self.execute:
            self.bridge = ServoBridge(
                self.get_parameter("serial_device").value,
                self.get_parameter("serial_baud").value,
            )
            self.send_targets("initial pose")
        self.timer = self.create_timer(0.08, self.update)
        self.publish_targets("waiting for red square")

    @staticmethod
    def load_config(config_path):
        defaults = {
            "initial": {"id1": 480, "id2": 466, "id4": 900},
            "limits": {
                "id1": [0, 1000], "id2": [0, 1000], "id4": [500, 2500],
            },
            "centering": {
                "stable_frames": 5, "deadband_pixels": 20,
                "id1_tick_per_pixel_y": 0.0, "id2_tick_per_pixel_x": 0.0,
                "command_interval_s": 0.35,
            },
            "grasp": {"enabled": False, "id1": 480, "id2": 466, "id4": 500},
        }
        if not config_path:
            return defaults
        with Path(config_path).expanduser().open(encoding="utf-8") as handle:
            supplied = json.load(handle)
        for key, value in supplied.items():
            if isinstance(value, dict) and isinstance(defaults.get(key), dict):
                defaults[key].update(value)
            else:
                defaults[key] = value
        return defaults

    def update(self):
        ok, frame = self.capture.read()
        if not ok:
            self.publish_targets("camera frame unavailable")
            return
        detections = self.detector.detect(frame)
        targets = [
            item for item in detections
            if item.get("color") == "red" and item.get("kind") == "square"
        ]
        if not targets:
            self.stable_frames = 0
            self.state = "SEARCH"
            self.publish_targets("searching for red square")
            return

        target = max(targets, key=lambda item: item["bbox"][2] * item["bbox"][3])
        self.stable_frames += 1
        centre_x, centre_y = target["center"]
        image_x = frame.shape[1] / 2.0
        image_y = frame.shape[0] / 2.0
        error_x = centre_x - image_x
        error_y = centre_y - image_y
        center = self.config["centering"]
        deadband = center["deadband_pixels"]

        if self.stable_frames < center["stable_frames"]:
            self.state = "CONFIRMING"
            self.publish_targets(f"confirming red square {self.stable_frames}/{center['stable_frames']}")
            return
        if abs(error_x) > deadband or abs(error_y) > deadband:
            self.state = "CENTERING"
            self.center_target(error_x, error_y)
            return

        self.state = "PREVIEW_READY"
        grasp = self.config["grasp"]
        if grasp.get("enabled", False):
            message = "square centred; calibrated grasp target ready"
        else:
            message = "square centred; configure calibrated grasp before closing claw"
        self.publish_targets(message, target, error_x, error_y)

    def center_target(self, error_x, error_y):
        center = self.config["centering"]
        next_id1 = clamp(
            self.id1 - error_y * center["id1_tick_per_pixel_y"],
            *self.config["limits"]["id1"],
        )
        next_id2 = clamp(
            self.id2 - error_x * center["id2_tick_per_pixel_x"],
            *self.config["limits"]["id2"],
        )
        self.id1, self.id2 = next_id1, next_id2
        now = time.monotonic()
        if self.execute and now - self.last_command_time >= center["command_interval_s"]:
            self.send_targets("visual centering")
        self.publish_targets(f"centering dx={error_x:.0f} dy={error_y:.0f}")

    def send_targets(self, reason):
        self.bridge.move_absolute(self.id1, self.id2, self.id4)
        self.last_command_time = time.monotonic()
        self.get_logger().info(f"{reason}: id1={self.id1} id2={self.id2} id4={self.id4}")

    def publish_targets(self, detail, target=None, error_x=None, error_y=None):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = "base_link"
        joint_state.name = JOINT_NAMES
        joint_state.position = arm_joint_positions(self.id1, self.id2, self.id4)
        self.joint_publisher.publish(joint_state)
        payload = {
            "state": self.state, "detail": detail, "id1": self.id1,
            "id2": self.id2, "id4": self.id4, "execute": self.execute,
        }
        if target is not None:
            payload.update({"center": target["center"], "bbox": target["bbox"],
                            "error_x": round(error_x, 1), "error_y": round(error_y, 1)})
        message = String(data=json.dumps(payload, ensure_ascii=False))
        self.status_publisher.publish(message)
        self.target_publisher.publish(message)

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
        if self.bridge is not None:
            self.bridge.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedSquareArm()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
