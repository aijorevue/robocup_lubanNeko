"""ROS 2 image-topic wrapper for :class:`ABCDDetector`."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .detector import ABCDDetector


def image_message_to_bgr(message):
    """Convert common ROS image encodings without cv_bridge or fixed paths."""

    channels = 1 if message.encoding in {"mono8", "8UC1"} else 3
    data = np.frombuffer(message.data, dtype=np.uint8)
    row_width = int(message.step)
    rows = data.reshape(int(message.height), row_width)
    rows = rows[:, : int(message.width) * channels]
    if channels == 1:
        return np.repeat(rows.reshape(message.height, message.width, 1), 3, axis=2)
    image = rows.reshape(message.height, message.width, channels)
    if message.encoding in {"rgb8", "rgba8"}:
        return image[:, :, :3][:, :, ::-1].copy()
    return image[:, :, :3].copy()


def _json_detection(detection):
    result = dict(detection)
    for key in ("box", "points"):
        if key in result and hasattr(result[key], "tolist"):
            result[key] = result[key].tolist()
    return result


def _default_config_path():
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("abcd_detector")) / "config" / "letter_detector.yaml"
    except (ImportError, LookupError):
        return Path(__file__).resolve().parents[1] / "config" / "letter_detector.yaml"


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class LetterDetectorNode(Node):
        def __init__(self):
            super().__init__("abcd_detector")
            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("detections_topic", "/abcd/detections")
            self.declare_parameter("config_path", str(_default_config_path()))
            config_path = self.get_parameter("config_path").get_parameter_value().string_value
            image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
            detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
            self.detector = ABCDDetector(config_path or None)
            self.publisher = self.create_publisher(String, detections_topic, 10)
            self.subscription = self.create_subscription(
                Image, image_topic, self._on_image, 10
            )

        def _on_image(self, message):
            try:
                image = image_message_to_bgr(message)
                detections = self.detector.detect(image)
                payload = String()
                payload.data = json.dumps(
                    [_json_detection(item) for item in detections],
                    ensure_ascii=True,
                )
                self.publisher.publish(payload)
            except (ValueError, cv2.error) as exc:
                self.get_logger().warning(f"letter detection failed: {exc}")

    rclpy.init(args=args)
    node = LetterDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
