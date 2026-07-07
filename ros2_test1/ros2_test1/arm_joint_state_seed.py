#!/usr/bin/env python3
"""Publish a steady default arm pose so RViz has valid joint states immediately."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .arm_kinematics import (
    BASE_YAW_HOME_TICK,
    GRIPPER_CLOSED_TICK,
    HOME_ID1_TICK,
    HOME_ID2_TICK,
    JOINT_NAMES,
    joint_positions,
)


def arm_joint_positions(
    id1_tick=HOME_ID1_TICK,
    id2_tick=HOME_ID2_TICK,
    id4_tick=GRIPPER_CLOSED_TICK,
    id6_tick=BASE_YAW_HOME_TICK,
):
    return joint_positions(id1_tick, id2_tick, id4_tick, id6_tick)


class ArmJointStateSeed(Node):
    def __init__(self):
        super().__init__("arm_joint_state_seed")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.positions = arm_joint_positions()
        self.timer = self.create_timer(0.1, self.publish_pose)

    def publish_pose(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = self.positions
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmJointStateSeed()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
