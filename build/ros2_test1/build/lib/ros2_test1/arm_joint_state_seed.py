#!/usr/bin/env python3
"""Publish a steady default arm pose so RViz has valid joint states immediately."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .arm_kinematics import JOINT_NAMES, READY_ID1_TICK, READY_ID2_TICK, joint_positions


def arm_joint_positions(id1_tick=READY_ID1_TICK, id2_tick=READY_ID2_TICK, id4_tick=500):
    return joint_positions(id1_tick, id2_tick, id4_tick)


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
