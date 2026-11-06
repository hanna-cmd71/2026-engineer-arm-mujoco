"""Publish a base-frame TCP target using the current JGarm orientation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from jgarm_control.kinematics import Kinematics
from jgarm_control.types import JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    return parser.parse_args()


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.copysign(
        np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
        / 2.0,
        rotation[2, 1] - rotation[1, 2],
    )
    y = np.copysign(
        np.sqrt(max(0.0, 1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]))
        / 2.0,
        rotation[0, 2] - rotation[2, 0],
    )
    z = np.copysign(
        np.sqrt(max(0.0, 1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]))
        / 2.0,
        rotation[1, 0] - rotation[0, 1],
    )
    return float(x), float(y), float(z), float(w)


def main() -> None:
    args = parse_args()
    position = np.array([args.x, args.y, args.z])
    urdf_path = (
        Path(get_package_share_directory("jgarm_description"))
        / "urdf"
        / "JGarm.urdf"
    )
    kinematics = Kinematics(urdf_path)

    rclpy.init()
    node = rclpy.create_node("jgarm_pose_target_publisher")
    state_holder: dict[str, JointState] = {}

    def state_callback(message: JointState) -> None:
        state_holder["state"] = message

    state_qos = QoSProfile(
        depth=10, reliability=ReliabilityPolicy.RELIABLE
    )
    node.create_subscription(
        JointState, "/joint_states", state_callback, state_qos
    )
    while rclpy.ok() and "state" not in state_holder:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "state" not in state_holder:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("No /joint_states received")

    state = state_holder["state"]
    state_index = {name: index for index, name in enumerate(state.name)}
    q = np.array(
        [state.position[state_index[name]] for name in JOINT_NAMES]
    )
    target = kinematics.forward(q)
    target[:3, 3] = position
    qx, qy, qz, qw = rotation_to_quaternion(target[:3, :3])

    publisher = node.create_publisher(PoseStamped, "/arm/target/pose", 10)
    deadline = time.monotonic() + 5.0
    while publisher.get_subscription_count() == 0 and rclpy.ok():
        if time.monotonic() >= deadline:
            raise RuntimeError("No subscriber found on /arm/target/pose")
        rclpy.spin_once(node, timeout_sec=0.1)

    message = PoseStamped()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "base_link"
    message.pose.position.x = args.x
    message.pose.position.y = args.y
    message.pose.position.z = args.z
    message.pose.orientation.x = qx
    message.pose.orientation.y = qy
    message.pose.orientation.z = qz
    message.pose.orientation.w = qw
    publisher.publish(message)
    time.sleep(0.2)

    print(f"Published TCP target: {position.tolist()}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
