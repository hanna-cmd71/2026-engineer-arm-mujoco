"""Numpad TCP teleoperation using KDL FK and IK."""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from jgarm_control.kinematics import Kinematics
from jgarm_control.types import JOINT_COUNT, JOINT_NAMES


class NumpadTcpTeleop(Node):
    def __init__(self, step_m: float, output_mode: str) -> None:
        super().__init__("jgarm_numpad_tcp_teleop")
        urdf_path = (
            Path(get_package_share_directory("jgarm_description"))
            / "urdf"
            / "JGarm.urdf"
        )
        self.kinematics = Kinematics(urdf_path)
        self.step = float(step_m)
        self.output_mode = output_mode
        self.current_q: np.ndarray | None = None
        self.target_q: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._state_callback,
            state_qos,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            "/arm/target/pose",
            10,
        )
        self.position_publisher = self.create_publisher(
            Float64MultiArray,
            "/arm/target/joint_position",
            10,
        )
        self.velocity_publisher = self.create_publisher(
            Float64MultiArray,
            "/arm/target/joint_velocity",
            10,
        )

    def _state_callback(self, message: JointState) -> None:
        index = {name: i for i, name in enumerate(message.name)}
        if not all(name in index for name in JOINT_NAMES):
            return
        q = np.array(
            [message.position[index[name]] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(q)):
            return
        self.current_q = q
        if self.target_pose is None:
            self.target_pose = self.kinematics.forward(q)
            self.target_q = q.copy()

    def wait_for_state(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.target_pose is not None:
                return
        raise RuntimeError(
            "no /joint_states received on ROS_DOMAIN_ID="
            f"{os.environ.get('ROS_DOMAIN_ID', '0')}"
        )

    def move(self, delta: np.ndarray) -> bool:
        if self.target_pose is None or self.target_q is None:
            return False
        candidate = self.target_pose.copy()
        candidate[:3, 3] += delta
        solution, error, iterations = _differential_ik(
            self.kinematics,
            candidate[:3, 3],
            self.target_q,
        )
        if error > 2e-4:
            self.get_logger().warning(
                "KDL differential IK did not converge: "
                f"{error:.3e} m after {iterations} iterations"
            )
            return False
        self.target_pose = candidate
        self.target_pose[:3, :3] = self.kinematics.forward(
            solution
        )[:3, :3]
        self.target_q = solution
        self._publish_target()
        print(
            "TCP target: "
            f"x={candidate[0, 3]:+.4f} "
            f"y={candidate[1, 3]:+.4f} "
            f"z={candidate[2, 3]:+.4f} "
            f"IK={error:.2e}m/{iterations}it"
        )
        return True

    def reset(self) -> None:
        if self.current_q is None:
            return
        self.target_q = self.current_q.copy()
        self.target_pose = self.kinematics.forward(self.current_q)
        self._publish_target()
        print(
            "Reset to current TCP: "
            f"{self.target_pose[:3, 3].round(4).tolist()}"
        )

    def _publish_target(self) -> None:
        if self.target_q is None or self.target_pose is None:
            return
        velocity = Float64MultiArray()
        velocity.data = [0.0] * JOINT_COUNT
        self.velocity_publisher.publish(velocity)
        if self.output_mode == "joint":
            position = Float64MultiArray()
            position.data = self.target_q.tolist()
            self.position_publisher.publish(position)
            return
        qx, qy, qz, qw = _rotation_to_quaternion(
            self.target_pose[:3, :3]
        )
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.pose.position.x = float(self.target_pose[0, 3])
        message.pose.position.y = float(self.target_pose[1, 3])
        message.pose.position.z = float(self.target_pose[2, 3])
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        self.pose_publisher.publish(message)


def _rotation_to_quaternion(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    def half_sqrt(value: float) -> float:
        return math.sqrt(max(0.0, value)) / 2.0

    trace = float(np.trace(rotation))
    w = half_sqrt(1.0 + trace)
    x = math.copysign(
        half_sqrt(
            1.0
            + rotation[0, 0]
            - rotation[1, 1]
            - rotation[2, 2]
        ),
        rotation[2, 1] - rotation[1, 2],
    )
    y = math.copysign(
        half_sqrt(
            1.0
            - rotation[0, 0]
            + rotation[1, 1]
            - rotation[2, 2]
        ),
        rotation[0, 2] - rotation[2, 0],
    )
    z = math.copysign(
        half_sqrt(
            1.0
            - rotation[0, 0]
            - rotation[1, 1]
            + rotation[2, 2]
        ),
        rotation[1, 0] - rotation[0, 1],
    )
    return float(x), float(y), float(z), float(w)


def _differential_ik(
    kinematics: Kinematics,
    target_position: np.ndarray,
    seed: np.ndarray,
    *,
    max_iterations: int = 200,
    tolerance_m: float = 1e-4,
    damping: float = 1e-3,
    step_scale: float = 0.5,
    max_step_rad: float = 0.05,
) -> tuple[np.ndarray, float, int]:
    q = np.asarray(seed, dtype=np.float64).copy()
    target = np.asarray(target_position, dtype=np.float64)
    lower = kinematics.joint_limits_rad[:, 0]
    upper = kinematics.joint_limits_rad[:, 1]
    best = q.copy()
    best_error = float("inf")
    for iteration in range(1, max_iterations + 1):
        position = kinematics.forward(q)[:3, 3]
        error = target - position
        error_norm = float(np.linalg.norm(error))
        if error_norm < best_error:
            best = q.copy()
            best_error = error_norm
        if error_norm <= tolerance_m:
            return q, error_norm, iteration
        jacobian = kinematics.jacobian(q)[:3]
        regularized = jacobian @ jacobian.T + (
            damping * damping
        ) * np.eye(3)
        dq = jacobian.T @ np.linalg.solve(regularized, error)
        norm = float(np.linalg.norm(dq))
        if norm > max_step_rad:
            dq *= max_step_rad / norm
        q = np.clip(q + step_scale * dq, lower, upper)
    return best, best_error, max_iterations


def _read_key(timeout_s: float = 0.05) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    sequence = first
    for _ in range(3):
        readable, _, _ = select.select([sys.stdin], [], [], 0.01)
        if not readable:
            break
        sequence += sys.stdin.read(1)
    return sequence


def _delta_for_key(key: str, step: float) -> np.ndarray | None:
    mapping = {
        "8": (step, 0.0, 0.0),
        "\x1b[A": (step, 0.0, 0.0),
        "2": (-step, 0.0, 0.0),
        "\x1b[B": (-step, 0.0, 0.0),
        "4": (0.0, step, 0.0),
        "\x1b[D": (0.0, step, 0.0),
        "6": (0.0, -step, 0.0),
        "\x1b[C": (0.0, -step, 0.0),
        "9": (0.0, 0.0, step),
        "\x1b[5~": (0.0, 0.0, step),
        "3": (0.0, 0.0, -step),
        "\x1b[6~": (0.0, 0.0, -step),
        "+": (0.0, 0.0, step),
        "-": (0.0, 0.0, -step),
        "7": (0.0, 0.0, step),
        "1": (0.0, 0.0, -step),
    }
    values = mapping.get(key)
    return None if values is None else np.asarray(values, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--output",
        choices=("joint", "pose"),
        default="joint",
        help="Publish the KDL IK joint target or the pose target.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=None,
        help="Set ROS_DOMAIN_ID before rclpy initialization.",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for /joint_states.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step <= 0.0:
        raise SystemExit("--step must be positive")
    if not sys.stdin.isatty():
        raise SystemExit(
            "Numpad teleoperation requires an interactive terminal"
        )
    if args.domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    rclpy.init()
    node = NumpadTcpTeleop(args.step, args.output)
    original = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("Waiting for /joint_states...")
        node.wait_for_state(args.state_timeout)
        print(
            "Numpad TCP teleop: 8/2 x+/- 4/6 y+/- "
            "9/3 z+/- 0 reset q quit"
        )
        print(
            "NumLock on: digits; NumLock off: arrow/PageUp/PageDown. "
            f"mode={args.output}, step={args.step:.4f} m"
        )
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            key = _read_key()
            if key is None:
                continue
            if key in ("q", "\x03", "\x1b"):
                break
            if key == "0":
                node.reset()
                continue
            delta = _delta_for_key(key, args.step)
            if delta is not None:
                node.move(delta)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
