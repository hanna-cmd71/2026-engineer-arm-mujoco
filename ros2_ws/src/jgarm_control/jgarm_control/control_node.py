"""High-rate JGarm joint controller and KDL pose target handler."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from jgarm_control.controller import PDGains, PDGravityController
from jgarm_control.dynamics import PinocchioDynamics
from jgarm_control.kinematics import IKOptions, Kinematics
from jgarm_control.types import JOINT_COUNT, JOINT_NAMES, JointFeedback


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-12:
        raise ValueError("Pose orientation quaternion is invalid")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class ArmControlNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_control_node")
        description_share = Path(
            get_package_share_directory("jgarm_description")
        )
        urdf_path = Path(
            self.declare_parameter(
                "urdf_path", str(description_share / "urdf" / "JGarm.urdf")
            ).value
        )
        kp = self.declare_parameter(
            "kp", [60.0, 100.0, 100.0, 40.0, 20.0, 10.0]
        ).value
        kd = self.declare_parameter(
            "kd", [2.0, 4.0, 4.0, 1.5, 1.0, 0.5]
        ).value
        effort_limits = self.declare_parameter(
            "effort_limits", [25.0, 63.0, 35.0, 18.0, 6.0, 3.0]
        ).value
        control_rate_hz = float(
            self.declare_parameter("control_rate_hz", 500.0).value
        )
        self._state_timeout_s = float(
            self.declare_parameter("state_timeout_s", 0.2).value
        )

        self.kinematics = Kinematics(urdf_path)
        self.dynamics = PinocchioDynamics(urdf_path)
        self.controller = PDGravityController(
            PDGains(np.asarray(kp), np.asarray(kd)),
            np.asarray(effort_limits),
            self.dynamics.gravity_torque,
        )
        self._target_position = np.array(
            self.declare_parameter(
                "target_position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).value,
            dtype=np.float64,
        )
        self._target_velocity = np.zeros(JOINT_COUNT)
        self._latest_state: JointFeedback | None = None
        self._latest_state_time = 0.0
        self._current_position = np.zeros(JOINT_COUNT)

        state_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(
            JointState, "/joint_states", self._state_callback, state_qos
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/target/joint_position",
            self._joint_target_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/target/joint_velocity",
            self._joint_velocity_callback,
            10,
        )
        self.create_subscription(
            PoseStamped, "/arm/target/pose", self._pose_target_callback, 10
        )
        self._effort_publisher = self.create_publisher(
            Float64MultiArray, "/arm/control/effort_cmd", 10
        )
        self.create_timer(1.0 / control_rate_hz, self._control_step)
        self.get_logger().info(
            f"Arm control started at {control_rate_hz:.0f} Hz"
        )

    def _state_callback(self, message: JointState) -> None:
        index_by_name = {
            name: index for index, name in enumerate(message.name)
        }
        position = np.full(JOINT_COUNT, np.nan)
        velocity = np.full(JOINT_COUNT, np.nan)
        for index, name in enumerate(JOINT_NAMES):
            source = index_by_name.get(name)
            if source is None:
                continue
            if source < len(message.position):
                position[index] = message.position[source]
            if source < len(message.velocity):
                velocity[index] = message.velocity[source]
        alive = np.isfinite(position) & np.isfinite(velocity)
        self._latest_state = JointFeedback(
            position,
            velocity,
            np.zeros(JOINT_COUNT),
            alive,
        )
        self._latest_state_time = time.monotonic()
        if np.all(alive):
            self._current_position = position.copy()

    def _joint_target_callback(
        self, message: Float64MultiArray
    ) -> None:
        if len(message.data) != JOINT_COUNT:
            self.get_logger().warning("Joint target must contain six values")
            return
        target = np.asarray(message.data, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            self.get_logger().warning("Joint target contains invalid values")
            return
        self._target_position = np.clip(
            target,
            self.kinematics.joint_limits_rad[:, 0],
            self.kinematics.joint_limits_rad[:, 1],
        )
        self._target_velocity = np.zeros(JOINT_COUNT)

    def _joint_velocity_callback(
        self, message: Float64MultiArray
    ) -> None:
        if len(message.data) != JOINT_COUNT:
            self.get_logger().warning(
                "Joint velocity target must contain six values"
            )
            return
        target = np.asarray(message.data, dtype=np.float64)
        if np.all(np.isfinite(target)):
            self._target_velocity = target

    def _pose_target_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id not in ("", "base_link"):
            self.get_logger().warning(
                "Pose target frame must be base_link"
            )
            return
        target = np.eye(4)
        target[:3, :3] = _quaternion_matrix(
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        )
        target[:3, 3] = [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ]
        result = self.kinematics.inverse(
            target,
            self._target_position,
            IKOptions(max_attempts=5),
        )
        if result.success:
            self._target_position = result.position_rad
            self._target_velocity = np.zeros(JOINT_COUNT)
            self.get_logger().info(
                "Pose IK accepted: "
                f"position_error={result.position_error_m:.6e} m"
            )
        else:
            self.get_logger().warning(result.message)

    def _control_step(self) -> None:
        if self._latest_state is None:
            return
        age = time.monotonic() - self._latest_state_time
        if age > self._state_timeout_s:
            return
        command = self.controller.compute(
            self._latest_state,
            self._target_position,
            self._target_velocity,
        )
        message = Float64MultiArray()
        message.data = command.effort_nm.tolist()
        self._effort_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                stop = Float64MultiArray()
                stop.data = [0.0] * JOINT_COUNT
                node._effort_publisher.publish(stop)
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
