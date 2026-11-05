"""FollowJointTrajectory action bridge for the JGarm torque controller."""

from __future__ import annotations

import threading
import time

from control_msgs.action import FollowJointTrajectory
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from jgarm_control.trajectory import JointTrajectory
from jgarm_control.types import JOINT_COUNT, JOINT_NAMES


class TrajectoryBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_trajectory_bridge_node")
        self._goal_tolerance_rad = float(
            self.declare_parameter("goal_tolerance_rad", 0.02).value
        )
        self._goal_hold_time_s = float(
            self.declare_parameter("goal_hold_time_s", 0.2).value
        )
        self._publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 200.0).value
        )
        self._state_timeout_s = float(
            self.declare_parameter("state_timeout_s", 0.2).value
        )
        self._lock = threading.Lock()
        self._latest_state: JointState | None = None
        self._latest_state_time = 0.0
        self._active_goal = None
        self._cancel_requested = False
        callback_group = ReentrantCallbackGroup()

        state_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(
            JointState, "/joint_states", self._state_callback, state_qos
        )
        self._position_publisher = self.create_publisher(
            Float64MultiArray, "/arm/target/joint_position", 10
        )
        self._velocity_publisher = self.create_publisher(
            Float64MultiArray, "/arm/target/joint_velocity", 10
        )
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/JG_arm_controller/follow_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=self._accept_goal,
            cancel_callback=self._cancel_goal,
            callback_group=callback_group,
        )
        self.get_logger().info(
            f"Trajectory bridge started at {self._publish_rate_hz:.0f} Hz"
        )

    def _state_callback(self, message: JointState) -> None:
        with self._lock:
            self._latest_state = message
            self._latest_state_time = time.monotonic()

    def _accept_goal(self, goal_request) -> GoalResponse:
        if self._active_goal is not None:
            self.get_logger().warning("Rejecting trajectory: goal already active")
            return GoalResponse.REJECT
        try:
            JointTrajectory.from_message(goal_request.trajectory)
        except ValueError as error:
            self.get_logger().warning(f"Rejecting trajectory: {error}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_goal(self, _goal_handle) -> CancelResponse:
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    def _ordered_joint_state(
        self, message: JointState
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return position, velocity

    def _publish_target(self, position: np.ndarray, velocity: np.ndarray) -> None:
        position_message = Float64MultiArray()
        position_message.data = position.tolist()
        self._position_publisher.publish(position_message)
        velocity_message = Float64MultiArray()
        velocity_message.data = velocity.tolist()
        self._velocity_publisher.publish(velocity_message)

    def _execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        try:
            trajectory = JointTrajectory.from_message(goal_handle.request.trajectory)
        except ValueError as error:
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(error)
            goal_handle.abort()
            return result

        self._active_goal = goal_handle
        self._cancel_requested = False
        start_time = time.monotonic()
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(JOINT_NAMES)
        period = 1.0 / self._publish_rate_hz
        feedback_divider = max(1, int(self._publish_rate_hz / 20.0))
        iteration = 0

        try:
            while rclpy.ok():
                if self._cancel_requested:
                    self._publish_target(
                        trajectory.positions_rad[-1],
                        np.zeros(JOINT_COUNT),
                    )
                    result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    result.error_string = "Trajectory execution canceled"
                    goal_handle.canceled()
                    return result

                with self._lock:
                    state = self._latest_state
                    state_time = self._latest_state_time
                if (
                    state is None
                    or time.monotonic() - state_time > self._state_timeout_s
                ):
                    time.sleep(period)
                    continue

                elapsed = time.monotonic() - start_time
                position, velocity = trajectory.sample(elapsed)
                self._publish_target(position, velocity)
                actual, actual_velocity = self._ordered_joint_state(state)

                if iteration % feedback_divider == 0:
                    feedback.header.stamp = self.get_clock().now().to_msg()
                    feedback.desired.positions = position.tolist()
                    feedback.desired.velocities = velocity.tolist()
                    feedback.actual.positions = actual.tolist()
                    feedback.actual.velocities = actual_velocity.tolist()
                    feedback.error.positions = (position - actual).tolist()
                    feedback.error.velocities = (
                        velocity - actual_velocity
                    ).tolist()
                    goal_handle.publish_feedback(feedback)

                if elapsed >= trajectory.duration_s + self._goal_hold_time_s:
                    final = trajectory.positions_rad[-1]
                    self._publish_target(final, np.zeros(JOINT_COUNT))
                    error = np.max(np.abs(actual - final))
                    if error <= self._goal_tolerance_rad:
                        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        result.error_string = "Trajectory completed"
                        goal_handle.succeed()
                    else:
                        result.error_code = (
                            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                        )
                        result.error_string = (
                            f"Final joint error {error:.4f} rad exceeds tolerance"
                        )
                        goal_handle.abort()
                    return result
                iteration += 1
                time.sleep(period)
        finally:
            self._active_goal = None
            self._cancel_requested = False

        result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
        result.error_string = "ROS context stopped"
        goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryBridgeNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
