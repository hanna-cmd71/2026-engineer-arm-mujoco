"""Plan and execute a six-joint goal through MoveIt 2."""

from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from jgarm_control.types import JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="0.1,-0.8,0.8,0.1,0.3,-0.1",
        help="Comma-separated six-joint target in radians.",
    )
    parser.add_argument("--planning-time", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.fromstring(args.target, sep=",")
    if target.shape != (6,):
        raise ValueError("--target must contain six values")

    rclpy.init()
    node = rclpy.create_node("jgarm_moveit_joint_goal")
    client = ActionClient(node, MoveGroup, "/move_action")
    if not client.wait_for_server(timeout_sec=10.0):
        raise RuntimeError("/move_action is unavailable")

    state_holder = {}

    def state_callback(message: JointState) -> None:
        state_holder["state"] = message
        state_holder["time"] = time.monotonic()

    state_qos = QoSProfile(
        depth=10, reliability=ReliabilityPolicy.RELIABLE
    )
    node.create_subscription(
        JointState, "/joint_states", state_callback, state_qos
    )
    stable_since = None
    deadline = time.monotonic() + 10.0
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        message = state_holder.get("state")
        fresh = (
            message is not None
            and time.monotonic() - state_holder.get("time", 0.0) < 0.2
        )
        stable = (
            fresh
            and message.velocity
            and max(abs(value) for value in message.velocity) < 0.01
        )
        if stable and stable_since is None:
            stable_since = time.monotonic()
        elif not stable:
            stable_since = None
        if stable_since is not None and time.monotonic() - stable_since >= 0.5:
            break
    else:
        raise RuntimeError("Robot did not settle before MoveIt planning")

    constraints = Constraints()
    for name, position in zip(JOINT_NAMES, target):
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = float(position)
        constraint.tolerance_above = 0.01
        constraint.tolerance_below = 0.01
        constraint.weight = 1.0
        constraints.joint_constraints.append(constraint)

    request = MotionPlanRequest()
    request.group_name = "JG_arm"
    request.num_planning_attempts = 10
    request.allowed_planning_time = args.planning_time
    request.max_velocity_scaling_factor = 0.2
    request.max_acceleration_scaling_factor = 0.2
    request.start_state.is_diff = True
    request.goal_constraints.append(constraints)
    request.workspace_parameters.header.frame_id = "base_link"
    request.workspace_parameters.min_corner.x = -0.6
    request.workspace_parameters.min_corner.y = -0.6
    request.workspace_parameters.min_corner.z = -0.2
    request.workspace_parameters.max_corner.x = 0.8
    request.workspace_parameters.max_corner.y = 0.8
    request.workspace_parameters.max_corner.z = 1.0

    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options = PlanningOptions()
    goal.planning_options.plan_only = False
    goal.planning_options.replan = True
    goal.planning_options.replan_attempts = 10

    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError("MoveIt rejected the goal")

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    response = result_future.result()
    if response is None:
        raise RuntimeError("MoveIt returned no result")

    result = response.result
    print(f"state={response.status} error_code={result.error_code}")
    print(result.error_code)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
