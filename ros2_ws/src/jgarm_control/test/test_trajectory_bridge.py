"""Concurrency guard tests for the trajectory bridge."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import GoalResponse
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from jgarm_control.trajectory_bridge_node import TrajectoryBridgeNode
from jgarm_control.types import JOINT_COUNT, JOINT_NAMES


class TrajectoryBridgeNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._log_dir = tempfile.mkdtemp(prefix="jgarm-bridge-")
        cls._old_ros_home = os.environ.get("ROS_HOME")
        cls._old_ros_log_dir = os.environ.get("ROS_LOG_DIR")
        os.environ["ROS_HOME"] = cls._log_dir
        os.environ["ROS_LOG_DIR"] = cls._log_dir
        cls._owns_context = False
        if not rclpy.ok():
            rclpy.init()
            cls._owns_context = True

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owns_context:
            rclpy.shutdown()
        if cls._old_ros_home is None:
            os.environ.pop("ROS_HOME", None)
        else:
            os.environ["ROS_HOME"] = cls._old_ros_home
        if cls._old_ros_log_dir is None:
            os.environ.pop("ROS_LOG_DIR", None)
        else:
            os.environ["ROS_LOG_DIR"] = cls._old_ros_log_dir
        shutil.rmtree(cls._log_dir, ignore_errors=True)

    @staticmethod
    def _goal() -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)
        for time_s in (0.0, 1.0):
            point = JointTrajectoryPoint()
            point.positions = [0.0] * JOINT_COUNT
            point.velocities = [0.0] * JOINT_COUNT
            point.time_from_start.sec = int(time_s)
            trajectory.points.append(point)
        goal.trajectory = trajectory
        return goal

    def test_second_goal_is_rejected_while_first_is_reserved(self) -> None:
        node = TrajectoryBridgeNode()
        try:
            first = node._accept_goal(self._goal())
            second = node._accept_goal(self._goal())
            self.assertEqual(first, GoalResponse.ACCEPT)
            self.assertEqual(second, GoalResponse.REJECT)
        finally:
            node.destroy_node()


if __name__ == "__main__":
    unittest.main()
