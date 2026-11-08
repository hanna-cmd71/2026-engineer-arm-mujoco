"""Parameter wiring tests for the arm control node."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.parameter import Parameter

from jgarm_control.controller import JointImpedanceController
from jgarm_control.control_node import ArmControlNode


class ArmControlNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._log_dir = tempfile.mkdtemp(prefix="jgarm-rclpy-")
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

    def test_gravity_enabled_override_reaches_controller(self) -> None:
        node = ArmControlNode(
            parameter_overrides=[
                Parameter("gravity_enabled", value=False)
            ]
        )
        try:
            self.assertFalse(
                bool(node.get_parameter("gravity_enabled").value)
            )
            self.assertFalse(node.controller.gravity_enabled)
        finally:
            node.destroy_node()

    def test_non_finite_pose_target_is_ignored(self) -> None:
        node = ArmControlNode()
        try:
            before = node._target_position.copy()
            message = PoseStamped()
            message.header.frame_id = "base_link"
            message.pose.position.x = float("nan")
            message.pose.orientation.w = 1.0

            node._pose_target_callback(message)

            np.testing.assert_allclose(node._target_position, before)
        finally:
            node.destroy_node()

    def test_default_gravity_uses_physical_mass_model(self) -> None:
        node = ArmControlNode()
        try:
            gravity = node.controller.gravity_torque(np.zeros(6))
            self.assertAlmostEqual(gravity[1], -3.544755, places=4)
            self.assertAlmostEqual(gravity[2], -3.544086, places=4)
        finally:
            node.destroy_node()

    def test_impedance_mode_selects_impedance_controller(self) -> None:
        node = ArmControlNode(
            parameter_overrides=[
                Parameter("controller_mode", value="impedance")
            ]
        )
        try:
            self.assertIsInstance(
                node.controller,
                JointImpedanceController,
            )
        finally:
            node.destroy_node()


if __name__ == "__main__":
    unittest.main()
