"""Parameter wiring tests for the arm control node."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import rclpy
from rclpy.parameter import Parameter

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


if __name__ == "__main__":
    unittest.main()
