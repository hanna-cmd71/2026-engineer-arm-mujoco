"""Tests for TCP tracking and viewer trail rendering."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import rclpy
from rclpy.parameter import Parameter

from jgarm_control.viewer_node import ArmViewerNode


class ArmViewerNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._log_dir = tempfile.mkdtemp(prefix="jgarm-viewer-rclpy-")
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

    def test_tcp_pose_matches_model_site(self) -> None:
        node = ArmViewerNode()
        try:
            pose = node.tcp_pose()

            np.testing.assert_allclose(
                pose[:3, 3],
                node.data.site_xpos[node._tcp_site_id],
            )
            np.testing.assert_allclose(
                pose[:3, :3],
                node.data.site_xmat[node._tcp_site_id].reshape(3, 3),
            )
            np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
        finally:
            node.destroy_node()

    def test_update_draws_current_tcp_and_trail(self) -> None:
        node = ArmViewerNode(
            parameter_overrides=[
                Parameter("tcp_trail_points", value=8),
                Parameter("tcp_trail_min_step_m", value=0.0),
            ]
        )
        try:
            scene = mujoco.MjvScene(node.model, maxgeom=16)
            viewer = SimpleNamespace(user_scn=scene)
            joint_id = node._joint_ids["joint_2"]
            qpos_address = node.model.jnt_qposadr[joint_id]

            node.update_tcp_visualization(viewer, now=1.0)
            first_position = node.tcp_pose()[:3, 3].copy()
            self.assertEqual(len(node._tcp_trail), 1)
            self.assertEqual(scene.ngeom, 2)

            node.data.qpos[qpos_address] += 0.1
            mujoco.mj_forward(node.model, node.data)
            node.update_tcp_visualization(viewer, now=1.1)

            self.assertEqual(len(node._tcp_trail), 2)
            self.assertEqual(scene.ngeom, 3)
            np.testing.assert_allclose(
                node._tcp_trail[0],
                first_position,
            )
            np.testing.assert_allclose(
                scene.geoms[2].pos,
                node.tcp_pose()[:3, 3],
            )
        finally:
            node.destroy_node()


if __name__ == "__main__":
    unittest.main()
