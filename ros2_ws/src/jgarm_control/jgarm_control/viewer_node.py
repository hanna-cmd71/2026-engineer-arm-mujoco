"""Read-only MuJoCo visualizer driven by /joint_states."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
from ament_index_python.packages import get_package_share_directory
from mujoco import viewer as mujoco_viewer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from jgarm_control.types import JOINT_NAMES


class ArmViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_viewer_node")
        description_share = Path(
            get_package_share_directory("jgarm_description")
        )
        model_path = Path(
            self.declare_parameter(
                "model_path",
                str(description_share / "model" / "scene.xml"),
            ).value
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._joint_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in JOINT_NAMES
        }
        self._latest_message: JointState | None = None
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(
            JointState, "/joint_states", self._state_callback, qos
        )
        self.get_logger().info(f"MuJoCo viewer model: {model_path}")

    def _state_callback(self, message: JointState) -> None:
        self._latest_message = message

    def apply_latest(self) -> None:
        if self._latest_message is None:
            return
        index_by_name = {
            name: index
            for index, name in enumerate(self._latest_message.name)
        }
        for name in JOINT_NAMES:
            source = index_by_name.get(name)
            if source is None:
                continue
            joint_id = self._joint_ids[name]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            dof_adr = self.model.jnt_dofadr[joint_id]
            if source < len(self._latest_message.position):
                self.data.qpos[qpos_adr] = self._latest_message.position[source]
            if source < len(self._latest_message.velocity):
                self.data.qvel[dof_adr] = self._latest_message.velocity[source]
        mujoco.mj_forward(self.model, self.data)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmViewerNode()
    with mujoco_viewer.launch_passive(node.model, node.data) as mj_viewer:
        mj_viewer.cam.distance = 1.2
        mj_viewer.cam.azimuth = 135.0
        mj_viewer.cam.elevation = -20.0
        while rclpy.ok() and mj_viewer.is_running():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.apply_latest()
            mj_viewer.sync()
            time.sleep(1.0 / 60.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
