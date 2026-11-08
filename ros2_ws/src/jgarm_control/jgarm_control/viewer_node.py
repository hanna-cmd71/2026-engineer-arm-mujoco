"""Interactive MuJoCo visualizer driven by /joint_states."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from geometry_msgs.msg import PoseStamped
import mujoco
from ament_index_python.packages import get_package_share_directory
from mujoco import viewer as mujoco_viewer
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from jgarm_control.energy_units import EnergyUnitExtractor
from jgarm_control.types import JOINT_NAMES


class ArmViewerNode(Node):
    def __init__(
        self, parameter_overrides: list[Parameter] | None = None
    ) -> None:
        super().__init__(
            "arm_viewer_node",
            parameter_overrides=parameter_overrides,
        )
        description_share = Path(
            get_package_share_directory("jgarm_description")
        )
        model_path = Path(
            self.declare_parameter(
                "model_path",
                str(description_share / "model" / "scene.xml"),
            ).value
        )
        self.show_frames = bool(
            self.declare_parameter("show_frames", False).value
        )
        self.draggable_energy_units = bool(
            self.declare_parameter(
                "draggable_energy_units",
                True,
            ).value
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        power_on_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "power_on"
        )
        if power_on_key_id < 0:
            raise ValueError("MuJoCo model is missing the power_on keyframe")
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, power_on_key_id
        )
        mujoco.mj_forward(self.model, self.data)
        self._joint_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in JOINT_NAMES
        }
        self._wheel_joint_ids = {
            name: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in (
                "wheel_front_left_spin",
                "wheel_front_right_spin",
                "wheel_rear_left_spin",
                "wheel_rear_right_spin",
            )
        }
        self._mobile_base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "mobile_base",
        )
        self._mobile_joint_ids = {
            name: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in ("base_x", "base_y", "base_yaw")
        }
        self._arm_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[self._joint_ids[name]]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._arm_dof_addresses = np.array(
            [
                self.model.jnt_dofadr[self._joint_ids[name]]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._energy_units = EnergyUnitExtractor(
            self.model,
            self.data,
            lateral_clearance_m=float(
                self.declare_parameter(
                    "extraction_lateral_clearance_m",
                    0.01,
                ).value
            ),
            angular_clearance_rad=np.deg2rad(
                float(
                    self.declare_parameter(
                        "extraction_angular_clearance_deg",
                        10.0,
                    ).value
                )
            ),
            log=self.get_logger().info,
        )
        self._tcp_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tcp",
        )
        if self._tcp_site_id < 0:
            raise ValueError("MuJoCo model is missing the tcp site")
        self._tcp_trail_enabled = bool(
            self.declare_parameter("tcp_trail_enabled", True).value
        )
        self._tcp_trail_points = max(
            1,
            int(self.declare_parameter("tcp_trail_points", 300).value),
        )
        self._tcp_trail_period_s = max(
            1e-3,
            float(
                self.declare_parameter("tcp_trail_period_s", 0.02).value
            ),
        )
        self._tcp_trail_min_step_m = max(
            0.0,
            float(
                self.declare_parameter(
                    "tcp_trail_min_step_m",
                    0.001,
                ).value
            ),
        )
        self._tcp_publish_period_s = 1.0 / max(
            1.0,
            float(
                self.declare_parameter("tcp_publish_rate_hz", 30.0).value
            ),
        )
        self._tcp_trail: deque[np.ndarray] = deque(
            maxlen=self._tcp_trail_points
        )
        self._last_tcp_trail_time = -np.inf
        self._last_tcp_publish_time = -np.inf
        self._tcp_pose_publisher = self.create_publisher(
            PoseStamped,
            "/arm/tcp_pose",
            10,
        )
        self._steps_per_render = max(
            1,
            round(1.0 / 60.0 / self.model.opt.timestep),
        )
        self._latest_message: JointState | None = None
        self._latest_odometry: Odometry | None = None
        qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(
            JointState, "/joint_states", self._state_callback, qos
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self._odometry_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.get_logger().info(f"MuJoCo viewer model: {model_path}")

    def _state_callback(self, message: JointState) -> None:
        self._latest_message = message

    def _odometry_callback(self, message: Odometry) -> None:
        self._latest_odometry = message

    def tcp_pose(self) -> np.ndarray:
        """Returns the TCP site pose as a 4x4 world-frame matrix."""
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self._tcp_site_id]
        return pose

    def _publish_tcp_pose(self) -> None:
        pose = self.tcp_pose()
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion,
            self.data.site_xmat[self._tcp_site_id],
        )
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.position.x = float(pose[0, 3])
        message.pose.position.y = float(pose[1, 3])
        message.pose.position.z = float(pose[2, 3])
        message.pose.orientation.x = float(quaternion[1])
        message.pose.orientation.y = float(quaternion[2])
        message.pose.orientation.z = float(quaternion[3])
        message.pose.orientation.w = float(quaternion[0])
        self._tcp_pose_publisher.publish(message)

    def update_tcp_visualization(
        self,
        mj_viewer,
        now: float | None = None,
    ) -> None:
        """Publishes the TCP pose and draws its trail in the viewer."""
        current_time = time.monotonic() if now is None else float(now)
        if (
            current_time - self._last_tcp_publish_time
            >= self._tcp_publish_period_s
        ):
            self._publish_tcp_pose()
            self._last_tcp_publish_time = current_time
        position = self.tcp_pose()[:3, 3].copy()
        if (
            self._tcp_trail_enabled
            and current_time - self._last_tcp_trail_time
            >= self._tcp_trail_period_s
        ):
            if (
                not self._tcp_trail
                or np.linalg.norm(position - self._tcp_trail[-1])
                >= self._tcp_trail_min_step_m
            ):
                self._tcp_trail.append(position)
            self._last_tcp_trail_time = current_time
        scene = mj_viewer.user_scn
        scene.ngeom = 0
        trail_capacity = max(0, int(scene.maxgeom) - 1)
        trail = list(self._tcp_trail)
        if len(trail) > trail_capacity:
            trail = trail[-trail_capacity:] if trail_capacity else []
        point_count = len(trail)
        axis = np.eye(3).reshape(-1)
        for index, point in enumerate(trail):
            fraction = (index + 1) / point_count
            rgba = np.array(
                [
                    0.05 + 0.05 * fraction,
                    0.35 + 0.55 * fraction,
                    0.95,
                    0.7 + 0.3 * fraction,
                ],
                dtype=np.float32,
            )
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.0045, 0.0, 0.0]),
                point,
                axis,
                rgba,
            )
            scene.ngeom += 1
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.01, 0.0, 0.0]),
                position,
                axis,
                np.array([1.0, 0.25, 0.05, 1.0], dtype=np.float32),
            )
            scene.ngeom += 1

    def release_selected_energy_unit(self, selected_body_id: int) -> None:
        """Releases a selected energy unit from its magnetic constraint."""
        if not self.draggable_energy_units:
            return
        self._energy_units.release(selected_body_id)

    def apply_latest(self) -> None:
        if self._latest_odometry is not None:
            pose = self._latest_odometry.pose.pose
            self.data.qpos[
                self.model.jnt_qposadr[
                    self._mobile_joint_ids["base_x"]
                ]
            ] = (
                pose.position.x
                - self.model.body_pos[self._mobile_base_body_id, 0]
            )
            self.data.qpos[
                self.model.jnt_qposadr[
                    self._mobile_joint_ids["base_y"]
                ]
            ] = (
                pose.position.y
                - self.model.body_pos[self._mobile_base_body_id, 1]
            )
            quaternion = pose.orientation
            yaw = np.arctan2(
                2.0 * (
                    quaternion.w * quaternion.z
                    + quaternion.x * quaternion.y
                ),
                1.0 - 2.0 * (
                    quaternion.y * quaternion.y
                    + quaternion.z * quaternion.z
                ),
            )
            self.data.qpos[
                self.model.jnt_qposadr[
                    self._mobile_joint_ids["base_yaw"]
                ]
            ] = yaw
        if self._latest_message is None:
            mujoco.mj_forward(self.model, self.data)
            return
        index_by_name = {
            name: index
            for index, name in enumerate(self._latest_message.name)
        }
        for name in (*JOINT_NAMES, *self._wheel_joint_ids):
            source = index_by_name.get(name)
            if source is None:
                continue
            joint_id = self._joint_ids.get(
                name,
                self._wheel_joint_ids.get(name),
            )
            qpos_adr = self.model.jnt_qposadr[joint_id]
            dof_adr = self.model.jnt_dofadr[joint_id]
            if source < len(self._latest_message.position):
                self.data.qpos[qpos_adr] = self._latest_message.position[source]
            if source < len(self._latest_message.velocity):
                self.data.qvel[dof_adr] = self._latest_message.velocity[source]
        mujoco.mj_forward(self.model, self.data)

    def step_released_energy_units(self) -> None:
        """Advances free-object physics while magnetic constraints are off."""
        if not self._energy_units.has_active_units:
            return
        if self._latest_message is None:
            arm_position = self.data.qpos[
                self._arm_qpos_addresses
            ].copy()
            arm_velocity = np.zeros(len(JOINT_NAMES))
        else:
            index_by_name = {
                name: index
                for index, name in enumerate(self._latest_message.name)
            }
            arm_position = np.array(
                [
                    self._latest_message.position[index_by_name[name]]
                    if name in index_by_name
                    else self.data.qpos[qpos_address]
                    for name, qpos_address in zip(
                        JOINT_NAMES,
                        self._arm_qpos_addresses,
                        strict=True,
                    )
                ]
            )
            arm_velocity = np.array(
                [
                    self._latest_message.velocity[index_by_name[name]]
                    if (
                        name in index_by_name
                        and index_by_name[name]
                        < len(self._latest_message.velocity)
                    )
                    else 0.0
                    for name in JOINT_NAMES
                ]
            )
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self._steps_per_render):
            self.data.qpos[self._arm_qpos_addresses] = arm_position
            self.data.qvel[self._arm_dof_addresses] = arm_velocity
            mujoco.mj_step(self.model, self.data)
            self._energy_units.constrain_pulling_units()
        self._energy_units.enable_pending_collisions()
        self.apply_latest()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmViewerNode()
    with mujoco_viewer.launch_passive(node.model, node.data) as mj_viewer:
        mj_viewer.opt.geomgroup[5] = 1 if node.show_frames else 0
        mj_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mj_viewer.cam.trackbodyid = -1
        mj_viewer.cam.lookat[:] = [-0.2, 0.0, 0.3]
        mj_viewer.cam.distance = 4.5
        mj_viewer.cam.azimuth = 135.0
        mj_viewer.cam.elevation = -30.0
        try:
            while rclpy.ok() and mj_viewer.is_running():
                rclpy.spin_once(node, timeout_sec=0.001)
                node.apply_latest()
                node.release_selected_energy_unit(mj_viewer.perturb.select)
                node.step_released_energy_units()
                node.update_tcp_visualization(mj_viewer)
                mj_viewer.sync()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
