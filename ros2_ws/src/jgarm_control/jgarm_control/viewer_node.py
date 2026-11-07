"""Interactive MuJoCo visualizer driven by /joint_states."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
from ament_index_python.packages import get_package_share_directory
from mujoco import viewer as mujoco_viewer
from nav_msgs.msg import Odometry
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
        self._energy_unit_equalities = {}
        self._energy_unit_qpos_addresses = {}
        self._energy_unit_dof_addresses = {}
        self._energy_unit_extraction_distances = {}
        for index in range(1, 7):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"energy_unit_{index:02d}",
            )
            equality_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                f"storage_magnet_{index:02d}",
            )
            if body_id >= 0 and equality_id >= 0:
                self._energy_unit_equalities[body_id] = equality_id
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"energy_unit_{index:02d}_free",
                )
                if joint_id >= 0:
                    self._energy_unit_qpos_addresses[body_id] = int(
                        self.model.jnt_qposadr[joint_id]
                    )
                    self._energy_unit_dof_addresses[body_id] = int(
                        self.model.jnt_dofadr[joint_id]
                    )
                    top_site_id = mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_SITE,
                        f"storage_post_{index:02d}_top",
                    )
                    self._energy_unit_extraction_distances[body_id] = (
                        2.0 * float(self.model.site_pos[top_site_id, 2])
                    )
        self._extraction_initial_poses: dict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._pulling_energy_units: set[int] = set()
        self._released_energy_units: set[int] = set()
        self._collision_pending_energy_units: set[int] = set()
        self._extraction_lateral_clearance_m = float(
            self.declare_parameter(
                "extraction_lateral_clearance_m",
                0.01,
            ).value
        )
        self._extraction_angular_clearance_rad = np.deg2rad(
            float(
                self.declare_parameter(
                    "extraction_angular_clearance_deg",
                    10.0,
                ).value
            )
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

    def release_selected_energy_unit(self, selected_body_id: int) -> None:
        """Releases a selected energy unit from its magnetic constraint."""
        if not self.draggable_energy_units:
            return
        if selected_body_id not in self._energy_unit_equalities:
            return
        if (
            selected_body_id in self._pulling_energy_units
            or selected_body_id in self._released_energy_units
        ):
            return
        equality_id = self._energy_unit_equalities[selected_body_id]
        self.data.eq_active[equality_id] = 0
        qpos_address = self._energy_unit_qpos_addresses[selected_body_id]
        pose = self.data.qpos[qpos_address : qpos_address + 7].copy()
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, pose[3:7])
        extraction_axis = rotation.reshape(3, 3)[:, 2].copy()
        self._extraction_initial_poses[selected_body_id] = (
            pose[:3].copy(),
            pose[3:7].copy(),
            extraction_axis,
        )
        self._pulling_energy_units.add(selected_body_id)
        self.get_logger().info(
            "Started magnetic extraction for energy unit: "
            f"{mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                selected_body_id,
            )}"
        )

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

    def _complete_energy_unit_extraction(self, body_id: int) -> None:
        self._pulling_energy_units.discard(body_id)
        self._released_energy_units.add(body_id)
        self._collision_pending_energy_units.add(body_id)
        self.get_logger().info(
            "Energy unit extraction complete: "
            f"{mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )}"
        )

    def _try_enable_energy_unit_collision(self, body_id: int) -> None:
        affected_geoms = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == body_id
        ]
        for geom_id in affected_geoms:
            self.model.geom_conaffinity[geom_id] |= 2
        self.model.body_conaffinity[body_id] |= 2
        mujoco.mj_forward(self.model, self.data)

        penetrates_station = False
        for contact in self.data.contact:
            body1 = int(
                self.model.geom_bodyid[int(contact.geom1)]
            )
            body2 = int(
                self.model.geom_bodyid[int(contact.geom2)]
            )
            if body_id not in (body1, body2):
                continue
            other_body = body2 if body1 == body_id else body1
            other_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                other_body,
            )
            if other_name and (
                other_name == "energy_storage_solids"
                or other_name.startswith("storage_post_")
            ):
                penetrates_station = True
                break

        if penetrates_station:
            for geom_id in affected_geoms:
                self.model.geom_conaffinity[geom_id] &= ~2
            self.model.body_conaffinity[body_id] &= ~2
            return

        self._collision_pending_energy_units.discard(body_id)
        self.get_logger().info(
            "Energy unit collision enabled: "
            f"{mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )}"
        )

    def _constrain_pulling_energy_units(self) -> None:
        for body_id in tuple(self._pulling_energy_units):
            qpos_address = self._energy_unit_qpos_addresses[body_id]
            dof_address = self._energy_unit_dof_addresses[body_id]
            (
                origin,
                initial_quaternion,
                extraction_axis,
            ) = self._extraction_initial_poses[body_id]
            displacement = (
                self.data.qpos[qpos_address : qpos_address + 3] - origin
            )
            axial_distance = float(displacement @ extraction_axis)
            extraction_distance = self._energy_unit_extraction_distances[
                body_id
            ]
            axial_distance = float(
                np.clip(
                    axial_distance,
                    0.0,
                    extraction_distance,
                )
            )
            lateral_offset = displacement - (
                float(displacement @ extraction_axis) * extraction_axis
            )
            lateral_distance = float(np.linalg.norm(lateral_offset))
            if lateral_distance > self._extraction_lateral_clearance_m:
                lateral_offset *= (
                    self._extraction_lateral_clearance_m / lateral_distance
                )
            self.data.qpos[qpos_address : qpos_address + 3] = (
                origin
                + extraction_axis * axial_distance
                + lateral_offset
            )
            current_quaternion = self.data.qpos[
                qpos_address + 3 : qpos_address + 7
            ]
            quaternion_dot = float(
                np.clip(
                    abs(current_quaternion @ initial_quaternion),
                    0.0,
                    1.0,
                )
            )
            rotation_angle = 2.0 * np.arccos(quaternion_dot)
            if rotation_angle > self._extraction_angular_clearance_rad:
                interpolation = (
                    self._extraction_angular_clearance_rad
                    / rotation_angle
                )
                reference_quaternion = initial_quaternion.copy()
                if current_quaternion @ reference_quaternion < 0.0:
                    reference_quaternion = -reference_quaternion
                limited_quaternion = (
                    (1.0 - interpolation) * reference_quaternion
                    + interpolation * current_quaternion
                )
                limited_quaternion /= np.linalg.norm(limited_quaternion)
                self.data.qpos[
                    qpos_address + 3 : qpos_address + 7
                ] = limited_quaternion
            linear_velocity = self.data.qvel[
                dof_address : dof_address + 3
            ]
            axial_velocity = float(linear_velocity @ extraction_axis)
            self.data.qvel[dof_address : dof_address + 3] = (
                axial_velocity * extraction_axis
            )
            self.data.qvel[dof_address + 3 : dof_address + 6] *= 0.5
            if axial_distance >= extraction_distance - 1e-6:
                self._complete_energy_unit_extraction(body_id)

    def step_released_energy_units(self) -> None:
        """Advances free-object physics while magnetic constraints are off."""
        if not (
            self._released_energy_units
            or self._pulling_energy_units
        ):
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
            self._constrain_pulling_energy_units()
        for body_id in tuple(self._collision_pending_energy_units):
            self._try_enable_energy_unit_collision(body_id)
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
                mj_viewer.sync()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
