"""Magnetic extraction state for RM 2027 energy units."""

from __future__ import annotations

from collections.abc import Callable
import math

import mujoco
import numpy as np


ENERGY_UNIT_COUNT = 6
_STATION_BODY_NAME = "energy_storage_solids"
_POST_BODY_PREFIX = "storage_post_"


class EnergyUnitExtractor:
    """Owns magnetic release, extraction limits, and collision enablement."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        lateral_clearance_m: float = 0.01,
        angular_clearance_rad: float = math.radians(10.0),
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not math.isfinite(lateral_clearance_m):
            raise ValueError("lateral_clearance_m must be finite")
        if lateral_clearance_m < 0.0:
            raise ValueError("lateral_clearance_m must be non-negative")
        if not math.isfinite(angular_clearance_rad):
            raise ValueError("angular_clearance_rad must be finite")
        if angular_clearance_rad < 0.0:
            raise ValueError("angular_clearance_rad must be non-negative")

        self.model = model
        self.data = data
        self._lateral_clearance_m = float(lateral_clearance_m)
        self._angular_clearance_rad = float(angular_clearance_rad)
        self._log = log if log is not None else _ignore_log
        self._equality_ids: dict[int, int] = {}
        self._qpos_addresses: dict[int, int] = {}
        self._dof_addresses: dict[int, int] = {}
        self._extraction_distances: dict[int, float] = {}
        self._initial_poses: dict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._pulling: set[int] = set()
        self._released: set[int] = set()
        self._collision_pending: set[int] = set()
        self._discover_units()

    @property
    def unit_body_ids(self) -> tuple[int, ...]:
        return tuple(self._equality_ids)

    @property
    def pulling_body_ids(self) -> frozenset[int]:
        return frozenset(self._pulling)

    @property
    def released_body_ids(self) -> frozenset[int]:
        return frozenset(self._released)

    @property
    def collision_pending_body_ids(self) -> frozenset[int]:
        return frozenset(self._collision_pending)

    @property
    def has_active_units(self) -> bool:
        return bool(self._pulling or self._released)

    def extraction_distance(self, body_id: int) -> float:
        return self._extraction_distances[body_id]

    def initial_pose(
        self, body_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin, quaternion, axis = self._initial_poses[body_id]
        return origin.copy(), quaternion.copy(), axis.copy()

    def release(self, body_id: int) -> bool:
        """Releases one unit from its magnetic constraint."""
        if body_id not in self._equality_ids:
            return False
        if body_id in self._pulling or body_id in self._released:
            return False
        equality_id = self._equality_ids[body_id]
        qpos_address = self._qpos_addresses[body_id]
        self.data.eq_active[equality_id] = 0
        pose = self.data.qpos[qpos_address : qpos_address + 7].copy()
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, pose[3:7])
        self._initial_poses[body_id] = (
            pose[:3].copy(),
            pose[3:7].copy(),
            rotation.reshape(3, 3)[:, 2].copy(),
        )
        self._pulling.add(body_id)
        self._log(
            "Started magnetic extraction for energy unit: "
            f"{self._body_name(body_id)}"
        )
        return True

    def constrain_pulling_units(self) -> None:
        """Limits pulling units to the post axis and clearance envelope."""
        for body_id in tuple(self._pulling):
            qpos_address = self._qpos_addresses[body_id]
            dof_address = self._dof_addresses[body_id]
            origin, initial_quaternion, extraction_axis = (
                self._initial_poses[body_id]
            )
            displacement = (
                self.data.qpos[qpos_address : qpos_address + 3] - origin
            )
            extraction_distance = self._extraction_distances[body_id]
            axial_distance = float(displacement @ extraction_axis)
            axial_distance = float(
                np.clip(axial_distance, 0.0, extraction_distance)
            )
            lateral_offset = displacement - (
                float(displacement @ extraction_axis) * extraction_axis
            )
            lateral_distance = float(np.linalg.norm(lateral_offset))
            if lateral_distance > self._lateral_clearance_m:
                lateral_offset *= (
                    self._lateral_clearance_m / lateral_distance
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
            if rotation_angle > self._angular_clearance_rad:
                interpolation = (
                    self._angular_clearance_rad / rotation_angle
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
                self._complete(body_id)

    def enable_pending_collisions(self) -> None:
        for body_id in tuple(self._collision_pending):
            self._try_enable_collision(body_id)

    def step(self, steps: int) -> None:
        """Advances released-unit physics for a fixed number of steps."""
        if steps < 1:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self.constrain_pulling_units()
        self.enable_pending_collisions()

    def _discover_units(self) -> None:
        for index in range(1, ENERGY_UNIT_COUNT + 1):
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
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"energy_unit_{index:02d}_free",
            )
            top_site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"storage_post_{index:02d}_top",
            )
            if min(body_id, equality_id, joint_id, top_site_id) < 0:
                continue
            extraction_distance = 2.0 * float(
                self.model.site_pos[top_site_id, 2]
            )
            if not math.isfinite(extraction_distance):
                raise ValueError(
                    f"storage_post_{index:02d}_top has an invalid offset"
                )
            if extraction_distance <= 0.0:
                raise ValueError(
                    f"storage_post_{index:02d}_top must have a positive "
                    "z offset"
                )
            self._equality_ids[body_id] = equality_id
            self._qpos_addresses[body_id] = int(
                self.model.jnt_qposadr[joint_id]
            )
            self._dof_addresses[body_id] = int(
                self.model.jnt_dofadr[joint_id]
            )
            self._extraction_distances[body_id] = extraction_distance

    def _complete(self, body_id: int) -> None:
        self._pulling.discard(body_id)
        self._released.add(body_id)
        self._collision_pending.add(body_id)
        self._log(
            "Energy unit extraction complete: "
            f"{self._body_name(body_id)}"
        )

    def _try_enable_collision(self, body_id: int) -> None:
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
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if body_id not in (body1, body2):
                continue
            other_body = body2 if body1 == body_id else body1
            if self._is_station_body(other_body):
                penetrates_station = True
                break

        if penetrates_station:
            for geom_id in affected_geoms:
                self.model.geom_conaffinity[geom_id] &= ~2
            self.model.body_conaffinity[body_id] &= ~2
            return

        self._collision_pending.discard(body_id)
        self._log(
            "Energy unit collision enabled: "
            f"{self._body_name(body_id)}"
        )

    def _is_station_body(self, body_id: int) -> bool:
        name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )
        return bool(
            name
            and (
                name == _STATION_BODY_NAME
                or name.startswith(_POST_BODY_PREFIX)
            )
        )

    def _body_name(self, body_id: int) -> str:
        name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )
        return name if name is not None else str(body_id)


def _ignore_log(_message: str) -> None:
    pass
