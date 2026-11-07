"""MuJoCo implementation of the JGarm low-level hardware boundary."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from jgarm_sim.types import JOINT_COUNT, JOINT_NAMES, JointFeedback, TorqueCommand


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "model" / "scene.xml"
)


class MuJoCoArmBackend:
    """Synchronous six-axis backend for MuJoCo simulation.

    The interface intentionally exposes joint feedback and torque commands only.
    Controllers do not receive direct access to ``MjData`` or simulation-only
    state. A future jRMCS adapter can implement the same boundary.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        control_period_s: float = 0.001,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        if self.model.nq < JOINT_COUNT or self.model.nv < JOINT_COUNT:
            raise ValueError(
                "The model requires at least six scalar joint coordinates"
            )
        if self.model.nu != JOINT_COUNT:
            raise ValueError("The backend requires exactly six actuators")

        self._joint_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._actuator_ids = np.array(
            [
                self._require_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor"
                )
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._joint_qpos_adr = self.model.jnt_qposadr[self._joint_ids].copy()
        self._joint_dof_adr = self.model.jnt_dofadr[self._joint_ids].copy()
        self._joint_ranges = self.model.jnt_range[self._joint_ids].copy()
        self._control_limits = self.model.actuator_ctrlrange[
            self._actuator_ids
        ].copy()
        self._power_on_key_id = self._require_id(
            mujoco.mjtObj.mjOBJ_KEY, "power_on"
        )

        self.control_period_s = float(control_period_s)
        if not np.isfinite(self.control_period_s) or self.control_period_s <= 0.0:
            raise ValueError("control_period_s must be finite and positive")
        steps = self.control_period_s / self.model.opt.timestep
        self.steps_per_control = int(round(steps))
        if self.steps_per_control < 1 or not np.isclose(
            steps, self.steps_per_control, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "control_period_s must be an integer multiple of the MuJoCo timestep"
            )

        self.last_command_was_safe = True
        self.last_applied_effort_nm = np.zeros(JOINT_COUNT, dtype=np.float64)
        self._closed = False
        self.reset()

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model does not contain {name!r}")
        return object_id

    @property
    def joint_ranges(self) -> np.ndarray:
        """Returns a copy of the six joint ranges in radians."""
        return self._joint_ranges.copy()

    @property
    def effort_limits(self) -> np.ndarray:
        """Returns the absolute torque limits in newton-metres."""
        return np.max(np.abs(self._control_limits), axis=1)

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    def reset(
        self,
        position_rad: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        """Resets simulation state and applies zero torque."""
        self._ensure_open()
        if position_rad is None:
            mujoco.mj_resetDataKeyframe(
                self.model, self.data, self._power_on_key_id
            )
        else:
            mujoco.mj_resetData(self.model, self.data)
            desired = np.asarray(position_rad, dtype=np.float64).reshape(-1)
            if desired.shape != (JOINT_COUNT,):
                raise ValueError(f"position_rad must contain {JOINT_COUNT} values")
            if not np.all(np.isfinite(desired)):
                raise ValueError("position_rad must contain only finite values")
            desired = np.clip(
                desired, self._joint_ranges[:, 0], self._joint_ranges[:, 1]
            )
            self.data.qpos[self._joint_qpos_adr] = desired

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_command_was_safe = True
        self.last_applied_effort_nm.fill(0.0)

    def read_feedback(self) -> JointFeedback:
        """Reads joint position, velocity, and applied actuator effort."""
        self._ensure_open()
        return JointFeedback(
            position_rad=self.data.qpos[self._joint_qpos_adr],
            velocity_rad_s=self.data.qvel[self._joint_dof_adr],
            effort_nm=self.data.actuator_force[self._actuator_ids],
            alive=np.ones(JOINT_COUNT, dtype=np.bool_),
            timestamp_s=self.time_s,
        )

    def step(self, command: TorqueCommand) -> JointFeedback:
        """Applies one control-cycle torque command and advances simulation."""
        self._ensure_open()
        requested = np.asarray(command.effort_nm, dtype=np.float64)
        safe = bool(command.enabled) and np.all(np.isfinite(requested))
        if safe:
            ctrl = np.clip(
                requested,
                self._control_limits[:, 0],
                self._control_limits[:, 1],
            )
        else:
            ctrl = np.zeros(JOINT_COUNT, dtype=np.float64)

        self.data.ctrl[self._actuator_ids] = ctrl
        self.last_command_was_safe = safe
        self.last_applied_effort_nm = ctrl.copy()
        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)
        return self.read_feedback()

    def tcp_position(self) -> np.ndarray:
        """Returns the current TCP site position in the world frame."""
        self._ensure_open()
        site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        return self.data.site_xpos[site_id].copy()

    def tcp_pose(self) -> np.ndarray:
        """Returns the current TCP site pose as a 4x4 world-frame matrix."""
        self._ensure_open()
        site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[site_id]
        return pose

    def close(self) -> None:
        """Closes the backend and prevents further state access."""
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The MuJoCo backend has been closed")
