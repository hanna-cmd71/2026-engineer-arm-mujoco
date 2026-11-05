"""Joint-space PD control with Pinocchio gravity feedforward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from jgarm_sim.types import JOINT_COUNT, JointFeedback, TorqueCommand


def _vector(
    values: np.ndarray | list[float] | tuple[float, ...], name: str
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must contain {JOINT_COUNT} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


@dataclass(frozen=True)
class PDGains:
    """Proportional and derivative gains in joint space."""

    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self) -> None:
        kp = _vector(self.kp, "kp")
        kd = _vector(self.kd, "kd")
        if np.any(kp < 0.0) or np.any(kd < 0.0):
            raise ValueError("PD gains must be non-negative")
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "kd", kd)


class PDGravityController:
    """Computes tau = Kp * q_error + Kd * dq_error + gravity(q)."""

    def __init__(
        self,
        gains: PDGains,
        effort_limits_nm: np.ndarray,
        gravity_torque: Callable[[np.ndarray], np.ndarray],
        gravity_enabled: bool = True,
    ) -> None:
        self.gains = gains
        self.effort_limits_nm = np.abs(
            _vector(effort_limits_nm, "effort_limits_nm")
        )
        self.gravity_torque = gravity_torque
        self.gravity_enabled = gravity_enabled
        self.last_pd_effort_nm = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.last_gravity_effort_nm = np.zeros(
            JOINT_COUNT, dtype=np.float64
        )
        self.last_unclipped_effort_nm = np.zeros(
            JOINT_COUNT, dtype=np.float64
        )

    def compute(
        self,
        feedback: JointFeedback,
        target_position_rad: np.ndarray | list[float] | tuple[float, ...],
        target_velocity_rad_s: (
            np.ndarray | list[float] | tuple[float, ...] | None
        ) = None,
        enabled: bool = True,
    ) -> TorqueCommand:
        if not enabled or not np.all(feedback.alive):
            return TorqueCommand(np.zeros(JOINT_COUNT), enabled=False)

        position = feedback.position_rad
        velocity = feedback.velocity_rad_s
        target_position = _vector(target_position_rad, "target_position_rad")
        target_velocity = (
            np.zeros(JOINT_COUNT, dtype=np.float64)
            if target_velocity_rad_s is None
            else _vector(target_velocity_rad_s, "target_velocity_rad_s")
        )
        if not (
            np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))
        ):
            return TorqueCommand(np.zeros(JOINT_COUNT), enabled=False)

        pd_effort = (
            self.gains.kp * (target_position - position)
            + self.gains.kd * (target_velocity - velocity)
        )
        gravity_effort = (
            _vector(self.gravity_torque(position), "gravity_effort")
            if self.gravity_enabled
            else np.zeros(JOINT_COUNT, dtype=np.float64)
        )
        unclipped = pd_effort + gravity_effort
        if not np.all(np.isfinite(unclipped)):
            return TorqueCommand(np.zeros(JOINT_COUNT), enabled=False)

        self.last_pd_effort_nm = pd_effort.copy()
        self.last_gravity_effort_nm = gravity_effort.copy()
        self.last_unclipped_effort_nm = unclipped.copy()
        effort = np.clip(
            unclipped, -self.effort_limits_nm, self.effort_limits_nm
        )
        return TorqueCommand(effort, enabled=True)
