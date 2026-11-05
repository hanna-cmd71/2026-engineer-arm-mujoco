"""Joint PD with Pinocchio gravity feedforward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from jgarm_control.types import (
    JOINT_COUNT,
    JointFeedback,
    TorqueCommand,
    vector6,
)


@dataclass(frozen=True)
class PDGains:
    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self) -> None:
        kp = vector6(self.kp, "kp")
        kd = vector6(self.kd, "kd")
        if np.any(kp < 0.0) or np.any(kd < 0.0):
            raise ValueError("PD gains must be non-negative")
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "kd", kd)


class PDGravityController:
    def __init__(
        self,
        gains: PDGains,
        effort_limits_nm,
        gravity_torque: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.gains = gains
        self.effort_limits_nm = np.abs(vector6(effort_limits_nm, "limits"))
        self.gravity_torque = gravity_torque

    def compute(
        self,
        feedback: JointFeedback,
        target_position_rad,
        target_velocity_rad_s=None,
    ) -> TorqueCommand:
        if not np.all(feedback.alive):
            return TorqueCommand(np.zeros(JOINT_COUNT), enabled=False)
        target_position = vector6(target_position_rad, "target position")
        target_velocity = (
            np.zeros(JOINT_COUNT)
            if target_velocity_rad_s is None
            else vector6(target_velocity_rad_s, "target velocity")
        )
        pd_effort = (
            self.gains.kp * (target_position - feedback.position_rad)
            + self.gains.kd * (target_velocity - feedback.velocity_rad_s)
        )
        effort = pd_effort + self.gravity_torque(feedback.position_rad)
        if not np.all(np.isfinite(effort)):
            return TorqueCommand(np.zeros(JOINT_COUNT), enabled=False)
        return TorqueCommand(
            np.clip(effort, -self.effort_limits_nm, self.effort_limits_nm)
        )
