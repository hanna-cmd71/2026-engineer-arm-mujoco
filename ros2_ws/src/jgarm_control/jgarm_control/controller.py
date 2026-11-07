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
from jgarm_control.hardware_profile import EngineerHardwareProfile


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
        gravity_torque: Callable[[np.ndarray], np.ndarray] | None = None,
        gravity_enabled: bool = True,
        hardware_profile: EngineerHardwareProfile | None = None,
    ) -> None:
        self.gains = gains
        self.effort_limits_nm = np.abs(vector6(effort_limits_nm, "limits"))
        self.hardware_profile = hardware_profile
        self.gravity_torque = (
            hardware_profile.gravity_torque
            if hardware_profile is not None
            else gravity_torque
        )
        if self.gravity_torque is None:
            raise ValueError("gravity_torque or hardware_profile is required")
        self.gravity_enabled = bool(gravity_enabled)
        self.last_pd_effort_nm = np.zeros(JOINT_COUNT)
        self.last_gravity_effort_nm = np.zeros(JOINT_COUNT)
        self.last_friction_effort_nm = np.zeros(JOINT_COUNT)
        self.last_breakaway_effort_nm = np.zeros(JOINT_COUNT)

    def set_gains(self, gains: PDGains) -> None:
        self.gains = gains

    def set_gravity_enabled(self, enabled: bool) -> None:
        self.gravity_enabled = bool(enabled)

    def safe_stop(self) -> TorqueCommand:
        self.last_pd_effort_nm = np.zeros(JOINT_COUNT)
        self.last_gravity_effort_nm = np.zeros(JOINT_COUNT)
        self.last_friction_effort_nm = np.zeros(JOINT_COUNT)
        self.last_breakaway_effort_nm = np.zeros(JOINT_COUNT)
        return TorqueCommand(np.zeros(JOINT_COUNT), enabled=False)

    def compute(
        self,
        feedback: JointFeedback,
        target_position_rad,
        target_velocity_rad_s=None,
    ) -> TorqueCommand:
        if not np.all(feedback.alive):
            return self.safe_stop()
        target_position = vector6(target_position_rad, "target position")
        target_velocity = (
            np.zeros(JOINT_COUNT)
            if target_velocity_rad_s is None
            else vector6(target_velocity_rad_s, "target velocity")
        )
        position_error = target_position - feedback.position_rad
        pd_effort = (
            self.gains.kp * position_error
            + self.gains.kd * (target_velocity - feedback.velocity_rad_s)
        )
        gravity_effort = (
            self.gravity_torque(feedback.position_rad)
            if self.gravity_enabled
            else np.zeros(JOINT_COUNT)
        )
        if self.hardware_profile is None:
            friction_effort = np.zeros(JOINT_COUNT)
            breakaway_effort = np.zeros(JOINT_COUNT)
        else:
            friction_effort = self.hardware_profile.friction_compensation(
                target_velocity
            )
            breakaway_effort = (
                self.hardware_profile.breakaway_compensation(
                    position_error,
                    target_velocity,
                    feedback.velocity_rad_s,
                )
            )
        effort = pd_effort + gravity_effort + friction_effort + breakaway_effort
        if not np.all(np.isfinite(effort)):
            return self.safe_stop()
        self.last_pd_effort_nm = pd_effort
        self.last_gravity_effort_nm = gravity_effort
        self.last_friction_effort_nm = friction_effort
        self.last_breakaway_effort_nm = breakaway_effort
        return TorqueCommand(
            np.clip(effort, -self.effort_limits_nm, self.effort_limits_nm)
        )
