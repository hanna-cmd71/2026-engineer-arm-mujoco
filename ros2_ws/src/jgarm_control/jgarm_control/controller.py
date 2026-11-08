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
from jgarm_control.dynamics import PinocchioDynamics


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


def _gain_scale(values, name: str) -> np.ndarray:
    scale = np.asarray(values, dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.full(JOINT_COUNT, float(scale[0]))
    if scale.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must contain one or six values")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return scale.copy()


class PDGravityController:
    def __init__(
        self,
        gains: PDGains,
        effort_limits_nm,
        gravity_torque: Callable[[np.ndarray], np.ndarray] | None = None,
        gravity_enabled: bool = True,
        hardware_profile: EngineerHardwareProfile | None = None,
        breakaway_enabled: bool = True,
        friction_enabled: bool = True,
    ) -> None:
        self.gains = gains
        self.effort_limits_nm = np.abs(vector6(effort_limits_nm, "limits"))
        self.hardware_profile = hardware_profile
        if gravity_torque is not None:
            self.gravity_torque = gravity_torque
        elif hardware_profile is not None:
            self.gravity_torque = hardware_profile.gravity_torque
        else:
            raise ValueError("gravity_torque or hardware_profile is required")
        self.gravity_enabled = bool(gravity_enabled)
        self.breakaway_enabled = bool(breakaway_enabled)
        self.friction_enabled = bool(friction_enabled)
        self.last_pd_effort_nm = np.zeros(JOINT_COUNT)
        self.last_gravity_effort_nm = np.zeros(JOINT_COUNT)
        self.last_friction_effort_nm = np.zeros(JOINT_COUNT)
        self.last_breakaway_effort_nm = np.zeros(JOINT_COUNT)

    def set_gains(self, gains: PDGains) -> None:
        self.gains = gains

    def set_gravity_enabled(self, enabled: bool) -> None:
        self.gravity_enabled = bool(enabled)

    def set_breakaway_enabled(self, enabled: bool) -> None:
        self.breakaway_enabled = bool(enabled)

    def set_friction_enabled(self, enabled: bool) -> None:
        self.friction_enabled = bool(enabled)

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
        target_acceleration_rad_s2=None,
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
            friction_effort = (
                self.hardware_profile.friction_compensation(
                    target_velocity
                )
                if self.friction_enabled
                else np.zeros(JOINT_COUNT)
            )
            breakaway_effort = (
                self.hardware_profile.breakaway_compensation(
                    position_error,
                    target_velocity,
                    feedback.velocity_rad_s,
                )
                if self.breakaway_enabled
                else np.zeros(JOINT_COUNT)
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


class JointImpedanceController:
    """Inverse-dynamics joint impedance with gravity/friction feedforward."""

    def __init__(
        self,
        gains: PDGains,
        effort_limits_nm,
        dynamics: PinocchioDynamics,
        gravity_enabled: bool = True,
        hardware_profile: EngineerHardwareProfile | None = None,
        breakaway_enabled: bool = True,
        friction_enabled: bool = True,
        kp_scale=1.0,
        kd_scale=1.0,
    ) -> None:
        self.gains = gains
        self.effort_limits_nm = np.abs(vector6(effort_limits_nm, "limits"))
        self.dynamics = dynamics
        self.gravity_enabled = bool(gravity_enabled)
        self.breakaway_enabled = bool(breakaway_enabled)
        self.friction_enabled = bool(friction_enabled)
        self.kp_scale = _gain_scale(kp_scale, "kp_scale")
        self.kd_scale = _gain_scale(kd_scale, "kd_scale")
        self.hardware_profile = hardware_profile
        self.last_pd_effort_nm = np.zeros(JOINT_COUNT)
        self.last_gravity_effort_nm = np.zeros(JOINT_COUNT)
        self.last_friction_effort_nm = np.zeros(JOINT_COUNT)
        self.last_breakaway_effort_nm = np.zeros(JOINT_COUNT)

    def set_gains(self, gains: PDGains) -> None:
        self.gains = gains

    def set_gravity_enabled(self, enabled: bool) -> None:
        self.gravity_enabled = bool(enabled)

    def set_breakaway_enabled(self, enabled: bool) -> None:
        self.breakaway_enabled = bool(enabled)

    def set_friction_enabled(self, enabled: bool) -> None:
        self.friction_enabled = bool(enabled)

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
        target_acceleration_rad_s2=None,
    ) -> TorqueCommand:
        if not np.all(feedback.alive):
            return self.safe_stop()
        target_position = vector6(target_position_rad, "target position")
        target_velocity = (
            np.zeros(JOINT_COUNT)
            if target_velocity_rad_s is None
            else vector6(target_velocity_rad_s, "target velocity")
        )
        target_acceleration = (
            np.zeros(JOINT_COUNT)
            if target_acceleration_rad_s2 is None
            else vector6(
                target_acceleration_rad_s2,
                "target acceleration",
            )
        )
        position_error = target_position - feedback.position_rad
        velocity_error = target_velocity - feedback.velocity_rad_s
        impedance_effort = (
            self.gains.kp * self.kp_scale * position_error
            + self.gains.kd * self.kd_scale * velocity_error
        )
        inertia_effort = self.dynamics.mass_matrix(
            feedback.position_rad
        ) @ target_acceleration
        coriolis_effort = self.dynamics.coriolis_torque(
            feedback.position_rad,
            feedback.velocity_rad_s,
        )
        gravity_effort = (
            self.dynamics.gravity_torque(feedback.position_rad)
            if self.gravity_enabled
            else np.zeros(JOINT_COUNT)
        )
        if self.hardware_profile is None:
            friction_effort = np.zeros(JOINT_COUNT)
            breakaway_effort = np.zeros(JOINT_COUNT)
        else:
            friction_effort = (
                self.hardware_profile.friction_compensation(
                    target_velocity
                )
                if self.friction_enabled
                else np.zeros(JOINT_COUNT)
            )
            breakaway_effort = (
                self.hardware_profile.breakaway_compensation(
                    position_error,
                    target_velocity,
                    feedback.velocity_rad_s,
                )
                if self.breakaway_enabled
                else np.zeros(JOINT_COUNT)
            )
        effort = (
            impedance_effort
            + inertia_effort
            + coriolis_effort
            + gravity_effort
            + friction_effort
            + breakaway_effort
        )
        if not np.all(np.isfinite(effort)):
            return self.safe_stop()
        self.last_pd_effort_nm = impedance_effort
        self.last_gravity_effort_nm = gravity_effort
        self.last_friction_effort_nm = friction_effort
        self.last_breakaway_effort_nm = breakaway_effort
        return TorqueCommand(
            np.clip(effort, -self.effort_limits_nm, self.effort_limits_nm)
        )
