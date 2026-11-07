"""Hardware model parameters extracted from the reATRM Engineer layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jgarm_control.types import JOINT_COUNT, vector6


@dataclass(frozen=True)
class EngineerHardwareProfile:
    profile_id: str
    motor_types: tuple[str, ...]
    joint_gear_ratio: np.ndarray
    effort_limits_nm: np.ndarray
    command_rate_hz: float
    state_rate_hz: float
    joint_viscous_damping_nm_s_rad: np.ndarray
    joint_armature_kg_m2: np.ndarray
    actuator_time_constant_s: np.ndarray
    gravity_j5_config_weight: float
    gravity_j5_conservative_coupling: bool
    gravity_p_nm: np.ndarray
    gravity_q_nm: np.ndarray
    gravity_bias_nm: np.ndarray
    friction_torque_nm: np.ndarray
    friction_velocity_scale_rad_s: np.ndarray
    breakaway_torque_nm: np.ndarray
    breakaway_error_scale_rad: float
    breakaway_target_velocity_threshold_rad_s: float
    breakaway_actual_velocity_threshold_rad_s: float
    plant_breakaway_velocity_scale_rad_s: float = 0.02
    plant_gravity_enabled: bool = False
    plant_friction_enabled: bool = True
    plant_breakaway_enabled: bool = True

    def __post_init__(self) -> None:
        if len(self.motor_types) != JOINT_COUNT:
            raise ValueError("motor_types must contain six values")
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")
        for name in (
            "joint_gear_ratio",
            "effort_limits_nm",
            "gravity_p_nm",
            "gravity_q_nm",
            "gravity_bias_nm",
            "joint_viscous_damping_nm_s_rad",
            "joint_armature_kg_m2",
            "actuator_time_constant_s",
            "friction_torque_nm",
            "friction_velocity_scale_rad_s",
            "breakaway_torque_nm",
        ):
            object.__setattr__(self, name, vector6(getattr(self, name), name))
        if np.any(self.joint_gear_ratio <= 0.0):
            raise ValueError("joint_gear_ratio must be positive")
        if np.any(self.effort_limits_nm <= 0.0):
            raise ValueError("effort_limits_nm must be positive")
        if np.any(self.friction_velocity_scale_rad_s <= 0.0):
            raise ValueError("friction velocity scales must be positive")
        for name in (
            "command_rate_hz",
            "state_rate_hz",
            "gravity_j5_config_weight",
            "breakaway_error_scale_rad",
            "breakaway_target_velocity_threshold_rad_s",
            "breakaway_actual_velocity_threshold_rad_s",
            "plant_breakaway_velocity_scale_rad_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if np.any(self.friction_torque_nm < 0.0):
            raise ValueError("friction_torque_nm must be non-negative")
        if np.any(self.breakaway_torque_nm < 0.0):
            raise ValueError("breakaway_torque_nm must be non-negative")
        if np.any(self.joint_viscous_damping_nm_s_rad < 0.0):
            raise ValueError(
                "joint_viscous_damping_nm_s_rad must be non-negative"
            )
        if np.any(self.joint_armature_kg_m2 < 0.0):
            raise ValueError("joint_armature_kg_m2 must be non-negative")
        if np.any(self.actuator_time_constant_s <= 0.0):
            raise ValueError("actuator_time_constant_s must be positive")

    def gravity_torque(self, position_rad) -> np.ndarray:
        """Returns the identified joint-side gravity load."""
        position = vector6(position_rad, "position_rad")
        theta_1 = position[1]
        theta_2 = theta_1 + position[2]
        theta_3 = theta_2 - position[4]
        phi_5 = (
            -position[4]
            if self.gravity_j5_conservative_coupling
            else self.gravity_j5_config_weight * theta_2 - position[4]
        )

        p = self.gravity_p_nm
        q = self.gravity_q_nm
        bias = self.gravity_bias_nm
        result = bias.copy()
        result[1] += (
            p[1] * np.cos(theta_1)
            + q[1] * np.sin(theta_1)
            + p[2] * np.cos(theta_2)
            + q[2] * np.sin(theta_2)
            + p[3] * np.cos(theta_3)
            + q[3] * np.sin(theta_3)
        )
        result[2] += (
            p[2] * np.cos(theta_2)
            + q[2] * np.sin(theta_2)
            + p[3] * np.cos(theta_3)
            + q[3] * np.sin(theta_3)
        )
        result[4] += p[4] * np.cos(phi_5) + q[4] * np.sin(phi_5)
        if self.gravity_j5_conservative_coupling:
            result[4] += p[3]
            result[4] -= p[3] * np.cos(theta_3) + q[3] * np.sin(theta_3)
        return result

    def friction_compensation(self, target_velocity_rad_s) -> np.ndarray:
        velocity = vector6(target_velocity_rad_s, "target_velocity_rad_s")
        return self.friction_torque_nm * np.tanh(
            velocity / self.friction_velocity_scale_rad_s
        )

    def breakaway_compensation(
        self,
        position_error_rad,
        target_velocity_rad_s,
        actual_velocity_rad_s,
    ) -> np.ndarray:
        error = vector6(position_error_rad, "position_error_rad")
        target_velocity = vector6(
            target_velocity_rad_s, "target_velocity_rad_s"
        )
        actual_velocity = vector6(
            actual_velocity_rad_s, "actual_velocity_rad_s"
        )
        active = (
            (
                np.abs(target_velocity)
                < self.breakaway_target_velocity_threshold_rad_s
            )
            & (
                np.abs(actual_velocity)
                < self.breakaway_actual_velocity_threshold_rad_s
            )
        )
        return np.where(
            active,
            self.breakaway_torque_nm
            * np.tanh(error / self.breakaway_error_scale_rad),
            0.0,
        )

    def passive_friction(self, velocity_rad_s) -> np.ndarray:
        """Returns load magnitude; the plant applies the opposite sign."""
        velocity = vector6(velocity_rad_s, "velocity_rad_s")
        friction = self.friction_torque_nm * np.tanh(
            velocity / self.friction_velocity_scale_rad_s
        )
        if self.plant_breakaway_enabled:
            friction += self.breakaway_torque_nm * np.tanh(
                velocity / self.plant_breakaway_velocity_scale_rad_s
            )
        if not self.plant_friction_enabled:
            return np.zeros(JOINT_COUNT)
        return friction
