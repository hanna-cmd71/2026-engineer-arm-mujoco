"""Tests for the reATRM Engineer hardware profile."""

import unittest

import numpy as np

from jgarm_control.hardware_profile import EngineerHardwareProfile


def _profile() -> EngineerHardwareProfile:
    return EngineerHardwareProfile(
        profile_id="reatrm_engineer",
        motor_types=(
            "DM4310",
            "LK_MF7015",
            "LK_MF7015",
            "DM4310",
            "DM4310",
            "DM4310",
        ),
        joint_gear_ratio=[1.0, 50.0, 50.0, 1.0, 1.5, 1.0],
        effort_limits_nm=[10.0, 63.0, 35.0, 10.0, 6.0, 3.0],
        command_rate_hz=500.0,
        state_rate_hz=111.111111,
        joint_viscous_damping_nm_s_rad=[0.05] * 6,
        joint_armature_kg_m2=[0.001] * 6,
        actuator_time_constant_s=[0.005] * 6,
        gravity_j5_config_weight=0.35,
        gravity_j5_conservative_coupling=True,
        gravity_p_nm=[0.0, 20.284, -15.112, -0.579, 16.368, 0.0],
        gravity_q_nm=[0.0, -60.824, 2.680, 0.380, -0.715, 0.0],
        gravity_bias_nm=[0.0, -22.244, 1.534, 0.0, -13.608, 0.0],
        friction_torque_nm=[0.0, 11.6, 7.4, 0.0, 0.75, 0.0],
        friction_velocity_scale_rad_s=[0.2] * 6,
        breakaway_torque_nm=[0.0, 11.25, 10.0, 0.0, 0.4, 0.2],
        breakaway_error_scale_rad=0.01,
        breakaway_target_velocity_threshold_rad_s=0.02,
        breakaway_actual_velocity_threshold_rad_s=0.03,
        plant_breakaway_velocity_scale_rad_s=0.02,
    )


class EngineerHardwareProfileTest(unittest.TestCase):
    def test_gravity_matches_reatrm_identified_model(self) -> None:
        profile = _profile()
        zero = profile.gravity_torque(np.zeros(6))
        power_on = profile.gravity_torque(
            [0.0, -1.3265, 1.35, 0.0, 0.0, 0.0]
        )

        np.testing.assert_allclose(
            zero,
            [0.0, -17.651, -14.157, 0.0, 2.760, 0.0],
            atol=0.002,
        )
        np.testing.assert_allclose(
            power_on,
            [0.0, 26.065, -14.081, 0.0, 2.754, 0.0],
            atol=0.005,
        )

    def test_friction_and_breakaway_use_identified_torques(self) -> None:
        profile = _profile()
        friction = profile.friction_compensation([0.0, 0.2, -0.2, 0, 0, 0])

        np.testing.assert_allclose(
            friction,
            [0.0, 11.6 * np.tanh(1.0), -7.4 * np.tanh(1.0), 0, 0, 0],
        )
        breakaway = profile.breakaway_compensation(
            [0.0, 0.01, -0.01, 0, 0, 0],
            np.zeros(6),
            np.zeros(6),
        )
        np.testing.assert_allclose(
            breakaway,
            [0.0, 11.25 * np.tanh(1.0), -10.0 * np.tanh(1.0), 0, 0, 0],
        )

    def test_hardware_effort_limits_include_dm_protocol_limit(self) -> None:
        profile = _profile()

        np.testing.assert_allclose(
            profile.effort_limits_nm,
            [10.0, 63.0, 35.0, 10.0, 6.0, 3.0],
        )


if __name__ == "__main__":
    unittest.main()
