"""Unit tests for the ROS 2 controller math dependencies."""

from pathlib import Path
import unittest

from ament_index_python.packages import get_package_share_directory
import numpy as np

from jgarm_control.controller import PDGains, PDGravityController
from jgarm_control.dynamics import PinocchioDynamics
from jgarm_control.hardware_profile import EngineerHardwareProfile
from jgarm_control.kinematics import Kinematics
from jgarm_control.types import JointFeedback, POWER_ON_POSITION_RAD


class ControlMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.urdf_path = (
            Path(get_package_share_directory("jgarm_description"))
            / "urdf"
            / "JGarm.urdf"
        )

    def test_kdl_forward_and_inverse(self) -> None:
        kinematics = Kinematics(self.urdf_path)
        joints = np.array([0.1, -0.8, 0.8, 0.1, 0.3, -0.1])
        target = kinematics.forward(joints)
        result = kinematics.inverse(target, np.zeros(6))
        self.assertTrue(result.success)
        self.assertLess(result.position_error_m, 1e-5)
        self.assertLess(result.rotation_error_rad, 1e-4)

    def test_pinocchio_gravity_is_finite(self) -> None:
        dynamics = PinocchioDynamics(self.urdf_path)
        gravity = dynamics.gravity_torque(np.zeros(6))
        self.assertEqual(gravity.shape, (6,))
        self.assertTrue(np.all(np.isfinite(gravity)))
        self.assertGreater(abs(gravity[1]), 0.1)

    def test_power_on_pose_matches_urdf_limits(self) -> None:
        kinematics = Kinematics(self.urdf_path)
        power_on = np.asarray(POWER_ON_POSITION_RAD)
        limits = kinematics.joint_limits_rad

        self.assertTrue(np.all(power_on >= limits[:, 0]))
        self.assertTrue(np.all(power_on <= limits[:, 1]))

    def test_pd_gravity_formula(self) -> None:
        gravity = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        controller = PDGravityController(
            PDGains(np.ones(6), 0.5 * np.ones(6)),
            np.full(6, 20.0),
            lambda _: gravity,
        )
        feedback = JointFeedback(
            np.zeros(6),
            np.ones(6),
            np.zeros(6),
            np.ones(6, dtype=bool),
        )
        command = controller.compute(feedback, 0.1 * np.ones(6))
        np.testing.assert_allclose(command.effort_nm, gravity - 0.4)

    def test_pd_runtime_gain_and_gravity_updates(self) -> None:
        gravity = np.ones(6)
        controller = PDGravityController(
            PDGains(np.zeros(6), np.zeros(6)),
            np.full(6, 20.0),
            lambda _: gravity,
        )
        feedback = JointFeedback(
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            np.ones(6, dtype=bool),
        )

        controller.set_gains(PDGains(np.ones(6), np.zeros(6)))
        controller.set_gravity_enabled(False)
        command = controller.compute(feedback, np.full(6, 0.5))
        np.testing.assert_allclose(command.effort_nm, np.full(6, 0.5))
        np.testing.assert_allclose(controller.last_gravity_effort_nm, 0.0)

    def test_safe_stop_clears_debug_torques(self) -> None:
        controller = PDGravityController(
            PDGains(np.ones(6), np.ones(6)),
            np.full(6, 20.0),
            lambda _: np.ones(6),
        )
        feedback = JointFeedback(
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            np.ones(6, dtype=bool),
        )
        controller.compute(feedback, np.ones(6))
        command = controller.safe_stop()

        self.assertFalse(command.enabled)
        np.testing.assert_allclose(command.effort_nm, 0.0)
        np.testing.assert_allclose(controller.last_pd_effort_nm, 0.0)
        np.testing.assert_allclose(controller.last_gravity_effort_nm, 0.0)

    def test_non_finite_gain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PDGains(np.full(6, np.inf), np.zeros(6))

    def test_hardware_profile_controller_combines_gravity_and_friction(
        self,
    ) -> None:
        profile = EngineerHardwareProfile(
            profile_id="test",
            motor_types=("DM4310",) * 6,
            joint_gear_ratio=np.ones(6),
            effort_limits_nm=np.full(6, 20.0),
            command_rate_hz=500.0,
            state_rate_hz=100.0,
            joint_viscous_damping_nm_s_rad=np.full(6, 0.05),
            joint_armature_kg_m2=np.full(6, 0.001),
            actuator_time_constant_s=np.full(6, 0.005),
            gravity_j5_config_weight=0.35,
            gravity_j5_conservative_coupling=True,
            gravity_p_nm=np.zeros(6),
            gravity_q_nm=np.zeros(6),
            gravity_bias_nm=np.ones(6),
            friction_torque_nm=np.ones(6),
            friction_velocity_scale_rad_s=np.ones(6),
            breakaway_torque_nm=np.zeros(6),
            breakaway_error_scale_rad=0.01,
            breakaway_target_velocity_threshold_rad_s=0.02,
            breakaway_actual_velocity_threshold_rad_s=0.03,
        )
        controller = PDGravityController(
            PDGains(np.zeros(6), np.zeros(6)),
            profile.effort_limits_nm,
            hardware_profile=profile,
        )
        feedback = JointFeedback(
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            np.ones(6, dtype=bool),
        )

        command = controller.compute(
            feedback,
            np.zeros(6),
            np.full(6, 0.2),
        )

        np.testing.assert_allclose(
            command.effort_nm,
            np.full(6, 1.0 + np.tanh(0.2)),
        )


if __name__ == "__main__":
    unittest.main()
