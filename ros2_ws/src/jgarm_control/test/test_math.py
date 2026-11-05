"""Unit tests for the ROS 2 controller math dependencies."""

from pathlib import Path
import unittest

from ament_index_python.packages import get_package_share_directory
import numpy as np

from jgarm_control.controller import PDGains, PDGravityController
from jgarm_control.dynamics import PinocchioDynamics
from jgarm_control.kinematics import Kinematics
from jgarm_control.types import JointFeedback


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


if __name__ == "__main__":
    unittest.main()
