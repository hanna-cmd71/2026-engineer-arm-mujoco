"""Joint trajectory sampling tests."""

from __future__ import annotations

import unittest

import numpy as np

from jgarm_control.trajectory import JointTrajectory


class JointTrajectoryTest(unittest.TestCase):
    def test_hermite_interpolation_matches_endpoints(self) -> None:
        trajectory = JointTrajectory(
            times_s=np.array([0.0, 1.0]),
            positions_rad=np.array(
                [[0.0] * 6, [1.0, -1.0, 0.5, 0.2, -0.2, 0.1]]
            ),
            velocities_rad_s=np.zeros((2, 6)),
        )
        start_position, start_velocity = trajectory.sample(0.0)
        end_position, end_velocity = trajectory.sample(1.0)
        middle_position, middle_velocity = trajectory.sample(0.5)

        np.testing.assert_allclose(start_position, 0.0)
        np.testing.assert_allclose(start_velocity, 0.0)
        np.testing.assert_allclose(end_position, trajectory.positions_rad[-1])
        np.testing.assert_allclose(end_velocity, 0.0)
        np.testing.assert_allclose(middle_position, 0.5 * trajectory.positions_rad[-1])
        self.assertTrue(np.all(np.isfinite(middle_velocity)))

    def test_non_monotonic_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JointTrajectory(
                np.array([0.0, 0.0]),
                np.zeros((2, 6)),
                np.zeros((2, 6)),
            )


if __name__ == "__main__":
    unittest.main()
