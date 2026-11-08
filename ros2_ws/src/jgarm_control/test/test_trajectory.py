"""Joint trajectory sampling tests."""

from __future__ import annotations

import unittest

import numpy as np

from jgarm_control.quintic import make_quintic_trajectory
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
        np.testing.assert_allclose(
            middle_position,
            0.5 * trajectory.positions_rad[-1],
        )
        self.assertTrue(np.all(np.isfinite(middle_velocity)))

    def test_non_monotonic_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JointTrajectory(
                np.array([0.0, 0.0]),
                np.zeros((2, 6)),
                np.zeros((2, 6)),
            )

    def test_quintic_trajectory_is_rest_to_rest(self) -> None:
        start = np.zeros(6)
        target = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.2])
        trajectory = make_quintic_trajectory(start, target, 2.0, 201)

        np.testing.assert_allclose(trajectory.positions_rad[0], start)
        np.testing.assert_allclose(trajectory.positions_rad[-1], target)
        np.testing.assert_allclose(trajectory.velocities_rad_s[0], 0.0)
        np.testing.assert_allclose(trajectory.velocities_rad_s[-1], 0.0)
        self.assertTrue(
            np.all(np.diff(trajectory.positions_rad[:, 0]) >= -1e-12)
        )

    def test_quintic_trajectory_rejects_invalid_sample_count(self) -> None:
        with self.assertRaises(TypeError):
            make_quintic_trajectory(
                np.zeros(6),
                np.ones(6),
                1.0,
                sample_count=2.5,
            )

    def test_hermite_acceleration_matches_finite_difference(self) -> None:
        trajectory = JointTrajectory(
            np.array([0.0, 1.0]),
            np.array(
                [[0.0] * 6, [0.4, -0.3, 0.2, -0.1, 0.3, -0.2]]
            ),
            np.array(
                [
                    [0.1, 0.0, -0.1, 0.0, 0.1, 0.0],
                    [0.0, 0.2, 0.0, -0.2, 0.0, 0.1],
                ]
            ),
        )
        time = 0.37
        step = 1e-6
        _, _, acceleration = trajectory.sample_with_acceleration(time)
        _, velocity_before = trajectory.sample(time - step)
        _, velocity_after = trajectory.sample(time + step)
        numerical = (velocity_after - velocity_before) / (2.0 * step)
        np.testing.assert_allclose(acceleration, numerical, atol=1e-5)

    def test_acceleration_is_zero_at_endpoints(self) -> None:
        trajectory = JointTrajectory(
            np.array([0.0, 1.0]),
            np.zeros((2, 6)),
            np.zeros((2, 6)),
        )
        _, _, start_acceleration = trajectory.sample_with_acceleration(0.0)
        _, _, end_acceleration = trajectory.sample_with_acceleration(1.0)
        np.testing.assert_allclose(start_acceleration, 0.0)
        np.testing.assert_allclose(end_acceleration, 0.0)


if __name__ == "__main__":
    unittest.main()
