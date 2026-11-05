"""Joint trajectory validation and C1-continuous sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trajectory_msgs.msg import JointTrajectory

from jgarm_control.types import JOINT_COUNT, JOINT_NAMES


@dataclass(frozen=True)
class JointTrajectory:
    times_s: np.ndarray
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64).reshape(-1)
        positions = np.asarray(self.positions_rad, dtype=np.float64)
        velocities = np.asarray(self.velocities_rad_s, dtype=np.float64)
        if times.size < 1:
            raise ValueError("Trajectory must contain at least one point")
        if positions.shape != (times.size, JOINT_COUNT):
            raise ValueError("Trajectory positions have an invalid shape")
        if velocities.shape != (times.size, JOINT_COUNT):
            raise ValueError("Trajectory velocities have an invalid shape")
        if not (
            np.all(np.isfinite(times))
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
        ):
            raise ValueError("Trajectory contains non-finite values")
        if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("Trajectory times must be strictly increasing")
        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(self, "positions_rad", positions.copy())
        object.__setattr__(self, "velocities_rad_s", velocities.copy())

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    @classmethod
    def from_message(cls, message: JointTrajectory) -> "JointTrajectory":
        if not message.joint_names:
            raise ValueError("Trajectory has no joint names")
        if not message.points:
            raise ValueError("Trajectory has no points")

        source_index = {
            name: index for index, name in enumerate(message.joint_names)
        }
        missing = [name for name in JOINT_NAMES if name not in source_index]
        if missing:
            raise ValueError(f"Trajectory is missing joints: {missing}")

        times = np.array(
            [
                point.time_from_start.sec
                + point.time_from_start.nanosec * 1e-9
                for point in message.points
            ],
            dtype=np.float64,
        )
        positions = np.zeros((len(message.points), JOINT_COUNT))
        velocities = np.zeros((len(message.points), JOINT_COUNT))
        for point_index, point in enumerate(message.points):
            if len(point.positions) < len(message.joint_names):
                raise ValueError("Trajectory point is missing joint positions")
            for target_index, name in enumerate(JOINT_NAMES):
                source = source_index[name]
                positions[point_index, target_index] = point.positions[source]
                if point.velocities:
                    if len(point.velocities) < len(message.joint_names):
                        raise ValueError(
                            "Trajectory point is missing joint velocities"
                        )
                    velocities[point_index, target_index] = (
                        point.velocities[source]
                    )

        if (
            not np.all(np.isfinite(times))
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(velocities))
        ):
            raise ValueError("Trajectory contains non-finite values")
        if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("Trajectory times must be strictly increasing")
        return cls(times, positions, velocities)

    def sample(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Samples position and velocity with cubic Hermite interpolation."""
        if elapsed_s <= self.times_s[0]:
            return self.positions_rad[0].copy(), self.velocities_rad_s[0].copy()
        if elapsed_s >= self.times_s[-1]:
            return self.positions_rad[-1].copy(), np.zeros(JOINT_COUNT)

        segment = int(np.searchsorted(self.times_s, elapsed_s, side="right") - 1)
        start_time = self.times_s[segment]
        end_time = self.times_s[segment + 1]
        duration = end_time - start_time
        ratio = (elapsed_s - start_time) / duration
        ratio2 = ratio * ratio
        ratio3 = ratio2 * ratio

        h00 = 2.0 * ratio3 - 3.0 * ratio2 + 1.0
        h10 = ratio3 - 2.0 * ratio2 + ratio
        h01 = -2.0 * ratio3 + 3.0 * ratio2
        h11 = ratio3 - ratio2
        dh00 = (6.0 * ratio2 - 6.0 * ratio) / duration
        dh10 = 3.0 * ratio2 - 4.0 * ratio + 1.0
        dh01 = (-6.0 * ratio2 + 6.0 * ratio) / duration
        dh11 = 3.0 * ratio2 - 2.0 * ratio

        p0 = self.positions_rad[segment]
        p1 = self.positions_rad[segment + 1]
        v0 = self.velocities_rad_s[segment]
        v1 = self.velocities_rad_s[segment + 1]
        position = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1
        velocity = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1
        return position, velocity
