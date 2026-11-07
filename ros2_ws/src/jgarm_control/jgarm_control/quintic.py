"""Quintic joint trajectory generation for interactive targets."""

from __future__ import annotations

import numpy as np

from jgarm_control.trajectory import JointTrajectory
from jgarm_control.types import JOINT_COUNT, vector6


def make_quintic_trajectory(
    start_position_rad,
    target_position_rad,
    duration_s: float,
    sample_count: int = 200,
) -> JointTrajectory:
    """Builds a rest-to-rest quintic trajectory with smooth acceleration."""
    start = vector6(start_position_rad, "start_position_rad")
    target = vector6(target_position_rad, "target_position_rad")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be an integer")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    times = np.linspace(0.0, duration_s, sample_count)
    ratio = times / duration_s
    ratio2 = ratio * ratio
    ratio3 = ratio2 * ratio
    ratio4 = ratio3 * ratio
    ratio5 = ratio4 * ratio
    blend = 10.0 * ratio3 - 15.0 * ratio4 + 6.0 * ratio5
    blend_derivative = (
        30.0 * ratio2 - 60.0 * ratio3 + 30.0 * ratio4
    ) / duration_s

    delta = target - start
    positions = start + blend[:, None] * delta
    velocities = blend_derivative[:, None] * delta
    return JointTrajectory(times, positions, velocities)
