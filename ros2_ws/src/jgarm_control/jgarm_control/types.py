"""Joint-space data structures used by the ROS 2 controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
JOINT_COUNT = len(JOINT_NAMES)
POWER_ON_POSITION_RAD = (0.0, -1.3265, 1.35, 0.0, 0.0, 0.0)
URDF_ZERO_POSITION_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def vector6(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must contain {JOINT_COUNT} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


@dataclass(frozen=True)
class JointFeedback:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    effort_nm: np.ndarray
    alive: np.ndarray


@dataclass(frozen=True)
class TorqueCommand:
    effort_nm: np.ndarray
    enabled: bool = True
