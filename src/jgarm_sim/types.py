"""Hardware-boundary data types shared by simulation and future adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
JOINT_COUNT = len(JOINT_NAMES)


def _as_vector(values: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    """Converts input to a detached six-element float vector."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must contain {JOINT_COUNT} values")
    return vector.copy()


@dataclass(frozen=True)
class JointFeedback:
    """Joint-space feedback received from a hardware backend.

    Coordinates and units match the planned jRMCS arm hardware boundary:
    radians, radians per second, newton-metres, and seconds.
    """

    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    effort_nm: np.ndarray
    alive: np.ndarray
    timestamp_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_rad", _as_vector(self.position_rad, "position_rad")
        )
        object.__setattr__(
            self, "velocity_rad_s", _as_vector(self.velocity_rad_s, "velocity_rad_s")
        )
        object.__setattr__(self, "effort_nm", _as_vector(self.effort_nm, "effort_nm"))
        alive = np.asarray(self.alive, dtype=np.bool_).reshape(-1)
        if alive.shape != (JOINT_COUNT,):
            raise ValueError(f"alive must contain {JOINT_COUNT} values")
        object.__setattr__(self, "alive", alive.copy())
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")


@dataclass(frozen=True)
class TorqueCommand:
    """Joint torque command written to a hardware backend."""

    effort_nm: np.ndarray
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort_nm", _as_vector(self.effort_nm, "effort_nm"))

