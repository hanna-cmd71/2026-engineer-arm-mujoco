"""Pinocchio rigid-body dynamics used by the Python controllers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio

from jgarm_sim.types import JOINT_COUNT


DEFAULT_URDF_PATH = Path(__file__).resolve().parents[2] / "JGarm.urdf"


class PinocchioDynamics:
    """Fixed-base Pinocchio model built from the same JGarm URDF."""

    def __init__(self, urdf_path: str | Path = DEFAULT_URDF_PATH) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.model = pinocchio.buildModelFromUrdf(str(self.urdf_path))
        if self.model.nq != JOINT_COUNT or self.model.nv != JOINT_COUNT:
            raise ValueError(
                "Pinocchio must load JGarm as a fixed-base six-joint model"
            )
        self.data = self.model.createData()
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

    @staticmethod
    def _coerce_joints(q: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        if values.shape != (JOINT_COUNT,):
            raise ValueError(f"q must contain {JOINT_COUNT} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("q must contain only finite values")
        return values.copy()

    def gravity_torque(
        self, q: np.ndarray | list[float] | tuple[float, ...]
    ) -> np.ndarray:
        """Returns the generalized gravity torque g(q) in N*m."""
        joints = self._coerce_joints(q)
        torque = pinocchio.computeGeneralizedGravity(
            self.model, self.data, joints
        )
        return np.asarray(torque, dtype=np.float64).reshape(JOINT_COUNT).copy()

