"""Pinocchio generalized gravity for JGarm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio

from jgarm_control.types import JOINT_COUNT, vector6


class PinocchioDynamics:
    def __init__(self, urdf_path: str | Path) -> None:
        self.model = pinocchio.buildModelFromUrdf(str(Path(urdf_path).resolve()))
        if self.model.nq != JOINT_COUNT or self.model.nv != JOINT_COUNT:
            raise ValueError("Pinocchio model must be fixed-base with six joints")
        self.data = self.model.createData()
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

    def gravity_torque(self, q) -> np.ndarray:
        joints = vector6(q, "q")
        torque = pinocchio.computeGeneralizedGravity(
            self.model, self.data, joints
        )
        return np.asarray(torque, dtype=np.float64)
