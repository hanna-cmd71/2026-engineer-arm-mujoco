"""Pinocchio dynamics tests against MuJoCo."""

from __future__ import annotations

import mujoco
import numpy as np

from jgarm_sim import MuJoCoArmBackend
from jgarm_sim.dynamics import PinocchioDynamics


def test_pinocchio_gravity_matches_mujoco() -> None:
    dynamics = PinocchioDynamics()
    backend = MuJoCoArmBackend()
    scratch = mujoco.MjData(backend.model)
    random = np.random.default_rng(15)

    for _ in range(20):
        joints = random.uniform(
            backend.joint_ranges[:, 0], backend.joint_ranges[:, 1]
        )
        scratch.qpos[:] = joints
        scratch.qvel[:] = 0.0
        scratch.qacc[:] = 0.0
        mujoco.mj_forward(backend.model, scratch)
        np.testing.assert_allclose(
            dynamics.gravity_torque(joints),
            scratch.qfrc_bias[:6],
            atol=5e-6,
        )
