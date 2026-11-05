"""PD and gravity compensation controller tests."""

from __future__ import annotations

import numpy as np
import pytest

from jgarm_sim import JointFeedback, MuJoCoArmBackend
from jgarm_sim.controllers import PDGains, PDGravityController
from jgarm_sim.dynamics import PinocchioDynamics


def make_feedback(
    position: np.ndarray,
    velocity: np.ndarray | None = None,
    alive: np.ndarray | None = None,
) -> JointFeedback:
    return JointFeedback(
        position_rad=position,
        velocity_rad_s=np.zeros(6) if velocity is None else velocity,
        effort_nm=np.zeros(6),
        alive=np.ones(6, dtype=bool) if alive is None else alive,
        timestamp_s=0.0,
    )


def test_pd_gravity_formula() -> None:
    gains = PDGains(np.ones(6), 0.5 * np.ones(6))
    gravity = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    controller = PDGravityController(gains, np.full(6, 20.0), lambda _: gravity)
    feedback = make_feedback(np.zeros(6), np.ones(6))

    command = controller.compute(feedback, 0.1 * np.ones(6))

    np.testing.assert_allclose(command.effort_nm, gravity - 0.4)
    np.testing.assert_allclose(controller.last_gravity_effort_nm, gravity)


def test_controller_clips_to_effort_limits() -> None:
    gains = PDGains(np.full(6, 100.0), np.zeros(6))
    limits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    controller = PDGravityController(gains, limits, lambda _: np.zeros(6))

    command = controller.compute(
        make_feedback(np.zeros(6)), np.ones(6)
    )

    np.testing.assert_allclose(command.effort_nm, limits)
    assert command.enabled


def test_controller_rejects_inactive_feedback() -> None:
    gains = PDGains(np.ones(6), np.ones(6))
    controller = PDGravityController(gains, np.ones(6), lambda _: np.zeros(6))
    alive = np.ones(6, dtype=bool)
    alive[2] = False

    command = controller.compute(
        make_feedback(np.zeros(6), alive=alive), np.ones(6)
    )

    assert not command.enabled
    np.testing.assert_array_equal(command.effort_nm, np.zeros(6))


def test_negative_pd_gain_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PDGains(-np.ones(6), np.ones(6))


def test_pd_gravity_holds_pose_in_mujoco() -> None:
    backend = MuJoCoArmBackend()
    dynamics = PinocchioDynamics()
    target = np.array([0.1, -0.8, 0.8, 0.1, 0.3, -0.1])
    gains = PDGains(
        kp=np.array([60.0, 100.0, 100.0, 40.0, 20.0, 10.0]),
        kd=np.array([2.0, 4.0, 4.0, 1.5, 1.0, 0.5]),
    )
    controller = PDGravityController(
        gains,
        backend.effort_limits,
        dynamics.gravity_torque,
    )
    backend.reset(target)

    for _ in range(5000):
        feedback = backend.read_feedback()
        command = controller.compute(feedback, target)
        feedback = backend.step(command)

    drift = np.max(np.abs(feedback.position_rad - target))
    assert drift < 0.02
