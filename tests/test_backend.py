"""Hardware-boundary backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from jgarm_sim import MuJoCoArmBackend, TorqueCommand


@pytest.mark.parametrize("joint_index", range(6))
def test_each_joint_responds_to_its_torque_command(joint_index: int) -> None:
    backend = MuJoCoArmBackend()
    backend.model.opt.gravity[:] = 0.0
    backend.reset()

    command = np.zeros(6)
    command[joint_index] = 1.0
    for _ in range(100):
        feedback = backend.step(TorqueCommand(command))

    assert feedback.position_rad[joint_index] > 0.005
    assert np.all(np.isfinite(feedback.position_rad))
    assert np.all(np.isfinite(feedback.velocity_rad_s))


def test_torque_command_is_clipped_to_actuator_limit() -> None:
    backend = MuJoCoArmBackend()
    backend.step(TorqueCommand(np.full(6, 1000.0)))
    np.testing.assert_allclose(
        backend.last_applied_effort_nm, backend.effort_limits
    )
    assert backend.last_command_was_safe


def test_invalid_torque_command_falls_back_to_zero() -> None:
    backend = MuJoCoArmBackend()
    command = TorqueCommand(np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0]))
    feedback = backend.step(command)

    assert not backend.last_command_was_safe
    np.testing.assert_array_equal(backend.last_applied_effort_nm, np.zeros(6))
    assert np.all(np.isfinite(feedback.position_rad))
    assert np.all(np.isfinite(feedback.velocity_rad_s))


def test_reset_restores_requested_position() -> None:
    backend = MuJoCoArmBackend()
    target = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
    backend.reset(target)
    np.testing.assert_allclose(backend.read_feedback().position_rad, target)
    assert backend.time_s == 0.0


def test_type_validation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="6 values"):
        TorqueCommand(np.zeros(5))


def test_backend_rejects_use_after_close() -> None:
    backend = MuJoCoArmBackend()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.read_feedback()
