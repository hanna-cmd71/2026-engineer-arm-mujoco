"""Tests for the physical mass/COM nominal model and its evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from jgarm_sim import MuJoCoArmBackend
from jgarm_sim.controllers import PDGains, PDGravityController
from jgarm_sim.nominal_model import (
    UncertaintyEnvelope,
    load_nominal_model,
)
from jgarm_sim.types import POWER_ON_POSITION_RAD


def test_urdf_mass_and_com_are_loaded() -> None:
    model = load_nominal_model()

    assert model.links["link_1"].mass_kg == pytest.approx(0.60157)
    np.testing.assert_allclose(
        model.links["link_1"].com_m,
        [0.025292, -0.0021277, 0.053006],
    )
    assert model.total_mass_kg == pytest.approx(2.76524, abs=1e-6)


def test_physical_gravity_matches_mujoco() -> None:
    model = load_nominal_model()
    backend = MuJoCoArmBackend()
    random = np.random.default_rng(31)
    for _ in range(20):
        position = random.uniform(
            backend.joint_ranges[:, 0],
            backend.joint_ranges[:, 1],
        )
        backend.reset(position)
        mujoco_gravity = backend.data.qfrc_bias[
            backend._joint_dof_adr
        ]
        np.testing.assert_allclose(
            model.physical_gravity_torque(position),
            mujoco_gravity,
            atol=1e-5,
        )


def test_empirical_gravity_differs_from_physical_model() -> None:
    model = load_nominal_model()
    position = np.zeros(6)
    physical = model.physical_gravity_torque(position)
    empirical = model.empirical_gravity_torque(position)

    assert abs(empirical[1] - physical[1]) > 10.0
    assert abs(empirical[2] - physical[2]) > 5.0


def test_uncertainty_sample_is_inside_envelope() -> None:
    envelope = UncertaintyEnvelope(
        mass_scale=(0.9, 1.1),
        feedback_delay_s=(0.0, 0.02),
    )
    sample = envelope.sample(np.random.default_rng(32))

    assert 0.9 <= sample["mass_scale"] <= 1.1
    assert 0.0 <= sample["feedback_delay_s"] <= 0.02


def test_physical_hold_reduces_gravity_offset() -> None:
    model = load_nominal_model()
    backend = MuJoCoArmBackend()
    target = np.asarray(POWER_ON_POSITION_RAD)

    physical_error = _hold_error(
        backend,
        model.physical_gravity_torque,
        target,
    )
    empirical_error = _hold_error(
        backend,
        model.empirical_gravity_torque,
        target,
    )

    assert physical_error < 0.01
    assert empirical_error > physical_error * 5.0
    backend.close()


def _hold_error(backend, gravity_torque, target) -> float:
    controller = PDGravityController(
        PDGains(np.ones(6) * 100.0, np.ones(6) * 2.0),
        backend.effort_limits,
        gravity_torque,
    )
    backend.reset(target)
    feedback = backend.read_feedback()
    for _ in range(1000):
        feedback = backend.step(controller.compute(feedback, target))
    return float(np.max(np.abs(target - feedback.position_rad)))
