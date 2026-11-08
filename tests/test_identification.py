"""Synthetic-data tests for offline hardware-profile identification."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jgarm_sim.identification import (
    BREAKAWAY_PHASE,
    FRICTION_PHASE,
    GRAVITY_PHASE,
    IdentificationSamples,
    fit_breakaway,
    fit_friction,
    fit_gravity,
    friction_load,
    gravity_load,
    identify_parameters,
    load_csv,
    write_profile_yaml,
    write_report,
)


GRAVITY_P = np.array([0.0, 20.284, -15.112, -0.579, 16.368, 0.0])
GRAVITY_Q = np.array([0.0, -60.824, 2.680, 0.380, -0.715, 0.0])
GRAVITY_BIAS = np.array([0.0, -22.244, 1.534, 0.0, -13.608, 0.0])
FRICTION_TORQUE = np.array([0.0, 11.6, 7.4, 0.0, 0.75, 0.0])
FRICTION_SCALE = np.full(6, 0.2)
BREAKAWAY_TORQUE = np.array([0.0, 11.25, 10.0, 0.0, 0.4, 0.2])
LIMITS = np.array(
    [
        [-3.14, 3.14],
        [-1.3265, 1.9024],
        [-0.5934, 1.35],
        [-3.14, 3.14],
        [-2.094395, 2.094395],
        [-3.14, 3.14],
    ]
)


def _gravity(position: np.ndarray) -> np.ndarray:
    return gravity_load(
        position,
        GRAVITY_P,
        GRAVITY_Q,
        GRAVITY_BIAS,
    )


def _gravity_samples(
    random: np.random.Generator,
    count: int,
) -> IdentificationSamples:
    position = random.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(count, 6))
    velocity = np.zeros_like(position)
    torque = _gravity(position)
    time = np.arange(count, dtype=float) * 0.001
    return IdentificationSamples(
        time,
        position,
        velocity,
        torque,
        target_position_rad=position,
        target_velocity_rad_s=velocity,
        phase=np.full(count, GRAVITY_PHASE),
    )


def _friction_samples(
    random: np.random.Generator,
    segments: int = 8,
    samples_per_segment: int = 40,
) -> IdentificationSamples:
    positions = []
    velocities = []
    torques = []
    for _ in range(segments):
        position = random.uniform(
            LIMITS[:, 0],
            LIMITS[:, 1],
            size=(samples_per_segment, 6),
        )
        velocity_value = random.uniform(-0.5, 0.5, size=6)
        velocity_value = np.where(
            np.abs(velocity_value) < 0.08,
            np.sign(velocity_value + 1e-12) * 0.1,
            velocity_value,
        )
        velocity = np.tile(velocity_value, (samples_per_segment, 1))
        positions.append(position)
        velocities.append(velocity)
        torques.append(
            _gravity(position)
            + friction_load(velocity, FRICTION_TORQUE, FRICTION_SCALE)
        )
    position = np.vstack(positions)
    velocity = np.vstack(velocities)
    torque = np.vstack(torques)
    count = position.shape[0]
    time = np.arange(count, dtype=float) * 0.01
    return IdentificationSamples(
        time,
        position,
        velocity,
        torque,
        target_position_rad=position,
        target_velocity_rad_s=velocity,
        phase=np.full(count, FRICTION_PHASE),
    )


def _breakaway_samples(
    random: np.random.Generator,
    count: int = 200,
) -> IdentificationSamples:
    position = random.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(count, 6))
    error = random.uniform(-0.05, 0.05, size=(count, 6))
    zero_velocity = np.zeros_like(position)
    torque = _gravity(position) + BREAKAWAY_TORQUE * np.tanh(error / 0.01)
    time = np.arange(count, dtype=float) * 0.01
    return IdentificationSamples(
        time,
        position,
        zero_velocity,
        torque,
        target_position_rad=position + error,
        target_velocity_rad_s=zero_velocity,
        phase=np.full(count, BREAKAWAY_PHASE),
    )


def test_gravity_fit_recovers_known_profile() -> None:
    random = np.random.default_rng(4)
    samples = _gravity_samples(random, 160)

    fit = fit_gravity(samples)

    np.testing.assert_allclose(fit.p_nm, GRAVITY_P, atol=1e-8)
    np.testing.assert_allclose(fit.q_nm, GRAVITY_Q, atol=1e-8)
    np.testing.assert_allclose(fit.bias_nm, GRAVITY_BIAS, atol=1e-8)
    assert fit.sample_count == samples.time_s.size
    assert np.max(fit.rmse_nm) < 1e-8


def test_friction_fit_recovers_known_profile() -> None:
    random = np.random.default_rng(5)
    gravity_samples = _gravity_samples(random, 120)
    gravity = fit_gravity(gravity_samples)
    friction_samples = _friction_samples(random)

    fit = fit_friction(
        friction_samples,
        gravity,
        velocity_rad_s=friction_samples.target_velocity_rad_s,
    )

    np.testing.assert_allclose(fit.torque_nm, FRICTION_TORQUE, atol=1e-5)
    np.testing.assert_allclose(
        fit.velocity_scale_rad_s,
        FRICTION_SCALE,
        atol=1e-4,
    )


def test_breakaway_fit_recovers_known_profile() -> None:
    random = np.random.default_rng(6)
    gravity = fit_gravity(_gravity_samples(random, 120))
    friction = fit_friction(
        _friction_samples(random),
        gravity,
        velocity_rad_s=None,
    )
    samples = _breakaway_samples(random)

    fit = fit_breakaway(samples, gravity, friction)

    np.testing.assert_allclose(
        fit.torque_nm,
        BREAKAWAY_TORQUE,
        atol=1e-5,
    )
    assert np.all(fit.identified)


def test_phase_aware_identification_combines_experiments() -> None:
    random = np.random.default_rng(7)
    gravity_samples = _gravity_samples(random, 120)
    friction_samples = _friction_samples(random)
    breakaway_samples = _breakaway_samples(random)
    samples = _combine(gravity_samples, friction_samples, breakaway_samples)

    result = identify_parameters(samples)

    np.testing.assert_allclose(result.gravity.p_nm, GRAVITY_P, atol=1e-7)
    np.testing.assert_allclose(result.gravity.q_nm, GRAVITY_Q, atol=1e-7)
    np.testing.assert_allclose(
        result.gravity.bias_nm,
        GRAVITY_BIAS,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.friction.torque_nm,
        FRICTION_TORQUE,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.breakaway.torque_nm,
        BREAKAWAY_TORQUE,
        atol=1e-4,
    )


def test_csv_and_report_outputs(tmp_path) -> None:
    random = np.random.default_rng(8)
    samples = _gravity_samples(random, 24)
    csv_path = tmp_path / "samples.csv"
    header = ["time_s"]
    header += [f"q{index}" for index in range(1, 7)]
    header += [f"dq{index}" for index in range(1, 7)]
    header += [f"tau{index}" for index in range(1, 7)]
    header += [f"q_target{index}" for index in range(1, 7)]
    header += [f"dq_target{index}" for index in range(1, 7)]
    header += ["phase"]
    lines = [",".join(header)]
    for index in range(samples.time_s.size):
        values = [samples.time_s[index]]
        values += samples.position_rad[index].tolist()
        values += samples.velocity_rad_s[index].tolist()
        values += samples.torque_nm[index].tolist()
        values += samples.target_position_rad[index].tolist()
        values += samples.target_velocity_rad_s[index].tolist()
        values.append(samples.phase[index])
        lines.append(",".join(str(value) for value in values))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = load_csv(csv_path)
    result = identify_parameters(loaded)
    output_path = tmp_path / "profile.yaml"
    report_path = tmp_path / "report.json"
    write_profile_yaml(output_path, result)
    write_report(report_path, result)

    assert loaded.position_rad.shape == samples.position_rad.shape
    assert "gravity_p_nm" in output_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "friction" in report


def test_gravity_load_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        gravity_load(
            np.zeros(5),
            GRAVITY_P,
            GRAVITY_Q,
            GRAVITY_BIAS,
        )


def _combine(*datasets: IdentificationSamples) -> IdentificationSamples:
    offsets = np.cumsum(
        [0.0]
        + [
            dataset.time_s[-1] - dataset.time_s[0] + 0.1
            for dataset in datasets[:-1]
        ]
    )
    time = np.concatenate(
        [
            dataset.time_s - dataset.time_s[0] + offset
            for dataset, offset in zip(
                datasets,
                offsets,
                strict=True,
            )
        ]
    )
    return IdentificationSamples(
        time,
        np.vstack([dataset.position_rad for dataset in datasets]),
        np.vstack([dataset.velocity_rad_s for dataset in datasets]),
        np.vstack([dataset.torque_nm for dataset in datasets]),
        np.vstack([dataset.target_position_rad for dataset in datasets]),
        np.vstack([dataset.target_velocity_rad_s for dataset in datasets]),
        np.concatenate([dataset.phase for dataset in datasets]),
    )
