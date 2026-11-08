"""Offline identification of JGarm hardware-profile parameters."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from jgarm_sim.types import JOINT_COUNT


GRAVITY_PHASE = "gravity"
FRICTION_PHASE = "friction"
BREAKAWAY_PHASE = "breakaway"
PHASES = (GRAVITY_PHASE, FRICTION_PHASE, BREAKAWAY_PHASE)


@dataclass(frozen=True)
class IdentificationSamples:
    time_s: np.ndarray
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    torque_nm: np.ndarray
    target_position_rad: np.ndarray | None = None
    target_velocity_rad_s: np.ndarray | None = None
    phase: np.ndarray | None = None

    def __post_init__(self) -> None:
        time = _vector(self.time_s, "time_s")
        if time.size < 2 or np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        position = _matrix(self.position_rad, "position_rad")
        velocity = _matrix(self.velocity_rad_s, "velocity_rad_s")
        torque = _matrix(self.torque_nm, "torque_nm")
        if not (
            position.shape[0] == velocity.shape[0] == torque.shape[0]
            == time.size
        ):
            raise ValueError("sample arrays must have the same length")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "position_rad", position)
        object.__setattr__(self, "velocity_rad_s", velocity)
        object.__setattr__(self, "torque_nm", torque)
        object.__setattr__(
            self,
            "target_position_rad",
            _optional_matrix(
                self.target_position_rad,
                "target_position_rad",
                time.size,
            ),
        )
        object.__setattr__(
            self,
            "target_velocity_rad_s",
            _optional_matrix(
                self.target_velocity_rad_s,
                "target_velocity_rad_s",
                time.size,
            ),
        )
        if self.phase is not None:
            phase = np.asarray(self.phase, dtype=str).reshape(-1)
            if phase.size != time.size:
                raise ValueError("phase must contain one value per sample")
            unknown = sorted(set(phase) - set(PHASES))
            if unknown:
                raise ValueError(f"unknown phases: {unknown}")
            object.__setattr__(self, "phase", phase.copy())


@dataclass(frozen=True)
class GravityFit:
    p_nm: np.ndarray
    q_nm: np.ndarray
    bias_nm: np.ndarray
    rmse_nm: np.ndarray
    sample_count: int
    j5_config_weight: float
    j5_conservative_coupling: bool


@dataclass(frozen=True)
class FrictionFit:
    torque_nm: np.ndarray
    velocity_scale_rad_s: np.ndarray
    rmse_nm: np.ndarray
    sample_count: np.ndarray


@dataclass(frozen=True)
class BreakawayFit:
    torque_nm: np.ndarray
    rmse_nm: np.ndarray
    sample_count: np.ndarray
    identified: np.ndarray


@dataclass(frozen=True)
class ParameterIdentification:
    gravity: GravityFit
    friction: FrictionFit
    breakaway: BreakawayFit

    def to_dict(self) -> dict[str, object]:
        return {
            "gravity": {
                "p_nm": _list(self.gravity.p_nm),
                "q_nm": _list(self.gravity.q_nm),
                "bias_nm": _list(self.gravity.bias_nm),
                "rmse_nm": _list(self.gravity.rmse_nm),
                "sample_count": self.gravity.sample_count,
                "j5_config_weight": self.gravity.j5_config_weight,
                "j5_conservative_coupling": (
                    self.gravity.j5_conservative_coupling
                ),
            },
            "friction": {
                "torque_nm": _list(self.friction.torque_nm),
                "velocity_scale_rad_s": _list(
                    self.friction.velocity_scale_rad_s
                ),
                "rmse_nm": _list(self.friction.rmse_nm),
                "sample_count": _list(self.friction.sample_count),
            },
            "breakaway": {
                "torque_nm": _list(self.breakaway.torque_nm),
                "rmse_nm": _list(self.breakaway.rmse_nm),
                "sample_count": _list(self.breakaway.sample_count),
                "identified": _list(self.breakaway.identified),
            },
        }


def gravity_load(
    position_rad,
    p_nm,
    q_nm,
    bias_nm,
    *,
    j5_config_weight: float = 0.35,
    j5_conservative_coupling: bool = True,
) -> np.ndarray:
    """Evaluates the identified gravity model for one or more samples."""
    raw_position = np.asarray(position_rad, dtype=np.float64)
    single_sample = raw_position.ndim == 1
    if single_sample:
        raw_position = raw_position.reshape(1, -1)
    position = _matrix(raw_position, "position_rad")
    p = _vector6(p_nm, "p_nm")
    q = _vector6(q_nm, "q_nm")
    bias = _vector6(bias_nm, "bias_nm")
    theta1 = position[:, 1]
    theta2 = theta1 + position[:, 2]
    theta3 = theta2 - position[:, 4]
    if j5_conservative_coupling:
        phi5 = -position[:, 4]
    else:
        phi5 = j5_config_weight * theta2 - position[:, 4]
    result = np.broadcast_to(bias, position.shape).copy()
    result[:, 1] += (
        p[1] * np.cos(theta1)
        + q[1] * np.sin(theta1)
        + p[2] * np.cos(theta2)
        + q[2] * np.sin(theta2)
        + p[3] * np.cos(theta3)
        + q[3] * np.sin(theta3)
    )
    result[:, 2] += (
        p[2] * np.cos(theta2)
        + q[2] * np.sin(theta2)
        + p[3] * np.cos(theta3)
        + q[3] * np.sin(theta3)
    )
    result[:, 4] += p[4] * np.cos(phi5) + q[4] * np.sin(phi5)
    if j5_conservative_coupling:
        result[:, 4] += p[3]
        result[:, 4] -= p[3] * np.cos(theta3) + q[3] * np.sin(theta3)
    return result[0] if single_sample else result


def friction_load(
    velocity_rad_s,
    torque_nm,
    velocity_scale_rad_s,
) -> np.ndarray:
    raw_velocity = np.asarray(velocity_rad_s, dtype=np.float64)
    single_sample = raw_velocity.ndim == 1
    if single_sample:
        raw_velocity = raw_velocity.reshape(1, -1)
    velocity = _matrix(raw_velocity, "velocity_rad_s")
    torque = _vector6(torque_nm, "torque_nm")
    scale = _vector6(velocity_scale_rad_s, "velocity_scale_rad_s")
    if np.any(scale <= 0.0):
        raise ValueError("velocity_scale_rad_s must be positive")
    result = torque * np.tanh(velocity / scale)
    return result[0] if single_sample else result


def fit_gravity(
    samples: IdentificationSamples,
    *,
    mask=None,
    velocity_threshold: float = 0.02,
    acceleration_threshold: float = 1.0,
    j5_config_weight: float = 0.35,
    j5_conservative_coupling: bool = True,
) -> GravityFit:
    """Fits the three-link gravity model to quasi-static samples."""
    active = _mask_or_default(
        mask,
        _quasi_static_mask(
            samples,
            velocity_threshold,
            acceleration_threshold,
        ),
    )
    if np.count_nonzero(active) < 20:
        raise ValueError("not enough quasi-static samples for gravity fit")
    position = samples.position_rad[active]
    torque = samples.torque_nm[active]
    theta1 = position[:, 1]
    theta2 = theta1 + position[:, 2]
    theta3 = theta2 - position[:, 4]

    rows = np.zeros((2 * position.shape[0], 8))
    values = np.empty(2 * position.shape[0])
    rows[0::2, 0] = np.cos(theta1)
    rows[0::2, 1] = np.sin(theta1)
    rows[0::2, 2] = np.cos(theta2)
    rows[0::2, 3] = np.sin(theta2)
    rows[0::2, 4] = np.cos(theta3)
    rows[0::2, 5] = np.sin(theta3)
    rows[0::2, 6] = 1.0
    rows[1::2, 2] = np.cos(theta2)
    rows[1::2, 3] = np.sin(theta2)
    rows[1::2, 4] = np.cos(theta3)
    rows[1::2, 5] = np.sin(theta3)
    rows[1::2, 7] = 1.0
    values[0::2] = torque[:, 1]
    values[1::2] = torque[:, 2]
    solution, *_ = np.linalg.lstsq(rows, values, rcond=None)
    if np.linalg.matrix_rank(rows) < rows.shape[1]:
        raise ValueError("gravity design matrix is rank deficient")

    p = np.zeros(JOINT_COUNT)
    q = np.zeros(JOINT_COUNT)
    bias = np.zeros(JOINT_COUNT)
    p[1], q[1], p[2], q[2], p[3], q[3] = solution[:6]
    bias[1], bias[2] = solution[6:]

    if j5_conservative_coupling:
        phi5 = -position[:, 4]
        target = torque[:, 4] - (
            p[3] - p[3] * np.cos(theta3) - q[3] * np.sin(theta3)
        )
    else:
        phi5 = j5_config_weight * theta2 - position[:, 4]
        target = torque[:, 4]
    design = np.column_stack(
        [np.cos(phi5), np.sin(phi5), np.ones(phi5.size)]
    )
    solution5, *_ = np.linalg.lstsq(design, target, rcond=None)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("J5 gravity design matrix is rank deficient")
    p[4], q[4], bias[4] = solution5

    predicted = gravity_load(
        position,
        p,
        q,
        bias,
        j5_config_weight=j5_config_weight,
        j5_conservative_coupling=j5_conservative_coupling,
    )
    rmse = np.sqrt(np.mean((predicted - torque) ** 2, axis=0))
    return GravityFit(
        p_nm=p,
        q_nm=q,
        bias_nm=bias,
        rmse_nm=rmse,
        sample_count=int(np.count_nonzero(active)),
        j5_config_weight=float(j5_config_weight),
        j5_conservative_coupling=bool(j5_conservative_coupling),
    )


def fit_friction(
    samples: IdentificationSamples,
    gravity: GravityFit,
    *,
    mask=None,
    velocity_rad_s=None,
    min_speed_rad_s: float = 0.05,
    acceleration_threshold: float = 2.0,
    min_torque_nm: float = 0.05,
) -> FrictionFit:
    """Fits positive tanh friction for each joint from moving samples."""
    velocity = (
        samples.velocity_rad_s
        if velocity_rad_s is None
        else _optional_matrix(
            velocity_rad_s,
            "velocity_rad_s",
            samples.time_s.size,
        )
    )
    if velocity is None:
        raise ValueError("velocity_rad_s must not be None")
    active = _mask_or_default(mask, np.ones(samples.time_s.size, dtype=bool))
    active &= _low_acceleration_mask(
        samples,
        velocity,
        acceleration_threshold,
    )
    predicted_gravity = gravity_load(
        samples.position_rad,
        gravity.p_nm,
        gravity.q_nm,
        gravity.bias_nm,
        j5_config_weight=gravity.j5_config_weight,
        j5_conservative_coupling=gravity.j5_conservative_coupling,
    )
    residual = samples.torque_nm - predicted_gravity
    torque = np.zeros(JOINT_COUNT)
    scale = np.full(JOINT_COUNT, 0.2)
    rmse = np.zeros(JOINT_COUNT)
    counts = np.zeros(JOINT_COUNT, dtype=int)
    for index in range(JOINT_COUNT):
        joint_active = active & (
            np.abs(velocity[:, index]) > min_speed_rad_s
        )
        count = int(np.count_nonzero(joint_active))
        counts[index] = count
        if count < 10:
            continue
        joint_velocity = velocity[joint_active, index]
        joint_residual = residual[joint_active, index]
        initial = max(float(np.median(np.abs(joint_residual))), 0.1)

        def residual_function(parameters):
            gain, log_scale = parameters
            model = gain * np.tanh(
                joint_velocity / np.exp(log_scale)
            )
            return model - joint_residual

        solution = least_squares(
            residual_function,
            x0=np.array([initial, np.log(0.2)]),
            bounds=([0.0, np.log(1e-3)], [np.inf, np.log(10.0)]),
            max_nfev=2000,
        )
        gain = max(0.0, float(solution.x[0]))
        joint_scale = float(np.exp(solution.x[1]))
        if gain < min_torque_nm:
            gain = 0.0
            joint_scale = 0.2
        torque[index] = gain
        scale[index] = joint_scale
        model = gain * np.tanh(joint_velocity / joint_scale)
        rmse[index] = float(
            np.sqrt(np.mean((model - joint_residual) ** 2))
        )
    return FrictionFit(
        torque_nm=torque,
        velocity_scale_rad_s=scale,
        rmse_nm=rmse,
        sample_count=counts,
    )


def fit_breakaway(
    samples: IdentificationSamples,
    gravity: GravityFit,
    friction: FrictionFit,
    *,
    mask=None,
    error_scale_rad: float = 0.01,
    target_velocity_threshold_rad_s: float = 0.02,
    actual_velocity_threshold_rad_s: float = 0.03,
    max_error_rad: float = 0.1,
    min_samples: int = 10,
) -> BreakawayFit:
    """Fits the low-speed breakaway torque when target positions exist."""
    torque = np.zeros(JOINT_COUNT)
    rmse = np.zeros(JOINT_COUNT)
    counts = np.zeros(JOINT_COUNT, dtype=int)
    identified = np.zeros(JOINT_COUNT, dtype=bool)
    if samples.target_position_rad is None:
        return BreakawayFit(torque, rmse, counts, identified)
    target_velocity = samples.target_velocity_rad_s
    if target_velocity is None:
        target_velocity = np.zeros_like(samples.velocity_rad_s)
    error = samples.target_position_rad - samples.position_rad
    predicted_gravity = gravity_load(
        samples.position_rad,
        gravity.p_nm,
        gravity.q_nm,
        gravity.bias_nm,
        j5_config_weight=gravity.j5_config_weight,
        j5_conservative_coupling=gravity.j5_conservative_coupling,
    )
    predicted_friction = friction_load(
        target_velocity,
        friction.torque_nm,
        friction.velocity_scale_rad_s,
    )
    residual = samples.torque_nm - predicted_gravity - predicted_friction
    active = _mask_or_default(mask, np.ones(samples.time_s.size, dtype=bool))
    for index in range(JOINT_COUNT):
        joint_active = (
            active
            & (
                np.abs(target_velocity[:, index])
                < target_velocity_threshold_rad_s
            )
            & (
                np.abs(samples.velocity_rad_s[:, index])
                < actual_velocity_threshold_rad_s
            )
            & (np.abs(error[:, index]) <= max_error_rad)
        )
        count = int(np.count_nonzero(joint_active))
        counts[index] = count
        if count < min_samples:
            continue
        basis = np.tanh(error[joint_active, index] / error_scale_rad)
        denominator = float(basis @ basis)
        if denominator <= 1e-12:
            continue
        gain = max(0.0, float(basis @ residual[joint_active, index]))
        gain /= denominator
        torque[index] = gain
        model = gain * basis
        rmse[index] = float(
            np.sqrt(
                np.mean(
                    (model - residual[joint_active, index]) ** 2
                )
            )
        )
        identified[index] = True
    return BreakawayFit(torque, rmse, counts, identified)


def identify_parameters(
    samples: IdentificationSamples,
    *,
    use_target_velocity: bool = True,
    j5_config_weight: float = 0.35,
    j5_conservative_coupling: bool = True,
    velocity_threshold: float = 0.02,
    acceleration_threshold: float = 1.0,
    min_friction_speed: float = 0.05,
) -> ParameterIdentification:
    """Runs gravity, friction, and breakaway identification."""
    gravity = fit_gravity(
        samples,
        mask=_phase_mask(samples, GRAVITY_PHASE),
        velocity_threshold=velocity_threshold,
        acceleration_threshold=acceleration_threshold,
        j5_config_weight=j5_config_weight,
        j5_conservative_coupling=j5_conservative_coupling,
    )
    friction_velocity = samples.velocity_rad_s
    if use_target_velocity and samples.target_velocity_rad_s is not None:
        friction_velocity = samples.target_velocity_rad_s
    friction = fit_friction(
        samples,
        gravity,
        mask=_phase_mask(samples, FRICTION_PHASE),
        velocity_rad_s=friction_velocity,
        min_speed_rad_s=min_friction_speed,
    )
    breakaway = fit_breakaway(
        samples,
        gravity,
        friction,
        mask=_phase_mask(samples, BREAKAWAY_PHASE),
    )
    return ParameterIdentification(gravity, friction, breakaway)


def load_csv(path: str | Path) -> IdentificationSamples:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV contains no samples")
    time = np.array([_value(row, "time_s") for row in rows])
    position = _columns(rows, "q")
    velocity = _columns(rows, "dq")
    torque = _columns(rows, "tau")
    target_position = _optional_columns(rows, "q_target")
    target_velocity = _optional_columns(rows, "dq_target")
    phase = (
        np.array(
            [row.get("phase") or "" for row in rows],
            dtype=str,
        )
        if any("phase" in row for row in rows)
        else None
    )
    if phase is not None and np.any(phase == ""):
        phase = None
    return IdentificationSamples(
        time,
        position,
        velocity,
        torque,
        target_position,
        target_velocity,
        phase,
    )


def write_profile_yaml(
    path: str | Path,
    result: ParameterIdentification,
    *,
    profile_id: str = "identified",
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by scripts/identify_hardware_profile.py.",
        "# Merge these values into hardware_profile.yaml after review.",
        "/**:",
        "  ros__parameters:",
        f'    hardware_profile.id: "{profile_id}"',
        "    hardware_profile.gravity_j5_config_weight: "
        f"{result.gravity.j5_config_weight:.6f}",
        "    hardware_profile.gravity_j5_conservative_coupling: "
        f"{str(result.gravity.j5_conservative_coupling).lower()}",
        "    hardware_profile.gravity_p_nm: "
        f"{_yaml_list(result.gravity.p_nm)}",
        "    hardware_profile.gravity_q_nm: "
        f"{_yaml_list(result.gravity.q_nm)}",
        "    hardware_profile.gravity_bias_nm: "
        f"{_yaml_list(result.gravity.bias_nm)}",
        "    hardware_profile.friction_torque_nm: "
        f"{_yaml_list(result.friction.torque_nm)}",
        "    hardware_profile.friction_velocity_scale_rad_s: "
        f"{_yaml_list(result.friction.velocity_scale_rad_s)}",
        "    # breakaway identified: "
        f"{str(result.breakaway.identified.tolist()).lower()}",
        "    hardware_profile.breakaway_torque_nm: "
        f"{_yaml_list(result.breakaway.torque_nm)}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: str | Path, result: ParameterIdentification) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _phase_mask(
    samples: IdentificationSamples,
    name: str,
) -> np.ndarray | None:
    if samples.phase is None:
        return None
    mask = samples.phase == name
    return mask if np.any(mask) else None


def _quasi_static_mask(
    samples: IdentificationSamples,
    velocity_threshold: float,
    acceleration_threshold: float,
) -> np.ndarray:
    speed = np.max(np.abs(samples.velocity_rad_s), axis=1)
    return (speed <= velocity_threshold) & _low_acceleration_mask(
        samples,
        samples.velocity_rad_s,
        acceleration_threshold,
    )


def _low_acceleration_mask(
    samples: IdentificationSamples,
    velocity: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if velocity.shape[0] < 3:
        return np.ones(velocity.shape[0], dtype=bool)
    acceleration = np.gradient(
        velocity,
        samples.time_s,
        axis=0,
    )
    return np.max(np.abs(acceleration), axis=1) <= threshold


def _mask_or_default(mask, default):
    return default if mask is None else np.asarray(mask, dtype=bool)


def _columns(rows, prefix: str) -> np.ndarray:
    result = np.empty((len(rows), JOINT_COUNT))
    for sample_index, row in enumerate(rows):
        for joint_index in range(JOINT_COUNT):
            result[sample_index, joint_index] = _value(
                row,
                f"{prefix}{joint_index + 1}",
            )
    return result


def _optional_columns(rows, prefix: str) -> np.ndarray | None:
    names = [f"{prefix}{index + 1}" for index in range(JOINT_COUNT)]
    if not all(name in rows[0] for name in names):
        return None
    return _columns(rows, prefix)


def _value(row, name: str) -> float:
    if name not in row or row.get(name) in (None, ""):
        raise ValueError(f"CSV is missing column {name}")
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"CSV column {name} contains a non-finite value")
    return value


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _vector6(values, name: str) -> np.ndarray:
    array = _vector(values, name)
    if array.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must contain six values")
    return array


def _matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != JOINT_COUNT:
        raise ValueError(f"{name} must have shape (N, {JOINT_COUNT})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _optional_matrix(values, name: str, length: int) -> np.ndarray | None:
    if values is None:
        return None
    array = _matrix(values, name)
    if array.shape[0] != length:
        raise ValueError(f"{name} must contain {length} rows")
    return array


def _yaml_list(values) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _list(values) -> list:
    return np.asarray(values).tolist()
