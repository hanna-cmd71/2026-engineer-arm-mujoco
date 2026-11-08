"""Evaluate the nominal mass/COM model and close-loop holding error."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jgarm_sim import MuJoCoArmBackend  # noqa: E402
from jgarm_sim.controllers import PDGains, PDGravityController  # noqa: E402
from jgarm_sim.nominal_model import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_nominal_model,
)
from jgarm_sim.types import POWER_ON_POSITION_RAD  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "nominal_model_evaluation.json"
KP = np.array([60.0, 100.0, 100.0, 40.0, 20.0, 10.0])
KD = np.array([2.0, 4.0, 4.0, 1.5, 1.0, 0.5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Nominal model configuration YAML.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=200,
        help="Random poses used for gravity comparison.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="PD holding duration in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON evaluation report.",
    )
    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=8,
        help="Mass/COM/payload samples for the hold envelope.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_nominal_model(args.config)
    backend = MuJoCoArmBackend()
    random = np.random.default_rng(21)
    positions = random.uniform(
        backend.joint_ranges[:, 0],
        backend.joint_ranges[:, 1],
        size=(args.positions, 6),
    )
    physical = np.array(
        [model.physical_gravity_torque(q) for q in positions]
    )
    empirical = np.array(
        [model.empirical_gravity_torque(q) for q in positions]
    )
    gravity_error = empirical - physical
    physical_hold = hold_pose(
        backend,
        model.physical_gravity_torque,
        args.duration,
    )
    empirical_hold = hold_pose(
        backend,
        model.empirical_gravity_torque,
        args.duration,
    )
    uncertainty_hold = evaluate_uncertainty(
        model,
        args.duration,
        args.uncertainty_samples,
    )
    backend.close()

    report = {
        "total_mass_kg": model.total_mass_kg,
        "links": model.mass_com_table(),
        "gravity_model": model.gravity_model,
        "uncertainty": dataclasses.asdict(model.uncertainty),
        "gravity_comparison": {
            "rmse_nm": _list(np.sqrt(np.mean(gravity_error**2, axis=0))),
            "max_abs_nm": _list(np.max(np.abs(gravity_error), axis=0)),
            "physical_mean_abs_nm": _list(
                np.mean(np.abs(physical), axis=0)
            ),
            "empirical_mean_abs_nm": _list(
                np.mean(np.abs(empirical), axis=0)
            ),
        },
        "hold": {
            "physical": physical_hold,
            "empirical": empirical_hold,
            "uncertainty": uncertainty_hold,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"total mass: {model.total_mass_kg:.4f} kg")
    print("links:")
    for link in model.links.values():
        print(
            f"  {link.name}: m={link.mass_kg:.5f} kg "
            f"com={np.array2string(link.com_m, precision=5)}"
        )
    print(
        "gravity physical-empirical RMSE: "
        + np.array2string(
            np.sqrt(np.mean(gravity_error**2, axis=0)),
            precision=4,
        )
    )
    print(
        "physical hold max error: "
        f"{physical_hold['max_position_error_rad']:.6f} rad"
    )
    print(
        "empirical hold max error: "
        f"{empirical_hold['max_position_error_rad']:.6f} rad"
    )
    print(
        "uncertainty hold max error: "
        f"{uncertainty_hold['max_error_rad']:.6f} rad"
    )
    print(f"report: {args.output}")
    return 0


def hold_pose(backend, gravity_torque, duration_s: float) -> dict[str, object]:
    target = np.asarray(POWER_ON_POSITION_RAD, dtype=np.float64)
    controller = PDGravityController(
        PDGains(KP, KD),
        backend.effort_limits,
        gravity_torque,
    )
    backend.reset(target)
    feedback = backend.read_feedback()
    steps = max(1, int(round(duration_s / backend.control_period_s)))
    for _ in range(steps):
        command = controller.compute(feedback, target)
        feedback = backend.step(command)
    error = target - feedback.position_rad
    return {
        "duration_s": duration_s,
        "steps": steps,
        "final_position_rad": _list(feedback.position_rad),
        "max_position_error_rad": float(np.max(np.abs(error))),
        "final_gravity_effort_nm": _list(
            controller.last_gravity_effort_nm
        ),
        "final_effort_nm": _list(controller.last_unclipped_effort_nm),
    }


def evaluate_uncertainty(
    model,
    duration_s: float,
    sample_count: int,
) -> dict[str, object]:
    random = np.random.default_rng(22)
    samples = []
    errors = []
    for _ in range(max(0, sample_count)):
        sample = model.uncertainty.sample(random)
        backend = MuJoCoArmBackend()
        apply_uncertainty(backend, sample, model.uncertainty, random)
        result = hold_pose(
            backend,
            model.physical_gravity_torque,
            duration_s,
        )
        backend.close()
        error = float(result["max_position_error_rad"])
        errors.append(error)
        samples.append({"parameters": sample, "max_error_rad": error})
    return {
        "samples": samples,
        "max_error_rad": float(max(errors)) if errors else 0.0,
        "mean_error_rad": float(np.mean(errors)) if errors else 0.0,
    }


def apply_uncertainty(backend, sample, envelope, random) -> None:
    """Perturbs link mass, COM, and payload in the MuJoCo plant."""
    com_offset = random.uniform(
        envelope.com_offset_m[0],
        envelope.com_offset_m[1],
        size=3,
    )
    for index in range(1, 7):
        body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"link_{index}",
        )
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing link_{index}")
        backend.model.body_mass[body_id] *= sample["mass_scale"]
        backend.model.body_ipos[body_id] += com_offset
    link6_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "link_6",
    )
    payload = sample["payload_mass_kg"]
    if payload > 0.0:
        old_mass = float(backend.model.body_mass[link6_id])
        old_com = backend.model.body_ipos[link6_id].copy()
        payload_com = old_com + np.array([0.0, 0.0, 0.05])
        total_mass = old_mass + payload
        backend.model.body_mass[link6_id] = total_mass
        backend.model.body_ipos[link6_id] = (
            old_mass * old_com + payload * payload_com
        ) / total_mass
    mujoco.mj_forward(backend.model, backend.data)


def _list(values) -> list:
    return np.asarray(values).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
