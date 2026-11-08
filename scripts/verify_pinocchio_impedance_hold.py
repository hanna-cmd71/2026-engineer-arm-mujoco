"""Verify Pinocchio inverse-dynamics impedance holding in MuJoCo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(
    0,
    str(PROJECT_ROOT / "ros2_ws" / "src" / "jgarm_control"),
)

from jgarm_control.controller import (  # noqa: E402
    JointImpedanceController,
    PDGains,
)
from jgarm_control.dynamics import PinocchioDynamics  # noqa: E402
from jgarm_sim import MuJoCoArmBackend  # noqa: E402
from jgarm_sim.types import POWER_ON_POSITION_RAD  # noqa: E402


KP = np.array([60.0, 100.0, 100.0, 40.0, 20.0, 10.0])
KD = np.array([2.0, 4.0, 4.0, 1.5, 1.0, 0.5])
OUTPUT_DIR = PROJECT_ROOT / "results" / "pinocchio_impedance_hold"


def main() -> int:
    backend = MuJoCoArmBackend()
    backend.model.dof_damping[backend._joint_dof_adr] = [
        0.05,
        0.05,
        0.05,
        0.5,
        0.5,
        0.5,
    ]
    backend.model.dof_armature[backend._joint_dof_adr] = 0.001
    effort_limits = np.array([10.0, 63.0, 35.0, 10.0, 6.0, 3.0])
    dynamics = PinocchioDynamics(PROJECT_ROOT / "JGarm.urdf")
    controller = JointImpedanceController(
        PDGains(KP, KD),
        effort_limits,
        dynamics,
        kp_scale=[3.0, 3.0, 3.0, 1.0, 3.0, 1.0],
        kd_scale=[3.0, 3.0, 3.0, 2.0, 3.0, 2.0],
    )
    target = np.asarray(POWER_ON_POSITION_RAD, dtype=np.float64)
    initial = target.copy()
    initial[3] += 0.005
    initial[5] += 0.005
    backend.reset(initial)
    backend.data.qvel[backend._joint_dof_adr] = [
        0.0,
        0.0,
        0.0,
        0.02,
        0.0,
        0.02,
    ]
    mujoco.mj_forward(backend.model, backend.data)

    times = []
    positions = []
    velocities = []
    commands = []
    predicted = []
    impedance = []
    gravity = []
    coriolis = []
    for step in range(3000):
        feedback = backend.read_feedback()
        command = controller.compute(feedback, target)
        q = feedback.position_rad
        dq = feedback.velocity_rad_s
        error = target - q
        mass = dynamics.mass_matrix(q)
        c_term = dynamics.coriolis_torque(q, dq)
        g_term = dynamics.gravity_torque(q)
        impedance_term = (
            controller.gains.kp * controller.kp_scale * error
            + controller.gains.kd * controller.kd_scale * (-dq)
        )
        tau_predicted = impedance_term + c_term + g_term + mass @ np.zeros(6)
        times.append(step * backend.control_period_s)
        positions.append(q.copy())
        velocities.append(dq.copy())
        commands.append(command.effort_nm.copy())
        predicted.append(tau_predicted.copy())
        impedance.append(impedance_term.copy())
        gravity.append(g_term.copy())
        coriolis.append(c_term.copy())
        feedback = backend.step(command)
    backend.close()

    times = np.asarray(times)
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    commands = np.asarray(commands)
    predicted = np.asarray(predicted)
    impedance = np.asarray(impedance)
    gravity = np.asarray(gravity)
    coriolis = np.asarray(coriolis)
    errors = target - positions
    residual = commands - predicted
    metrics = {
        "initial_error_rad": errors[0].tolist(),
        "final_error_rad": errors[-1].tolist(),
        "max_error_rad": np.max(np.abs(errors), axis=0).tolist(),
        "final_velocity_rad_s": velocities[-1].tolist(),
        "max_velocity_rad_s": np.max(np.abs(velocities), axis=0).tolist(),
        "gravity_at_initial_nm": gravity[0].tolist(),
        "max_torque_residual_nm": float(np.max(np.abs(residual))),
        "max_command_nm": np.max(np.abs(commands), axis=0).tolist(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(times, errors[:, 3], label="J4 error")
    axes[0].plot(times, errors[:, 5], label="J6 error")
    axes[0].set_ylabel("position error / rad")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, commands[:, 3], label="command", linewidth=1.8)
    axes[1].plot(times, predicted[:, 3], "--", label="Pinocchio prediction")
    axes[1].plot(times, impedance[:, 3], label="impedance")
    axes[1].plot(times, gravity[:, 3], label="gravity")
    axes[1].plot(times, coriolis[:, 3], label="coriolis")
    axes[1].set_xlabel("time / s")
    axes[1].set_ylabel("J4 torque / N*m")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    figure.suptitle("Pinocchio inverse-dynamics impedance hold in MuJoCo")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "hold_J4_J6.png", dpi=160)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"plot: {OUTPUT_DIR / 'hold_J4_J6.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
