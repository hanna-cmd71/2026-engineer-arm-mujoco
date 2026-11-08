"""Gravity-compensated MuJoCo drag-teaching demo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jgarm_sim import MuJoCoArmBackend  # noqa: E402
from jgarm_sim.controllers import PDGains, PDGravityController  # noqa: E402
from jgarm_sim.dynamics import PinocchioDynamics  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "model" / "scene.xml"
DEFAULT_KD = np.array([2.0, 4.0, 4.0, 1.5, 1.0, 0.5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    parser.add_argument("--steps-per-frame", type=int, default=17)
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.damping_scale < 0.0:
        raise SystemExit("--damping-scale must be non-negative")
    backend = MuJoCoArmBackend(args.model)
    dynamics = PinocchioDynamics()
    controller = PDGravityController(
        PDGains(np.zeros(6), DEFAULT_KD * args.damping_scale),
        backend.effort_limits,
        dynamics.gravity_torque,
    )
    start = time.monotonic()
    with mujoco_viewer.launch_passive(
        backend.model,
        backend.data,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[:] = [-0.2, 0.0, 0.3]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -30.0
        print(
            "Drag teaching: select a link in the viewer and use the "
            "perturb drag tool. Gravity compensation is active; "
            "zero stiffness, damping only."
        )
        while viewer.is_running():
            if (
                args.duration > 0.0
                and time.monotonic() - start >= args.duration
            ):
                break
            for _ in range(max(1, args.steps_per_frame)):
                feedback = backend.read_feedback()
                command = controller.compute(
                    feedback,
                    feedback.position_rad,
                )
                backend.step(command)
            viewer.sync()
            time.sleep(1.0 / 60.0)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
