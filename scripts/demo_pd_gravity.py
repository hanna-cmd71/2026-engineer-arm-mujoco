"""Run joint-space PD control with Pinocchio gravity compensation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from mujoco import viewer as mujoco_viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jgarm_sim import MuJoCoArmBackend  # noqa: E402
from jgarm_sim.controllers import PDGains, PDGravityController  # noqa: E402
from jgarm_sim.dynamics import PinocchioDynamics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="0.1,-0.8,0.8,0.1,0.3,-0.1",
        help="Comma-separated six-joint target in radians.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Close the viewer after this many seconds; zero means no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.fromstring(args.target, sep=",")
    if target.shape != (6,):
        raise ValueError("--target must contain six comma-separated values")

    backend = MuJoCoArmBackend()
    dynamics = PinocchioDynamics()
    gains = PDGains(
        kp=np.array([60.0, 100.0, 100.0, 40.0, 20.0, 10.0]),
        kd=np.array([2.0, 4.0, 4.0, 1.5, 1.0, 0.5]),
    )
    controller = PDGravityController(
        gains,
        backend.effort_limits,
        dynamics.gravity_torque,
    )
    start_time = time.monotonic()
    next_tick = start_time

    with mujoco_viewer.launch_passive(
        backend.model, backend.data
    ) as mj_viewer:
        mj_viewer.cam.distance = 1.2
        mj_viewer.cam.azimuth = 135.0
        mj_viewer.cam.elevation = -20.0
        while mj_viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break

            while next_tick <= now:
                feedback = backend.read_feedback()
                command = controller.compute(feedback, target)
                backend.step(command)
                next_tick += backend.control_period_s
            mj_viewer.sync()
            time.sleep(1.0 / 60.0)

    feedback = backend.read_feedback()
    print(
        "final joint error:",
        np.array2string(
            target - feedback.position_rad, precision=6
        ),
    )
    backend.close()


if __name__ == "__main__":
    main()
