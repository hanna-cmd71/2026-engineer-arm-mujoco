"""Solve a KDL IK target and display the resulting JGarm pose."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer as mujoco_viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jgarm_sim import MuJoCoArmBackend  # noqa: E402
from jgarm_sim.kinematics import IKOptions, Kinematics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.50)
    parser.add_argument("--y", type=float, default=0.00)
    parser.add_argument("--z", type=float, default=0.50)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Keep the seed orientation instead of specifying full pose orientation.",
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
    kinematics = Kinematics()
    backend = MuJoCoArmBackend()
    seed = np.zeros(6)
    target = kinematics.forward(seed)
    target[:3, 3] = [args.x, args.y, args.z]

    result = kinematics.inverse(
        target,
        seed,
        options=IKOptions(),
        position_only=args.position_only,
    )
    print(
        f"success={result.success} attempts={result.attempts} "
        f"iterations={result.iterations} "
        f"position_error={result.position_error_m:.6e} m "
        f"rotation_error={result.rotation_error_rad:.6e} rad"
    )
    print("q =", np.array2string(result.position_rad, precision=6))
    if not result.success:
        return

    backend.reset(result.position_rad)
    start_time = time.monotonic()
    with mujoco_viewer.launch_passive(
        backend.model, backend.data
    ) as mj_viewer:
        mj_viewer.cam.distance = 1.2
        mj_viewer.cam.azimuth = 135.0
        mj_viewer.cam.elevation = -20.0
        while mj_viewer.is_running():
            if args.duration > 0.0 and time.monotonic() - start_time >= args.duration:
                break
            mj_viewer.sync()
            time.sleep(1.0 / 60.0)

    backend.close()


if __name__ == "__main__":
    main()

