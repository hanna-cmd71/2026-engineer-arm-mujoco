"""Open the JGarm scene in the MuJoCo passive viewer."""

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

from jgarm_sim import MuJoCoArmBackend, TorqueCommand  # noqa: E402


DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "scene.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MuJoCo scene XML.",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Advance physics with zero actuator torque instead of showing a static pose.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Close automatically after this many seconds; zero means no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = MuJoCoArmBackend(args.model)
    zero_command = TorqueCommand(np.zeros(6, dtype=np.float64))
    start_time = time.monotonic()
    last_render_time = start_time

    with mujoco_viewer.launch_passive(backend.model, backend.data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        while viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break
            if args.physics:
                elapsed = now - last_render_time
                steps = max(1, int(elapsed / backend.control_period_s))
                for _ in range(steps):
                    backend.step(zero_command)
            viewer.sync()
            last_render_time = now
            time.sleep(1.0 / 60.0)

    backend.close()


if __name__ == "__main__":
    main()
