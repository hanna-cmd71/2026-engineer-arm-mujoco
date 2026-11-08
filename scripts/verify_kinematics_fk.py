"""Compare KDL, Pinocchio, and MuJoCo forward kinematics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jgarm_sim import MuJoCoArmBackend  # noqa: E402
from jgarm_sim.dynamics import PinocchioDynamics  # noqa: E402
from jgarm_sim.kinematics import Kinematics  # noqa: E402
from jgarm_sim.types import POWER_ON_POSITION_RAD  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "kinematics_fk_check"
SAMPLE_COUNT = 200


def main() -> int:
    kdl = Kinematics()
    pinocchio_model = PinocchioDynamics()
    backend = MuJoCoArmBackend()
    frame_id = pinocchio_model.model.getFrameId("link_6")
    if frame_id >= pinocchio_model.model.nframes:
        raise ValueError("Pinocchio model has no link_6 frame")
    base_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base_link",
    )
    if base_id < 0:
        raise ValueError("MuJoCo model has no base_link body")

    random = np.random.default_rng(41)
    positions = random.uniform(
        backend.joint_ranges[:, 0],
        backend.joint_ranges[:, 1],
        size=(SAMPLE_COUNT, 6),
    )
    kdl_poses = []
    pin_poses = []
    mujoco_poses = []
    for position in positions:
        kdl_poses.append(kdl.forward(position))
        pinocchio.framesForwardKinematics(
            pinocchio_model.model,
            pinocchio_model.data,
            position,
        )
        pinocchio.updateFramePlacements(
            pinocchio_model.model,
            pinocchio_model.data,
        )
        pin_pose = np.asarray(
            pinocchio_model.data.oMf[frame_id].homogeneous,
            dtype=np.float64,
        ).copy()
        pin_poses.append(pin_pose)
        backend.reset(position)
        world_tcp = backend.tcp_pose()
        world_base = np.eye(4)
        world_base[:3, :3] = backend.data.xmat[base_id].reshape(3, 3)
        world_base[:3, 3] = backend.data.xpos[base_id]
        mujoco_poses.append(np.linalg.inv(world_base) @ world_tcp)
    backend.close()

    kdl_poses = np.asarray(kdl_poses)
    pin_poses = np.asarray(pin_poses)
    mujoco_poses = np.asarray(mujoco_poses)
    comparisons = {
        "kdl_vs_pinocchio": _errors(kdl_poses, pin_poses),
        "kdl_vs_mujoco": _errors(kdl_poses, mujoco_poses),
        "pinocchio_vs_mujoco": _errors(pin_poses, mujoco_poses),
    }
    report = {
        "sample_count": SAMPLE_COUNT,
        "power_on_kdl": kdl.forward(POWER_ON_POSITION_RAD).tolist(),
        "comparisons": comparisons,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for name, values in comparisons.items():
        axes[0].plot(values["position_m"], label=name)
        axes[1].plot(values["rotation_rad"], label=name)
    axes[0].set_ylabel("position error / m")
    axes[1].set_ylabel("rotation error / rad")
    axes[1].set_xlabel("sample")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle("JGarm forward-kinematics consistency")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "errors.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2))
    print(f"plot: {OUTPUT_DIR / 'errors.png'}")
    return 0


def _errors(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    positions = np.linalg.norm(
        reference[:, :3, 3] - actual[:, :3, 3],
        axis=1,
    )
    rotations = np.array(
        [
            Rotation.from_matrix(
                reference[index, :3, :3] @ actual[index, :3, :3].T
            ).magnitude()
            for index in range(reference.shape[0])
        ]
    )
    return {
        "position_m": positions.tolist(),
        "rotation_rad": rotations.tolist(),
        "max_position_m": float(np.max(positions)),
        "rms_position_m": float(np.sqrt(np.mean(positions**2))),
        "max_rotation_rad": float(np.max(rotations)),
        "rms_rotation_rad": float(np.sqrt(np.mean(rotations**2))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
