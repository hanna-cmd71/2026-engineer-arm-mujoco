"""Validate KDL IK solutions against Pinocchio and MuJoCo FK."""

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
from jgarm_sim.kinematics import IKOptions, Kinematics  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "kinematics_ik_check"
SAMPLE_COUNT = 20


def main() -> int:
    kdl = Kinematics()
    dynamics = PinocchioDynamics()
    frame_id = dynamics.model.getFrameId("link_6")
    backend = MuJoCoArmBackend()
    base_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base_link",
    )
    random = np.random.default_rng(43)
    results = []
    for _ in range(SAMPLE_COUNT):
        target_position = random.uniform(
            kdl.joint_limits_rad[:, 0],
            kdl.joint_limits_rad[:, 1],
        )
        target_pose = kdl.forward(target_position)
        seed = random.uniform(
            kdl.joint_limits_rad[:, 0],
            kdl.joint_limits_rad[:, 1],
        )
        ik = kdl.inverse(
            target_pose,
            seed,
            IKOptions(
                max_iterations=300,
                max_attempts=20,
                position_tolerance_m=1e-5,
                rotation_tolerance_rad=1e-5,
            ),
        )
        if not ik.success:
            results.append(
                {
                    "success": False,
                    "message": ik.message,
                }
            )
            continue
        pinocchio.framesForwardKinematics(
            dynamics.model,
            dynamics.data,
            ik.position_rad,
        )
        pinocchio.updateFramePlacements(
            dynamics.model,
            dynamics.data,
        )
        pin_pose = np.asarray(
            dynamics.data.oMf[frame_id].homogeneous,
            dtype=np.float64,
        ).copy()
        backend.reset(ik.position_rad)
        world_tcp = backend.tcp_pose()
        world_base = np.eye(4)
        world_base[:3, :3] = backend.data.xmat[base_id].reshape(3, 3)
        world_base[:3, 3] = backend.data.xpos[base_id]
        mujoco_pose = np.linalg.inv(world_base) @ world_tcp
        results.append(
            {
                "success": True,
                "kdl": _pose_error(target_pose, kdl.forward(ik.position_rad)),
                "pinocchio": _pose_error(target_pose, pin_pose),
                "mujoco": _pose_error(target_pose, mujoco_pose),
                "attempts": ik.attempts,
                "iterations": ik.iterations,
            }
        )
    backend.close()
    successful = [result for result in results if result["success"]]
    report = {
        "sample_count": SAMPLE_COUNT,
        "success_count": len(successful),
        "results": results,
        "summary": {
            backend_name: {
                "max_position_m": max(
                    result[backend_name]["position_m"]
                    for result in successful
                ),
                "max_rotation_rad": max(
                    result[backend_name]["rotation_rad"]
                    for result in successful
                ),
            }
            for backend_name in ("kdl", "pinocchio", "mujoco")
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for backend_name in ("kdl", "pinocchio", "mujoco"):
        values = [
            result[backend_name]["position_m"] for result in successful
        ]
        axis.plot(values, "o-", label=backend_name)
    axis.set_yscale("log")
    axis.set_xlabel("IK sample")
    axis.set_ylabel("position error / m")
    axis.set_title("KDL IK solution checked by each FK backend")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "errors.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report["summary"], indent=2))
    print(f"success: {len(successful)}/{SAMPLE_COUNT}")
    print(f"plot: {OUTPUT_DIR / 'errors.png'}")
    return 0


def _pose_error(target: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    position = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    rotation = float(
        Rotation.from_matrix(
            target[:3, :3] @ actual[:3, :3].T
        ).magnitude()
    )
    return {"position_m": position, "rotation_rad": rotation}


if __name__ == "__main__":
    raise SystemExit(main())
