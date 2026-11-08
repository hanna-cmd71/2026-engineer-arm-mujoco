"""Compare KDL, Pinocchio, and MuJoCo TCP Jacobians."""

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


OUTPUT_DIR = PROJECT_ROOT / "results" / "kinematics_jacobian_check"
SAMPLE_COUNT = 100


def main() -> int:
    kdl = Kinematics()
    dynamics = PinocchioDynamics()
    frame_id = dynamics.model.getFrameId("link_6")
    backend = MuJoCoArmBackend()
    site_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "tcp",
    )
    random = np.random.default_rng(42)
    positions = random.uniform(
        backend.joint_ranges[:, 0],
        backend.joint_ranges[:, 1],
        size=(SAMPLE_COUNT, 6),
    )
    comparisons = {
        "kdl_analytic_vs_fd": [],
        "kdl_vs_pinocchio": [],
        "kdl_vs_mujoco": [],
        "pinocchio_vs_mujoco": [],
    }
    for position in positions:
        kdl_jacobian = kdl.jacobian(position)
        finite_difference = _finite_difference_jacobian(kdl, position)
        pinocchio.computeJointJacobians(
            dynamics.model,
            dynamics.data,
            position,
        )
        pinocchio.updateFramePlacements(
            dynamics.model,
            dynamics.data,
        )
        pin_jacobian = np.asarray(
            pinocchio.computeFrameJacobian(
                dynamics.model,
                dynamics.data,
                position,
                frame_id,
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ),
            dtype=np.float64,
        )
        backend.reset(position)
        linear = np.zeros((3, backend.model.nv))
        angular = np.zeros((3, backend.model.nv))
        mujoco.mj_jacSite(
            backend.model,
            backend.data,
            linear,
            angular,
            site_id,
        )
        mujoco_jacobian = np.vstack(
            [
                linear[:, backend._joint_dof_adr],
                angular[:, backend._joint_dof_adr],
            ]
        )
        comparisons["kdl_analytic_vs_fd"].append(
            _difference(kdl_jacobian, finite_difference)
        )
        comparisons["kdl_vs_pinocchio"].append(
            _difference(kdl_jacobian, pin_jacobian)
        )
        comparisons["kdl_vs_mujoco"].append(
            _difference(kdl_jacobian, mujoco_jacobian)
        )
        comparisons["pinocchio_vs_mujoco"].append(
            _difference(pin_jacobian, mujoco_jacobian)
        )
    backend.close()

    report = {
        "sample_count": SAMPLE_COUNT,
        "comparisons": {
            name: {
                "max_abs": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(np.square(values)))),
                "values": values,
            }
            for name, values in comparisons.items()
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for name, values in comparisons.items():
        axis.plot(values, label=name)
    axis.set_yscale("log")
    axis.set_xlabel("sample")
    axis.set_ylabel("max absolute Jacobian error")
    axis.set_title("JGarm TCP Jacobian consistency")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "errors.png", dpi=160)
    plt.close(figure)
    print(json.dumps({
        name: {
            "max_abs": float(np.max(values)),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
        }
        for name, values in comparisons.items()
    }, indent=2))
    print(f"plot: {OUTPUT_DIR / 'errors.png'}")
    return 0


def _finite_difference_jacobian(
    kinematics: Kinematics,
    position: np.ndarray,
) -> np.ndarray:
    step = 1e-6
    jacobian = np.zeros((6, 6))
    for index in range(6):
        plus = position.copy()
        minus = position.copy()
        plus[index] += step
        minus[index] -= step
        plus_pose = kinematics.forward(plus)
        minus_pose = kinematics.forward(minus)
        jacobian[:3, index] = (
            plus_pose[:3, 3] - minus_pose[:3, 3]
        ) / (2.0 * step)
        delta_rotation = plus_pose[:3, :3] @ minus_pose[:3, :3].T
        jacobian[3:, index] = (
            Rotation.from_matrix(delta_rotation).as_rotvec()
            / (2.0 * step)
        )
    return jacobian


def _difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


if __name__ == "__main__":
    raise SystemExit(main())
