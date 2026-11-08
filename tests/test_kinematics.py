"""KDL kinematics tests against MuJoCo and finite differences."""

from __future__ import annotations

import mujoco
import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation

from jgarm_sim import MuJoCoArmBackend
from jgarm_sim.dynamics import PinocchioDynamics
from jgarm_sim.kinematics import IKOptions, Kinematics


def test_kdl_fk_matches_mujoco_tcp() -> None:
    kinematics = Kinematics()
    backend = MuJoCoArmBackend()
    base_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base_link",
    )
    assert base_id >= 0
    random = np.random.default_rng(12)
    max_position_error = 0.0
    max_rotation_error = 0.0

    for _ in range(50):
        joints = random.uniform(
            kinematics.joint_limits_rad[:, 0],
            kinematics.joint_limits_rad[:, 1],
        )
        kdl_pose = kinematics.forward(joints)
        backend.reset(joints)
        mujoco_pose = backend.tcp_pose()
        mujoco_position = (
            mujoco_pose[:3, 3] - backend.data.xpos[base_id]
        )
        max_position_error = max(
            max_position_error,
            float(np.linalg.norm(kdl_pose[:3, 3] - mujoco_position)),
        )
        max_rotation_error = max(
            max_rotation_error,
            float(
                Rotation.from_matrix(
                    kdl_pose[:3, :3] @ mujoco_pose[:3, :3].T
                ).magnitude()
            ),
        )

    assert max_position_error < 1e-5
    assert max_rotation_error < 1e-4


def test_kdl_fk_matches_pinocchio() -> None:
    kinematics = Kinematics()
    dynamics = PinocchioDynamics()
    frame_id = dynamics.model.getFrameId("link_6")
    random = np.random.default_rng(13)
    max_position_error = 0.0
    max_rotation_error = 0.0
    for _ in range(50):
        joints = random.uniform(
            kinematics.joint_limits_rad[:, 0],
            kinematics.joint_limits_rad[:, 1],
        )
        kdl_pose = kinematics.forward(joints)
        pinocchio.framesForwardKinematics(
            dynamics.model,
            dynamics.data,
            joints,
        )
        pinocchio.updateFramePlacements(
            dynamics.model,
            dynamics.data,
        )
        pin_pose = np.asarray(
            dynamics.data.oMf[frame_id].homogeneous,
            dtype=np.float64,
        ).copy()
        max_position_error = max(
            max_position_error,
            float(np.linalg.norm(kdl_pose[:3, 3] - pin_pose[:3, 3])),
        )
        max_rotation_error = max(
            max_rotation_error,
            float(
                Rotation.from_matrix(
                    kdl_pose[:3, :3] @ pin_pose[:3, :3].T
                ).magnitude()
            ),
        )
    assert max_position_error < 1e-5
    assert max_rotation_error < 1e-4


def test_ik_solution_matches_pinocchio_fk() -> None:
    kinematics = Kinematics()
    dynamics = PinocchioDynamics()
    frame_id = dynamics.model.getFrameId("link_6")
    target_joints = np.array([0.1, -0.8, 0.8, 0.1, 0.3, -0.1])
    target_pose = kinematics.forward(target_joints)
    ik = kinematics.inverse(
        target_pose,
        np.zeros(6),
        IKOptions(max_attempts=5),
    )
    assert ik.success
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
    position_error = np.linalg.norm(target_pose[:3, 3] - pin_pose[:3, 3])
    rotation_error = Rotation.from_matrix(
        target_pose[:3, :3] @ pin_pose[:3, :3].T
    ).magnitude()
    assert position_error < 1e-4
    assert rotation_error < 1e-4


def test_kdl_jacobian_matches_finite_difference() -> None:
    kinematics = Kinematics()
    joints = np.array([0.2, -0.7, 0.6, 0.1, 0.3, -0.2])
    analytic = kinematics.jacobian(joints)
    numerical = np.zeros((6, 6))
    step = 1e-6
    base_pose = kinematics.forward(joints)

    for index in range(6):
        moved_joints = joints.copy()
        moved_joints[index] += step
        moved_pose = kinematics.forward(moved_joints)
        numerical[:3, index] = (
            moved_pose[:3, 3] - base_pose[:3, 3]
        ) / step
        numerical[3:, index] = (
            Rotation.from_matrix(
                moved_pose[:3, :3] @ base_pose[:3, :3].T
            ).as_rotvec()
            / step
        )

    np.testing.assert_allclose(analytic, numerical, atol=5e-4)


def test_kdl_ik_reaches_random_reachable_poses() -> None:
    kinematics = Kinematics()
    random = np.random.default_rng(8)
    success_count = 0
    max_position_error = 0.0
    max_rotation_error = 0.0

    for _ in range(100):
        joints = random.uniform(
            kinematics.joint_limits_rad[:, 0],
            kinematics.joint_limits_rad[:, 1],
        )
        result = kinematics.inverse(kinematics.forward(joints))
        if result.success:
            success_count += 1
            max_position_error = max(
                max_position_error, result.position_error_m
            )
            max_rotation_error = max(
                max_rotation_error, result.rotation_error_rad
            )

    assert success_count >= 95
    assert max_position_error < 1e-4
    assert max_rotation_error < 1e-3


def test_position_only_ik_preserves_seed_orientation() -> None:
    kinematics = Kinematics()
    seed = np.array([0.1, -0.7, 0.7, 0.1, 0.3, -0.1])
    target = kinematics.forward(seed)
    target[2, 3] += 0.02
    result = kinematics.inverse(
        target,
        seed,
        options=IKOptions(max_attempts=5),
        position_only=True,
    )
    assert result.success
