"""Model structure and geometry tests."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation

from jgarm_sim import JOINT_NAMES, MuJoCoArmBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RANGES = np.array(
    [
        [-3.14, 3.14],
        [-1.3265, 1.9024],
        [-0.5934, 1.35],
        [-3.14, 3.14],
        [-2.094395, 2.094395],
        [-3.14, 3.14],
    ]
)


def _urdf_tcp_pose(joints: np.ndarray) -> np.ndarray:
    """Computes link_6 pose directly from JGarm.urdf transforms."""
    root = ElementTree.parse(PROJECT_ROOT / "JGarm.urdf").getroot()
    by_child = {
        joint.find("child").get("link"): joint for joint in root.findall("joint")
    }
    chain = []
    child = "link_6"
    while child in by_child:
        joint = by_child[child]
        chain.append(joint)
        child = joint.find("parent").get("link")
    chain.reverse()

    transform = np.eye(4)
    for joint, position in zip(chain, joints):
        origin = joint.find("origin")
        joint_transform = np.eye(4)
        joint_transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy"), sep=" ")
        ).as_matrix()
        joint_transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")

        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec(
            axis / np.linalg.norm(axis) * position
        ).as_matrix()
        transform = transform @ joint_transform @ rotation
    return transform


def test_model_has_six_joints_and_actuators() -> None:
    backend = MuJoCoArmBackend()
    assert backend.model.nq == 6
    assert backend.model.nv == 6
    assert backend.model.nu == 6


def test_joint_names_and_limits_match_urdf() -> None:
    backend = MuJoCoArmBackend()
    actual_names = tuple(
        backend.model.joint(index).name for index in range(backend.model.njnt)
    )
    assert actual_names == JOINT_NAMES
    np.testing.assert_allclose(backend.joint_ranges, EXPECTED_RANGES)


def test_mesh_complexity_is_below_mujoco_limit() -> None:
    backend = MuJoCoArmBackend()
    assert backend.model.nmeshface < 200_000


def test_initial_state_is_finite_and_collision_free() -> None:
    backend = MuJoCoArmBackend()
    feedback = backend.read_feedback()
    assert np.all(np.isfinite(feedback.position_rad))
    assert np.all(np.isfinite(feedback.velocity_rad_s))
    assert np.all(feedback.alive)
    assert backend.data.ncon == 0


def test_mjcf_tcp_pose_matches_urdf_chain() -> None:
    backend = MuJoCoArmBackend()
    random = np.random.default_rng(7)
    max_position_error = 0.0
    max_rotation_error = 0.0

    for _ in range(100):
        joints = random.uniform(
            backend.joint_ranges[:, 0], backend.joint_ranges[:, 1]
        )
        backend.reset(joints)
        urdf_pose = _urdf_tcp_pose(joints)
        mjcf_pose = backend.tcp_pose()
        mjcf_position = mjcf_pose[:3, 3]
        mjcf_rotation = mjcf_pose[:3, :3]
        max_position_error = max(
            max_position_error,
            float(np.linalg.norm(urdf_pose[:3, 3] - mjcf_position)),
        )
        max_rotation_error = max(
            max_rotation_error,
            float(
                Rotation.from_matrix(
                    urdf_pose[:3, :3] @ mjcf_rotation.T
                ).magnitude()
            ),
        )

    assert max_position_error < 1e-5
    assert max_rotation_error < 1e-4
