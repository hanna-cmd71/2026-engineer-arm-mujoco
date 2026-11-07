"""Model structure and geometry tests."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from jgarm_sim import JOINT_NAMES, MuJoCoArmBackend
from jgarm_sim.types import POWER_ON_POSITION_RAD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_BASE_HEIGHT_M = 0.097
ARM_BASE_TRANSLATION_M = np.array(
    [-0.20985272, 0.0, ARM_BASE_HEIGHT_M]
)
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


def test_model_has_six_arm_joints_and_actuators() -> None:
    backend = MuJoCoArmBackend()
    assert backend.model.nq == 55
    assert backend.model.nv == 49
    assert backend.model.nu == 6


def test_arm_base_mounts_on_mecanum_base() -> None:
    backend = MuJoCoArmBackend()
    base_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base_link",
    )
    chassis_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "mobile_base_chassis",
    )
    np.testing.assert_allclose(
        backend.data.xpos[base_id],
        ARM_BASE_TRANSLATION_M,
        atol=1e-8,
    )
    assert chassis_id >= 0
    for index in range(1, 5):
        assert (
            mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                (
                    "wheel_front_left_spin",
                    "wheel_front_right_spin",
                    "wheel_rear_left_spin",
                    "wheel_rear_right_spin",
                )[index - 1],
            )
            >= 0
        )


def test_joint_names_and_limits_match_urdf() -> None:
    backend = MuJoCoArmBackend()
    actual_names = tuple(
        backend.model.joint(backend._joint_ids[index]).name
        for index in range(len(JOINT_NAMES))
    )
    assert actual_names == JOINT_NAMES
    np.testing.assert_allclose(backend.joint_ranges, EXPECTED_RANGES)


def test_mesh_complexity_is_below_mujoco_limit() -> None:
    backend = MuJoCoArmBackend()
    assert backend.model.nmeshface < 300_000


def test_task_scene_contains_interactive_objects() -> None:
    backend = MuJoCoArmBackend()
    fixed_bodies = ("exchange_core", "energy_storage")
    for body_name in fixed_bodies:
        assert (
            mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            >= 0
        )

    for index in range(1, 7):
        body_name = f"energy_unit_{index:02d}"
        joint_name = f"energy_unit_{index:02d}_free"
        body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        joint_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        assert body_id >= 0
        assert joint_id >= 0
        assert (
            backend.model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_FREE
        )
        assert backend.model.body_mass[body_id] == 0.4
        magnet_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"storage_magnet_{index:02d}",
        )
        assert magnet_id >= 0
        assert backend.model.eq_active0[magnet_id]


def test_scene_assets_preserve_cad_scale_and_ground_origin() -> None:
    backend = MuJoCoArmBackend()
    floor_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    floor_z = backend.model.geom_pos[floor_id, 2]
    expected_sizes = {
        "exchange_core": [0.945, 0.905, 1.304],
        "energy_storage": [2.360, 0.902, 1.172],
        "energy_unit_01": [0.095, 0.095, 0.150],
    }

    for body_name, expected_size in expected_sizes.items():
        body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        world_vertices = []
        for geom_id in range(backend.model.ngeom):
            if int(backend.model.geom_bodyid[geom_id]) != body_id:
                continue
            if backend.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if int(backend.model.geom_group[geom_id]) != 2:
                continue
            mesh_id = backend.model.geom_dataid[geom_id]
            start = backend.model.mesh_vertadr[mesh_id]
            count = backend.model.mesh_vertnum[mesh_id]
            vertices = backend.model.mesh_vert[start : start + count]
            rotation = backend.data.geom_xmat[geom_id].reshape(3, 3)
            translation = backend.data.geom_xpos[geom_id]
            world_vertices.append(vertices @ rotation.T + translation)

        vertices = np.concatenate(world_vertices)
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        np.testing.assert_allclose(
            upper - lower,
            expected_size,
            rtol=0.0,
            atol=3e-3,
        )
        if body_name != "energy_unit_01":
            assert abs(lower[2] - floor_z) < 0.004


def test_scene_entity_orientation_matches_layout() -> None:
    backend = MuJoCoArmBackend()
    core_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "exchange_core",
    )
    storage_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "energy_storage",
    )
    floor_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    core_rotation = backend.data.xmat[core_id].reshape(3, 3)
    storage_rotation = backend.data.xmat[storage_id].reshape(3, 3)

    np.testing.assert_allclose(
        core_rotation[:, 0],
        [0.0, 1.0, 0.0],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        storage_rotation[:, 1],
        [0.0, 0.0, 1.0],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        backend.model.geom_size[floor_id],
        [5.0, 5.0, 0.05],
    )


def test_exchange_cylinder_is_marked_and_simplified() -> None:
    backend = MuJoCoArmBackend()
    marker_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "exchange_cylinder_marker",
    )
    cylinder_mesh_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_MESH,
        "exchange_cylinder_visual",
    )
    assert marker_id >= 0
    assert cylinder_mesh_id >= 0
    np.testing.assert_allclose(
        backend.data.site_xpos[marker_id],
        [0.9122086, -0.0799920, 0.6882113],
        atol=1e-5,
    )
    assert backend.model.mesh_facenum[cylinder_mesh_id] <= 300


def test_storage_cylinders_are_marked_and_simplified() -> None:
    backend = MuJoCoArmBackend()
    install_target_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "energy_unit_install_target",
    )
    assert install_target_id >= 0
    np.testing.assert_allclose(
        backend.data.site_xpos[install_target_id],
        [0.9122084, -0.0293238, 0.6882113],
        atol=1e-5,
    )

    for index in range(1, 7):
        site_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"energy_storage_cylinder_{index:02d}",
        )
        mesh_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_MESH,
            f"energy_storage_cylinder_{index:02d}_visual",
        )
        assert site_id >= 0
        assert mesh_id >= 0
        assert backend.model.mesh_facenum[mesh_id] <= 300


def test_resource_units_insert_along_z_in_real_gravity() -> None:
    backend = MuJoCoArmBackend()
    np.testing.assert_allclose(
        backend.model.opt.gravity,
        [0.0, 0.0, -9.81],
    )
    expected_axes = (
        [0.000549, 0.000033, -0.9999999],
        [0.86602, 0.00011, -0.50000],
        [-0.86604, -0.00014, -0.49998],
        [0.86603, 0.00011, 0.50000],
        [-0.86603, -0.00014, 0.50000],
        [-0.00055, 0.00003, 1.00000],
    )

    for index, expected_axis in enumerate(expected_axes, start=1):
        body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"energy_unit_{index:02d}",
        )
        post_body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"storage_post_{index:02d}",
        )
        insert_site_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"energy_unit_{index:02d}_insert_depth",
        )
        post_top_site_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"storage_post_{index:02d}_top",
        )
        unit_rotation = backend.data.xmat[body_id].reshape(3, 3)
        post_rotation = backend.data.xmat[post_body_id].reshape(3, 3)
        np.testing.assert_allclose(
            unit_rotation[:, 2],
            expected_axis,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            unit_rotation[:, 2],
            post_rotation[:, 2],
            atol=1e-6,
        )
        unit_insert_depth = (
            backend.data.xpos[body_id]
            + unit_rotation[:, 2] * 0.08
        )
        np.testing.assert_allclose(
            unit_insert_depth,
            backend.data.site_xpos[post_top_site_id],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.data.site_xpos[insert_site_id],
            backend.data.site_xpos[post_top_site_id],
            atol=1e-6,
        )


def test_extraction_distance_matches_each_post_length() -> None:
    backend = MuJoCoArmBackend()
    expected_lengths = (
        0.10203645,
        0.02299403,
        0.02314527,
        0.02323817,
        0.02345881,
        0.10210295,
    )
    for index, expected_length in enumerate(expected_lengths, start=1):
        top_site_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"storage_post_{index:02d}_top",
        )
        removal_distance = 2.0 * float(
            backend.model.site_pos[top_site_id, 2]
        )
        np.testing.assert_allclose(
            removal_distance,
            expected_length,
            atol=1e-6,
        )


def test_storage_posts_have_independent_z_frames() -> None:
    backend = MuJoCoArmBackend()
    hub = np.array([-0.20985272, 0.59829011, 0.59563672])
    for index in range(1, 7):
        post_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"storage_post_{index:02d}",
        )
        top_site_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"storage_post_{index:02d}_top",
        )
        assert post_id >= 0
        assert top_site_id >= 0
        rotation = backend.data.xmat[post_id].reshape(3, 3)
        assert np.isclose(np.linalg.norm(rotation[:, 2]), 1.0)
        toward_hub = hub - backend.data.xpos[post_id]
        toward_hub /= np.linalg.norm(toward_hub)
        np.testing.assert_allclose(
            rotation[:, 2],
            -toward_hub,
            atol=1e-4,
        )


def test_debug_frame_axes_are_available_in_group_five() -> None:
    backend = MuJoCoArmBackend()
    assert int(np.count_nonzero(backend.model.geom_group == 5)) == 44

    for index in range(1, 7):
        for prefix in ("storage_post", "energy_unit"):
            for axis in ("x", "y", "z"):
                geom_id = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{prefix}_{index:02d}_axis_{axis}",
                )
                assert geom_id >= 0

    for geom_name in (
        "world_axis_x",
        "world_axis_y",
        "world_axis_z",
        "assembly_E_axis_x",
        "assembly_E_axis_y",
        "assembly_E_axis_z",
        "assembly_insert_axis_minus_x",
    ):
        geom_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        assert geom_id >= 0

    target_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "energy_unit_install_target",
    )
    target_rotation = backend.data.site_xmat[target_id].reshape(3, 3)
    np.testing.assert_allclose(
        target_rotation[:, 2],
        [0.0, 1.0, 0.0],
        atol=1e-5,
    )


def test_both_mechanisms_have_physical_collision_geometry() -> None:
    backend = MuJoCoArmBackend()
    exchange_body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "exchange_core",
    )
    storage_body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "energy_storage_solids",
    )
    assert exchange_body_id >= 0
    assert storage_body_id >= 0
    exchange_colliders = 0
    storage_colliders = 0
    for geom_id in range(backend.model.ngeom):
        body_id = int(backend.model.geom_bodyid[geom_id])
        if body_id == exchange_body_id:
            exchange_colliders += 1
        if body_id == storage_body_id:
            storage_colliders += 1
    assert exchange_colliders >= 2
    assert storage_colliders >= 30


def test_released_energy_unit_collides_with_resource_station() -> None:
    backend = MuJoCoArmBackend()
    body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "energy_unit_01",
    )
    geom_ids = [
        geom_id
        for geom_id in range(backend.model.ngeom)
        if int(backend.model.geom_bodyid[geom_id]) == body_id
    ]
    assert geom_ids
    for geom_id in geom_ids:
        backend.model.geom_conaffinity[geom_id] |= 2
        assert backend.model.geom_conaffinity[geom_id] & 2
    backend.model.body_conaffinity[body_id] |= 2
    assert backend.model.body_conaffinity[body_id] & 2


def test_released_energy_unit_cannot_pass_through_resource_station() -> None:
    backend = MuJoCoArmBackend()
    body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "energy_unit_01",
    )
    equality_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        "storage_magnet_01",
    )
    backend.data.eq_active[equality_id] = 0
    for geom_id in range(backend.model.ngeom):
        if int(backend.model.geom_bodyid[geom_id]) == body_id:
            backend.model.geom_conaffinity[geom_id] |= 2
    backend.model.body_conaffinity[body_id] |= 2

    storage_collision = False
    for _ in range(1000):
        backend.data.xfrc_applied[body_id, 1] = 2.0
        mujoco.mj_step(backend.model, backend.data)
        for contact in backend.data.contact:
            geom1 = backend.model.geom(int(contact.geom1))
            geom2 = backend.model.geom(int(contact.geom2))
            names = {geom1.name, geom2.name}
            if any(
                name and "energy_storage_part" in name
                for name in names
            ):
                storage_collision = True
    assert storage_collision


def test_energy_unit_center_of_mass_and_collision_are_stable() -> None:
    backend = MuJoCoArmBackend()
    for index in range(1, 7):
        body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"energy_unit_{index:02d}",
        )
        np.testing.assert_allclose(
            backend.model.body_ipos[body_id],
            [0.0, 0.0, 0.0789116693050644],
            atol=1e-8,
        )
        collision_geoms = [
            geom_id
            for geom_id in range(backend.model.ngeom)
            if (
                int(backend.model.geom_bodyid[geom_id]) == body_id
                and backend.model.geom_contype[geom_id] == 4
            )
        ]
        assert len(collision_geoms) == 1
        geom_id = collision_geoms[0]
        assert (
            backend.model.geom_type[geom_id]
            == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        np.testing.assert_allclose(
            backend.model.geom_size[geom_id, :2],
            [0.0475, 0.075],
            atol=1e-8,
        )


def test_initial_state_is_finite_and_wheels_are_grounded() -> None:
    backend = MuJoCoArmBackend()
    feedback = backend.read_feedback()
    np.testing.assert_allclose(
        feedback.position_rad, POWER_ON_POSITION_RAD
    )
    assert np.all(np.isfinite(feedback.position_rad))
    assert np.all(np.isfinite(feedback.velocity_rad_s))
    assert np.all(feedback.alive)
    assert backend.data.ncon == 4


def test_gravity_load_is_visible_at_urdf_zero() -> None:
    backend = MuJoCoArmBackend()
    backend.reset(np.zeros(6))
    bias = np.array(
        [
            backend.data.qfrc_bias[
                backend.model.jnt_dofadr[
                    backend._joint_ids[index]
                ]
            ]
            for index in range(len(JOINT_NAMES))
        ]
    )

    assert abs(bias[1]) > 3.0
    assert abs(bias[2]) > 3.0


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
        mjcf_position = (
            mjcf_pose[:3, 3] - ARM_BASE_TRANSLATION_M
        )
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
