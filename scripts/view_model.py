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
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Show world, installation, post, and energy-unit frames.",
    )
    parser.add_argument(
        "--locked-units",
        action="store_true",
        help="Keep energy units magnetically locked while viewing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = MuJoCoArmBackend(args.model)
    zero_command = TorqueCommand(np.zeros(6, dtype=np.float64))
    start_time = time.monotonic()
    last_render_time = start_time
    energy_unit_equalities = {}
    energy_unit_qpos_addresses = {}
    energy_unit_dof_addresses = {}
    energy_unit_extraction_distances = {}
    for index in range(1, 7):
        body_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"energy_unit_{index:02d}",
        )
        equality_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"storage_magnet_{index:02d}",
        )
        if body_id >= 0 and equality_id >= 0:
            energy_unit_equalities[body_id] = equality_id
            joint_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"energy_unit_{index:02d}_free",
            )
            if joint_id >= 0:
                energy_unit_qpos_addresses[body_id] = int(
                    backend.model.jnt_qposadr[joint_id]
                )
                energy_unit_dof_addresses[body_id] = int(
                    backend.model.jnt_dofadr[joint_id]
                )
                top_site_id = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"storage_post_{index:02d}_top",
                )
                energy_unit_extraction_distances[body_id] = (
                    2.0 * float(backend.model.site_pos[top_site_id, 2])
                )
    extraction_initial_poses = {}
    pulling_energy_units = set()
    released_energy_units: set[int] = set()
    collision_pending_units: set[int] = set()

    def try_enable_collision(body_id: int) -> bool:
        affected_geoms = [
            geom_id
            for geom_id in range(backend.model.ngeom)
            if int(backend.model.geom_bodyid[geom_id]) == body_id
        ]
        for geom_id in affected_geoms:
            backend.model.geom_conaffinity[geom_id] |= 2
        backend.model.body_conaffinity[body_id] |= 2
        mujoco.mj_forward(backend.model, backend.data)
        penetrates_station = False
        for contact in backend.data.contact:
            body1 = int(backend.model.geom_bodyid[int(contact.geom1)])
            body2 = int(backend.model.geom_bodyid[int(contact.geom2)])
            if body_id not in (body1, body2):
                continue
            other_body = body2 if body1 == body_id else body1
            other_name = mujoco.mj_id2name(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                other_body,
            )
            if other_name and (
                other_name == "energy_storage_solids"
                or other_name.startswith("storage_post_")
            ):
                penetrates_station = True
                break
        if penetrates_station:
            for geom_id in affected_geoms:
                backend.model.geom_conaffinity[geom_id] &= ~2
            backend.model.body_conaffinity[body_id] &= ~2
            return False
        collision_pending_units.discard(body_id)
        print(
            "Enabled resource-station collision for "
            f"{mujoco.mj_id2name(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )}"
        )
        return True

    with mujoco_viewer.launch_passive(backend.model, backend.data) as viewer:
        viewer.opt.geomgroup[5] = 1 if args.show_frames else 0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[:] = [-0.2, 0.0, 0.3]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -30.0
        while viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break
            selected_body_id = viewer.perturb.select
            if (
                not args.locked_units
                and selected_body_id in energy_unit_equalities
                and selected_body_id not in pulling_energy_units
                and selected_body_id not in released_energy_units
            ):
                equality_id = energy_unit_equalities[selected_body_id]
                backend.data.eq_active[equality_id] = 0
                qpos_address = energy_unit_qpos_addresses[selected_body_id]
                pose = backend.data.qpos[
                    qpos_address : qpos_address + 7
                ].copy()
                rotation = np.empty(9, dtype=np.float64)
                mujoco.mju_quat2Mat(rotation, pose[3:7])
                extraction_initial_poses[selected_body_id] = (
                    pose[:3].copy(),
                    pose[3:7].copy(),
                    rotation.reshape(3, 3)[:, 2].copy(),
                )
                pulling_energy_units.add(selected_body_id)
                name = mujoco.mj_id2name(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    selected_body_id,
                )
                print(f"Started magnetic extraction for {name}")
            if args.physics:
                elapsed = now - last_render_time
                steps = max(1, int(elapsed / backend.control_period_s))
                for _ in range(steps):
                    backend.step(zero_command)
            elif released_energy_units or pulling_energy_units:
                steps = max(
                    1,
                    round(1.0 / 60.0 / backend.model.opt.timestep),
                )
                for _ in range(steps):
                    mujoco.mj_step(backend.model, backend.data)
                    for body_id in tuple(pulling_energy_units):
                        qpos_address = energy_unit_qpos_addresses[body_id]
                        dof_address = energy_unit_dof_addresses[body_id]
                        (
                            origin,
                            initial_quaternion,
                            extraction_axis,
                        ) = extraction_initial_poses[body_id]
                        displacement = (
                            backend.data.qpos[
                                qpos_address : qpos_address + 3
                            ]
                            - origin
                        )
                        extraction_distance = (
                            energy_unit_extraction_distances[body_id]
                        )
                        axial_distance = float(
                            np.clip(
                                displacement @ extraction_axis,
                                0.0,
                                extraction_distance,
                            )
                        )
                        lateral_offset = displacement - (
                            float(displacement @ extraction_axis)
                            * extraction_axis
                        )
                        lateral_distance = float(
                            np.linalg.norm(lateral_offset)
                        )
                        if lateral_distance > 0.01:
                            lateral_offset *= 0.01 / lateral_distance
                        backend.data.qpos[
                            qpos_address : qpos_address + 3
                        ] = (
                            origin
                            + extraction_axis * axial_distance
                            + lateral_offset
                        )
                        current_quaternion = backend.data.qpos[
                            qpos_address + 3 : qpos_address + 7
                        ]
                        quaternion_dot = float(
                            np.clip(
                                abs(
                                    current_quaternion
                                    @ initial_quaternion
                                ),
                                0.0,
                                1.0,
                            )
                        )
                        rotation_angle = 2.0 * np.arccos(quaternion_dot)
                        if rotation_angle > np.deg2rad(10.0):
                            interpolation = np.deg2rad(10.0) / rotation_angle
                            reference_quaternion = initial_quaternion.copy()
                            if (
                                current_quaternion @ reference_quaternion
                                < 0.0
                            ):
                                reference_quaternion = -reference_quaternion
                            limited_quaternion = (
                                (1.0 - interpolation)
                                * reference_quaternion
                                + interpolation * current_quaternion
                            )
                            limited_quaternion /= np.linalg.norm(
                                limited_quaternion
                            )
                            backend.data.qpos[
                                qpos_address + 3 : qpos_address + 7
                            ] = limited_quaternion
                        linear_velocity = backend.data.qvel[
                            dof_address : dof_address + 3
                        ]
                        axial_velocity = float(
                            linear_velocity @ extraction_axis
                        )
                        backend.data.qvel[
                            dof_address : dof_address + 3
                        ] = axial_velocity * extraction_axis
                        backend.data.qvel[
                            dof_address + 3 : dof_address + 6
                        ] *= 0.5
                        if axial_distance >= extraction_distance - 1e-6:
                            pulling_energy_units.discard(body_id)
                            released_energy_units.add(body_id)
                            collision_pending_units.add(body_id)
                for body_id in tuple(collision_pending_units):
                    try_enable_collision(body_id)
            viewer.sync()
            last_render_time = now
            time.sleep(1.0 / 60.0)

    backend.close()


if __name__ == "__main__":
    main()
