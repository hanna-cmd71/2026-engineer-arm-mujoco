"""Tests for the shared energy-unit extraction logic."""

from __future__ import annotations

from pathlib import Path
import unittest

from ament_index_python.packages import get_package_share_directory
import mujoco
import numpy as np

from jgarm_control.energy_units import EnergyUnitExtractor


class EnergyUnitExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_path = (
            Path(get_package_share_directory("jgarm_description"))
            / "model"
            / "scene.xml"
        )
        cls.model = mujoco.MjModel.from_xml_path(str(model_path))
        cls.power_on_key_id = mujoco.mj_name2id(
            cls.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "power_on",
        )
        if cls.power_on_key_id < 0:
            raise ValueError("scene.xml is missing the power_on keyframe")

    def setUp(self) -> None:
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.power_on_key_id,
        )
        mujoco.mj_forward(self.model, self.data)
        self.extractor = EnergyUnitExtractor(self.model, self.data)

    def test_discovers_all_six_units(self) -> None:
        self.assertEqual(len(self.extractor.unit_body_ids), 6)
        for body_id in self.extractor.unit_body_ids:
            self.assertGreater(
                self.extractor.extraction_distance(body_id),
                0.0,
            )

    def test_release_disables_magnet_and_marks_pulling(self) -> None:
        body_id = self.extractor.unit_body_ids[0]

        self.assertTrue(self.extractor.release(body_id))
        self.assertIn(body_id, self.extractor.pulling_body_ids)
        self.assertFalse(self.extractor.release(body_id))

        equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "storage_magnet_01",
        )
        self.assertEqual(self.data.eq_active[equality_id], 0)

    def test_extraction_completes_at_post_length(self) -> None:
        body_id = self.extractor.unit_body_ids[0]
        self.assertTrue(self.extractor.release(body_id))
        origin, _, axis = self.extractor.initial_pose(body_id)
        distance = self.extractor.extraction_distance(body_id)
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "energy_unit_01_free",
        )
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_address : qpos_address + 3] = (
            origin + axis * distance
        )

        self.extractor.constrain_pulling_units()

        self.assertIn(body_id, self.extractor.released_body_ids)
        self.assertIn(
            body_id,
            self.extractor.collision_pending_body_ids,
        )

    def test_constrain_limits_lateral_offset(self) -> None:
        body_id = self.extractor.unit_body_ids[0]
        self.assertTrue(self.extractor.release(body_id))
        origin, _, axis = self.extractor.initial_pose(body_id)
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "energy_unit_01_free",
        )
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        lateral = np.array([0.05, 0.0, 0.0])
        self.data.qpos[qpos_address : qpos_address + 3] = (
            origin + lateral
        )

        self.extractor.constrain_pulling_units()

        displacement = self.data.qpos[
            qpos_address : qpos_address + 3
        ] - origin
        lateral_offset = displacement - (
            float(displacement @ axis) * axis
        )
        self.assertLessEqual(np.linalg.norm(lateral_offset), 0.0101)

    def test_negative_clearance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnergyUnitExtractor(
                self.model,
                self.data,
                lateral_clearance_m=-0.01,
            )


if __name__ == "__main__":
    unittest.main()
