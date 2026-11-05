"""KDL forward, differential, and numerical inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf as kdl_urdf

from jgarm_control.types import JOINT_COUNT, JOINT_NAMES


BASE_LINK = "base_link"
TIP_LINK = "link_6"


@dataclass(frozen=True)
class IKOptions:
    max_iterations: int = 300
    max_attempts: int = 20
    position_tolerance_m: float = 1e-4
    rotation_tolerance_rad: float = 1e-3
    damping: float = 1e-4
    random_seed: int = 10


@dataclass(frozen=True)
class IKResult:
    success: bool
    position_rad: np.ndarray
    attempts: int
    iterations: int
    position_error_m: float
    rotation_error_rad: float
    message: str


class Kinematics:
    def __init__(self, urdf_path: str | Path) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        success, tree = kdl_urdf.treeFromFile(str(self.urdf_path))
        if not success:
            raise ValueError(f"kdl_parser could not parse {self.urdf_path}")
        self.chain = tree.getChain(BASE_LINK, TIP_LINK)
        if self.chain.getNrOfJoints() != JOINT_COUNT:
            raise ValueError("KDL chain must contain six joints")
        self.joint_limits_rad = self._load_joint_limits()
        self._q_min = self._to_jnt_array(self.joint_limits_rad[:, 0])
        self._q_max = self._to_jnt_array(self.joint_limits_rad[:, 1])
        self._fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self._jacobian_solver = kdl.ChainJntToJacSolver(self.chain)

    def _load_joint_limits(self) -> np.ndarray:
        root = ElementTree.parse(self.urdf_path).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        limits = np.zeros((JOINT_COUNT, 2), dtype=np.float64)
        for index, name in enumerate(JOINT_NAMES):
            limit = joints[name].find("limit")
            if limit is None:
                raise ValueError(f"URDF joint {name} has no limits")
            limits[index] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )
        return limits

    @staticmethod
    def _coerce_joints(values) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (JOINT_COUNT,):
            raise ValueError(f"q must contain {JOINT_COUNT} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("q must contain only finite values")
        return vector.copy()

    @staticmethod
    def _to_jnt_array(values: np.ndarray) -> kdl.JntArray:
        joints = kdl.JntArray(JOINT_COUNT)
        for index, value in enumerate(values):
            joints[index] = float(value)
        return joints

    @staticmethod
    def _from_jnt_array(values: kdl.JntArray) -> np.ndarray:
        return np.array([values[index] for index in range(JOINT_COUNT)])

    @staticmethod
    def _to_kdl_frame(pose: np.ndarray) -> kdl.Frame:
        rotation = pose[:3, :3]
        rotation_kdl = kdl.Rotation(
            rotation[0, 0],
            rotation[0, 1],
            rotation[0, 2],
            rotation[1, 0],
            rotation[1, 1],
            rotation[1, 2],
            rotation[2, 0],
            rotation[2, 1],
            rotation[2, 2],
        )
        return kdl.Frame(rotation_kdl, kdl.Vector(*pose[:3, 3]))

    @staticmethod
    def _from_kdl_frame(frame: kdl.Frame) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        for row in range(3):
            for column in range(3):
                pose[row, column] = frame.M[row, column]
            pose[row, 3] = frame.p[row]
        return pose

    def forward(self, q) -> np.ndarray:
        frame = kdl.Frame()
        status = self._fk_solver.JntToCart(
            self._to_jnt_array(self._coerce_joints(q)), frame
        )
        if status < 0:
            raise RuntimeError(f"KDL FK failed with status {status}")
        return self._from_kdl_frame(frame)

    def pose_error(
        self, target: np.ndarray, actual: np.ndarray
    ) -> tuple[float, float]:
        position_error = float(
            np.linalg.norm(target[:3, 3] - actual[:3, 3])
        )
        target_rotation = self._to_kdl_frame(target).M
        actual_rotation = self._to_kdl_frame(actual).M
        rotation_error = abs(
            (target_rotation * actual_rotation.Inverse()).GetRotAngle()[0]
        )
        return position_error, float(rotation_error)

    def inverse(
        self,
        target_pose: np.ndarray,
        seed,
        options: IKOptions | None = None,
    ) -> IKResult:
        options = IKOptions() if options is None else options
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_pose must be a finite 4x4 matrix")
        initial = np.clip(
            self._coerce_joints(seed),
            self.joint_limits_rad[:, 0],
            self.joint_limits_rad[:, 1],
        )
        random = np.random.default_rng(options.random_seed)
        attempts = np.vstack(
            [
                initial,
                random.uniform(
                    self.joint_limits_rad[:, 0],
                    self.joint_limits_rad[:, 1],
                    size=(options.max_attempts - 1, JOINT_COUNT),
                ),
            ]
        )
        target_frame = self._to_kdl_frame(target)
        best_q = initial
        best_position_error = np.inf
        best_rotation_error = np.inf
        last_iterations = 0

        for attempt_index, q_start in enumerate(attempts, start=1):
            velocity_solver = kdl.ChainIkSolverVel_pinv(
                self.chain, eps=options.damping
            )
            position_solver = kdl.ChainIkSolverPos_NR_JL(
                self.chain,
                self._q_min,
                self._q_max,
                self._fk_solver,
                velocity_solver,
                maxiter=1,
                eps=1e-6,
            )
            q_current = q_start.copy()
            status = -1
            for iteration in range(1, options.max_iterations + 1):
                q_next = kdl.JntArray(JOINT_COUNT)
                status = position_solver.CartToJnt(
                    self._to_jnt_array(q_current), target_frame, q_next
                )
                q_current = np.clip(
                    self._from_jnt_array(q_next),
                    self.joint_limits_rad[:, 0],
                    self.joint_limits_rad[:, 1],
                )
                last_iterations = iteration
                if status == 0:
                    break

            actual = self.forward(q_current)
            position_error, rotation_error = self.pose_error(target, actual)
            if position_error < best_position_error:
                best_q = q_current
                best_position_error = position_error
                best_rotation_error = rotation_error
            success = (
                status == 0
                and position_error <= options.position_tolerance_m
                and rotation_error <= options.rotation_tolerance_rad
            )
            if success:
                return IKResult(
                    True,
                    q_current,
                    attempt_index,
                    last_iterations,
                    position_error,
                    rotation_error,
                    position_solver.strError(status),
                )

        return IKResult(
            False,
            best_q,
            options.max_attempts,
            last_iterations,
            best_position_error,
            best_rotation_error,
            "IK did not converge",
        )
