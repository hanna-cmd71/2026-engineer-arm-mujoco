"""KDL-based forward, differential, and numerical inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf

from jgarm_sim.types import JOINT_COUNT, JOINT_NAMES


DEFAULT_URDF_PATH = Path(__file__).resolve().parents[2] / "JGarm.urdf"
BASE_LINK = "base_link"
TIP_LINK = "link_6"


@dataclass(frozen=True)
class IKOptions:
    """Numerical IK limits and tolerances."""

    max_iterations: int = 300
    max_attempts: int = 20
    position_tolerance_m: float = 1e-4
    rotation_tolerance_rad: float = 1e-3
    damping: float = 1e-4
    random_seed: int = 10

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if self.rotation_tolerance_rad <= 0.0:
            raise ValueError("rotation_tolerance_rad must be positive")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")


@dataclass(frozen=True)
class IKResult:
    """Result of a multi-attempt KDL numerical IK solve."""

    success: bool
    position_rad: np.ndarray
    attempts: int
    iterations: int
    position_error_m: float
    rotation_error_rad: float
    solver_status: int
    message: str


class Kinematics:
    """KDL kinematics for the fixed-base JGarm chain."""

    def __init__(self, urdf_path: str | Path = DEFAULT_URDF_PATH) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        success, tree = kdl_parser_py.urdf.treeFromFile(str(self.urdf_path))
        if not success:
            raise ValueError(f"kdl_parser could not parse {self.urdf_path}")

        self.chain = tree.getChain(BASE_LINK, TIP_LINK)
        if self.chain.getNrOfJoints() != JOINT_COUNT:
            raise ValueError("The KDL chain must contain exactly six joints")

        self.joint_names = tuple(
            self.chain.getSegment(index).getJoint().getName()
            for index in range(self.chain.getNrOfSegments())
        )
        if self.joint_names != JOINT_NAMES:
            raise ValueError(
                f"Unexpected KDL joint order: {self.joint_names}"
            )

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
            joint = joints.get(name)
            if joint is None or joint.find("limit") is None:
                raise ValueError(f"URDF joint {name!r} has no limit")
            limit = joint.find("limit")
            limits[index] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )
        return limits

    @staticmethod
    def _coerce_joints(q: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        if values.shape != (JOINT_COUNT,):
            raise ValueError(f"q must contain {JOINT_COUNT} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("q must contain only finite values")
        return values.copy()

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

    def _coerce_pose(self, pose: np.ndarray) -> np.ndarray:
        target = np.asarray(pose, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError("target_pose must have shape (4, 4)")
        if not np.all(np.isfinite(target)):
            raise ValueError("target_pose must contain only finite values")
        rotation = target[:3, :3]
        if not np.allclose(
            rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-6
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("target_pose rotation must be orthonormal")
        if not np.allclose(target[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("target_pose must be homogeneous")
        return target.copy()

    def forward(self, q: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        """Returns the base-to-TCP pose as a 4x4 matrix."""
        joints = self._to_jnt_array(self._coerce_joints(q))
        frame = kdl.Frame()
        status = self._fk_solver.JntToCart(joints, frame)
        if status < 0:
            raise RuntimeError(f"KDL FK failed with status {status}")
        return self._from_kdl_frame(frame)

    def jacobian(
        self, q: np.ndarray | list[float] | tuple[float, ...]
    ) -> np.ndarray:
        """Returns the 6x6 spatial Jacobian in the KDL base frame."""
        joints = self._to_jnt_array(self._coerce_joints(q))
        jacobian = kdl.Jacobian(JOINT_COUNT)
        status = self._jacobian_solver.JntToJac(joints, jacobian)
        if status < 0:
            raise RuntimeError(f"KDL Jacobian failed with status {status}")
        result = np.zeros((6, JOINT_COUNT), dtype=np.float64)
        for row in range(6):
            for column in range(JOINT_COUNT):
                result[row, column] = jacobian[row, column]
        return result

    def pose_error(
        self, target_pose: np.ndarray, actual_pose: np.ndarray
    ) -> tuple[float, float]:
        """Returns translational and rotational pose error magnitudes."""
        position_error = float(
            np.linalg.norm(target_pose[:3, 3] - actual_pose[:3, 3])
        )
        target_rotation = self._to_kdl_frame(target_pose).M
        actual_rotation = self._to_kdl_frame(actual_pose).M
        rotation_error = float(
            abs((target_rotation * actual_rotation.Inverse()).GetRotAngle()[0])
        )
        return position_error, rotation_error

    def inverse(
        self,
        target_pose: np.ndarray,
        seed: np.ndarray | list[float] | tuple[float, ...] | None = None,
        options: IKOptions | None = None,
        position_only: bool = False,
    ) -> IKResult:
        """Solves numerical IK with KDL and deterministic random restarts."""
        options = IKOptions() if options is None else options
        target = self._coerce_pose(target_pose)
        initial = (
            np.zeros(JOINT_COUNT, dtype=np.float64)
            if seed is None
            else self._coerce_joints(seed)
        )
        initial = np.clip(
            initial, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1]
        )
        if position_only:
            target[:3, :3] = self.forward(initial)[:3, :3]

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
        best = initial.copy()
        best_position_error = np.inf
        best_rotation_error = np.inf
        last_status = -1
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
            for iteration in range(1, options.max_iterations + 1):
                q_next = kdl.JntArray(JOINT_COUNT)
                last_status = position_solver.CartToJnt(
                    self._to_jnt_array(q_current), target_frame, q_next
                )
                q_current = np.clip(
                    self._from_jnt_array(q_next),
                    self.joint_limits_rad[:, 0],
                    self.joint_limits_rad[:, 1],
                )
                last_iterations = iteration
                if last_status == 0:
                    break

            actual = self.forward(q_current)
            position_error, rotation_error = self.pose_error(target, actual)
            if (
                position_error < best_position_error
                or (
                    np.isclose(position_error, best_position_error)
                    and rotation_error < best_rotation_error
                )
            ):
                best = q_current
                best_position_error = position_error
                best_rotation_error = rotation_error

            success = (
                last_status == 0
                and position_error <= options.position_tolerance_m
                and rotation_error <= options.rotation_tolerance_rad
            )
            if success:
                return IKResult(
                    success=True,
                    position_rad=q_current,
                    attempts=attempt_index,
                    iterations=last_iterations,
                    position_error_m=position_error,
                    rotation_error_rad=rotation_error,
                    solver_status=last_status,
                    message=position_solver.strError(last_status),
                )

        return IKResult(
            success=False,
            position_rad=best,
            attempts=options.max_attempts,
            iterations=last_iterations,
            position_error_m=best_position_error,
            rotation_error_rad=best_rotation_error,
            solver_status=last_status,
            message="IK did not converge",
        )
