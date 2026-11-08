"""Nominal physical model and uncertainty envelope for JGarm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pinocchio
import yaml

from jgarm_sim.dynamics import PinocchioDynamics
from jgarm_sim.identification import gravity_load
from jgarm_sim.types import JOINT_COUNT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "ros2_ws"
    / "src"
    / "jgarm_bringup"
    / "config"
    / "nominal_model.yaml"
)


@dataclass(frozen=True)
class LinkInertial:
    name: str
    mass_kg: float
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray


@dataclass(frozen=True)
class UncertaintyEnvelope:
    mass_scale: tuple[float, float] = (0.9, 1.1)
    com_offset_m: tuple[float, float] = (-0.005, 0.005)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    breakaway_scale: tuple[float, float] = (0.5, 1.5)
    actuator_time_constant_s: tuple[float, float] = (0.002, 0.01)
    feedback_delay_s: tuple[float, float] = (0.0, 0.02)
    payload_mass_kg: tuple[float, float] = (0.0, 0.5)

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            low, high = getattr(self, field_name)
            if not (
                np.isfinite(low)
                and np.isfinite(high)
                and low <= high
            ):
                raise ValueError(
                    f"{field_name} must be a finite ordered range"
                )

    def sample(self, random: np.random.Generator) -> dict[str, float]:
        return {
            name: float(random.uniform(low, high))
            for name, (low, high) in (
                (field_name, getattr(self, field_name))
                for field_name in self.__dataclass_fields__
            )
        }


class NominalArmModel:
    """URDF mass/COM model with the reATRM profile kept as a comparison."""

    def __init__(
        self,
        urdf_path: str | Path,
        profile_path: str | Path,
        *,
        gravity_model: str = "physical",
        uncertainty: UncertaintyEnvelope | None = None,
    ) -> None:
        if gravity_model not in ("physical", "empirical"):
            raise ValueError("gravity_model must be physical or empirical")
        self.urdf_path = Path(urdf_path).resolve()
        self.profile_path = Path(profile_path).resolve()
        self.gravity_model = gravity_model
        self.uncertainty = uncertainty or UncertaintyEnvelope()
        self.links = load_link_inertials(self.urdf_path)
        self.dynamics = PinocchioDynamics(self.urdf_path)
        self.profile = load_hardware_profile(self.profile_path)

    @property
    def total_mass_kg(self) -> float:
        return float(sum(link.mass_kg for link in self.links.values()))

    def center_of_mass(self, position_rad) -> np.ndarray:
        position = np.asarray(position_rad, dtype=np.float64).reshape(-1)
        if position.shape != (JOINT_COUNT,):
            raise ValueError("position_rad must contain six values")
        pinocchio.forwardKinematics(
            self.dynamics.model,
            self.dynamics.data,
            position,
        )
        return np.asarray(
            pinocchio.centerOfMass(
                self.dynamics.model,
                self.dynamics.data,
                position,
            ),
            dtype=np.float64,
        ).copy()

    def physical_gravity_torque(self, position_rad) -> np.ndarray:
        return self.dynamics.gravity_torque(position_rad)

    def empirical_gravity_torque(self, position_rad) -> np.ndarray:
        position = np.asarray(position_rad, dtype=np.float64).reshape(-1)
        if position.shape != (JOINT_COUNT,):
            raise ValueError("position_rad must contain six values")
        return gravity_load(
            position,
            self.profile["gravity_p_nm"],
            self.profile["gravity_q_nm"],
            self.profile["gravity_bias_nm"],
            j5_config_weight=self.profile["gravity_j5_config_weight"],
            j5_conservative_coupling=(
                self.profile["gravity_j5_conservative_coupling"]
            ),
        )

    def gravity_torque(self, position_rad) -> np.ndarray:
        if self.gravity_model == "physical":
            return self.physical_gravity_torque(position_rad)
        return self.empirical_gravity_torque(position_rad)

    def mass_com_table(self) -> list[dict[str, object]]:
        return [
            {
                "name": link.name,
                "mass_kg": link.mass_kg,
                "com_m": link.com_m.tolist(),
                "inertia_kg_m2": link.inertia_kg_m2.tolist(),
            }
            for link in self.links.values()
        ]


def load_link_inertials(
    urdf_path: str | Path,
) -> dict[str, LinkInertial]:
    root = ElementTree.parse(str(urdf_path)).getroot()
    result = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            continue
        origin = inertial.find("origin")
        com = np.zeros(3)
        if origin is not None and origin.get("xyz"):
            com = np.fromstring(origin.get("xyz"), sep=" ")
        matrix = np.array(
            [
                [
                    float(inertia.get("ixx")),
                    float(inertia.get("ixy")),
                    float(inertia.get("ixz")),
                ],
                [
                    float(inertia.get("ixy")),
                    float(inertia.get("iyy")),
                    float(inertia.get("iyz")),
                ],
                [
                    float(inertia.get("ixz")),
                    float(inertia.get("iyz")),
                    float(inertia.get("izz")),
                ],
            ]
        )
        name = link.get("name")
        if name is None:
            continue
        result[name] = LinkInertial(
            name=name,
            mass_kg=float(mass.get("value")),
            com_m=com,
            inertia_kg_m2=matrix,
        )
    return result


def load_hardware_profile(path: str | Path) -> dict[str, object]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    parameters = document["/**"]["ros__parameters"]
    return {
        key.removeprefix("hardware_profile."): value
        for key, value in parameters.items()
        if key.startswith("hardware_profile.")
    }


def load_nominal_model(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    urdf_path: str | Path | None = None,
) -> NominalArmModel:
    path = Path(config_path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = document["nominal_model"]
    source_urdf = Path(config["source_urdf"])
    if not source_urdf.is_absolute():
        source_urdf = PROJECT_ROOT / source_urdf
    profile_path = Path(config["hardware_profile"])
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    uncertainty = UncertaintyEnvelope(**{
        key: tuple(value)
        for key, value in config.get("uncertainty", {}).items()
    })
    return NominalArmModel(
        urdf_path or source_urdf,
        profile_path,
        gravity_model=str(config.get("gravity_model", "physical")),
        uncertainty=uncertainty,
    )
