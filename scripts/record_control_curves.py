"""Record target, state, and torque curves from a running JGarm simulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from jgarm_control.types import (
    JOINT_COUNT,
    POWER_ON_POSITION_RAD,
)


class CurveRecorder(Node):
    def __init__(self) -> None:
        super().__init__("jgarm_control_curve_recorder")
        self.latest_position = np.full(JOINT_COUNT, np.nan)
        self.latest_velocity = np.full(JOINT_COUNT, np.nan)
        self.latest_command = np.full(JOINT_COUNT, np.nan)
        self.latest_gravity = np.full(JOINT_COUNT, np.nan)
        self.latest_pd = np.full(JOINT_COUNT, np.nan)
        self.latest_friction = np.full(JOINT_COUNT, np.nan)
        self.latest_breakaway = np.full(JOINT_COUNT, np.nan)
        self.latest_mujoco_friction = np.full(JOINT_COUNT, np.nan)
        self.target = np.asarray(POWER_ON_POSITION_RAD, dtype=np.float64)
        self.base_target = self.target.copy()
        self.target_velocity = np.zeros(JOINT_COUNT)
        self.target_acceleration = np.zeros(JOINT_COUNT)
        self.samples = []
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._state_callback,
            qos,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/control/effort_cmd",
            lambda message: self._vector_callback(
                message, "latest_command"
            ),
            10,
        )
        for topic, attribute in (
            ("/arm/debug/gravity_effort", "latest_gravity"),
            ("/arm/debug/pd_effort", "latest_pd"),
            ("/arm/debug/friction_effort", "latest_friction"),
            ("/arm/debug/breakaway_effort", "latest_breakaway"),
            (
                "/arm/debug/mujoco_friction_effort",
                "latest_mujoco_friction",
            ),
        ):
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda message, name=attribute: self._vector_callback(
                    message, name
                ),
                10,
            )
        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/arm/target/joint_position",
            10,
        )
        self.velocity_publisher = self.create_publisher(
            Float64MultiArray,
            "/arm/target/joint_velocity",
            10,
        )
        self.acceleration_publisher = self.create_publisher(
            Float64MultiArray,
            "/arm/target/joint_acceleration",
            10,
        )

    def _state_callback(self, message: JointState) -> None:
        index = {name: i for i, name in enumerate(message.name)}
        for joint_index in range(JOINT_COUNT):
            name = f"joint_{joint_index + 1}"
            source = index.get(name)
            if source is None:
                continue
            if source < len(message.position):
                self.latest_position[joint_index] = message.position[source]
            if source < len(message.velocity):
                self.latest_velocity[joint_index] = message.velocity[source]

    def _vector_callback(
        self,
        message: Float64MultiArray,
        attribute: str,
    ) -> None:
        if len(message.data) != JOINT_COUNT:
            return
        values = np.asarray(message.data, dtype=np.float64)
        if np.all(np.isfinite(values)):
            setattr(self, attribute, values.copy())

    def wait_for_state(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if np.all(np.isfinite(self.latest_position)):
                self.target = self.latest_position.copy()
                self.base_target = self.target.copy()
                return
        raise RuntimeError("no fresh /joint_states received")

    def publish_target(self) -> None:
        position = Float64MultiArray()
        position.data = self.target.tolist()
        self.publisher.publish(position)
        velocity = Float64MultiArray()
        velocity.data = self.target_velocity.tolist()
        self.velocity_publisher.publish(velocity)
        acceleration = Float64MultiArray()
        acceleration.data = self.target_acceleration.tolist()
        self.acceleration_publisher.publish(acceleration)

    def record(self, elapsed_s: float) -> None:
        self.samples.append(
            (
                elapsed_s,
                *self.target.tolist(),
                *self.latest_position.tolist(),
                *self.latest_velocity.tolist(),
                *self.latest_command.tolist(),
                *self.latest_gravity.tolist(),
                *self.latest_pd.tolist(),
                *self.latest_friction.tolist(),
                *self.latest_breakaway.tolist(),
                *self.latest_mujoco_friction.tolist(),
            )
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ["time_s"]
        header += [f"target{i}" for i in range(1, 7)]
        header += [f"q{i}" for i in range(1, 7)]
        header += [f"dq{i}" for i in range(1, 7)]
        header += [f"tau_cmd{i}" for i in range(1, 7)]
        header += [f"gravity{i}" for i in range(1, 7)]
        header += [f"pd{i}" for i in range(1, 7)]
        header += [f"friction{i}" for i in range(1, 7)]
        header += [f"breakaway{i}" for i in range(1, 7)]
        header += [f"mujoco_friction{i}" for i in range(1, 7)]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="pd")
    parser.add_argument("--duration", type=float, default=3.5)
    parser.add_argument("--step-time", type=float, default=1.0)
    parser.add_argument("--step-joint", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument(
        "--hold-power-on",
        action="store_true",
        help="Record a hold at the configured power-on pose.",
    )
    parser.add_argument(
        "--profile",
        choices=("step", "sine", "sweep"),
        default="step",
    )
    parser.add_argument("--frequency-hz", type=float, default=0.5)
    parser.add_argument("--amplitude-rad", type=float, default=0.05)
    parser.add_argument(
        "--center-offset-rad",
        type=float,
        default=0.0,
        help="Offset the selected joint away from its power-on limit.",
    )
    parser.add_argument(
        "--frequencies",
        default="0.2,0.5,1.0,2.0,3.0",
        help="Comma-separated sweep frequencies in Hz.",
    )
    parser.add_argument("--dwell", type=float, default=1.5)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/control_curves_pd.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.step_joint <= JOINT_COUNT:
        raise SystemExit("--step-joint must be between 1 and 6")
    frequencies = [
        float(value)
        for value in args.frequencies.split(",")
        if value.strip()
    ]
    if args.profile == "sweep" and (
        not frequencies
        or any(not np.isfinite(value) or value <= 0.0 for value in frequencies)
    ):
        raise SystemExit("--frequencies must contain positive values")
    if args.dwell <= 0.0:
        raise SystemExit("--dwell must be positive")
    rclpy.init()
    node = CurveRecorder()
    try:
        node.wait_for_state()
        if args.hold_power_on:
            node.target = np.asarray(
                POWER_ON_POSITION_RAD,
                dtype=np.float64,
            )
            node.base_target = node.target.copy()
        node.base_target[args.step_joint - 1] += args.center_offset_rad
        node.target = node.base_target.copy()
        start = time.monotonic()
        next_publish = 0.0
        next_sample = 0.0
        stepped = False
        while rclpy.ok() and time.monotonic() - start < args.duration:
            rclpy.spin_once(node, timeout_sec=0.002)
            elapsed = time.monotonic() - start
            if elapsed >= next_publish:
                node.publish_target()
                next_publish += 0.02
            if elapsed >= args.step_time and not stepped:
                if args.profile == "step":
                    node.target[args.step_joint - 1] += args.step_size
                stepped = True
            if stepped and args.profile == "sine":
                phase = (
                    2.0
                    * math.pi
                    * args.frequency_hz
                    * (elapsed - args.step_time)
                )
                node.target = node.base_target.copy()
                node.target[args.step_joint - 1] += (
                    args.amplitude_rad * math.sin(phase)
                )
                node.target_velocity.fill(0.0)
                node.target_velocity[args.step_joint - 1] = (
                    args.amplitude_rad
                    * 2.0
                    * math.pi
                    * args.frequency_hz
                    * math.cos(phase)
                )
                node.target_acceleration.fill(0.0)
                node.target_acceleration[args.step_joint - 1] = (
                    -args.amplitude_rad
                    * (2.0 * math.pi * args.frequency_hz) ** 2
                    * math.sin(phase)
                )
            if stepped and args.profile == "sweep":
                sweep_time = elapsed - args.step_time
                index = min(
                    int(sweep_time / args.dwell),
                    len(frequencies) - 1,
                )
                frequency = frequencies[index]
                segment_start = (
                    args.step_time + index * args.dwell
                )
                phase = (
                    2.0
                    * math.pi
                    * frequency
                    * (elapsed - segment_start)
                )
                node.target = node.base_target.copy()
                node.target[args.step_joint - 1] += (
                    args.amplitude_rad * math.sin(phase)
                )
                node.target_velocity.fill(0.0)
                node.target_velocity[args.step_joint - 1] = (
                    args.amplitude_rad
                    * 2.0
                    * math.pi
                    * frequency
                    * math.cos(phase)
                )
                node.target_acceleration.fill(0.0)
                node.target_acceleration[args.step_joint - 1] = (
                    -args.amplitude_rad
                    * (2.0 * math.pi * frequency) ** 2
                    * math.sin(phase)
                )
            if elapsed >= next_sample:
                node.record(elapsed)
                next_sample += 0.01
        node.save(args.output)
        metadata_path = (
            args.metadata
            if args.metadata is not None
            else args.output.with_suffix(".json")
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "profile": args.profile,
                    "joint": args.step_joint,
                    "step_time_s": args.step_time,
                    "amplitude_rad": args.amplitude_rad,
                    "center_offset_rad": args.center_offset_rad,
                    "frequency_hz": args.frequency_hz,
                    "frequencies_hz": frequencies,
                    "dwell_s": args.dwell,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved {args.output} ({len(node.samples)} samples)")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
