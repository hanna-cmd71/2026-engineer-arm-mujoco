"""High-rate JGarm joint controller and KDL pose target handler."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool

from jgarm_control.controller import (
    JointImpedanceController,
    PDGains,
    PDGravityController,
)
from jgarm_control.dynamics import PinocchioDynamics
from jgarm_control.hardware_profile import EngineerHardwareProfile
from jgarm_control.kinematics import IKOptions, Kinematics
from jgarm_control.types import (
    JOINT_COUNT,
    JOINT_NAMES,
    JointFeedback,
    POWER_ON_POSITION_RAD,
    TorqueCommand,
    vector6,
)


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-12:
        raise ValueError("Pose orientation quaternion is invalid")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class ArmControlNode(Node):
    def __init__(
        self, parameter_overrides: list[Parameter] | None = None
    ) -> None:
        super().__init__(
            "arm_control_node",
            parameter_overrides=parameter_overrides,
        )
        description_share = Path(
            get_package_share_directory("jgarm_description")
        )
        urdf_path = Path(
            self.declare_parameter(
                "urdf_path", str(description_share / "urdf" / "JGarm.urdf")
            ).value
        )
        kp = self.declare_parameter(
            "kp", [60.0, 100.0, 100.0, 40.0, 20.0, 10.0]
        ).value
        kd = self.declare_parameter(
            "kd", [2.0, 4.0, 4.0, 1.5, 1.0, 0.5]
        ).value
        gravity_enabled = bool(
            self.declare_parameter("gravity_enabled", True).value
        )
        breakaway_enabled = bool(
            self.declare_parameter("breakaway_enabled", True).value
        )
        friction_enabled = bool(
            self.declare_parameter("friction_enabled", True).value
        )
        impedance_kp_scale = np.asarray(
            self.declare_parameter(
                "impedance_kp_scale",
                [3.0, 3.0, 3.0, 1.0, 3.0, 1.0],
            ).value,
            dtype=np.float64,
        ).reshape(-1)
        impedance_kd_scale = np.asarray(
            self.declare_parameter(
                "impedance_kd_scale",
                [3.0, 3.0, 3.0, 2.0, 3.0, 2.0],
            ).value,
            dtype=np.float64,
        ).reshape(-1)
        control_rate_hz = float(
            self.declare_parameter("control_rate_hz", 500.0).value
        )
        self._state_timeout_s = float(
            self.declare_parameter("state_timeout_s", 0.2).value
        )
        gravity_model = str(
            self.declare_parameter("gravity_model", "physical").value
        )
        if gravity_model not in ("physical", "empirical"):
            raise ValueError("gravity_model must be physical or empirical")
        controller_mode = str(
            self.declare_parameter("controller_mode", "pd").value
        )
        if controller_mode not in ("pd", "impedance"):
            raise ValueError("controller_mode must be pd or impedance")
        if controller_mode == "impedance" and gravity_model != "physical":
            raise ValueError(
                "impedance control requires physical gravity"
            )
        if not np.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and positive")
        if (
            not np.isfinite(self._state_timeout_s)
            or self._state_timeout_s <= 0.0
        ):
            raise ValueError("state_timeout_s must be finite and positive")

        self.kinematics = Kinematics(urdf_path)
        hardware_profile = self._load_hardware_profile()
        self.hardware_profile = hardware_profile
        dynamics = PinocchioDynamics(urdf_path)
        gravity_torque = (
            dynamics.gravity_torque
            if gravity_model == "physical"
            else hardware_profile.gravity_torque
        )
        gains = PDGains(np.asarray(kp), np.asarray(kd))
        if controller_mode == "impedance":
            self.controller = JointImpedanceController(
                gains,
                hardware_profile.effort_limits_nm,
                dynamics,
                gravity_enabled=gravity_enabled,
                hardware_profile=hardware_profile,
                breakaway_enabled=breakaway_enabled,
                friction_enabled=friction_enabled,
                kp_scale=impedance_kp_scale,
                kd_scale=impedance_kd_scale,
            )
        else:
            self.controller = PDGravityController(
                gains,
                hardware_profile.effort_limits_nm,
                gravity_torque=gravity_torque,
                gravity_enabled=gravity_enabled,
                hardware_profile=hardware_profile,
                breakaway_enabled=breakaway_enabled,
                friction_enabled=friction_enabled,
            )
        target_position = vector6(
            self.declare_parameter(
                "target_position", list(POWER_ON_POSITION_RAD)
            ).value,
            "target_position",
        )
        self._target_position = np.clip(
            target_position,
            self.kinematics.joint_limits_rad[:, 0],
            self.kinematics.joint_limits_rad[:, 1],
        )
        self._target_velocity = np.zeros(JOINT_COUNT)
        self._target_acceleration = np.zeros(JOINT_COUNT)
        self._target_acceleration_time = 0.0
        self._latest_state: JointFeedback | None = None
        self._latest_state_time = 0.0
        self._current_position = np.zeros(JOINT_COUNT)
        self._enabled = True
        self._control_counter = 0
        debug_divider = max(1, round(control_rate_hz / 50.0))

        state_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(
            JointState,
            "/arm/control/joint_state",
            self._state_callback,
            state_qos,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/target/joint_position",
            self._joint_target_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/target/joint_velocity",
            self._joint_velocity_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/target/joint_acceleration",
            self._joint_acceleration_callback,
            10,
        )
        self.create_subscription(
            PoseStamped, "/arm/target/pose", self._pose_target_callback, 10
        )
        self._effort_publisher = self.create_publisher(
            Float64MultiArray, "/arm/control/effort_cmd", 10
        )
        self._pd_publisher = self.create_publisher(
            Float64MultiArray, "/arm/debug/pd_effort", 10
        )
        self._gravity_publisher = self.create_publisher(
            Float64MultiArray, "/arm/debug/gravity_effort", 10
        )
        self._friction_publisher = self.create_publisher(
            Float64MultiArray, "/arm/debug/friction_effort", 10
        )
        self._breakaway_publisher = self.create_publisher(
            Float64MultiArray, "/arm/debug/breakaway_effort", 10
        )
        self._error_publisher = self.create_publisher(
            Float64MultiArray, "/arm/debug/position_error", 10
        )
        self._enable_service = self.create_service(
            SetBool, "/arm/control/enable", self._enable_callback
        )
        self.add_on_set_parameters_callback(self._parameter_callback)
        self._debug_divider = debug_divider
        self.create_timer(1.0 / control_rate_hz, self._control_step)
        self.get_logger().info(
            f"Arm control started at {control_rate_hz:.0f} Hz with "
            f"profile {hardware_profile.profile_id}, {controller_mode} "
            f"controller, {gravity_model} gravity"
        )

    def _load_hardware_profile(self) -> EngineerHardwareProfile:
        return EngineerHardwareProfile(
            profile_id=str(
                self.declare_parameter(
                    "hardware_profile.id", "reatrm_engineer"
                ).value
            ),
            motor_types=tuple(
                self.declare_parameter(
                    "hardware_profile.motor_types",
                    [
                        "DM4310",
                        "LK_MF7015",
                        "LK_MF7015",
                        "DM4310",
                        "DM4310",
                        "DM4310",
                    ],
                ).value
            ),
            joint_gear_ratio=self.declare_parameter(
                "hardware_profile.joint_gear_ratio",
                [1.0, 50.0, 50.0, 1.0, 1.5, 1.0],
            ).value,
            effort_limits_nm=self.declare_parameter(
                "hardware_profile.effort_limits_nm",
                [10.0, 63.0, 35.0, 10.0, 6.0, 3.0],
            ).value,
            command_rate_hz=float(
                self.declare_parameter(
                    "hardware_profile.command_rate_hz", 500.0
                ).value
            ),
            state_rate_hz=float(
                self.declare_parameter(
                    "hardware_profile.state_rate_hz", 111.111111
                ).value
            ),
            joint_viscous_damping_nm_s_rad=self.declare_parameter(
                "hardware_profile.joint_viscous_damping_nm_s_rad",
                [0.05] * JOINT_COUNT,
            ).value,
            joint_armature_kg_m2=self.declare_parameter(
                "hardware_profile.joint_armature_kg_m2",
                [0.001] * JOINT_COUNT,
            ).value,
            actuator_time_constant_s=self.declare_parameter(
                "hardware_profile.actuator_time_constant_s",
                [0.005] * JOINT_COUNT,
            ).value,
            gravity_j5_config_weight=float(
                self.declare_parameter(
                    "hardware_profile.gravity_j5_config_weight", 0.35
                ).value
            ),
            gravity_j5_conservative_coupling=bool(
                self.declare_parameter(
                    "hardware_profile."
                    "gravity_j5_conservative_coupling",
                    True,
                ).value
            ),
            gravity_p_nm=self.declare_parameter(
                "hardware_profile.gravity_p_nm",
                [0.0, 20.284, -15.112, -0.579, 16.368, 0.0],
            ).value,
            gravity_q_nm=self.declare_parameter(
                "hardware_profile.gravity_q_nm",
                [0.0, -60.824, 2.680, 0.380, -0.715, 0.0],
            ).value,
            gravity_bias_nm=self.declare_parameter(
                "hardware_profile.gravity_bias_nm",
                [0.0, -22.244, 1.534, 0.0, -13.608, 0.0],
            ).value,
            friction_torque_nm=self.declare_parameter(
                "hardware_profile.friction_torque_nm",
                [0.0, 11.6, 7.4, 0.0, 0.75, 0.0],
            ).value,
            friction_velocity_scale_rad_s=self.declare_parameter(
                "hardware_profile.friction_velocity_scale_rad_s",
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            ).value,
            breakaway_torque_nm=self.declare_parameter(
                "hardware_profile.breakaway_torque_nm",
                [0.0, 11.25, 10.0, 0.0, 0.4, 0.2],
            ).value,
            breakaway_error_scale_rad=float(
                self.declare_parameter(
                    "hardware_profile.breakaway_error_scale_rad", 0.01
                ).value
            ),
            breakaway_target_velocity_threshold_rad_s=float(
                self.declare_parameter(
                    "hardware_profile."
                    "breakaway_target_velocity_threshold_rad_s",
                    0.02,
                ).value
            ),
            breakaway_actual_velocity_threshold_rad_s=float(
                self.declare_parameter(
                    "hardware_profile."
                    "breakaway_actual_velocity_threshold_rad_s",
                    0.03,
                ).value
            ),
            plant_breakaway_velocity_scale_rad_s=float(
                self.declare_parameter(
                    "hardware_profile."
                    "plant_breakaway_velocity_scale_rad_s",
                    0.02,
                ).value
            ),
            plant_gravity_enabled=bool(
                self.declare_parameter(
                    "hardware_profile.plant_gravity_enabled", False
                ).value
            ),
            plant_friction_enabled=bool(
                self.declare_parameter(
                    "hardware_profile.plant_friction_enabled", True
                ).value
            ),
            plant_breakaway_enabled=bool(
                self.declare_parameter(
                    "hardware_profile.plant_breakaway_enabled", True
                ).value
            ),
        )

    def _state_callback(self, message: JointState) -> None:
        index_by_name = {
            name: index for index, name in enumerate(message.name)
        }
        position = np.full(JOINT_COUNT, np.nan)
        velocity = np.full(JOINT_COUNT, np.nan)
        effort = np.zeros(JOINT_COUNT)
        for index, name in enumerate(JOINT_NAMES):
            source = index_by_name.get(name)
            if source is None:
                continue
            if source < len(message.position):
                position[index] = message.position[source]
            if source < len(message.velocity):
                velocity[index] = message.velocity[source]
            if source < len(message.effort):
                effort[index] = message.effort[source]
        alive = np.isfinite(position) & np.isfinite(velocity)
        self._latest_state = JointFeedback(
            position,
            velocity,
            effort,
            alive,
        )
        self._latest_state_time = time.monotonic()
        if np.all(alive):
            self._current_position = position.copy()

    def _joint_target_callback(
        self, message: Float64MultiArray
    ) -> None:
        if len(message.data) != JOINT_COUNT:
            self.get_logger().warning("Joint target must contain six values")
            return
        target = np.asarray(message.data, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            self.get_logger().warning("Joint target contains invalid values")
            return
        self._target_position = np.clip(
            target,
            self.kinematics.joint_limits_rad[:, 0],
            self.kinematics.joint_limits_rad[:, 1],
        )
        self._target_velocity = np.zeros(JOINT_COUNT)

    def _enable_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self._enabled = bool(request.data)
        if not self._enabled:
            self._target_position = self._current_position.copy()
            self._target_velocity = np.zeros(JOINT_COUNT)
            self._publish_command(self.controller.safe_stop())
        response.success = True
        response.message = "Arm controller enabled" if self._enabled else (
            "Arm controller disabled; zero torque requested"
        )
        self.get_logger().info(response.message)
        return response

    def _parameter_callback(
        self, parameters
    ) -> SetParametersResult:
        kp = self.get_parameter("kp").value
        kd = self.get_parameter("kd").value
        gravity_enabled = self.get_parameter("gravity_enabled").value
        breakaway_enabled = self.get_parameter(
            "breakaway_enabled"
        ).value
        friction_enabled = self.get_parameter("friction_enabled").value
        for parameter in parameters:
            if parameter.name == "kp":
                kp = parameter.value
            elif parameter.name == "kd":
                kd = parameter.value
            elif parameter.name == "gravity_enabled":
                if not isinstance(parameter.value, bool):
                    return SetParametersResult(
                        successful=False,
                        reason="gravity_enabled must be a boolean",
                    )
                gravity_enabled = parameter.value
            elif parameter.name == "breakaway_enabled":
                if not isinstance(parameter.value, bool):
                    return SetParametersResult(
                        successful=False,
                        reason="breakaway_enabled must be a boolean",
                    )
                breakaway_enabled = parameter.value
            elif parameter.name == "friction_enabled":
                if not isinstance(parameter.value, bool):
                    return SetParametersResult(
                        successful=False,
                        reason="friction_enabled must be a boolean",
                    )
                friction_enabled = parameter.value
        try:
            gains = PDGains(np.asarray(kp), np.asarray(kd))
        except (TypeError, ValueError) as error:
            return SetParametersResult(successful=False, reason=str(error))
        self.controller.set_gains(gains)
        self.controller.set_gravity_enabled(bool(gravity_enabled))
        self.controller.set_breakaway_enabled(bool(breakaway_enabled))
        self.controller.set_friction_enabled(bool(friction_enabled))
        return SetParametersResult(successful=True)

    def _joint_velocity_callback(
        self, message: Float64MultiArray
    ) -> None:
        if len(message.data) != JOINT_COUNT:
            self.get_logger().warning(
                "Joint velocity target must contain six values"
            )
            return
        target = np.asarray(message.data, dtype=np.float64)
        if np.all(np.isfinite(target)):
            self._target_velocity = target

    def _joint_acceleration_callback(
        self, message: Float64MultiArray
    ) -> None:
        if len(message.data) != JOINT_COUNT:
            self.get_logger().warning(
                "Joint acceleration target must contain six values"
            )
            return
        target = np.asarray(message.data, dtype=np.float64)
        if np.all(np.isfinite(target)):
            self._target_acceleration = target
            self._target_acceleration_time = time.monotonic()

    def _pose_target_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id not in ("", "base_link"):
            self.get_logger().warning(
                "Pose target frame must be base_link"
            )
            return
        pose = message.pose
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(position)) or not np.all(
            np.isfinite(quaternion)
        ):
            self.get_logger().warning(
                "Pose target contains non-finite values"
            )
            return
        target = np.eye(4)
        target[:3, :3] = _quaternion_matrix(*quaternion)
        target[:3, 3] = position
        result = self.kinematics.inverse(
            target,
            self._target_position,
            IKOptions(max_attempts=5),
        )
        if result.success:
            self._target_position = result.position_rad
            self._target_velocity = np.zeros(JOINT_COUNT)
            self._target_acceleration = np.zeros(JOINT_COUNT)
            self._target_acceleration_time = 0.0
            self.get_logger().info(
                "Pose IK accepted: "
                f"position_error={result.position_error_m:.6e} m"
            )
        else:
            self.get_logger().warning(result.message)

    def _control_step(self) -> None:
        if self._latest_state is None:
            return
        age = time.monotonic() - self._latest_state_time
        if age > self._state_timeout_s:
            self._publish_command(self.controller.safe_stop())
            return
        if self._enabled:
            acceleration = (
                self._target_acceleration
                if time.monotonic() - self._target_acceleration_time
                <= self._state_timeout_s
                else np.zeros(JOINT_COUNT)
            )
            command = self.controller.compute(
                self._latest_state,
                self._target_position,
                self._target_velocity,
                acceleration,
            )
        else:
            command = self.controller.safe_stop()
        self._publish_command(command)

    def _publish_command(self, command: TorqueCommand) -> None:
        message = Float64MultiArray()
        message.data = command.effort_nm.tolist()
        self._effort_publisher.publish(message)
        self._control_counter += 1
        if self._control_counter % self._debug_divider == 0:
            self._pd_publisher.publish(
                Float64MultiArray(
                    data=self.controller.last_pd_effort_nm.tolist()
                )
            )
            self._gravity_publisher.publish(
                Float64MultiArray(
                    data=self.controller.last_gravity_effort_nm.tolist()
                )
            )
            self._friction_publisher.publish(
                Float64MultiArray(
                    data=self.controller.last_friction_effort_nm.tolist()
                )
            )
            self._breakaway_publisher.publish(
                Float64MultiArray(
                    data=self.controller.last_breakaway_effort_nm.tolist()
                )
            )
            if self._latest_state is not None:
                error = (
                    self._target_position
                    - self._latest_state.position_rad
                )
                self._error_publisher.publish(
                    Float64MultiArray(data=error.tolist())
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                stop = Float64MultiArray()
                stop.data = [0.0] * JOINT_COUNT
                node._effort_publisher.publish(stop)
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
