"""Qt dashboard for joint targets, PD tuning, and torque visualization."""

from __future__ import annotations

from collections import deque
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMessage
from trajectory_msgs.msg import JointTrajectoryPoint

from jgarm_control.quintic import make_quintic_trajectory
from jgarm_control.types import (
    JOINT_COUNT,
    JOINT_NAMES,
    POWER_ON_POSITION_RAD,
    URDF_ZERO_POSITION_RAD,
)


JOINT_LIMITS = np.array(
    [
        [-3.14, 3.14],
        [-1.3265, 1.9024],
        [-0.5934, 1.35],
        [-3.14, 3.14],
        [-2.094395, 2.094395],
        [-3.14, 3.14],
    ]
)
DEFAULT_KP = np.array([60.0, 100.0, 100.0, 40.0, 20.0, 10.0])
DEFAULT_KD = np.array([2.0, 4.0, 4.0, 1.5, 1.0, 0.5])
PLOT_REFRESH_PERIOD_S = 0.05
PLOT_MAX_SAMPLES = 250


class DashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("jgarm_dashboard")
        state_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(
            JointState, "/joint_states", self._state_callback, state_qos
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/control/effort_cmd",
            self._command_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/debug/pd_effort",
            self._pd_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/debug/gravity_effort",
            self._gravity_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/debug/mujoco_bias_effort",
            self._mujoco_bias_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/debug/friction_effort",
            self._friction_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm/debug/breakaway_effort",
            self._breakaway_callback,
            10,
        )
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/JG_arm_controller/follow_joint_trajectory",
        )
        self.parameter_client = self.create_client(
            SetParameters, "/arm_control_node/set_parameters"
        )
        self.enable_client = self.create_client(
            SetBool, "/arm/control/enable"
        )
        self.position = np.zeros(JOINT_COUNT)
        self.velocity = np.zeros(JOINT_COUNT)
        self.actual_effort = np.zeros(JOINT_COUNT)
        self.command_effort = np.zeros(JOINT_COUNT)
        self.pd_effort = np.zeros(JOINT_COUNT)
        self.gravity_effort = np.zeros(JOINT_COUNT)
        self.mujoco_bias_effort = np.zeros(JOINT_COUNT)
        self.friction_effort = np.zeros(JOINT_COUNT)
        self.breakaway_effort = np.zeros(JOINT_COUNT)
        self.has_state = False

    def _joint_vector(
        self, message: JointState, field: str
    ) -> np.ndarray:
        result = np.zeros(JOINT_COUNT)
        values = getattr(message, field)
        index_by_name = {
            name: index for index, name in enumerate(message.name)
        }
        for index, name in enumerate(JOINT_NAMES):
            source = index_by_name.get(name)
            if source is not None and source < len(values):
                result[index] = values[source]
        return result

    def _state_callback(self, message: JointState) -> None:
        self.position = self._joint_vector(message, "position")
        self.velocity = self._joint_vector(message, "velocity")
        self.actual_effort = self._joint_vector(message, "effort")
        self.has_state = True

    def _command_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) == JOINT_COUNT:
            self.command_effort = np.asarray(message.data)

    def _pd_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) == JOINT_COUNT:
            self.pd_effort = np.asarray(message.data)

    def _gravity_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) == JOINT_COUNT:
            self.gravity_effort = np.asarray(message.data)

    def _mujoco_bias_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) == JOINT_COUNT:
            self.mujoco_bias_effort = np.asarray(message.data)

    def _friction_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) == JOINT_COUNT:
            self.friction_effort = np.asarray(message.data)

    def _breakaway_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) == JOINT_COUNT:
            self.breakaway_effort = np.asarray(message.data)


class TorquePlotWidget(QtWidgets.QWidget):
    """Draws six torque histories with native Qt painting."""

    _SERIES = (
        ("actual", "#d62728"),
        ("hold", "#1f77b4"),
        ("PD", "#2ca02c"),
        ("gravity", "#9467bd"),
        ("friction", "#ff7f0e"),
        ("breakaway", "#8c564b"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._time_history = deque(maxlen=PLOT_MAX_SAMPLES)
        self._history = [
            {
                "actual": deque(maxlen=PLOT_MAX_SAMPLES),
                "hold": deque(maxlen=PLOT_MAX_SAMPLES),
                "PD": deque(maxlen=PLOT_MAX_SAMPLES),
                "gravity": deque(maxlen=PLOT_MAX_SAMPLES),
                "friction": deque(maxlen=PLOT_MAX_SAMPLES),
                "breakaway": deque(maxlen=PLOT_MAX_SAMPLES),
            }
            for _ in range(JOINT_COUNT)
        ]
        self.setMinimumSize(760, 720)

    def append(
        self,
        actual_effort: np.ndarray,
        command_effort: np.ndarray,
        pd_effort: np.ndarray,
        gravity_effort: np.ndarray,
        friction_effort: np.ndarray | None = None,
        breakaway_effort: np.ndarray | None = None,
    ) -> None:
        if friction_effort is None:
            friction_effort = np.zeros(JOINT_COUNT)
        if breakaway_effort is None:
            breakaway_effort = np.zeros(JOINT_COUNT)
        self._time_history.append(time.monotonic())
        values = (
            actual_effort,
            command_effort,
            pd_effort,
            gravity_effort,
            friction_effort,
            breakaway_effort,
        )
        keys = ("actual", "hold", "PD", "gravity", "friction", "breakaway")
        for joint_index, joint_history in enumerate(self._history):
            for key, joint_values in zip(keys, values):
                joint_history[key].append(float(joint_values[joint_index]))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), self.palette().window())

            gap = 20.0
            margin = 12.0
            panel_width = (self.width() - 2.0 * margin - gap) / 2.0
            panel_height = (
                self.height() - 2.0 * margin - 2.0 * gap
            ) / 3.0
            time_values = np.asarray(self._time_history)
            duration = (
                max(time_values[-1] - time_values[0], 1e-6)
                if time_values.size > 1
                else 1.0
            )

            for joint_index in range(JOINT_COUNT):
                row = joint_index // 2
                column = joint_index % 2
                panel = QtCore.QRectF(
                    margin + column * (panel_width + gap),
                    margin + row * (panel_height + gap),
                    panel_width,
                    panel_height,
                )
                self._draw_panel(
                    painter,
                    panel,
                    joint_index,
                    time_values,
                    duration,
                )
        finally:
            painter.end()

    def _draw_panel(
        self,
        painter: QtGui.QPainter,
        panel: QtCore.QRectF,
        joint_index: int,
        time_values: np.ndarray,
        duration: float,
    ) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor("#c7d0d9"), 1.0))
        painter.setBrush(QtGui.QColor("#fbfcfd"))
        painter.drawRoundedRect(panel, 5.0, 5.0)

        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSize(10)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#202630"))
        painter.drawText(
            panel.adjusted(10.0, 5.0, -10.0, -5.0),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
            f"joint_{joint_index + 1}",
        )

        history = self._history[joint_index]
        magnitude = max(
            (
                abs(value)
                for key, _ in self._SERIES
                for value in history[key]
            ),
            default=0.0,
        )
        span = max(1.0, 1.2 * magnitude)
        plot_rect = panel.adjusted(48.0, 30.0, -12.0, -24.0)
        self._draw_grid(painter, plot_rect, span)

        painter.save()
        painter.setClipRect(plot_rect)
        if time_values.size > 1:
            normalized_time = (
                time_values - (time_values[-1] - duration)
            ) / duration
            for key, color in self._SERIES:
                points = QtGui.QPolygonF()
                for x_ratio, value in zip(normalized_time, history[key]):
                    point = QtCore.QPointF(
                        plot_rect.left() + x_ratio * plot_rect.width(),
                        plot_rect.center().y()
                        - value / span * plot_rect.height() / 2.0,
                    )
                    points.append(point)
                pen = QtGui.QPen(QtGui.QColor(color), 1.4)
                if key in ("PD", "gravity", "friction", "breakaway"):
                    pen.setStyle(QtCore.Qt.DashLine)
                    pen.setWidthF(1.0)
                painter.setPen(pen)
                painter.drawPolyline(points)
        painter.restore()
        self._draw_legend(painter, plot_rect)

    def _draw_grid(
        self,
        painter: QtGui.QPainter,
        plot_rect: QtCore.QRectF,
        span: float,
    ) -> None:
        grid_pen = QtGui.QPen(QtGui.QColor("#dfe5eb"), 1.0)
        text_pen = QtGui.QPen(QtGui.QColor("#56606b"), 1.0)
        label_font = painter.font()
        label_font.setBold(False)
        label_font.setPointSize(7)
        painter.setFont(label_font)
        for index in range(5):
            ratio = index / 4.0
            y_value = plot_rect.bottom() - ratio * plot_rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(
                QtCore.QPointF(plot_rect.left(), y_value),
                QtCore.QPointF(plot_rect.right(), y_value),
            )
            torque = span * (2.0 * ratio - 1.0)
            painter.setPen(text_pen)
            painter.drawText(
                QtCore.QRectF(
                    2.0,
                    y_value - 7.0,
                    plot_rect.left() - 8.0,
                    14.0,
                ),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{torque:.1f}",
            )

    def _draw_legend(
        self,
        painter: QtGui.QPainter,
        plot_rect: QtCore.QRectF,
    ) -> None:
        legend_font = painter.font()
        legend_font.setPointSize(7)
        painter.setFont(legend_font)
        x_position = plot_rect.right() - 152.0
        for index, (key, color) in enumerate(self._SERIES):
            y_position = plot_rect.top() + 7.0 + index * 11.0
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
            painter.drawLine(
                QtCore.QPointF(x_position, y_position),
                QtCore.QPointF(x_position + 16.0, y_position),
            )
            painter.setPen(QtGui.QColor("#303843"))
            painter.drawText(
                QtCore.QPointF(x_position + 20.0, y_position + 3.0),
                key,
            )


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, node: DashboardNode) -> None:
        super().__init__()
        self._node = node
        self._goal_handle = None
        self._initialized = False
        self._sync_guard = False
        self.setWindowTitle("JGarm Control Dashboard")
        self.resize(1360, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(440)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        root_layout.addWidget(controls, 0)
        root_layout.addWidget(self._build_plot(), 1)

        controls_layout.addWidget(self._build_joint_controls())
        controls_layout.addWidget(self._build_gain_controls())
        controls_layout.addWidget(self._build_torque_readout())
        self._status_label = QtWidgets.QLabel("Waiting for /joint_states")
        self._status_label.setWordWrap(True)
        controls_layout.addWidget(self._status_label)

        self._spin_timer = QtCore.QTimer(self)
        self._spin_timer.timeout.connect(self._spin_ros)
        self._spin_timer.start(10)
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.timeout.connect(self._refresh_plot)
        self._plot_timer.start(round(PLOT_REFRESH_PERIOD_S * 1000.0))

    def _build_joint_controls(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Joint targets")
        layout = QtWidgets.QGridLayout(group)
        layout.addWidget(QtWidgets.QLabel("Joint"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Current"), 0, 1)
        layout.addWidget(QtWidgets.QLabel("Target"), 0, 2)
        self._current_labels = []
        self._target_spins = []
        self._target_sliders = []
        for index in range(JOINT_COUNT):
            current_label = QtWidgets.QLabel("0.000")
            current_label.setMinimumWidth(64)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(*JOINT_LIMITS[index])
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-1000, 1000)
            spin.valueChanged.connect(
                lambda value, i=index: self._spin_to_slider(i, value)
            )
            slider.valueChanged.connect(
                lambda value, i=index: self._slider_to_spin(i, value)
            )
            layout.addWidget(
                QtWidgets.QLabel(f"joint_{index + 1}"),
                index + 1,
                0,
            )
            layout.addWidget(current_label, index + 1, 1)
            layout.addWidget(spin, index + 1, 2)
            layout.addWidget(slider, index + 1, 3)
            self._current_labels.append(current_label)
            self._target_spins.append(spin)
            self._target_sliders.append(slider)

        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(0.5, 10.0)
        self._duration_spin.setValue(2.0)
        self._duration_spin.setSuffix(" s")
        layout.addWidget(QtWidgets.QLabel("Move time"), JOINT_COUNT + 1, 0)
        layout.addWidget(self._duration_spin, JOINT_COUNT + 1, 1, 1, 3)

        preset_layout = QtWidgets.QHBoxLayout()
        power_on_button = QtWidgets.QPushButton("Power-on pose")
        zero_button = QtWidgets.QPushButton("URDF zero")
        power_on_button.clicked.connect(
            lambda: self._apply_pose_preset(POWER_ON_POSITION_RAD)
        )
        zero_button.clicked.connect(
            lambda: self._apply_pose_preset(URDF_ZERO_POSITION_RAD)
        )
        preset_layout.addWidget(power_on_button)
        preset_layout.addWidget(zero_button)
        layout.addLayout(preset_layout, JOINT_COUNT + 2, 0, 1, 4)

        button_layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply target")
        hold_button = QtWidgets.QPushButton("Hold current")
        stop_button = QtWidgets.QPushButton("Disable drive")
        enable_button = QtWidgets.QPushButton("Enable drive")
        apply_button.clicked.connect(self._apply_target)
        hold_button.clicked.connect(self._hold_current)
        stop_button.clicked.connect(lambda: self._set_enabled(False))
        enable_button.clicked.connect(lambda: self._set_enabled(True))
        button_layout.addWidget(apply_button)
        button_layout.addWidget(hold_button)
        button_layout.addWidget(stop_button)
        button_layout.addWidget(enable_button)
        layout.addLayout(button_layout, JOINT_COUNT + 3, 0, 1, 4)
        return group

    def _build_gain_controls(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("PD controller")
        layout = QtWidgets.QGridLayout(group)
        self._kp_spins = []
        self._kd_spins = []
        for index in range(JOINT_COUNT):
            kp = QtWidgets.QDoubleSpinBox()
            kp.setRange(0.0, 1000.0)
            kp.setDecimals(3)
            kp.setValue(DEFAULT_KP[index])
            kd = QtWidgets.QDoubleSpinBox()
            kd.setRange(0.0, 200.0)
            kd.setDecimals(3)
            kd.setValue(DEFAULT_KD[index])
            layout.addWidget(QtWidgets.QLabel(f"J{index + 1} Kp"), index, 0)
            layout.addWidget(kp, index, 1)
            layout.addWidget(QtWidgets.QLabel("Kd"), index, 2)
            layout.addWidget(kd, index, 3)
            self._kp_spins.append(kp)
            self._kd_spins.append(kd)
        self._gravity_checkbox = QtWidgets.QCheckBox("Gravity compensation")
        self._gravity_checkbox.setChecked(True)
        layout.addWidget(self._gravity_checkbox, JOINT_COUNT, 0, 1, 2)
        apply_button = QtWidgets.QPushButton("Apply PD")
        apply_button.clicked.connect(self._apply_pd)
        layout.addWidget(apply_button, JOINT_COUNT, 2, 1, 2)
        return group

    def _build_torque_readout(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Static torque readout (N·m)")
        layout = QtWidgets.QVBoxLayout(group)
        self._torque_table = QtWidgets.QTableWidget(JOINT_COUNT, 6)
        self._torque_table.setHorizontalHeaderLabels(
            ["Joint", "MuJoCo", "Gravity", "PD", "Hold", "Actual"]
        )
        self._torque_table.verticalHeader().setVisible(False)
        self._torque_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self._torque_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection
        )
        self._torque_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self._torque_table.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self._torque_table.setFixedHeight(190)
        self._torque_items = []
        for row in range(JOINT_COUNT):
            table_row = []
            for column in range(6):
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self._torque_table.setItem(row, column, item)
                table_row.append(item)
            table_row[0].setText(f"J{row + 1}")
            self._torque_items.append(table_row)
        layout.addWidget(self._torque_table)
        return group

    def _build_plot(self) -> QtWidgets.QWidget:
        self._plot = TorquePlotWidget()
        return self._plot

    def _spin_ros(self) -> None:
        if rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.0)
        if self._node.has_state:
            for index, label in enumerate(self._current_labels):
                label.setText(f"{self._node.position[index]: .3f}")
            if not self._initialized:
                for index, spin in enumerate(self._target_spins):
                    spin.setValue(float(self._node.position[index]))
                self._initialized = True

    def _slider_to_spin(self, index: int, slider_value: int) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        lower, upper = JOINT_LIMITS[index]
        value = lower + (slider_value + 1000.0) * (upper - lower) / 2000.0
        self._target_spins[index].setValue(value)
        self._sync_guard = False

    def _spin_to_slider(self, index: int, spin_value: float) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        lower, upper = JOINT_LIMITS[index]
        slider_value = round(
            -1000.0 + 2000.0 * (spin_value - lower) / (upper - lower)
        )
        self._target_sliders[index].setValue(int(slider_value))
        self._sync_guard = False

    def _apply_target(self) -> None:
        if not self._node.has_state:
            self._status_label.setText("Cannot move: no /joint_states received")
            return
        if not self._node.action_client.server_is_ready():
            self._status_label.setText("Trajectory action server is not ready")
            return
        if self._goal_handle is not None:
            self._status_label.setText("A trajectory is already active")
            return

        target = np.array(
            [spin.value() for spin in self._target_spins],
            dtype=np.float64,
        )
        source = make_quintic_trajectory(
            self._node.position,
            target,
            self._duration_spin.value(),
        )
        trajectory = JointTrajectoryMessage()
        trajectory.joint_names = list(JOINT_NAMES)
        for time_value, position, velocity in zip(
            source.times_s, source.positions_rad, source.velocities_rad_s
        ):
            point = JointTrajectoryPoint()
            point.positions = position.tolist()
            point.velocities = velocity.tolist()
            whole_seconds = int(time_value)
            point.time_from_start.sec = whole_seconds
            point.time_from_start.nanosec = int(
                (time_value - whole_seconds) * 1e9
            )
            trajectory.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        for name in JOINT_NAMES:
            tolerance = JointTolerance()
            tolerance.name = name
            tolerance.position = 0.02
            tolerance.velocity = 0.1
            goal.goal_tolerance.append(tolerance)
        goal.goal_time_tolerance.sec = 2
        future = self._node.action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)
        self._status_label.setText("Sending smooth trajectory...")

    def _apply_pose_preset(self, target) -> None:
        for spin, value in zip(self._target_spins, target):
            spin.setValue(float(value))
        self._status_label.setText("Pose preset loaded; planning target")
        self._apply_target()

    def _goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:  # noqa: BLE001
            self._status_label.setText(f"Trajectory request failed: {error}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self._status_label.setText("Trajectory rejected")
            return
        self._goal_handle = goal_handle
        self._status_label.setText("Trajectory executing")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_response)

    def _result_response(self, future) -> None:
        try:
            wrapped = future.result()
        except Exception as error:  # noqa: BLE001
            self._goal_handle = None
            self._status_label.setText(f"Trajectory result failed: {error}")
            return
        self._goal_handle = None
        if wrapped is None:
            self._status_label.setText("Trajectory ended without result")
            return
        result = wrapped.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self._status_label.setText("Target reached")
        else:
            self._status_label.setText(
                f"Trajectory failed: {result.error_string}"
            )

    def _hold_current(self) -> None:
        if not self._node.has_state:
            return
        for index, spin in enumerate(self._target_spins):
            spin.setValue(float(self._node.position[index]))
        self._duration_spin.setValue(min(self._duration_spin.value(), 1.0))
        self._apply_target()

    def _apply_pd(self) -> None:
        if not self._node.parameter_client.service_is_ready():
            self._status_label.setText("PD parameter service is not ready")
            return
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="kp",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    double_array=[spin.value() for spin in self._kp_spins],
                ),
            ),
            Parameter(
                name="kd",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    double_array=[spin.value() for spin in self._kd_spins],
                ),
            ),
            Parameter(
                name="gravity_enabled",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=self._gravity_checkbox.isChecked(),
                ),
            ),
        ]
        future = self._node.parameter_client.call_async(request)
        future.add_done_callback(self._parameter_response)
        self._status_label.setText("Applying PD parameters...")

    def _parameter_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self._status_label.setText(f"PD parameter request failed: {error}")
            return
        if response is not None and all(
            result.successful for result in response.results
        ):
            self._status_label.setText("PD parameters applied")
        else:
            self._status_label.setText("PD parameters rejected")

    def _set_enabled(self, enabled: bool, wait: bool = False) -> None:
        if not self._node.enable_client.service_is_ready():
            self._status_label.setText("Enable service is not ready")
            return
        if not enabled and self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        request = SetBool.Request()
        request.data = enabled
        future = self._node.enable_client.call_async(request)
        future.add_done_callback(self._enable_response)
        self._status_label.setText(
            "Enabling drive..." if enabled else "Disabling drive..."
        )
        if wait:
            deadline = time.monotonic() + 0.5
            while rclpy.ok() and not future.done():
                if time.monotonic() >= deadline:
                    self._node.get_logger().warning(
                        "Timed out waiting for drive disable confirmation"
                    )
                    break
                rclpy.spin_once(self._node, timeout_sec=0.02)

    def _enable_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self._status_label.setText(f"Enable request failed: {error}")
            return
        if response is not None and response.success:
            self._status_label.setText(response.message)
        else:
            self._status_label.setText("Enable request rejected")

    def _refresh_plot(self) -> None:
        table_values = (
            self._node.mujoco_bias_effort,
            self._node.gravity_effort,
            self._node.pd_effort,
            self._node.command_effort,
            self._node.actual_effort,
        )
        for row, items in enumerate(self._torque_items):
            for column, values in enumerate(table_values, start=1):
                items[column].setText(f"{values[row]:+.3f}")
        self._plot.append(
            self._node.actual_effort,
            self._node.command_effort,
            self._node.pd_effort,
            self._node.gravity_effort,
            self._node.friction_effort,
            self._node.breakaway_effort,
        )

    def closeEvent(self, event) -> None:
        self._set_enabled(False, wait=True)
        event.accept()


def main(args=None) -> None:
    rclpy.init(args=args)
    app = QtWidgets.QApplication(sys.argv)
    node = DashboardNode()
    window = DashboardWindow(node)
    window.show()
    try:
        app.exec_()
    finally:
        window._set_enabled(False, wait=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
