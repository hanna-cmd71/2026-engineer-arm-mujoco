"""Dashboard dependency and module import tests."""

import os
from pathlib import Path
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeFuture:
    def add_done_callback(self, _callback) -> None:
        pass


class _FakeActionClient:
    def __init__(self) -> None:
        self.goal = None

    def server_is_ready(self) -> bool:
        return True

    def send_goal_async(self, goal):
        self.goal = goal
        return _FakeFuture()


class _FakeNode:
    def __init__(self) -> None:
        self.has_state = True
        self.position = np.zeros(6)
        self.actual_effort = np.zeros(6)
        self.command_effort = np.zeros(6)
        self.pd_effort = np.zeros(6)
        self.gravity_effort = np.zeros(6)
        self.mujoco_bias_effort = np.zeros(6)
        self.friction_effort = np.zeros(6)
        self.breakaway_effort = np.zeros(6)
        self.action_client = _FakeActionClient()


class DashboardImportTest(unittest.TestCase):
    def test_dashboard_module_imports(self) -> None:
        from jgarm_control import dashboard_node

        self.assertTrue(hasattr(dashboard_node, "DashboardNode"))
        self.assertEqual(dashboard_node.JOINT_LIMITS.shape, (6, 2))

    def test_joint_limits_match_urdf(self) -> None:
        from ament_index_python.packages import get_package_share_directory

        from jgarm_control.dashboard_node import JOINT_LIMITS
        from jgarm_control.kinematics import Kinematics

        urdf_path = (
            Path(get_package_share_directory("jgarm_description"))
            / "urdf"
            / "JGarm.urdf"
        )
        urdf_limits = Kinematics(urdf_path).joint_limits_rad
        np.testing.assert_allclose(JOINT_LIMITS, urdf_limits, atol=0.0)

    def test_torque_plot_renders_offscreen(self) -> None:
        from PyQt5 import QtWidgets

        from jgarm_control.dashboard_node import TorquePlotWidget

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        widget = TorquePlotWidget()
        widget.resize(900, 760)
        self.assertEqual(
            {key for key, _ in widget._SERIES},
            set(widget._history[0]),
        )
        widget.append(
            np.ones(6),
            2.0 * np.ones(6),
            0.5 * np.ones(6),
            0.3 * np.ones(6),
        )
        widget.append(
            2.0 * np.ones(6),
            -1.0 * np.ones(6),
            np.ones(6),
            -0.2 * np.ones(6),
        )

        widget.show()
        app.processEvents()
        image = widget.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertGreater(image.width(), 0)
        panel_width = image.width() / 2.0
        panel_height = image.height() / 3.0
        for row in range(3):
            for column in range(2):
                fill_count = 0
                for y in range(
                    int(row * panel_height + 30),
                    int((row + 1) * panel_height - 10),
                    12,
                ):
                    for x in range(
                        int(column * panel_width + 30),
                        int((column + 1) * panel_width - 10),
                        12,
                    ):
                        if image.pixelColor(x, y).name() == "#fbfcfd":
                            fill_count += 1
                self.assertGreater(fill_count, 10)
        widget.deleteLater()

    def test_dashboard_window_builds_offscreen(self) -> None:
        from PyQt5 import QtWidgets

        from jgarm_control.dashboard_node import DashboardWindow

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        window = DashboardWindow(object())
        self.assertIsNotNone(window.centralWidget())
        window.deleteLater()

    def test_apply_target_builds_follow_trajectory_goal(self) -> None:
        from PyQt5 import QtWidgets

        from jgarm_control.dashboard_node import DashboardWindow
        from jgarm_control.types import JOINT_NAMES

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        node = _FakeNode()
        node.mujoco_bias_effort[1] = -3.545
        node.gravity_effort[1] = -3.545
        node.pd_effort[1] = 0.0
        node.command_effort[1] = -3.545
        node.actual_effort[1] = -3.544
        window = DashboardWindow(node)
        window._duration_spin.setValue(1.0)
        window._refresh_plot()

        window._apply_target()

        goal = node.action_client.goal
        self.assertIsNotNone(goal)
        self.assertEqual(window._torque_items[1][1].text(), "-3.545")
        self.assertEqual(window._torque_items[1][2].text(), "-3.545")
        self.assertEqual(window._torque_items[1][4].text(), "-3.545")
        self.assertEqual(window._torque_items[1][5].text(), "-3.544")
        self.assertEqual(len(goal.goal_tolerance), len(JOINT_NAMES))
        self.assertEqual(
            [tolerance.name for tolerance in goal.goal_tolerance],
            list(JOINT_NAMES),
        )
        self.assertTrue(
            all(
                tolerance.position == 0.02
                and tolerance.velocity == 0.1
                for tolerance in goal.goal_tolerance
            )
        )
        window.deleteLater()

    def test_presets_match_power_on_and_zero(self) -> None:
        from jgarm_control.types import (
            POWER_ON_POSITION_RAD,
            URDF_ZERO_POSITION_RAD,
        )

        np.testing.assert_allclose(
            POWER_ON_POSITION_RAD,
            [0.0, -1.3265, 1.35, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(URDF_ZERO_POSITION_RAD, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
