"""Terminal WASD teleoperation for the Mecanum mobile base."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class WasdTeleopNode(Node):
    """Publishes planar velocity commands from non-blocking terminal input."""

    def __init__(self) -> None:
        super().__init__("wasd_teleop_node")
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self._linear_speed = float(
            self.declare_parameter("linear_speed_m_s", 0.25).value
        )
        self._angular_speed = float(
            self.declare_parameter("angular_speed_rad_s", 0.6).value
        )
        self._key_timeout_s = float(
            self.declare_parameter("key_timeout_s", 0.3).value
        )
        self._command = Twist()
        self._active = False
        self._last_key_time = 0.0

    def _set_from_key(self, key: str) -> None:
        command = Twist()
        normalized_key = key.lower()
        if normalized_key == "w":
            command.linear.x = self._linear_speed
        elif normalized_key == "s":
            command.linear.x = -self._linear_speed
        elif normalized_key == "a":
            command.linear.y = self._linear_speed
        elif normalized_key == "d":
            command.linear.y = -self._linear_speed
        elif normalized_key == "q":
            command.angular.z = self._angular_speed
        elif normalized_key == "e":
            command.angular.z = -self._angular_speed
        elif normalized_key == " ":
            self._command = Twist()
            self._active = False
            self._publisher.publish(self._command)
            return
        else:
            return
        self._command = command
        self._active = True
        self._last_key_time = time.monotonic()
        self._publisher.publish(self._command)

    def run(self) -> None:
        """Reads terminal keys until ROS or Ctrl-C stops the node."""
        if not sys.stdin.isatty():
            raise RuntimeError(
                "WASD teleoperation requires an interactive terminal"
            )
        original_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setcbreak(sys.stdin.fileno())
            print(
                "WASD: move, Q/E: rotate, Space: stop, "
                "Q/E/other MuJoCo UI keys no longer affect the viewer."
            )
            while rclpy.ok():
                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.05,
                )
                if readable:
                    key = sys.stdin.read(1)
                    self._set_from_key(key)
                if (
                    self._active
                    and time.monotonic() - self._last_key_time
                    > self._key_timeout_s
                ):
                    self._command = Twist()
                    self._active = False
                    self._publisher.publish(self._command)
        finally:
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                original_settings,
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WasdTeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
