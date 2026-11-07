"""Launch the JGarm MuJoCo simulation and controller."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("jgarm_description"))
    urdf_path = description_share / "urdf" / "JGarm.ros.urdf"
    model_path = description_share / "model" / "scene.xml"

    viewer = LaunchConfiguration("viewer")
    moveit = LaunchConfiguration("moveit")
    dashboard = LaunchConfiguration("dashboard")
    show_frames = LaunchConfiguration("show_frames")
    draggable_units = LaunchConfiguration("draggable_units")
    control_config = PathJoinSubstitution(
        [FindPackageShare("jgarm_bringup"), "config", "control.yaml"]
    )
    sim_config = PathJoinSubstitution(
        [FindPackageShare("jgarm_bringup"), "config", "sim.yaml"]
    )
    hardware_profile = PathJoinSubstitution(
        [FindPackageShare("jgarm_bringup"), "config", "hardware_profile.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "viewer",
                default_value="true",
                description="Open the MuJoCo viewer window.",
            ),
            DeclareLaunchArgument(
                "moveit",
                default_value="true",
                description="Start the MoveIt 2 move_group node.",
            ),
            DeclareLaunchArgument(
                "dashboard",
                default_value="true",
                description="Open the joint and torque dashboard.",
            ),
            DeclareLaunchArgument(
                "show_frames",
                default_value="false",
                description="Show model, post, and energy-unit frames.",
            ),
            DeclareLaunchArgument(
                "draggable_units",
                default_value="true",
                description="Release a magnetic constraint when selecting a unit.",
            ),
            Node(
                package="jgarm_sim_cpp",
                executable="mujoco_sim_node",
                name="mujoco_sim_node",
                output="screen",
                parameters=[
                    hardware_profile,
                    sim_config,
                    {"model_path": str(model_path)},
                ],
            ),
            Node(
                package="jgarm_control",
                executable="control_node",
                name="arm_control_node",
                output="screen",
                parameters=[
                    hardware_profile,
                    control_config,
                    {"urdf_path": str(urdf_path)},
                ],
            ),
            Node(
                package="jgarm_control",
                executable="trajectory_bridge_node",
                name="arm_trajectory_bridge_node",
                output="screen",
                parameters=[control_config],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {"robot_description": urdf_path.read_text(encoding="utf-8")}
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("jgarm_moveit_config"),
                            "launch",
                            "move_group.launch.py",
                        ]
                    )
                ),
                condition=IfCondition(moveit),
            ),
            Node(
                package="jgarm_control",
                executable="viewer_node",
                name="arm_viewer_node",
                output="screen",
                condition=IfCondition(viewer),
                parameters=[
                    {
                        "model_path": str(model_path),
                        "show_frames": ParameterValue(
                            show_frames,
                            value_type=bool,
                        ),
                        "draggable_energy_units": ParameterValue(
                            draggable_units,
                            value_type=bool,
                        ),
                    }
                ],
            ),
            Node(
                package="jgarm_control",
                executable="dashboard_node",
                name="jgarm_dashboard",
                output="screen",
                condition=IfCondition(dashboard),
            ),
        ]
    )
