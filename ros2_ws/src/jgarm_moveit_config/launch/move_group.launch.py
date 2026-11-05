"""Launch MoveIt 2 move_group for JGarm."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description() -> LaunchDescription:
    config_share = Path(get_package_share_directory("jgarm_moveit_config"))
    description_share = Path(get_package_share_directory("jgarm_description"))
    urdf_path = description_share / "urdf" / "JGarm.ros.urdf"

    moveit_config = (
        MoveItConfigsBuilder("JGarm", package_name="jgarm_moveit_config")
        .robot_description(file_path=str(urdf_path))
        .robot_description_semantic(
            file_path=str(config_share / "config" / "JGarm.srdf")
        )
        .robot_description_kinematics(
            file_path=str(config_share / "config" / "kinematics.yaml")
        )
        .joint_limits(
            file_path=str(config_share / "config" / "joint_limits.yaml")
        )
        .planning_pipelines(pipelines=["ompl"])
        .trajectory_execution(
            file_path=str(config_share / "config" / "moveit_controllers.yaml")
        )
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "publish_robot_description": True,
                "octomap_monitor": {"enabled": False},
            },
        ],
    )
    return LaunchDescription([move_group])
