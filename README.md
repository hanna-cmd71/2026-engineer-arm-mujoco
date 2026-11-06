# JGarm MuJoCo Simulation

ROS 2 Jazzy simulation for the six-axis JGarm manipulator.

## Architecture

```text
C++23 jgarm_sim_cpp
  MuJoCo 3.17 physics
  1 kHz simulation timer
  /joint_states publisher
  /arm/control/effort_cmd subscriber
  /arm/sim/reset service

Python 3.12 jgarm_control
  KDL numerical IK
  Pinocchio generalized gravity
  PD gravity feedforward controller
  MuJoCo passive viewer driven by /joint_states

MoveIt 2
  KDL kinematics plugin
  OMPL RRT-Connect/RRT* planning
  FollowJointTrajectory action

Python 3.12 trajectory bridge
  cubic Hermite interpolation
  position and velocity reference publishing
  goal tolerance and completion checks
```

The C++ node owns physics. The Python nodes never advance the dynamic model;
they compute control and render the current ROS joint state.

## Environment

The ROS 2 runtime uses ROS 2 Jazzy with system Python 3.12. A project-local
`.ros-venv` adds MuJoCo, Pinocchio, and URDF parsing without mixing in the
legacy Python 3.10 environment.

The default C++ SDK path is `/sim-workspaces/mujoco-3.17.0`. Override it
before building with `export MUJOCO_ROOT=/path/to/mujoco`.

Install or refresh the runtime:

```bash
cd /sim-workspaces/JGarm
./scripts/install_ros2_env.sh
```

Activate ROS 2, the Python 3.12 paths, and the built workspace:

```bash
source scripts/activate_ros2_env.sh
```

The script can also be used before the first build; it sources the workspace
setup only when it exists.

## Build

```bash
cd /sim-workspaces/JGarm
source scripts/activate_ros2_env.sh

cd ros2_ws
colcon build --symlink-install
```

After building, open a new shell or reload the environment:

```bash
cd /sim-workspaces/JGarm
source scripts/activate_ros2_env.sh
```

## Run

Start the C++ MuJoCo simulation, Python controller, robot state publisher,
and live MuJoCo viewer:

```bash
ros2 launch jgarm_bringup sim.launch.py
```

Run headless:

```bash
ros2 launch jgarm_bringup sim.launch.py viewer:=false
```

The MuJoCo viewer process subscribes to `/joint_states` and updates the
display continuously without running a second physics simulation.

## Control interfaces

```text
/joint_states                    sensor_msgs/JointState
/arm/target/joint_position       std_msgs/Float64MultiArray
/arm/target/pose                 geometry_msgs/PoseStamped
/arm/control/effort_cmd          std_msgs/Float64MultiArray
/arm/sim/reset                   std_srvs/Trigger
```

Move to a joint target through MoveIt planning and trajectory execution:

```bash
python3 scripts/ros2_moveit_joint_goal.py \
  --target 0.1,-0.8,0.8,0.1,0.3,-0.1
```

The MoveIt client waits until the robot is stationary before planning, so a
new command does not start from a stale or moving state. The trajectory
bridge publishes position and velocity references together at 200 Hz, avoiding
position-only target jumps.

`/arm/target/joint_position` remains available for low-level debugging, but
directly publishing to it bypasses MoveIt planning and should not be used for
normal motion tests.

The pose target frame is `base_link`. The Python controller uses
`kdl_parser_py` and KDL numerical IK, then PD plus Pinocchio gravity
compensation.

Publish a TCP target while preserving the current TCP orientation:

```bash
python3 scripts/ros2_send_pose_target.py \
  --x 0.25 --y 0.05 --z 0.42
```

Plan and execute another joint goal:

```bash
python3 scripts/ros2_moveit_joint_goal.py \
  --target 0.2,-0.7,0.7,0.1,0.4,-0.1
```

## Tests

Build test:

```bash
source scripts/activate_ros2_env.sh
cd ros2_ws
colcon test
colcon test-result --verbose
```

Legacy Python algorithm regression tests are still available through the
Python 3.10 environment:

```bash
source scripts/activate_env.sh
./scripts/run_tests.sh -q
```

The ROS 2 runtime does not use the micromamba environment.
