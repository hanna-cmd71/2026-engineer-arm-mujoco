# JGarm MuJoCo Simulation

ROS 2 Jazzy simulation for the six-axis JGarm manipulator.

## Architecture

```text
C++23 jgarm_sim_cpp
  MuJoCo 3.17 physics
  1 kHz simulation timer
  power-on keyframe and gravity-load diagnostics
  /joint_states publisher
  /arm/control/effort_cmd subscriber
  /arm/sim/reset service

Python 3.12 jgarm_control
  KDL numerical IK
  reATRM Engineer hardware profile
  gravity, friction, breakaway, and PD feedforward
  MuJoCo passive viewer driven by /joint_states
  Qt dashboard for joint targets, PD gains, and six-axis torque plots

MoveIt 2
  KDL kinematics plugin
  OMPL RRT-Connect/RRT* planning
  FollowJointTrajectory action

Python 3.12 trajectory bridge
  cubic Hermite interpolation
  position and velocity reference publishing
  goal tolerance and completion checks
  start-state validation and no-jump cancellation
```

The C++ node owns physics. The Python nodes never advance the dynamic model;
they compute control and render the current ROS joint state.

## Hardware profile

The shared [hardware_profile.yaml](ros2_ws/src/jgarm_bringup/config/hardware_profile.yaml)
extracts the arm hardware information from reATRM's Engineer layer:

- DM4310 on J1/J4/J5/J6 with a protocol torque limit of `10 N*m`;
- LK MF7015 on J2/J3;
- joint gear ratio `[1, 50, 50, 1, 1.5, 1]`;
- the identified gravity, friction, and breakaway compensation model;
- a 500 Hz internal command/control boundary;
- an approximately 111 Hz external `/joint_states` rate.

The controller reads a separate `/arm/control/joint_state` fast channel, matching
the way reATRM's controller consumes hardware feedback internally. MoveIt and the
Dashboard continue to consume `/joint_states`.

The reATRM J5 gravity fit contains a non-conservative empirical coupling. For
MuJoCo's passive plant it is projected to a conservative force field. The
projection preserves the two acceptance poses within about `0.005 N*m`.

The joint viscous damping (`0.05 N*m*s/rad` for J1-J3 and
`0.5 N*m*s/rad` for J4-J6), `0.001 kg*m^2` armature, and `5 ms` actuator
response are explicitly marked simulation placeholders. reATRM does not identify
these values, so they must not be presented as measured hardware data.

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
live MuJoCo viewer, and the control dashboard:

```bash
ros2 launch jgarm_bringup sim.launch.py
```

Run headless:

```bash
ros2 launch jgarm_bringup sim.launch.py \
  viewer:=false dashboard:=false
```

Show world, assembly-target, placement-post, and energy-unit frames:

```bash
ros2 launch jgarm_bringup sim.launch.py show_frames:=true
```

The frame axes are in MuJoCo visual group 5: red X, green Y, and blue Z.
Energy units are draggable by default. Select a unit in the MuJoCo viewer and
use its body-drag controls. The first selection starts a magnetic extraction:
the unit may move along the placement-post axis with 10 mm lateral and 10 degree
angular clearance. The magnet releases after the unit travels one full post
length. Keep units locked with:

```bash
ros2 launch jgarm_bringup sim.launch.py draggable_units:=false
```

The MuJoCo viewer process subscribes to `/joint_states` and updates the
display continuously without running a second physics simulation.

## Task scene

[scene.xml](model/scene.xml) combines the arm with the converted URDF assets
from the ground-station workspace:

- `exchange_core` is the static transfer target in front of the arm;
- `energy_storage` is the static resource station behind the arm;
- `energy_unit_01` through `energy_unit_06` are the six 0.4 kg movable units
  held at the resource-station magnetic posts.

The scene uses the arm base as the frame origin. The six units follow RM 2027
rules: 0.4 kg nominal mass, magnetic retention, and extraction along their
symmetry axes. MuJoCo retains its real 9.81 m/s^2 world gravity; the arm
controller compensates that load through the reATRM hardware profile. Exact
rule constants and assembly motion limits are recorded in
[rm2027.yaml](ros2_ws/src/jgarm_bringup/config/rm2027.yaml).

The arm is mounted on a planar mobile base with four Mecanum-style wheels.
The base supports forward, lateral, and yaw velocity commands on `/cmd_vel`
and publishes `/odom` plus the dynamic `world -> base_link` transform.
Start the terminal teleoperation node in a second terminal:

```bash
ros2 run jgarm_control wasd_teleop_node
```

Use `W`/`S` to move forward/backward, `A`/`D` to strafe left/right, and
`Q`/`E` to rotate left/right. Space stops immediately. The terminal input
avoids MuJoCo's own keyboard shortcuts, so movement cannot switch the viewer
to wireframe mode. The camera remains fixed over a 10 m by 10 m workspace.

Regenerate the simplified scene meshes from the source URDF workspaces with:

```bash
source scripts/activate_env.sh
python scripts/prepare_scene_meshes.py --force
```

The generated meshes are installed with `jgarm_description`; simulation does
not read the source URDF packages at runtime.

## Control dashboard

The dashboard opens two visualization windows by default:

- the live MuJoCo model;
- six joint target controls, runtime Kp/Kd inputs, a gravity-compensation
  switch, six torque plots, and a static torque value table.

The default pose matches the powered robot:

```text
[0, -1.3265, 1.35, 0, 0, 0] rad
```

`J2=-1.327` on the real display corresponds to the URDF lower limit
`-1.3265 rad` after three-decimal rounding, so the model keeps the valid URDF
value to avoid starting outside the joint range.

For each joint, set a target with the spin box or slider and select
`Apply target`. The dashboard generates a rest-to-rest quintic trajectory,
sends it through the `FollowJointTrajectory` action, and lets MoveIt 2 execute
the planned motion. The target therefore changes through smooth position and
velocity references instead of a direct torque step.

Each torque plot shows:

- `actual`: torque reported by MuJoCo in `/joint_states[].effort`;
- `command`: the clipped torque sent to MuJoCo;
- `PD`: the proportional and derivative contribution;
- `gravity`: the identified reATRM gravity feedforward contribution.
- `friction`: the identified velocity-friction compensation;
- `breakaway`: the identified low-speed breakaway compensation.

The numeric table adds:

- `MuJoCo`: the hardware-profile load used by the simulator;
- `Gravity`: the same hardware-profile gravity model used by the controller;
- `PD`: the proportional and derivative contribution;
- `Hold`: the total torque command used to hold the arm;
- actual actuator torque remains visible in the plot.

Use `Power-on pose` to return to the real power-on posture and `URDF zero` to
move smoothly to the all-zero URDF posture. At URDF zero, the table normally
shows approximately `-17.651 N·m` on J2, `-14.157 N·m` on J3, and
`+2.760 N·m` on J5 even
though every joint angle is zero. This confirms that gravity is still active.

`Apply PD` updates all Kp/Kd values and the gravity switch atomically.
`Hold current` plans a short move back to the measured pose. `Disable drive`
publishes zero torque and cancels the dashboard-owned trajectory; in the real
gravity model the arm can then fall, so use `Hold current` for a controlled
stop and reserve `Disable drive` for an intentional emergency stop.

The default dashboard requires a desktop display. Use the headless command
above on a server or in automated tests.

## Control interfaces

```text
/joint_states                    sensor_msgs/JointState
/arm/control/joint_state         sensor_msgs/JointState (500 Hz internal)
/arm/target/joint_position       std_msgs/Float64MultiArray
/arm/target/pose                 geometry_msgs/PoseStamped
/arm/control/effort_cmd          std_msgs/Float64MultiArray
/arm/debug/pd_effort             std_msgs/Float64MultiArray
/arm/debug/gravity_effort        std_msgs/Float64MultiArray
/arm/debug/mujoco_bias_effort    std_msgs/Float64MultiArray
/arm/debug/mujoco_friction_effort std_msgs/Float64MultiArray
/arm/debug/friction_effort       std_msgs/Float64MultiArray
/arm/debug/breakaway_effort      std_msgs/Float64MultiArray
/arm/debug/position_error        std_msgs/Float64MultiArray
/arm/control/enable              std_srvs/SetBool
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
`kdl_parser_py` and KDL numerical IK, then the shared Engineer hardware
profile for PD, gravity, friction, and breakaway compensation.

The trajectory bridge rejects a trajectory when its first point differs from
the measured joint state by more than `0.03 rad`. This prevents the initial
target step that previously caused arm jitter. Canceling a trajectory holds
the last commanded reference instead of jumping to the final point.

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

Inspect the torque and control diagnostics while the simulation is running:

```bash
ros2 topic echo /arm/debug/pd_effort --once
ros2 topic echo /arm/debug/gravity_effort --once
ros2 topic echo /arm/debug/position_error --once
ros2 topic info /joint_states -v
```

Legacy Python algorithm regression tests are still available through the
Python 3.10 environment:

```bash
source scripts/activate_env.sh
./scripts/run_tests.sh -q
```

The ROS 2 runtime does not use the micromamba environment.
