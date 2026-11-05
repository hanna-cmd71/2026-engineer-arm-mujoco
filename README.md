# JGarm MuJoCo Simulation

Python-first MuJoCo simulation for the six-axis JGarm manipulator. The
simulation is independent of ROS and jRMCS hardware runtime code. Its
low-level state and torque command semantics remain compatible with the
planned jRMCS arm interface.

## Environment

This project uses one Micromamba root and one environment:

```text
Micromamba:  /opt/sim-envs/bin/micromamba
Root:        /home/hanner/micromamba
Environment: mujoco_env
Python:      3.10
```

ROS Jazzy must not be sourced in the same shell because it injects Python 3.12
packages into this Python 3.10 environment.

Create the environment:

```bash
MAMBA_ROOT_PREFIX="$HOME/micromamba" \
/opt/sim-envs/bin/micromamba env create \
  -f environment.yml
```

Activate it:

```bash
micromamba activate mujoco_env
```

Build and install the official Python 3.10 KDL bindings and `kdl_parser_py`:

```bash
./scripts/install_kdl.sh
```

This step installs Pinocchio 4.1.0, PyKDL 1.5.1, and the URDF parser used by
the numerical KDL inverse kinematics.

If the current terminal previously sourced ROS, use the project activation
script instead. It removes the stale ROS Python paths before activation:

```bash
cd /sim-workspaces/JGarm
source scripts/activate_env.sh
```

Verify the active interpreter:

```bash
which python
python --version
echo "$PYTHONPATH"
```

Expected:

```text
/home/hanner/micromamba/envs/mujoco_env/bin/python
Python 3.10.x
<empty PYTHONPATH>
```

## Prepare assets

The CAD-exported STL files remain unchanged. Generate simplified visual
meshes and convex collision meshes with:

```bash
python scripts/prepare_meshes.py --force
```

## Inspect the model

```bash
python scripts/view_model.py
```

Advance physics with zero actuator torque:

```bash
python scripts/view_model.py --physics
```

## Run tests

```bash
./scripts/run_tests.sh -q
```

The test wrapper disables pytest plugin auto-discovery so ROS Jazzy pytest
plugins cannot be loaded by the Python 3.10 MuJoCo environment.

Running pytest directly is also supported after activating the environment:

```bash
python -m pytest -q
```

## Kinematics and control

The Python package provides:

```text
Kinematics              KDL FK, Jacobian, and numerical IK
PinocchioDynamics       fixed-base generalized gravity
PDGravityController     joint PD plus gravity feedforward
```

Solve and display an IK target:

```bash
python scripts/demo_ik.py --x 0.50 --y 0.00 --z 0.50
```

Run joint PD control with Pinocchio gravity compensation:

```bash
python scripts/demo_pd_gravity.py \
  --target 0.1,-0.8,0.8,0.1,0.3,-0.1
```

## Use the Python API

The test and viewer entry points configure the source path themselves. For
an interactive control script, run it from the project root with:

```bash
PYTHONPATH=src python your_controller.py
```

Joint order is fixed to `joint_1` through `joint_6`. The hardware boundary
uses radians, radians per second, newton-metres, and seconds.
