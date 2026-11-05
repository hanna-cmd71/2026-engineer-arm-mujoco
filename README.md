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

## Use the Python API

The test and viewer entry points configure the source path themselves. For
an interactive control script, run it from the project root with:

```bash
PYTHONPATH=src python your_controller.py
```

Joint order is fixed to `joint_1` through `joint_6`. The hardware boundary
uses radians, radians per second, newton-metres, and seconds.
