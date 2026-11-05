# JGarm MuJoCo Simulation

Python-first MuJoCo simulation for the six-axis JGarm manipulator. The
simulation is intentionally independent of ROS, reATRM, and jRMCS hardware
runtime code. Its low-level state and torque command semantics are kept
compatible with the planned jRMCS arm interface.

## Environment

The project currently targets Python 3.10 and MuJoCo 3.8.1.

```bash
python3 -m pip install -r requirements.txt
```

## Prepare assets

The CAD-exported STL files are kept unchanged. Run the mesh preparation script
to generate simplified visual assets and convex collision assets:

```bash
python3 scripts/prepare_meshes.py --force
```

## Inspect the model

```bash
python3 scripts/view_model.py
```

## Run tests

```bash
python3 -m pytest
```

