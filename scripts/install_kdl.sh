#!/usr/bin/env bash

set -euo pipefail

project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
python_bin="${HOME}/micromamba/envs/mujoco_env/bin/python"
environment_prefix="${HOME}/micromamba/envs/mujoco_env"
kdl_version="v1.5.1"
kdl_parser_version="2.11.0"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/jgarm-kdl.XXXXXX")"

if [[ ! -x "${python_bin}" ]]; then
  echo "MuJoCo Python was not found at ${python_bin}" >&2
  exit 1
fi

echo "Installing Python dependencies..."
"${python_bin}" -m pip install \
  "pin==4.1.0" \
  "urdf-parser-py==0.0.4" \
  "catkin_pkg==1.1.1"

echo "Fetching Orocos KDL ${kdl_version}..."
git clone \
  --depth 1 \
  --branch "${kdl_version}" \
  --recurse-submodules \
  https://github.com/orocos/orocos_kinematics_dynamics.git \
  "${build_root}/orocos-kdl"

echo "Building PyKDL for Python 3.10..."
ROS_PYTHON_VERSION=3 cmake \
  -S "${build_root}/orocos-kdl/python_orocos_kdl" \
  -B "${build_root}/pykdl-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${environment_prefix}" \
  -DPYTHON_EXECUTABLE="${python_bin}"
cmake --build "${build_root}/pykdl-build" --parallel
cmake --install "${build_root}/pykdl-build"

site_packages="${environment_prefix}/lib/python3.10/site-packages"
dist_packages="${environment_prefix}/lib/python3.10/dist-packages"
cp "${dist_packages}/PyKDL.so" "${site_packages}/PyKDL.so"

echo "Fetching kdl_parser ${kdl_parser_version}..."
git clone \
  --depth 1 \
  --branch "${kdl_parser_version}" \
  https://github.com/ros/kdl_parser.git \
  "${build_root}/kdl_parser"

# The last Python package published in this tag predates Python 3 syntax
# restrictions and uses the removed PyKDL.Joint.None name.
kdl_parser_file="${build_root}/kdl_parser/kdl_parser_py/kdl_parser_py/urdf.py"
"${python_bin}" - "${kdl_parser_file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
source = source.replace("kdl.Joint.None", "kdl.Joint.Fixed")
path.write_text(source)
PY

cp -a \
  "${build_root}/kdl_parser/kdl_parser_py/kdl_parser_py" \
  "${site_packages}/"

echo "Verifying PyKDL and kdl_parser..."
PYTHONPATH="${project_root}/src" "${python_bin}" - <<'PY'
import PyKDL
import kdl_parser_py.urdf
import pinocchio

assert hasattr(PyKDL, "ChainIkSolverPos_NR_JL")
assert hasattr(PyKDL, "ChainIkSolverVel_pinv")
print(f"PyKDL: {PyKDL.__file__}")
print(f"Pinocchio: {pinocchio.__version__}")
PY

echo "KDL build files kept at ${build_root}"
