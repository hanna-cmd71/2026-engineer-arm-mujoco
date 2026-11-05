#!/usr/bin/env bash

set -euo pipefail

project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
venv_root="${project_root}/.ros-venv"
site_packages="${venv_root}/lib/python3.12/site-packages"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/jgarm-ros-env.XXXXXX")"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy was not found under /opt/ros/jazzy" >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash
if ! /usr/bin/python3 -c "import PyKDL, rclpy"; then
  echo "System PyKDL or rclpy is unavailable." >&2
  echo "Install the ROS 2 Jazzy Python and PyKDL packages first." >&2
  exit 1
fi

if [[ ! -x "${venv_root}/bin/python" ]]; then
  /usr/bin/python3 -m venv --system-site-packages "${venv_root}"
fi

"${venv_root}/bin/python" -m pip install \
  -r "${project_root}/requirements-ros2.txt"

git clone \
  --depth 1 \
  --branch 2.11.0 \
  https://github.com/ros/kdl_parser.git \
  "${build_root}/kdl_parser"

kdl_parser_file="${build_root}/kdl_parser/kdl_parser_py/kdl_parser_py/urdf.py"
"${venv_root}/bin/python" - "${kdl_parser_file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
source = source.replace("kdl.Joint.None", "kdl.Joint.Fixed")
source = source.replace(
    "urdf.URDF.from_xml_string(urdf_file.read())",
    'urdf.URDF.from_xml_string(urdf_file.read().encode("utf-8"))',
)
path.write_text(source)
PY

rm -rf "${site_packages}/kdl_parser_py"
cp -a \
  "${build_root}/kdl_parser/kdl_parser_py/kdl_parser_py" \
  "${site_packages}/"

echo "Verifying ROS 2 Python runtime..."
PYTHONPATH="${site_packages}:${site_packages}/cmeel.prefix/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}" \
  /usr/bin/python3 - <<'PY'
import rclpy
import PyKDL
import mujoco
import pinocchio
import kdl_parser_py.urdf

print(f"rclpy: {rclpy.__file__}")
print(f"PyKDL: {PyKDL.__file__}")
print(f"MuJoCo: {mujoco.__version__}")
print(f"Pinocchio: {pinocchio.__version__}")
PY

echo "ROS 2 environment files kept at ${build_root}"
