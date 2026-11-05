#!/usr/bin/env bash

set -euo pipefail

project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
python_bin="${HOME}/micromamba/envs/mujoco_env/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "MuJoCo Python was not found at ${python_bin}" >&2
  exit 1
fi

cd "${project_root}"

# Do not discover pytest plugins from ROS or other globally installed packages.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="${project_root}/src"

exec "${python_bin}" -m pytest "$@"

