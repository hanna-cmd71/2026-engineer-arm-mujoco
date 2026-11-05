#!/usr/bin/env bash
#
# Activate the project's single Micromamba environment in a clean runtime.
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/activate_env.sh" >&2
  exit 1
fi

_jgarm_project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"

# Remove ROS Jazzy and stale Python paths inherited from an existing shell.
unset PYTHONPATH
unset PYTHONHOME
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset LD_LIBRARY_PATH
unset PKG_CONFIG_PATH

export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$(/opt/sim-envs/bin/micromamba shell hook --shell bash)"
micromamba activate mujoco_env
hash -r

cd "${_jgarm_project_root}" || return 1
unset _jgarm_project_root

echo "JGarm environment active: $(command -v python)"

