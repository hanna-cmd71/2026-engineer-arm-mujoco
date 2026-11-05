#!/usr/bin/env bash
#
# Activate the ROS 2 Jazzy Python 3.12 runtime and the built ROS workspace.
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/activate_ros2_env.sh" >&2
  exit 1
fi

_jgarm_project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
_jgarm_site="$(
  cd "${_jgarm_project_root}/.ros-venv/lib/python3.12/site-packages" \
    >/dev/null 2>&1 && pwd
)"
_jgarm_cmeel_site="${_jgarm_site}/cmeel.prefix/lib/python3.12/site-packages"

if declare -F micromamba >/dev/null 2>&1; then
  micromamba deactivate >/dev/null 2>&1 || true
fi
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

source /opt/ros/jazzy/setup.bash
export PYTHONPATH="${_jgarm_site}:${_jgarm_cmeel_site}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${_jgarm_project_root}/ros2_ws/install/setup.bash" ]]; then
  source "${_jgarm_project_root}/ros2_ws/install/setup.bash"
fi

unset _jgarm_project_root
unset _jgarm_site
unset _jgarm_cmeel_site

echo "ROS 2 JGarm environment active: $(command -v python)"
