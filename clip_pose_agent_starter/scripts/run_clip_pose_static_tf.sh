#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd -- "${PACKAGE_DIR}/../../.." && pwd)"
POSE_FILE="${POSE_FILE:-${1:-}}"
CHILD_FRAME="${CHILD_FRAME:-clip_task}"

if [[ -z "${POSE_FILE}" ]]; then
  echo "usage: POSE_FILE=~/clip_pose_sessions/<session>/hybrid/results/clip_pose.yaml $0" >&2
  exit 2
fi

POSE_FILE="$(realpath -m "${POSE_FILE/#\~/${HOME}}")"
SESSION_DIR="$(cd -- "$(dirname -- "${POSE_FILE}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SESSION_DIR}/hybrid/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/static_tf_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[clip_static_tf] workspace: ${WS_DIR}"
echo "[clip_static_tf] pose:      ${POSE_FILE}"
echo "[clip_static_tf] child:     ${CHILD_FRAME}"
echo "[clip_static_tf] log:       ${LOG_FILE}"

cd "${WS_DIR}"
colcon build --packages-select clip_pose_agent_starter --symlink-install
: "${COLCON_TRACE:=}"
set +u
source "${WS_DIR}/install/setup.bash"
set -u

ros2 run clip_pose_agent_starter clip_pose_static_tf --ros-args \
  -p pose_file:="${POSE_FILE}" \
  -p child_frame:="${CHILD_FRAME}"
