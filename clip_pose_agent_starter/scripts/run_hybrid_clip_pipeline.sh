#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd -- "${PACKAGE_DIR}/../../.." && pwd)"
MODE="${1:-all}"
SESSION_DIR="${SESSION_DIR:-${2:-}}"
CONFIG_FILE="${HYBRID_CONFIG:-${PACKAGE_DIR}/configs/hybrid_default.yaml}"

if [[ -z "${SESSION_DIR}" ]]; then
  echo "usage: SESSION_DIR=~/clip_pose_sessions/<session> $0 [prepare|annotate|reconstruct|all]" >&2
  exit 2
fi

SESSION_DIR="$(realpath -m "${SESSION_DIR/#\~/${HOME}}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SESSION_DIR}/hybrid/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/hybrid_${MODE}_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[clip_hybrid] workspace: ${WS_DIR}"
echo "[clip_hybrid] session:   ${SESSION_DIR}"
echo "[clip_hybrid] command:   ${MODE}"
echo "[clip_hybrid] config:    ${CONFIG_FILE}"
echo "[clip_hybrid] log:       ${LOG_FILE}"

cd "${WS_DIR}"
colcon build --packages-select clip_pose_agent_starter --symlink-install
: "${COLCON_TRACE:=}"
set +u
source "${WS_DIR}/install/setup.bash"
set -u

ros2 run clip_pose_agent_starter clip_hybrid_pipeline "${MODE}" \
  --session "${SESSION_DIR}" \
  --config "${CONFIG_FILE}"
