#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${PKG_DIR}/../../.." && pwd)"

PACKAGE="${OAK_DRIVER_PACKAGE:-oak_camera_calibration}"
LAUNCH_FILE="${OAK_LAUNCH_FILE:-oak4_pro_af_gui.launch.py}"
ROBOT_NAME="${ROBOT_NAME:-mur620d}"
PARENT_FRAME="${OAK_PARENT_FRAME:-${ROBOT_NAME}/UR10_r/tool0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/clip_pose_sessions}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${CLIP_CAPTURE_LOG_DIR:-${OUTPUT_ROOT}/node_logs}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/oak_driver_${STAMP}.log"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${LOG_ROOT}/ros_oak_${STAMP}}"
mkdir -p "${ROS_LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[oak_driver] workspace: ${WS_DIR}"
echo "[oak_driver] package:   ${PACKAGE}"
echo "[oak_driver] launch:    ${LAUNCH_FILE}"
echo "[oak_driver] parent:    ${PARENT_FRAME}"
echo "[oak_driver] log file:  ${LOG_FILE}"
echo "[oak_driver] ros logs:  ${ROS_LOG_DIR}"
echo

cd "${WS_DIR}"
echo "[oak_driver] building ${PACKAGE}..."
colcon build --packages-select "${PACKAGE}" --symlink-install

echo
echo "[oak_driver] sourcing install/setup.bash..."
set +u
source "${WS_DIR}/install/setup.bash"
set -u

echo
echo "[oak_driver] installed executables:"
ros2 pkg executables "${PACKAGE}" || true

echo
echo "[oak_driver] launch arguments:"
ros2 launch "${PACKAGE}" "${LAUNCH_FILE}" --show-args || true

echo
echo "[oak_driver] launching..."
ros2 launch "${PACKAGE}" "${LAUNCH_FILE}" parent_frame:="${PARENT_FRAME}"
