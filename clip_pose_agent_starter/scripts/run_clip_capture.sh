#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${PKG_DIR}/../../.." && pwd)"

PACKAGE="clip_pose_agent_starter"
LAUNCH_FILE="mur620_clip_capture.launch.py"

IMAGE_TOPIC="${IMAGE_TOPIC:-/oak/rgb/image_raw/compressed}"
IMAGE_COMPRESSED="${IMAGE_COMPRESSED:-true}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/oak/rgb/camera_info}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/oak/rgbd/points}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/clip_pose_sessions}"
SESSION_NAME="${SESSION_NAME:-}"
ROBOT_BASE_FRAME="${ROBOT_BASE_FRAME:-mur620/UR10_r/base_link}"
ROBOT_TCP_FRAME="${ROBOT_TCP_FRAME:-mur620/UR10_r/tool0}"
CAMERA_FRAME="${CAMERA_FRAME:-}"
PLANNING_FRAME="${PLANNING_FRAME:-mur620/UR10_r/base_link}"
ACTION_NAME="${ACTION_NAME:-/mur620/jparse_move_r}"
MOVE_ENABLED="${MOVE_ENABLED:-false}"
KEYBOARD_JOG_ENABLED="${KEYBOARD_JOG_ENABLED:-false}"
JOG_TWIST_TOPIC="${JOG_TWIST_TOPIC:-/mur620/jparse_velocity_controller_r/twist_cmd}"
JOG_FRAME="${JOG_FRAME:-UR10_r/base_link}"
SAMPLES="${SAMPLES:-18}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${CLIP_CAPTURE_LOG_DIR:-${OUTPUT_ROOT}/node_logs}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/clip_capture_${STAMP}.log"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${LOG_ROOT}/ros_${STAMP}}"
mkdir -p "${ROS_LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[clip_capture] workspace: ${WS_DIR}"
echo "[clip_capture] package:   ${PACKAGE}"
echo "[clip_capture] log file:  ${LOG_FILE}"
echo "[clip_capture] ros logs:  ${ROS_LOG_DIR}"
echo "[clip_capture] image:     ${IMAGE_TOPIC} compressed=${IMAGE_COMPRESSED}"
echo "[clip_capture] info:      ${CAMERA_INFO_TOPIC}"
echo "[clip_capture] points:    ${POINTCLOUD_TOPIC}"
echo "[clip_capture] move:      ${MOVE_ENABLED}; jog: ${KEYBOARD_JOG_ENABLED}"
echo

cd "${WS_DIR}"
echo "[clip_capture] building ${PACKAGE}..."
colcon build --packages-select "${PACKAGE}" --symlink-install

echo
echo "[clip_capture] sourcing install/setup.bash..."
source "${WS_DIR}/install/setup.bash"

echo
echo "[clip_capture] installed executables:"
ros2 pkg executables "${PACKAGE}" || true

echo
echo "[clip_capture] currently visible relevant topics:"
ros2 topic list 2>/dev/null | grep -E '(^/oak|jparse|tf)' || true
for topic in "${IMAGE_TOPIC}" "${CAMERA_INFO_TOPIC}" "${POINTCLOUD_TOPIC}"; do
  echo -n "[clip_capture] topic type ${topic}: "
  ros2 topic type "${topic}" 2>/dev/null || true
done

echo
echo "[clip_capture] launching..."
ros2 launch "${PACKAGE}" "${LAUNCH_FILE}" \
  image_topic:="${IMAGE_TOPIC}" \
  image_compressed:="${IMAGE_COMPRESSED}" \
  camera_info_topic:="${CAMERA_INFO_TOPIC}" \
  pointcloud_topic:="${POINTCLOUD_TOPIC}" \
  output_root:="${OUTPUT_ROOT}" \
  session_name:="${SESSION_NAME}" \
  robot_base_frame:="${ROBOT_BASE_FRAME}" \
  robot_tcp_frame:="${ROBOT_TCP_FRAME}" \
  camera_frame:="${CAMERA_FRAME}" \
  planning_frame:="${PLANNING_FRAME}" \
  action_name:="${ACTION_NAME}" \
  move_enabled:="${MOVE_ENABLED}" \
  keyboard_jog_enabled:="${KEYBOARD_JOG_ENABLED}" \
  jog_twist_topic:="${JOG_TWIST_TOPIC}" \
  jog_frame:="${JOG_FRAME}" \
  samples:="${SAMPLES}"
