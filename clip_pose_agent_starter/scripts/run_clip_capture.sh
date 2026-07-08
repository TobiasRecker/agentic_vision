#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${PKG_DIR}/../../.." && pwd)"

PACKAGE="clip_pose_agent_starter"
LAUNCH_FILE="mur620_clip_capture.launch.py"
ROBOT_NAME="${ROBOT_NAME:-mur620d}"

IMAGE_TOPIC="${IMAGE_TOPIC:-/oak/rgb/image_raw/compressed}"
IMAGE_COMPRESSED="${IMAGE_COMPRESSED:-true}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/oak/rgb/camera_info}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/oak/rgbd/points}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/clip_pose_sessions}"
SESSION_NAME="${SESSION_NAME:-}"
ROBOT_BASE_FRAME="${ROBOT_BASE_FRAME:-${ROBOT_NAME}/UR10_r/base_link}"
ROBOT_TCP_FRAME="${ROBOT_TCP_FRAME:-${ROBOT_NAME}/UR10_r/tool0}"
CAMERA_FRAME="${CAMERA_FRAME:-}"
EXTRA_TF_TOPICS="${EXTRA_TF_TOPICS:-}"
EXTRA_TF_STATIC_TOPICS="${EXTRA_TF_STATIC_TOPICS:-}"
USE_CONFIGURED_TCP_TO_CAMERA="${USE_CONFIGURED_TCP_TO_CAMERA:-true}"
TCP_TO_CAMERA_TRANSLATION_XYZ="${TCP_TO_CAMERA_TRANSLATION_XYZ:-0.0068564203,-0.0892312561,0.1018930213}"
TCP_TO_CAMERA_QUATERNION_XYZW="${TCP_TO_CAMERA_QUATERNION_XYZW:-0.0241307793,-0.0030488269,-0.0062149980,0.9996848423}"
PLANNING_FRAME="${PLANNING_FRAME:-${ROBOT_NAME}/UR10_r/base_link}"
ACTION_NAME="${ACTION_NAME:-/${ROBOT_NAME}/jparse_move_r}"
MOVE_ENABLED="${MOVE_ENABLED:-false}"
KEYBOARD_JOG_ENABLED="${KEYBOARD_JOG_ENABLED:-false}"
JOG_TWIST_TOPIC="${JOG_TWIST_TOPIC:-/${ROBOT_NAME}/jparse_velocity_controller_r/twist_cmd}"
JOG_FRAME="${JOG_FRAME:-UR10_r/base_link}"
SAMPLES="${SAMPLES:-18}"
ALLOW_2D_CENTER_FALLBACK="${ALLOW_2D_CENTER_FALLBACK:-true}"
FALLBACK_CENTER_DEPTH_M="${FALLBACK_CENTER_DEPTH_M:-0.45}"
WAIT_FOR_CAMERA_SEC="${WAIT_FOR_CAMERA_SEC:-12}"
REQUIRE_POINTCLOUD="${REQUIRE_POINTCLOUD:-false}"

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
echo "[clip_capture] robot:    ${ROBOT_NAME}"
echo "[clip_capture] image:     ${IMAGE_TOPIC} compressed=${IMAGE_COMPRESSED}"
echo "[clip_capture] info:      ${CAMERA_INFO_TOPIC}"
echo "[clip_capture] points:    ${POINTCLOUD_TOPIC}"
echo "[clip_capture] move:      ${MOVE_ENABLED}; jog: ${KEYBOARD_JOG_ENABLED}"
echo "[clip_capture] note:      z prepares a target; g sends it, and only if MOVE_ENABLED=true"
echo "[clip_capture] 2D center: ${ALLOW_2D_CENTER_FALLBACK}; fallback depth ${FALLBACK_CENTER_DEPTH_M} m"
echo "[clip_capture] extra TF:  ${EXTRA_TF_TOPICS}"
echo "[clip_capture] extra S-TF:${EXTRA_TF_STATIC_TOPICS}"
echo "[clip_capture] handeye:  configured=${USE_CONFIGURED_TCP_TO_CAMERA}"
echo "[clip_capture] handeye t:${TCP_TO_CAMERA_TRANSLATION_XYZ}"
echo "[clip_capture] handeye q:${TCP_TO_CAMERA_QUATERNION_XYZW}"
echo "[clip_capture] wait cam: ${WAIT_FOR_CAMERA_SEC}s; require pointcloud=${REQUIRE_POINTCLOUD}"
echo

topic_publisher_count() {
  local topic="$1"
  local topic_info
  topic_info="$(ros2 topic info "${topic}" 2>/dev/null || true)"
  printf '%s\n' "${topic_info}" | awk '/Publisher count:/ {print $3; found=1} END {if (!found) print 0}'
}

topic_type_or_empty() {
  local topic="$1"
  ros2 topic type "${topic}" 2>/dev/null || true
}

topic_has_publisher() {
  local topic="$1"
  local count
  count="$(topic_publisher_count "${topic}")"
  [[ "${count}" =~ ^[1-9][0-9]*$ ]]
}

wait_for_camera_publishers() {
  local required_topics=("${IMAGE_TOPIC}" "${CAMERA_INFO_TOPIC}")
  if [[ "${REQUIRE_POINTCLOUD}" == "true" ]]; then
    required_topics+=("${POINTCLOUD_TOPIC}")
  fi
  if [[ "${WAIT_FOR_CAMERA_SEC}" == "0" ]]; then
    return
  fi

  echo
  echo "[clip_capture] waiting for camera publishers..."
  local start now missing topic count topic_type
  start="$(date +%s)"
  while true; do
    missing=()
    for topic in "${required_topics[@]}"; do
      count="$(topic_publisher_count "${topic}")"
      topic_type="$(topic_type_or_empty "${topic}")"
      if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
        missing+=("${topic}{type=${topic_type:-none},pubs=${count:-0}}")
      fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
      echo "[clip_capture] camera publishers are visible."
      return
    fi
    now="$(date +%s)"
    if (( now - start >= WAIT_FOR_CAMERA_SEC )); then
      echo "[clip_capture] WARNING: camera publishers still missing after ${WAIT_FOR_CAMERA_SEC}s: ${missing[*]}"
      for topic in "${required_topics[@]}"; do
        echo "[clip_capture] topic info -v ${topic}:"
        timeout 3 ros2 topic info -v "${topic}" 2>/dev/null || true
      done
      return
    fi
    echo "[clip_capture] waiting: ${missing[*]}"
    sleep 1
  done
}

cd "${WS_DIR}"
echo "[clip_capture] building ${PACKAGE}..."
colcon build --packages-select "${PACKAGE}" --symlink-install

echo
echo "[clip_capture] sourcing install/setup.bash..."
set +u
source "${WS_DIR}/install/setup.bash"
set -u

echo
echo "[clip_capture] installed executables:"
ros2 pkg executables "${PACKAGE}" || true

echo
echo "[clip_capture] currently visible relevant topics:"
ros2 topic list 2>/dev/null | grep -E '(^/oak|jparse|tf)' || true
for topic in "${IMAGE_TOPIC}" "${CAMERA_INFO_TOPIC}" "${POINTCLOUD_TOPIC}" \
  ${EXTRA_TF_TOPICS//,/ } ${EXTRA_TF_STATIC_TOPICS//,/ }; do
  [[ -n "${topic}" ]] || continue
  topic_type="$(topic_type_or_empty "${topic}")"
  publisher_count="$(topic_publisher_count "${topic}")"
  echo "[clip_capture] topic ${topic}: type=${topic_type:-<not visible>} publishers=${publisher_count}"
done

wait_for_camera_publishers

echo
echo "[clip_capture] currently visible relevant actions:"
visible_actions="$(timeout 3 ros2 action list 2>/dev/null || true)"
printf '%s\n' "${visible_actions}" | grep -E 'jparse|move|UR10|mur620' || true
if ! printf '%s\n' "${visible_actions}" | grep -Fxq "${ACTION_NAME}"; then
  echo "[clip_capture] WARNING: action ${ACTION_NAME} is not currently visible"
fi

echo
echo "[clip_capture] camera_info sample:"
timeout 3 ros2 topic echo --once "${CAMERA_INFO_TOPIC}" 2>/dev/null | grep -E 'frame_id|height|width' || true

echo
echo "[clip_capture] sampled TF frames relevant to robot/camera:"
TF_SAMPLE_TOPICS="/tf_static /tf ${EXTRA_TF_STATIC_TOPICS//,/ } ${EXTRA_TF_TOPICS//,/ }"
for tf_topic in ${TF_SAMPLE_TOPICS}; do
  [[ -n "${tf_topic}" ]] || continue
  echo "[clip_capture] ${tf_topic}:"
  timeout 3 ros2 topic echo --once "${tf_topic}" 2>/dev/null \
    | grep -E 'frame_id|child_frame_id' \
    | grep -E 'mur620|mur620d|UR10|tool0|base_link|oak|camera|rgb|optical' \
    | sort -u || true
done

echo
echo "[clip_capture] launching..."
LAUNCH_ARGS=(
  image_topic:="${IMAGE_TOPIC}"
  image_compressed:="${IMAGE_COMPRESSED}"
  camera_info_topic:="${CAMERA_INFO_TOPIC}"
  pointcloud_topic:="${POINTCLOUD_TOPIC}"
  output_root:="${OUTPUT_ROOT}"
  robot_base_frame:="${ROBOT_BASE_FRAME}"
  robot_tcp_frame:="${ROBOT_TCP_FRAME}"
  use_configured_tcp_to_camera:="${USE_CONFIGURED_TCP_TO_CAMERA}"
  tcp_to_camera_translation_xyz:="${TCP_TO_CAMERA_TRANSLATION_XYZ}"
  tcp_to_camera_quaternion_xyzw:="${TCP_TO_CAMERA_QUATERNION_XYZW}"
  planning_frame:="${PLANNING_FRAME}"
  action_name:="${ACTION_NAME}"
  move_enabled:="${MOVE_ENABLED}"
  keyboard_jog_enabled:="${KEYBOARD_JOG_ENABLED}"
  jog_twist_topic:="${JOG_TWIST_TOPIC}"
  jog_frame:="${JOG_FRAME}"
  samples:="${SAMPLES}"
  allow_2d_center_fallback:="${ALLOW_2D_CENTER_FALLBACK}"
  fallback_center_depth_m:="${FALLBACK_CENTER_DEPTH_M}"
)
if [[ -n "${SESSION_NAME}" ]]; then
  LAUNCH_ARGS+=(session_name:="${SESSION_NAME}")
fi
if [[ -n "${CAMERA_FRAME}" ]]; then
  LAUNCH_ARGS+=(camera_frame:="${CAMERA_FRAME}")
fi
if [[ -n "${EXTRA_TF_TOPICS}" ]]; then
  LAUNCH_ARGS+=(extra_tf_topics:="${EXTRA_TF_TOPICS}")
fi
if [[ -n "${EXTRA_TF_STATIC_TOPICS}" ]]; then
  LAUNCH_ARGS+=(extra_tf_static_topics:="${EXTRA_TF_STATIC_TOPICS}")
fi

printf '[clip_capture] launch arg: %q\n' "${LAUNCH_ARGS[@]}"
ros2 launch "${PACKAGE}" "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}"
