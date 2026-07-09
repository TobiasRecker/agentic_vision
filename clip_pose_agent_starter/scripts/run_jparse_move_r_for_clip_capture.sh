#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${PKG_DIR}/../../.." && pwd)"

PACKAGE="${JPARSE_PACKAGE:-mur_control}"
ROBOT_NAME="${ROBOT_NAME:-mur620d}"
ARM="${ARM:-r}"
ARM_NAME="${ARM_NAME:-UR10_${ARM}}"
BASE_LINK="${BASE_LINK:-${ARM_NAME}/base_link}"
TIP_LINK="${TIP_LINK:-${ARM_NAME}/tool0}"
ACTION_NAME="${ACTION_NAME:-/${ROBOT_NAME}/jparse_move_${ARM}}"
TWIST_TOPIC="${TWIST_TOPIC:-/${ROBOT_NAME}/jparse_velocity_controller_${ARM}/twist_cmd}"
SAFE_COMMAND_TOPIC="${SAFE_COMMAND_TOPIC:-/${ROBOT_NAME}/${ARM_NAME}/safe_forward_velocity_controller/commands}"
DIRECT_COMMAND_TOPIC="${DIRECT_COMMAND_TOPIC:-/${ROBOT_NAME}/${ARM_NAME}/forward_velocity_controller/commands}"
COMMAND_TOPIC="${COMMAND_TOPIC:-auto}"
JOINT_STATES_TOPIC="${JOINT_STATES_TOPIC:-/joint_states}"
ROBOT_DESCRIPTION_TOPIC="${ROBOT_DESCRIPTION_TOPIC:-/${ROBOT_NAME}/robot_description}"
DEBUG_TWIST_TOPIC="${DEBUG_TWIST_TOPIC:-/${ROBOT_NAME}/jparse_velocity_controller_${ARM}/debug_twist}"
SINGULAR_VALUES_TOPIC="${SINGULAR_VALUES_TOPIC:-/${ROBOT_NAME}/jparse_velocity_controller_${ARM}/singular_values}"
CONTROLLER_MANAGER="${CONTROLLER_MANAGER:-/${ROBOT_NAME}/${ARM_NAME}/controller_manager}"
REQUIRED_CONTROLLER="${REQUIRED_CONTROLLER:-forward_velocity_controller}"
CONFLICTING_CONTROLLERS="${CONFLICTING_CONTROLLERS:-integrated_cartesian_admittance_controller scaled_joint_trajectory_controller joint_trajectory_controller passthrough_trajectory_controller forward_position_controller forward_effort_controller}"
SWITCH_TO_FORWARD_CONTROLLER="${SWITCH_TO_FORWARD_CONTROLLER:-true}"
RESTORE_CONTROLLERS_ON_EXIT="${RESTORE_CONTROLLERS_ON_EXIT:-true}"
CONTROLLER_SWITCH_TIMEOUT="${CONTROLLER_SWITCH_TIMEOUT:-5.0}"
RATE_HZ="${RATE_HZ:-500.0}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-0.12}"
INVERSE_MODE="${INVERSE_MODE:-jparse}"
DAMPING="${DAMPING:-0.03}"
MAX_JOINT_VELOCITY="${MAX_JOINT_VELOCITY:-0.6}"
MAX_LINEAR_VELOCITY="${MAX_LINEAR_VELOCITY:-0.12}"
MAX_ANGULAR_VELOCITY="${MAX_ANGULAR_VELOCITY:-0.5}"
WAIT_FOR_DISCOVERY_SEC="${WAIT_FOR_DISCOVERY_SEC:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/clip_pose_sessions}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${CLIP_CAPTURE_LOG_DIR:-${OUTPUT_ROOT}/node_logs}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/jparse_move_${ARM}_${STAMP}.log"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${LOG_ROOT}/ros_jparse_${ARM}_${STAMP}}"
mkdir -p "${ROS_LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[jparse_capture] workspace: ${WS_DIR}"
echo "[jparse_capture] package:   ${PACKAGE}"
echo "[jparse_capture] log file:  ${LOG_FILE}"
echo "[jparse_capture] ros logs:  ${ROS_LOG_DIR}"
echo "[jparse_capture] robot:    ${ROBOT_NAME}"
echo "[jparse_capture] arm:      ${ARM} (${ARM_NAME})"
echo "[jparse_capture] base/tip: ${BASE_LINK} -> ${TIP_LINK}"
echo "[jparse_capture] action:   ${ACTION_NAME}"
echo "[jparse_capture] twist:    ${TWIST_TOPIC}"
echo "[jparse_capture] command:  ${COMMAND_TOPIC} (safe=${SAFE_COMMAND_TOPIC}, direct=${DIRECT_COMMAND_TOPIC})"
echo "[jparse_capture] control:  manager=${CONTROLLER_MANAGER}, required=${REQUIRED_CONTROLLER}, switch=${SWITCH_TO_FORWARD_CONTROLLER}, restore=${RESTORE_CONTROLLERS_ON_EXIT}"
echo

PIDS=()
SWITCHED_CONTROLLERS=false
ACTIVE_CONTROLLERS_BEFORE=()
CONTROLLER_LIST_BEFORE=""

topic_count_field() {
  local topic="$1"
  local label="$2"
  local topic_info
  topic_info="$(ros2 topic info "${topic}" 2>/dev/null || true)"
  printf '%s\n' "${topic_info}" | awk -v label="${label}" '$1 == label && $2 == "count:" {print $3; found=1} END {if (!found) print 0}'
}

topic_type_or_empty() {
  local topic="$1"
  ros2 topic type "${topic}" 2>/dev/null || true
}

subscriber_count() {
  topic_count_field "$1" "Subscription"
}

publisher_count() {
  topic_count_field "$1" "Publisher"
}

resolve_command_topic() {
  if [[ "${COMMAND_TOPIC}" != "auto" ]]; then
    printf '%s\n' "${COMMAND_TOPIC}"
    return
  fi

  local safe_subs direct_subs
  safe_subs="$(subscriber_count "${SAFE_COMMAND_TOPIC}")"
  direct_subs="$(subscriber_count "${DIRECT_COMMAND_TOPIC}")"
  if [[ "${safe_subs}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' "${SAFE_COMMAND_TOPIC}"
  elif [[ "${direct_subs}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' "${DIRECT_COMMAND_TOPIC}"
  else
    printf '%s\n' "${DIRECT_COMMAND_TOPIC}"
  fi
}

print_topic_summary() {
  local topic="$1"
  local topic_type pubs subs
  topic_type="$(topic_type_or_empty "${topic}")"
  pubs="$(publisher_count "${topic}")"
  subs="$(subscriber_count "${topic}")"
  echo "[jparse_capture] topic ${topic}: type=${topic_type:-<not visible>} publishers=${pubs} subscribers=${subs}"
}

list_controllers() {
  timeout 8 ros2 control list_controllers -c "${CONTROLLER_MANAGER}" 2>/dev/null || true
}

controller_state_from_list() {
  local list_text="$1"
  local controller="$2"
  printf '%s\n' "${list_text}" | awk -v controller="${controller}" '$1 == controller {print $3; found=1; exit} END {if (!found) print ""}'
}

collect_active_conflicting_controllers() {
  local list_text="$1"
  local controller state
  ACTIVE_CONTROLLERS_BEFORE=()
  for controller in ${CONFLICTING_CONTROLLERS}; do
    state="$(controller_state_from_list "${list_text}" "${controller}")"
    if [[ "${state}" == "active" ]]; then
      ACTIVE_CONTROLLERS_BEFORE+=("${controller}")
    fi
  done
}

ensure_required_controller_active() {
  CONTROLLER_LIST_BEFORE="$(list_controllers)"
  if [[ -z "${CONTROLLER_LIST_BEFORE}" ]]; then
    echo "[jparse_capture] WARNING: could not read controllers from ${CONTROLLER_MANAGER}"
    return
  fi

  echo "${CONTROLLER_LIST_BEFORE}"
  local required_state
  required_state="$(controller_state_from_list "${CONTROLLER_LIST_BEFORE}" "${REQUIRED_CONTROLLER}")"
  collect_active_conflicting_controllers "${CONTROLLER_LIST_BEFORE}"

  if [[ "${required_state}" == "active" ]]; then
    echo "[jparse_capture] ${REQUIRED_CONTROLLER} is already active."
    return
  fi

  echo "[jparse_capture] ${REQUIRED_CONTROLLER} is ${required_state:-not listed}; active conflicting controllers: ${ACTIVE_CONTROLLERS_BEFORE[*]:-<none>}"
  if [[ "${SWITCH_TO_FORWARD_CONTROLLER}" != "true" ]]; then
    echo "[jparse_capture] WARNING: controller switching is disabled. J-PARSE will likely time out without robot motion."
    return
  fi

  local switch_cmd=(
    timeout 12
    ros2 control switch_controllers
    -c "${CONTROLLER_MANAGER}"
    --activate "${REQUIRED_CONTROLLER}"
    --strict
    --switch-timeout "${CONTROLLER_SWITCH_TIMEOUT}"
  )
  if [[ "${#ACTIVE_CONTROLLERS_BEFORE[@]}" -gt 0 ]]; then
    switch_cmd+=(--deactivate "${ACTIVE_CONTROLLERS_BEFORE[@]}")
  fi

  echo "[jparse_capture] switching controllers: ${switch_cmd[*]}"
  if "${switch_cmd[@]}"; then
    SWITCHED_CONTROLLERS=true
    echo "[jparse_capture] controller switch succeeded."
    echo "[jparse_capture] controller state after switch:"
    list_controllers
  else
    echo "[jparse_capture] ERROR: controller switch failed. J-PARSE cannot move the robot in the current controller state."
  fi
}

restore_controllers() {
  if [[ "${SWITCHED_CONTROLLERS}" != "true" || "${RESTORE_CONTROLLERS_ON_EXIT}" != "true" ]]; then
    return
  fi

  local restore_cmd=(
    timeout 12
    ros2 control switch_controllers
    -c "${CONTROLLER_MANAGER}"
    --deactivate "${REQUIRED_CONTROLLER}"
    --best-effort
    --switch-timeout "${CONTROLLER_SWITCH_TIMEOUT}"
  )
  if [[ "${#ACTIVE_CONTROLLERS_BEFORE[@]}" -gt 0 ]]; then
    restore_cmd+=(--activate "${ACTIVE_CONTROLLERS_BEFORE[@]}")
  fi

  echo "[jparse_capture] restoring controllers: ${restore_cmd[*]}"
  "${restore_cmd[@]}" || true
}

cd "${WS_DIR}"
echo "[jparse_capture] building ${PACKAGE}..."
colcon build --packages-select "${PACKAGE}" --symlink-install

echo
echo "[jparse_capture] sourcing install/setup.bash..."
set +u
source "${WS_DIR}/install/setup.bash"
set -u

echo
echo "[jparse_capture] installed executables:"
ros2 pkg executables "${PACKAGE}" | grep -E 'jparse|mur_control' || ros2 pkg executables "${PACKAGE}" || true

echo
echo "[jparse_capture] current controller state:"
ensure_required_controller_active

echo
echo "[jparse_capture] relevant topics before start:"
for topic in \
  "${ROBOT_DESCRIPTION_TOPIC}" \
  "${JOINT_STATES_TOPIC}" \
  "${TWIST_TOPIC}" \
  "${SAFE_COMMAND_TOPIC}" \
  "${DIRECT_COMMAND_TOPIC}" \
  "${DEBUG_TWIST_TOPIC}" \
  "${SINGULAR_VALUES_TOPIC}"; do
  print_topic_summary "${topic}"
done

COMMAND_TOPIC_RESOLVED="$(resolve_command_topic)"
echo
echo "[jparse_capture] resolved command topic: ${COMMAND_TOPIC_RESOLVED}"
if [[ "${COMMAND_TOPIC_RESOLVED}" == "${SAFE_COMMAND_TOPIC}" ]]; then
  echo "[jparse_capture] using safety input topic because a subscriber is visible there."
else
  echo "[jparse_capture] using direct forward velocity topic; no safety input subscriber was visible."
fi

echo
echo "[jparse_capture] quick TF sample ${ROBOT_NAME}/${BASE_LINK} -> ${ROBOT_NAME}/${TIP_LINK}:"
timeout 4 ros2 run tf2_ros tf2_echo "${ROBOT_NAME}/${BASE_LINK}" "${ROBOT_NAME}/${TIP_LINK}" 2>/dev/null | head -30 || true

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo
    echo "[jparse_capture] stopping child processes..."
    for pid in "${PIDS[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
  restore_controllers
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo
echo "[jparse_capture] starting jparse_velocity_controller..."
ros2 run "${PACKAGE}" jparse_velocity_controller --ros-args \
  -r __node:="${ROBOT_NAME}_jparse_velocity_controller_${ARM}" \
  -p robot_name:="${ROBOT_NAME}" \
  -p arm:="${ARM}" \
  -p base_link:="${BASE_LINK}" \
  -p tip_link:="${TIP_LINK}" \
  -p robot_description_topic:="${ROBOT_DESCRIPTION_TOPIC}" \
  -p twist_topic:="${TWIST_TOPIC}" \
  -p command_topic:="${COMMAND_TOPIC_RESOLVED}" \
  -p joint_states_topic:="${JOINT_STATES_TOPIC}" \
  -p singular_values_topic:="${SINGULAR_VALUES_TOPIC}" \
  -p debug_twist_topic:="${DEBUG_TWIST_TOPIC}" \
  -p rate_hz:="${RATE_HZ}" \
  -p command_timeout:="${COMMAND_TIMEOUT}" \
  -p inverse_mode:="${INVERSE_MODE}" \
  -p damping:="${DAMPING}" \
  -p max_joint_velocity:="${MAX_JOINT_VELOCITY}" &
PIDS+=("$!")

sleep 1

echo
echo "[jparse_capture] starting jparse_move_action_server.py..."
ros2 run "${PACKAGE}" jparse_move_action_server.py \
  --robot-name "${ROBOT_NAME}" \
  --arm "${ARM}" \
  --base-link "${BASE_LINK}" \
  --tip-link "${TIP_LINK}" \
  --action-name "${ACTION_NAME}" \
  --twist-topic "${TWIST_TOPIC}" \
  --joint-velocity-topic "${COMMAND_TOPIC_RESOLVED}" \
  --joint-states-topic "${JOINT_STATES_TOPIC}" \
  --max-linear-velocity "${MAX_LINEAR_VELOCITY}" \
  --max-angular-velocity "${MAX_ANGULAR_VELOCITY}" \
  --max-joint-velocity "${MAX_JOINT_VELOCITY}" &
PIDS+=("$!")

sleep "${WAIT_FOR_DISCOVERY_SEC}"

echo
echo "[jparse_capture] relevant topics after start:"
for topic in "${TWIST_TOPIC}" "${COMMAND_TOPIC_RESOLVED}" "${DEBUG_TWIST_TOPIC}" "${SINGULAR_VALUES_TOPIC}"; do
  print_topic_summary "${topic}"
done

echo
echo "[jparse_capture] visible jparse actions:"
timeout 4 ros2 action list 2>/dev/null | grep -E 'jparse|move|mur620' || true
if timeout 4 ros2 action list 2>/dev/null | grep -Fxq "${ACTION_NAME}"; then
  echo "[jparse_capture] action is visible: ${ACTION_NAME}"
else
  echo "[jparse_capture] WARNING: action is still not visible: ${ACTION_NAME}"
fi

echo
echo "[jparse_capture] running. Leave this terminal open while using run_clip_capture.sh."
echo "[jparse_capture] press Ctrl-C here to stop the J-PARSE helper nodes."

set +e
wait -n "${PIDS[@]}"
STATUS=$?
set -e
echo "[jparse_capture] a child process exited with status ${STATUS}"
exit "${STATUS}"
