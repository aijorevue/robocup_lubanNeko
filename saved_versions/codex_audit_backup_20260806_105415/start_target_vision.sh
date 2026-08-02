#!/usr/bin/env bash
set -eo pipefail

LOG=/home/cat/ros2_ws/target_vision_desktop.log
LOCK=/tmp/target_vision_desktop.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  msg="target vision is already running; log: $LOG"
  echo "$msg"
  echo "==== $(date '+%F %T') $msg ====" >>"$LOG"
  exit 0
fi

exec >>"$LOG" 2>&1

REQUESTED_MODE="${SERVO_MODE:-rk}"
case "$REQUESTED_MODE" in
  rk|rk_direct|direct)
    MODE="rk"
    LAUNCH_FILE="red_square_grasp_rk_direct.launch.py"
    export DIRECT_SERVO_BUS=1
    export DIRECT_ZP_UART="${DIRECT_ZP_UART:-/dev/ttyS0}"
    export DIRECT_ARM_UART="${DIRECT_ARM_UART:-/dev/ttyS9}"
    export DIRECT_ZP_TIME_MS="${DIRECT_ZP_TIME_MS:-450}"
    export DIRECT_ARM_TIME_MS="${DIRECT_ARM_TIME_MS:-700}"
    ;;
  rc|rct6)
    MODE="rc"
    LAUNCH_FILE="red_square_grasp_rc.launch.py"
    unset DIRECT_SERVO_BUS
    unset DIRECT_ZP_UART
    unset DIRECT_ARM_UART
    unset DIRECT_ZP_TIME_MS
    unset DIRECT_ARM_TIME_MS
    ;;
  *)
    echo "Unknown SERVO_MODE=$REQUESTED_MODE. Use rk or rc."
    exit 2
    ;;
esac

echo "==== $(date '+%F %T') start target vision mode=$MODE launch=$LAUNCH_FILE ===="
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video20}"
export PYTHONUNBUFFERED=1

echo "camera=$CAMERA_DEVICE display=$DISPLAY"
if [ "$MODE" = "rk" ]; then
  echo "rk direct ports: arm85=$DIRECT_ARM_UART zp=$DIRECT_ZP_UART arm_time_ms=$DIRECT_ARM_TIME_MS zp_time_ms=$DIRECT_ZP_TIME_MS"
else
  echo "rc mode: RCT6 bridge on SERVO_UART=${SERVO_UART:-/dev/ttyS0}"
fi
echo "target color=${VISION_TARGET_COLOR:-red} kind=${VISION_TARGET_KIND:-any} execute=${RED_SQUARE_EXECUTE:-true}"

XAUTH_FILE="$(find "$XDG_RUNTIME_DIR" -maxdepth 1 -name '.mutter-Xwaylandauth.*' -print -quit 2>/dev/null)"
if [ -n "$XAUTH_FILE" ]; then
  export XAUTHORITY="$XAUTH_FILE"
fi

source /opt/ros/humble/setup.bash
source /home/cat/ros2_ws/install/local_setup.bash
export PYTHONPATH="/home/cat/ros2_ws/build/ros2_test1:/home/cat/ros2_ws/install/ros2_test1/lib/python3.10/site-packages:${PYTHONPATH:-}"
export AMENT_PREFIX_PATH="/home/cat/ros2_ws/install/ros2_test1:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="/home/cat/ros2_ws/install/ros2_test1:${CMAKE_PREFIX_PATH:-}"
cd /home/cat/ros2_ws

pkill -u "$(id -un)" -f "ros2_test1.target_vision|ros2_test1.target_vision_rc|/ros2_test1/target_vision|/ros2_test1/target_vision_rc" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "[t]arget_vision.*--enable-red-square-grasp" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "[t]arget_vision_rc.*--enable-red-square-grasp" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "/home/cat/bin/[s]ervo_slider_gui" 2>/dev/null || true
pkill -u "$(id -un)" -f "[r]ed_square_grasp_rc.launch.py|[r]ed_square_grasp_rk_direct.launch.py|[r]ed_square_chassis_rk_direct.launch.py" 2>/dev/null || true
pkill -u "$(id -un)" -f "/rviz2/rviz2.*arm_5.rviz" 2>/dev/null || true
pkill -u "$(id -un)" -f "/robot_state_publisher/[r]obot_state_publisher" 2>/dev/null || true
sleep 1

exec ros2 launch ros2_test1 "$LAUNCH_FILE" \
  execute:="${RED_SQUARE_EXECUTE:-true}" \
  target_color:="${VISION_TARGET_COLOR:-red}" \
  target_kind:="${VISION_TARGET_KIND:-any}"
