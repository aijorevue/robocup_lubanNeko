#!/usr/bin/env bash
set -eo pipefail

REQUESTED_MODE="${SERVO_MODE:-rk}"
REQUESTED_PROFILE="${VISION_PROFILE:-chassis}"
case "$REQUESTED_MODE" in
  rk|rk_direct|direct)
    MODE="rk"
    case "$REQUESTED_PROFILE" in
      chassis|linked|h7)
        PROFILE="chassis"
        LAUNCH_FILE="red_square_chassis_rk_direct.launch.py"
        LOG=/home/cat/ros2_ws/chassis_arm_link.log
        ;;
      standalone|desktop|manual)
        PROFILE="standalone"
        LAUNCH_FILE="red_square_grasp_rk_direct.launch.py"
        LOG=/home/cat/ros2_ws/target_vision_desktop.log
        ;;
      *)
        echo "Unknown VISION_PROFILE=$REQUESTED_PROFILE. Use chassis or standalone."
        exit 2
        ;;
    esac
    export HTD85_UART="${HTD85_UART:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82109853-if00}"
    export HTD85_BAUD="${HTD85_BAUD:-115200}"
    export HTD85_ARM_TIME_MS="${HTD85_ARM_TIME_MS:-600}"
    export HTD85_AUX_TIME_MS="${HTD85_AUX_TIME_MS:-200}"
if [ -n "${RED_SQUARE_EXECUTE:-}" ] && [ -z "${ABCD_EXECUTE:-}" ]; then
  ABCD_EXECUTE="$RED_SQUARE_EXECUTE"
fi
: "${ABCD_EXECUTE:=true}"
: "${VISION_TARGET_COLOR:=red}"
: "${VISION_TARGET_KIND:=letter}"
: "${VISION_TARGET_LETTERS:=A,B,C,D}"
    ;;
  *)
    echo "Unknown SERVO_MODE=$REQUESTED_MODE. This project supports rk only."
    exit 2
    ;;
esac

LOCK=/tmp/robocup_target_vision.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  msg="target vision is already running; waiting for the owner; profile=$PROFILE"
  echo "$msg"
  echo "==== $(date '+%F %T') $msg ====" >>"$LOG"
  while ! flock -n 9; do
    sleep 2
  done
fi

exec >>"$LOG" 2>&1

echo "==== $(date '+%F %T') start target vision mode=$MODE profile=$PROFILE launch=$LAUNCH_FILE ===="
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video20}"
export PYTHONUNBUFFERED=1

wait_for_device() {
  local device="$1"
  while [ ! -e "$device" ]; do
    echo "waiting for required device: $device"
    sleep 1
  done
}

if [ "$MODE" = "rk" ] && [ "$PROFILE" = "chassis" ]; then
  wait_for_device "$HTD85_UART"
fi

echo "camera=$CAMERA_DEVICE display=$DISPLAY"
echo "rk servo bus: htd85=$HTD85_UART baud=$HTD85_BAUD arm_time_ms=$HTD85_ARM_TIME_MS aux_time_ms=$HTD85_AUX_TIME_MS"
echo "target color=$VISION_TARGET_COLOR kind=$VISION_TARGET_KIND letters=$VISION_TARGET_LETTERS execute=$ABCD_EXECUTE"

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

pkill -u "$(id -un)" -f "ros2_test1.target_vision|/ros2_test1/target_vision" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "[t]arget_vision.*--enable-letter-grasp" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "/home/cat/bin/[s]ervo_slider_gui" 2>/dev/null || true
pkill -u "$(id -un)" -f "[r]ed_square_grasp_rk_direct.launch.py|[r]ed_square_chassis_rk_direct.launch.py" 2>/dev/null || true
pkill -u "$(id -un)" -f "/rviz2/rviz2.*arm_5.rviz" 2>/dev/null || true
pkill -u "$(id -un)" -f "/robot_state_publisher/[r]obot_state_publisher" 2>/dev/null || true
sleep 1

exec ros2 launch ros2_test1 "$LAUNCH_FILE" execute:="$ABCD_EXECUTE" target_color:="$VISION_TARGET_COLOR" target_kind:="$VISION_TARGET_KIND" target_letters:="$VISION_TARGET_LETTERS"
