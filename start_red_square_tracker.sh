#!/usr/bin/env bash

LOCK=/tmp/red_square_tracker_desktop.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

LOG=/home/cat/ros2_ws/red_square_tracker.log
exec >>"$LOG" 2>&1

echo "==== $(date '+%F %T') start red square tracker ===="
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export CAMERA_DEVICE="${CAMERA_DEVICE:-auto}"
export PYTHONUNBUFFERED=1

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

pkill -INT -u "$(id -un)" -f "/ros2_test1/[t]arget_vision|python3 -m ros2_test1.[t]arget_vision" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "/ros2_test1/[r]ed_square_tracker|python3 -m ros2_test1.[r]ed_square_tracker" 2>/dev/null || true
pkill -INT -u "$(id -un)" -f "/home/cat/bin/[s]ervo_slider_gui" 2>/dev/null || true
pkill -u "$(id -un)" -f "[r]ed_square_grasp_all.launch.py" 2>/dev/null || true
sleep 1

flock -u 9
exec 9>&-
exec ros2 run ros2_test1 red_square_tracker --execute
