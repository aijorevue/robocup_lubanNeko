#!/usr/bin/env bash
set -eo pipefail

LOG=/home/cat/ros2_ws/chassis_arm_link_autostart.log
exec >>"$LOG" 2>&1

echo "==== $(date '+%F %T') autostart chassis arm link ===="

export RED_SQUARE_EXECUTE="${RED_SQUARE_EXECUTE:-true}"
export VISION_TARGET_COLOR="${VISION_TARGET_COLOR:-red}"
export VISION_TARGET_KIND="${VISION_TARGET_KIND:-any}"
export CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video20}"

BOOT_DELAY_S="${CHASSIS_ARM_BOOT_DELAY_S:-2}"
echo "boot delay ${BOOT_DELAY_S}s; camera=$CAMERA_DEVICE"
sleep "$BOOT_DELAY_S"

for _ in $(seq 1 60); do
  if [ -x /home/cat/ros2_ws/start_chassis_arm_link.sh ]; then
    break
  fi
  echo "waiting for /home/cat/ros2_ws/start_chassis_arm_link.sh"
  sleep 1
done

for _ in $(seq 1 30); do
  if [ -e "$CAMERA_DEVICE" ]; then
    break
  fi
  echo "waiting for camera $CAMERA_DEVICE"
  sleep 1
done

exec /home/cat/ros2_ws/start_chassis_arm_link.sh
