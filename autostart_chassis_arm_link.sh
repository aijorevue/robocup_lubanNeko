#!/usr/bin/env bash
set -eo pipefail

LOG=/home/cat/ros2_ws/chassis_arm_link_autostart.log
exec >>"$LOG" 2>&1

echo "==== $(date '+%F %T') autostart chassis arm link ===="

export RED_SQUARE_EXECUTE="${RED_SQUARE_EXECUTE:-true}"
export VISION_TARGET_COLOR="${VISION_TARGET_COLOR:-red}"
export VISION_TARGET_KIND="${VISION_TARGET_KIND:-any}"
# Main camera: the arm-mounted USB camera. Keep this as a stable V4L2 by-path
# path so adding the future secondary camera cannot renumber the main camera.
export CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/v4l/by-path/platform-fc800000.usb-usb-0:1:1.0-video-index0}"
# The secondary camera is used only for task-two letter preselection.
export SECONDARY_CAMERA_DEVICE="${SECONDARY_CAMERA_DEVICE:-/dev/v4l/by-path/platform-fc880000.usb-usb-0:1.3:1.0-video-index0}"

BOOT_DELAY_S="${CHASSIS_ARM_BOOT_DELAY_S:-0}"
echo "boot delay ${BOOT_DELAY_S}s; camera=$CAMERA_DEVICE"
if [ "$BOOT_DELAY_S" -gt 0 ] 2>/dev/null; then
  sleep "$BOOT_DELAY_S"
fi

for _ in $(seq 1 60); do
  if [ -x /home/cat/ros2_ws/start_chassis_arm_link.sh ]; then
    break
  fi
  echo "waiting for /home/cat/ros2_ws/start_chassis_arm_link.sh"
  sleep 1
done

echo "camera startup is managed by target_vision hot reconnect: $CAMERA_DEVICE"

exec /home/cat/ros2_ws/start_chassis_arm_link.sh
