#!/usr/bin/env bash
set -eo pipefail

export SERVO_MODE="${SERVO_MODE:-rk}"
export VISION_PROFILE=chassis
exec /home/cat/ros2_ws/start_target_vision.sh "$@"
