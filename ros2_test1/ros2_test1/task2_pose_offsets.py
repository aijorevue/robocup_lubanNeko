"""Runtime-configurable final offsets for formal task-two special bands."""

from __future__ import annotations

import json
import os
from pathlib import Path


CONFIG_ENV = "ROBOCUP_TASK2_POSE_OFFSETS"
DEFAULT_CONFIG_PATH = Path("/home/cat/ros2_ws/config/task2_pose_offsets.json")
OFFSET_MIN_TICKS = -100
OFFSET_MAX_TICKS = 100

BANDS = {
    "near": {
        "label": "7 <= d <= 13 cm",
        "contains": lambda depth: 7.0 <= depth <= 13.0,
    },
    "mid": {
        "label": "15 < d <= 19 cm",
        "contains": lambda depth: 15.0 < depth <= 19.0,
    },
    "far": {
        "label": "20.5 <= d <= 24 cm",
        "contains": lambda depth: 20.5 <= depth <= 24.0,
    },
}
TARGET_KINDS = ("letter", "ring")
JOINTS = ("id1", "id2")


def config_path() -> Path:
    value = os.environ.get(CONFIG_ENV, "").strip()
    return Path(value) if value else DEFAULT_CONFIG_PATH


def default_offsets():
    return {
        band: {
            kind: {joint: 0 for joint in JOINTS}
            for kind in TARGET_KINDS
        }
        for band in BANDS
    }


def validate_offsets(data):
    if not isinstance(data, dict):
        raise ValueError("configuration must be an object")
    result = default_offsets()
    for band in BANDS:
        band_data = data.get(band, {})
        if not isinstance(band_data, dict):
            raise ValueError(f"{band} must be an object")
        for kind in TARGET_KINDS:
            kind_data = band_data.get(kind, {})
            if not isinstance(kind_data, dict):
                raise ValueError(f"{band}.{kind} must be an object")
            for joint in JOINTS:
                value = kind_data.get(joint, 0)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{band}.{kind}.{joint} must be an integer")
                if not OFFSET_MIN_TICKS <= value <= OFFSET_MAX_TICKS:
                    raise ValueError(
                        f"{band}.{kind}.{joint} must be between "
                        f"{OFFSET_MIN_TICKS} and {OFFSET_MAX_TICKS}"
                    )
                result[band][kind][joint] = value
    return result


def load_offsets(path=None, *, strict=False):
    target = Path(path) if path is not None else config_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return validate_offsets(data)
    except FileNotFoundError:
        return default_offsets()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if strict:
            raise ValueError(f"invalid task-two pose offset config {target}: {exc}") from exc
        print(
            f"PLATFORM_PICK TASK2_OFFSET_CONFIG_INVALID path={target} "
            f"error={exc}; using_zero_offsets",
            flush=True,
        )
        return default_offsets()


def save_offsets(offsets, path=None):
    target = Path(path) if path is not None else config_path()
    validated = validate_offsets(offsets)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    payload = json.dumps(validated, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def band_for_depth(depth_cm):
    depth = float(depth_cm)
    for name, definition in BANDS.items():
        if definition["contains"](depth):
            return name
    return None


def apply_final_offsets(depth_cm, target_kind, id1, id2, offsets):
    band = band_for_depth(depth_cm)
    kind = str(target_kind).strip().lower()
    if band is None or kind not in TARGET_KINDS:
        return int(id1), int(id2), None, 0, 0
    validated = validate_offsets(offsets)
    extra_id1 = validated[band][kind]["id1"]
    extra_id2 = validated[band][kind]["id2"]
    return (
        int(id1) + extra_id1,
        int(id2) + extra_id2,
        band,
        extra_id1,
        extra_id2,
    )
