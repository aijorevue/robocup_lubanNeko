#!/usr/bin/env python3
"""Interactive editor for formal task-two special-band final offsets."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys

SOURCE_PARENT = Path("/home/cat/ros2_ws/ros2_test1")
if SOURCE_PARENT.is_dir():
    sys.path.insert(0, str(SOURCE_PARENT))

try:
    from ros2_test1.task2_pose_offsets import (
        BANDS,
        JOINTS,
        OFFSET_MAX_TICKS,
        OFFSET_MIN_TICKS,
        TARGET_KINDS,
        config_path,
        default_offsets,
        load_offsets,
        save_offsets,
    )
except ModuleNotFoundError:
    from task2_pose_offsets import (
        BANDS,
        JOINTS,
        OFFSET_MAX_TICKS,
        OFFSET_MIN_TICKS,
        TARGET_KINDS,
        config_path,
        default_offsets,
        load_offsets,
        save_offsets,
    )


BAND_ALIASES = {"1": "near", "2": "mid", "3": "far", **{key: key for key in BANDS}}
KIND_ALIASES = {"1": "letter", "2": "ring", **{key: key for key in TARGET_KINDS}}
JOINT_ALIASES = {"1": "id1", "2": "id2", **{key: key for key in JOINTS}}


def signed(value):
    return f"{value:+d}"


def service_active():
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "robocup-chassis-arm.service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "active"


def show(offsets):
    print("\n任务二三段特殊距离最终附加刻度")
    print(f"配置文件: {config_path()}")
    print("正数=增加刻度，负数=减少刻度；只影响任务二最终抓取解算。")
    for index, (band, definition) in enumerate(BANDS.items(), 1):
        print(f"\n{index}. {definition['label']} ({band})")
        print(
            "   字母块: "
            f"ID1 {signed(offsets[band]['letter']['id1'])}, "
            f"ID2 {signed(offsets[band]['letter']['id2'])}"
        )
        print(
            "   圆环:   "
            f"ID1 {signed(offsets[band]['ring']['id1'])}, "
            f"ID2 {signed(offsets[band]['ring']['id2'])}"
        )
    if service_active():
        print("\n注意: 正式服务正在运行，本次保存将在下次服务启动时生效。")
    else:
        print("\n正式服务未运行；配置将在下次任务启动时生效。")


def resolve(value, aliases, label):
    key = aliases.get(str(value).strip().lower())
    if key is None:
        raise ValueError(f"无效{label}: {value}")
    return key


def parse_ticks(value):
    try:
        ticks = int(str(value).strip())
    except ValueError as exc:
        raise ValueError("刻度必须是整数，例如 5 或 -5") from exc
    if not OFFSET_MIN_TICKS <= ticks <= OFFSET_MAX_TICKS:
        raise ValueError(f"刻度范围必须为 {OFFSET_MIN_TICKS}..{OFFSET_MAX_TICKS}")
    return ticks


def set_value(offsets, band_value, kind_value, joint_value, ticks_value):
    band = resolve(band_value, BAND_ALIASES, "距离段")
    kind = resolve(kind_value, KIND_ALIASES, "目标类型")
    joint = resolve(joint_value, JOINT_ALIASES, "舵机")
    ticks = parse_ticks(ticks_value)
    updated = copy.deepcopy(offsets)
    old = updated[band][kind][joint]
    updated[band][kind][joint] = ticks
    path = save_offsets(updated)
    print(
        f"已保存: {BANDS[band]['label']} {kind} {joint.upper()} "
        f"{signed(old)} -> {signed(ticks)}"
    )
    print(f"文件: {path}")
    print("下次任务进程启动时生效。")
    return updated


def choose(prompt, aliases):
    value = input(prompt).strip().lower()
    if value == "0":
        return None
    return resolve(value, aliases, prompt.rstrip())


def interactive():
    offsets = load_offsets(strict=True)
    while True:
        show(offsets)
        print("\n操作: 1 修改一项  2 清零一个距离段  3 全部清零  0 退出")
        action = input("请选择: ").strip()
        try:
            if action == "0":
                return 0
            if action == "1":
                band = choose("距离段 1近/2中/3远/0取消: ", BAND_ALIASES)
                if band is None:
                    continue
                kind = choose("目标 1字母块/2圆环/0取消: ", KIND_ALIASES)
                if kind is None:
                    continue
                joint = choose("舵机 1=ID1/2=ID2/0取消: ", JOINT_ALIASES)
                if joint is None:
                    continue
                current = offsets[band][kind][joint]
                ticks = input(
                    f"当前 {signed(current)}，输入新的附加刻度 "
                    f"({OFFSET_MIN_TICKS}..{OFFSET_MAX_TICKS}): "
                )
                offsets = set_value(offsets, band, kind, joint, ticks)
            elif action == "2":
                band = choose("清零距离段 1近/2中/3远/0取消: ", BAND_ALIASES)
                if band is None:
                    continue
                updated = copy.deepcopy(offsets)
                updated[band] = default_offsets()[band]
                save_offsets(updated)
                offsets = updated
                print(f"已清零: {BANDS[band]['label']}")
            elif action == "3":
                confirm = input("输入 YES 确认全部清零: ").strip()
                if confirm == "YES":
                    offsets = default_offsets()
                    save_offsets(offsets)
                    print("已全部清零。")
            else:
                print("无效选项。")
        except ValueError as exc:
            print(f"错误: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="设置任务二三段特殊距离的最终 ID1/ID2 附加刻度"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("show", help="显示当前配置")
    set_parser = subparsers.add_parser("set", help="直接设置一项")
    set_parser.add_argument("band", help="1/2/3 或 near/mid/far")
    set_parser.add_argument("kind", help="letter/ring")
    set_parser.add_argument("joint", help="id1/id2")
    set_parser.add_argument("ticks", help="-100..100 的整数")
    reset_parser = subparsers.add_parser("reset", help="清零一个距离段或全部")
    reset_parser.add_argument("band", nargs="?", default="all")
    args = parser.parse_args()

    try:
        offsets = load_offsets(strict=True)
        if args.command is None:
            return interactive()
        if args.command == "show":
            show(offsets)
            return 0
        if args.command == "set":
            set_value(offsets, args.band, args.kind, args.joint, args.ticks)
            return 0
        if args.command == "reset":
            if args.band == "all":
                save_offsets(default_offsets())
                print("已全部清零。")
            else:
                band = resolve(args.band, BAND_ALIASES, "距离段")
                offsets[band] = default_offsets()[band]
                save_offsets(offsets)
                print(f"已清零: {BANDS[band]['label']}")
            return 0
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
