#!/usr/bin/env python3
"""Rotate the two 180-degree PWM servos (SG90/MG90S) on the 24-channel
Hiwonder-style ZP controller, channels 12 and 23.

Same ASCII protocol and serial settings as the vision app's
DirectBusServoBridge: 115200 8N1 raw, packets "#012P1500T1000!".

Examples:
  python3 test_zp_aux_servos.py                     # gentle +-45 deg sweep, both servos
  python3 test_zp_aux_servos.py --min 500 --max 2500 --cycles 1   # full 0..180 sweep
  python3 test_zp_aux_servos.py --pulse 1500        # single move to 90 deg and hold
  python3 test_zp_aux_servos.py --channels 12 --pulse 1300
"""

import argparse
import glob
import os
import subprocess
import sys
import time

DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
FALLBACK_PORTS = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM1"]
SERVO_BAUD = 115200
PULSE_MIN = 500
PULSE_MAX = 2500


def find_port(requested):
    if requested:
        return requested
    for candidate in [DEFAULT_PORT] + FALLBACK_PORTS:
        if os.path.exists(candidate):
            return candidate
    glob_hits = sorted(glob.glob("/dev/ttyUSB*"))
    if glob_hits:
        return glob_hits[0]
    return None


def port_holders(device):
    """PIDs holding the device open, so the user knows the vision app is bound."""
    holders = []
    real = os.path.realpath(device)
    for pid_dir in glob.glob("/proc/[0-9]*/fd"):
        try:
            pid = pid_dir.split("/")[2]
            with open(os.path.join(pid_dir, "..", "comm")) as fh:
                comm = fh.read().strip()
            for fd in os.listdir(pid_dir):
                try:
                    link = os.readlink(os.path.join(pid_dir, fd))
                except OSError:
                    continue
                if link == real:
                    holders.append(f"{comm}({pid})")
                    break
        except (OSError, IndexError):
            continue
    return sorted(set(holders))


def open_port(device):
    subprocess.run(
        ["stty", "-F", device, str(SERVO_BAUD), "cs8", "-cstopb", "-parenb",
         "-ixon", "-ixoff", "-crtscts", "raw", "-echo"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)


def move(fd, channel, pulse, time_ms):
    pulse = max(PULSE_MIN, min(PULSE_MAX, int(pulse)))
    time_ms = max(0, min(9999, int(time_ms)))
    packet = f"#{channel:03d}P{pulse:04d}T{time_ms:04d}!".encode("ascii")
    os.write(fd, packet)
    print(f"TX {packet.decode('ascii')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=None, help=f"serial device (default {DEFAULT_PORT})")
    parser.add_argument("--channels", default="12,23", help="channels to drive (default 12,23)")
    parser.add_argument("--pulse", type=int, default=None,
                        help="single fixed move to this pulse and exit")
    parser.add_argument("--min", type=int, default=1000, help="sweep low pulse (default 1000)")
    parser.add_argument("--max", type=int, default=2000, help="sweep high pulse (default 2000)")
    parser.add_argument("--time", type=int, default=800, help="move time ms (default 800)")
    parser.add_argument("--cycles", type=int, default=2, help="sweep cycles (default 2)")
    parser.add_argument("--settle", type=float, default=1.2,
                        help="seconds between steps (default 1.2)")
    args = parser.parse_args()

    device = find_port(args.port)
    if not device:
        sys.exit("no ZP serial port found; pass --port /dev/ttyUSBx")
    holders = port_holders(device)
    if holders:
        print(f"NOTE: {device} also held by: {', '.join(holders)}"
              " (vision app is idle, short writes may interleave)", flush=True)
    channels = [int(c) for c in args.channels.split(",") if c.strip()]
    for channel in channels:
        if not 0 <= channel <= 31:
            sys.exit(f"channel {channel} out of range 0..31")

    fd = open_port(device)
    print(f"ZP port {device} @ {SERVO_BAUD} 8N1; channels={channels}", flush=True)
    try:
        if args.pulse is not None:
            for channel in channels:
                move(fd, channel, args.pulse, args.time)
            time.sleep(args.settle)
            return
        lo, hi = max(PULSE_MIN, args.min), min(PULSE_MAX, args.max)
        if lo > hi:
            sys.exit("--min must be <= --max")
        for cycle in range(args.cycles):
            for pulse in (hi, lo):
                for channel in channels:
                    move(fd, channel, pulse, args.time)
                time.sleep(max(args.settle, args.time / 1000.0 + 0.4))
        for channel in channels:
            move(fd, channel, 1500, args.time)  # back to 90 deg
        time.sleep(args.settle)
        print("done", flush=True)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
