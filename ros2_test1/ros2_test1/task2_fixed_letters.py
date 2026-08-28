"""Persistent task-two letter fallback for the formal RK application."""

import json
import os
import re
import tempfile


DEFAULT_PATH = "/home/cat/.config/robocup/task2_fixed_letters.json"
VALID_LETTERS = frozenset("ABCD")


def _normalise(values):
    if isinstance(values, str):
        values = re.findall(r"[ABCD]", values.upper())
    try:
        letters = [str(value).upper() for value in values]
    except (TypeError, ValueError):
        return ()
    result = []
    for letter in letters:
        if letter in VALID_LETTERS and letter not in result:
            result.append(letter)
    return tuple(result[:2]) if len(result) == 2 else ()


def load_fixed_letters(path=None):
    """Return two configured distinct letters, or an empty tuple."""
    path = path or os.environ.get("TASK2_FIXED_LETTERS_PATH", DEFAULT_PATH)
    try:
        with open(path, "r", encoding="ascii") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return ()
    if isinstance(payload, dict):
        payload = payload.get("letters")
    return _normalise(payload)


def save_fixed_letters(letters, path=None):
    """Atomically persist two distinct A/B/C/D letters."""
    selected = _normalise(letters)
    if len(selected) != 2:
        raise ValueError("enter two different letters from A, B, C, D")
    path = path or os.environ.get("TASK2_FIXED_LETTERS_PATH", DEFAULT_PATH)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    payload = {"letters": list(selected), "version": 1}
    fd, temporary = tempfile.mkstemp(prefix=".task2_fixed_letters.", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return selected


def main(argv=None):
    import argparse
    import getpass

    parser = argparse.ArgumentParser(description="Set task-two persistent letter fallback")
    parser.add_argument("letters", nargs="?", help="two different letters, for example CD")
    parser.add_argument("--show", action="store_true", help="show the saved pair")
    parser.add_argument("--clear", action="store_true", help="remove the saved pair")
    parser.add_argument("--path", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = args.path or os.environ.get("TASK2_FIXED_LETTERS_PATH", DEFAULT_PATH)

    if args.clear:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        print("TASK2_FIXED_LETTERS cleared", flush=True)
        return 0
    if args.show:
        selected = load_fixed_letters(path)
        print("TASK2_FIXED_LETTERS " + (" ".join(selected) if selected else "not set"), flush=True)
        return 0

    value = args.letters
    if not value:
        current = load_fixed_letters(path)
        if current:
            print("当前已保存: " + " ".join(current))
        value = input("请输入两个不同字母（A/B/C/D，例如 CD）: ")
    selected = save_fixed_letters(value, path)
    print("TASK2_FIXED_LETTERS saved: " + " ".join(selected), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
