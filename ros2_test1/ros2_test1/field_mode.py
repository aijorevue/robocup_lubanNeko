"""Field selection and target policy shared by the vision and link layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FieldMode(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def wire_name(self):
        return self.value.upper()


@dataclass(frozen=True)
class FieldTargetPolicy:
    """Allowed target colors for each RK station."""

    field: FieldMode

    @property
    def disc_colors(self):
        # Yellow is the neutral disc target on both mirrored fields.
        return frozenset((self.field.value, "yellow"))

    @property
    def platform_ring_color(self):
        return self.field.value

    @property
    def platform_color(self):
        return self.platform_ring_color

    def matches_disc_ball(self, detection):
        return (
            detection.get("kind") == "ball"
            and detection.get("color") in self.disc_colors
        )

    def matches_platform_target(self, detection, target_kind, target_letters):
        kind = detection.get("kind")
        if target_kind not in {"any", kind}:
            return False
        if kind == "letter":
            return detection.get("letter") in target_letters
        if kind == "ring":
            return detection.get("color") == self.platform_ring_color
        return False

    @staticmethod
    def matches_column_letter(detection, target_letters):
        return (
            detection.get("kind") == "letter"
            and detection.get("letter") in target_letters
        )


def parse_field(value, default=FieldMode.RED):
    """Parse a wire/CLI value without allowing an unknown field to leak in."""

    if isinstance(value, FieldMode):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"red", "r"}:
        return FieldMode.RED
    if normalized in {"blue", "b"}:
        return FieldMode.BLUE
    return default


def policy_for(field):
    return FieldTargetPolicy(parse_field(field))
