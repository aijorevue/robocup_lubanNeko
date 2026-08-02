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
    def platform_color(self):
        return self.field.value

    @property
    def column_color(self):
        return self.field.value

    def matches_disc_ball(self, detection):
        return (
            detection.get("kind") == "ball"
            and detection.get("color") in self.disc_colors
        )

    def matches_column_ball(self, detection):
        return (
            detection.get("kind") == "ball"
            and detection.get("color") == self.column_color
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
