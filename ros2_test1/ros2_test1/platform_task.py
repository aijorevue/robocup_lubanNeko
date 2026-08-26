"""Formal task-two sequencing; hardware writes are injected for replay tests.

PRESELECT_DONE is a timed high-pose barrier. Slot DONE is emitted only after
the final high pose has settled. Depth uses the standalone measured mapping.
"""
from collections import Counter, deque
import math
import time

from .grasp_calibration import calibrated_grasp_ticks

HIGH = (650, 600, 350)
LETTER_PLACE = (500, 350, 600)
RING_PLACE = (520, 340, 120)
HTD85_AUX_HIGH = (300, 600, 450)  # physical ID14, ID15, ID17; ID14 retracted
PLATFORM_GRIPPER_CLOSED = 430
PLATFORM_GRIPPER_OPEN = 600
PLATFORM_ARM_TIME_MS = 600
PLATFORM_AUX_TIME_MS = 200
PLATFORM_GRIPPER_TIME_MS = 200
PLATFORM_RING_PLACE_TIME_MS = 700
PLATFORM_RING_AXIS_TIME_MS = 500
PLATFORM_LETTER_PLACE_TIME_MS = 500
PLATFORM_CENTER_TIME_MS = 100
POST_OPEN_ID2_RETREAT_TICKS_BY_KIND = {
    "letter": 30,
    "ring": 50,
}
NEAR_GRASP_MIN_DISTANCE_CM = 7.0
NEAR_GRASP_MAX_DISTANCE_CM = 10.0
NEAR_RING_ID2_RETREAT_EXTRA_TICKS = 30
NEAR_LETTER_DESCENT_ID2_OFFSET_TICKS = 20
NEAR_LETTER_DESCENT_ID2_MIN_TICKS = 0
PLATFORM_GRASP_ID1_OFFSET_TICKS = 40
CENTER_DEADBAND_PX = 45
CENTER_ID6_STEP_TICKS = 5
CENTER_ID2_STEP_TICKS = 7
CENTER_ID2_RANGE = (450, 700)
CENTER_ID6_RANGE = (0, 700)
# Only the target-observation phase is bounded.  Once a valid selected
# letter/own-field ring has been seen, the grasp transaction is allowed to
# finish without the no-target watchdog interrupting it.
PLATFORM_NO_TARGET_TIMEOUT_S = 3.5
TARGET_WINDOW_SIZE_PX = 400
TARGET_WINDOW_MIN_AREA_FRACTION = 0.80


def _target_in_center_window(target, frame_shape):
    """Require at least 80% of a target bbox inside the centered 400x400 window."""
    bbox = target.get("bbox")
    if bbox is None or len(bbox) != 4:
        return False
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    frame_height, frame_width = frame_shape[:2]
    window_size = min(
        float(TARGET_WINDOW_SIZE_PX), float(frame_width), float(frame_height)
    )
    window_left = (float(frame_width) - window_size) / 2.0
    window_top = (float(frame_height) - window_size) / 2.0
    intersection_width = max(
        0.0,
        min(x + width, window_left + window_size) - max(x, window_left),
    )
    intersection_height = max(
        0.0,
        min(y + height, window_top + window_size) - max(y, window_top),
    )
    inside_fraction = (intersection_width * intersection_height) / (width * height)
    return inside_fraction >= TARGET_WINDOW_MIN_AREA_FRACTION


class PlatformTask:
    def __init__(self, pose, gripper, center, *, ring_place_id6=None,
                 ring_place_id2=None, ring_place_id1=None,
                 ring_return_high_id1=None, ring_return_high_id2=None,
                 ring_return_high_id6=None, letter_place_id6=None,
                 letter_place_id2=None, letter_place_id1=None,
                 letter_return_high_id1=None, letter_return_high_id2=None,
                 letter_return_high_id6=None, clock=time.monotonic):
        self.pose = pose
        self.gripper = gripper
        self.center = center
        self.ring_place_id6 = ring_place_id6
        self.ring_place_id2 = ring_place_id2
        self.ring_place_id1 = ring_place_id1
        self.ring_return_high_id1 = ring_return_high_id1
        self.ring_return_high_id2 = ring_return_high_id2
        self.ring_return_high_id6 = ring_return_high_id6
        self.letter_place_id6 = letter_place_id6
        self.letter_place_id2 = letter_place_id2
        self.letter_place_id1 = letter_place_id1
        self.letter_return_high_id1 = letter_return_high_id1
        self.letter_return_high_id2 = letter_return_high_id2
        self.letter_return_high_id6 = letter_return_high_id6
        self.clock = clock
        self.reset()

    def reset(self):
        self.stage = None
        self.selected = ()
        self.done = self.error = None
        self.deadline = 0.0
        self.actions = deque()
        self.high_ready = False
        self.target_seen = False
        self.status = "PLATFORM_PICK idle"

    def _fail(self, reason):
        self.error = reason
        self.done = None
        self.stage = "platform_fault"
        self.high_ready = False
        self.actions.clear()
        self.status = f"PLATFORM_PICK ERR {reason}"

    def _command(self, label, callback, *args):
        try:
            settle = callback(*args)
            if settle is None or settle is False:
                raise RuntimeError("write failed")
        except Exception as exc:
            self._fail(f"SERVO_WRITE_FAILED_{label}")
            print(f"{self.status}: {exc}", flush=True)
            return False
        self.deadline = self.clock() + float(settle)
        self.status = f"PLATFORM_PICK {label}"
        print(self.status, flush=True)
        return True

    def begin_preselect(self):
        self.reset()
        self.stage = "platform_preselect"
        self.pairs = deque(maxlen=12)
        self.warmup = 20
        self.timeout = self.clock() + 25.0
        self.status = "PLATFORM_PICK secondary: waiting two distinct letters"

    def preselect(self, detections, fresh=True):
        if self.stage != "platform_preselect" or not fresh:
            return
        if self.warmup:
            self.warmup -= 1
            return
        candidates = sorted(
            (d for d in detections if d.get("kind") == "letter"
             and d.get("letter") in {"A", "B", "C", "D"}
             and float(d.get("confidence") or 0) >= 38),
            key=lambda d: d["center"][0],
        )
        pair = ((candidates[0]["letter"], candidates[-1]["letter"])
                if len(candidates) >= 2 else None)
        self.pairs.append(pair if pair and pair[0] != pair[1] else None)
        if len(self.pairs) < 8:
            return
        # Vote complete co-visible pairs: never combine letters seen alone.
        counts = Counter(p for p in self.pairs if p is not None)
        if not counts:
            return
        selected, votes = counts.most_common(1)[0]
        if votes < 6:
            return
        self.selected = selected
        print(f"PLATFORM_PICK LETTERS_LOCKED L1={selected[0]} L2={selected[1]}", flush=True)
        self.stage = "platform_raise"
        self._command("ARM_HIGH", self.pose, HIGH, True)

    def begin_slot(self, field):
        if not self.high_ready or len(self.selected) != 2:
            self._fail("PRESELECT_HIGH_NOT_READY")
            return
        self.done = self.error = None
        self.field = str(field).lower()
        self.stage = "platform_detect"
        self.timeout = self.clock() + PLATFORM_NO_TARGET_TIMEOUT_S
        self.deadline = 0.0
        self.discard = 8  # Flush approach/motion frames as in the standalone App.
        self.votes = deque(maxlen=8)
        self.target_key = None
        self.target_center = None
        self.target_seen = False
        self.center_id1 = HIGH[0]
        self.center_id2, self.center_id6 = HIGH[1:]
        self.status = "PLATFORM_PICK main camera: waiting target"

    @staticmethod
    def _key(target):
        return (target.get("kind"), target.get("letter") if target.get("kind") == "letter"
                else target.get("color"))

    def _allowed(self, target):
        kind, label = self._key(target)
        # The secondary camera is authoritative for platform letters.  An
        # empty selection must reject every letter, never use a launch-time
        # A/B/C/D fallback.
        return ((kind == "letter" and bool(self.selected) and label in self.selected)
                or (kind == "ring" and label == self.field))

    def skip(self, reason):
        self.high_ready = False
        self.finish_reason = f"SKIPPED:{reason}"
        self.actions = deque([("RETURN_HIGH", self.pose, (HIGH, True))])
        self.stage = "platform_actions"
        self.deadline = 0.0

    def _detect(self, detections, shape):
        if self.discard:
            self.discard -= 1
            return
        if shape is None or len(shape) < 2:
            return
        height, width = shape[:2]
        candidates = [d for d in detections
                      if d.get("kind") in {"letter", "ring"}
                      and d.get("observed", True)
                      and (float(d.get("confidence") or 0) >= 45 if d.get("kind") == "letter"
                           else float(d.get("score") or 0) >= 0.55)
                      and d.get("center") is not None
                      and _target_in_center_window(d, shape)]
        # Ignore opponent rings and letters that were not selected by the
        # secondary camera.  Waiting for a valid candidate lets the station
        # timeout path advance safely without ever grabbing the wrong letter.
        candidates = [d for d in candidates if self._allowed(d)]
        if self.target_key is not None:
            candidates = [d for d in candidates if self._key(d) == self.target_key
                          and math.dist(d["center"], self.target_center) <= 320]
        target = min(candidates, key=lambda d: math.dist(
            d["center"], self.target_center or (width / 2, height / 2)), default=None)
        if target is None:
            self.votes.append(None)
            return
        key = self._key(target)
        point = tuple(target["center"])
        if not self.target_seen:
            self.target_seen = True
            self.status = (
                "PLATFORM_PICK target observed; "
                "grasp transaction has no deadline"
            )
        self.votes.append((key, point))
        if sum(v is not None and v[0] == key and math.dist(v[1], point) <= 140
               for v in self.votes) < 3:
            return
        if not self._allowed(target):
            self.skip("TARGET_NOT_SELECTED")
            return
        self.target_key, self.target_center = key, point
        dx, dy = point[0] - width / 2, point[1] - height / 2
        if abs(dx) > CENTER_DEADBAND_PX or abs(dy) > CENTER_DEADBAND_PX:
            id2 = max(CENTER_ID2_RANGE[0], min(CENTER_ID2_RANGE[1], self.center_id2 + (
                -CENTER_ID2_STEP_TICKS if dy > CENTER_DEADBAND_PX
                else CENTER_ID2_STEP_TICKS if dy < -CENTER_DEADBAND_PX else 0
            )))
            id6 = max(
                CENTER_ID6_RANGE[0],
                min(CENTER_ID6_RANGE[1], self.center_id6 + (
                    -CENTER_ID6_STEP_TICKS if dx > CENTER_DEADBAND_PX
                    else CENTER_ID6_STEP_TICKS if dx < -CENTER_DEADBAND_PX else 0
                )),
            )
            if (id2, id6) == (self.center_id2, self.center_id6):
                self.skip("CENTER_LIMIT")
                return
            self.high_ready = False
            self.center_id2, self.center_id6 = id2, id6
            self._command("CENTER_STEP", self.center, id2, id6)
            self.votes.clear()
            self.discard = 1
            return
        try:
            depth = float(target.get("distance_cm"))
            if not math.isfinite(depth):
                raise ValueError("non-finite depth")
            id1, id2 = calibrated_grasp_ticks(
                depth, id1_offset_ticks=PLATFORM_GRASP_ID1_OFFSET_TICKS,
            )
        except (ValueError, TypeError):
            self.status = "PLATFORM_PICK waiting valid measured depth 7..30cm"
            return
        near_grasp = NEAR_GRASP_MIN_DISTANCE_CM <= depth <= NEAR_GRASP_MAX_DISTANCE_CM
        if key[0] == "letter" and near_grasp:
            id2 = max(
                NEAR_LETTER_DESCENT_ID2_MIN_TICKS,
                id2 - NEAR_LETTER_DESCENT_ID2_OFFSET_TICKS,
            )
        self.high_ready = False
        placement = LETTER_PLACE if key[0] == "letter" else RING_PLACE
        print(f"PLATFORM_PICK TARGET kind={key[0]} label={key[1]} depth_cm={depth:.2f} "
              f"down={id1}/{id2}/{self.center_id6} model=measured_7_30cm", flush=True)
        self.finish_reason = f"PICKED_{key[0].upper()}"
        retreat_ticks = POST_OPEN_ID2_RETREAT_TICKS_BY_KIND[key[0]]
        if key[0] == "ring" and near_grasp:
            retreat_ticks += NEAR_RING_ID2_RETREAT_EXTRA_TICKS
        retreat_id2 = max(
            CENTER_ID2_RANGE[0],
            self.center_id2 - retreat_ticks,
        )
        actions = [
            ("GRIPPER_OPEN", self.gripper, (PLATFORM_GRIPPER_OPEN,)),
            ("POST_OPEN_ID2_RETREAT", self.pose,
             ((self.center_id1, retreat_id2, self.center_id6), False)),
            ("DESCEND", self.pose, ((id1, id2, self.center_id6), False)),
            ("GRIPPER_CLOSE", self.gripper, (PLATFORM_GRIPPER_CLOSED,)),
            ("LIFT_HIGH", self.pose, (HIGH, True)),
        ]
        if key[0] == "ring":
            # Ring placement is deliberately sequenced ID6 -> ID2 -> ID1,
            # followed by the reverse high-pose order ID1 -> ID2 -> ID6.
            actions.extend([
                ("PLACE_RING_ID6", self.ring_place_id6,
                 (RING_PLACE[2],)),
                ("PLACE_RING_ID2", self.ring_place_id2,
                 (RING_PLACE[1],)),
                ("PLACE_RING_ID1", self.ring_place_id1,
                (RING_PLACE[0],)),
            ])
        else:
            # Letter placement uses the same staged joint order as the
            # standalone app: ID6 -> ID2 -> ID1, 500 ms each.
            actions.extend([
                ("PLACE_LETTER_ID6", self.letter_place_id6,
                 (placement[2],)),
                ("PLACE_LETTER_ID2", self.letter_place_id2,
                 (placement[1],)),
                ("PLACE_LETTER_ID1", self.letter_place_id1,
                 (placement[0],)),
            ])
        actions.extend([
            ("PLACE_OPEN", self.gripper, (PLATFORM_GRIPPER_OPEN,)),
            ("PLACE_CLOSE", self.gripper, (PLATFORM_GRIPPER_CLOSED,)),
        ])
        if key[0] == "ring":
            actions.extend([
                ("RETURN_RING_HIGH_ID1", self.ring_return_high_id1,
                 (HIGH[0],)),
                ("RETURN_RING_HIGH_ID2", self.ring_return_high_id2,
                 (HIGH[1],)),
                ("RETURN_RING_HIGH_ID6", self.ring_return_high_id6,
                (HIGH[2],)),
            ])
        else:
            actions.extend([
                ("RETURN_LETTER_HIGH_ID1", self.letter_return_high_id1,
                 (HIGH[0],)),
                ("RETURN_LETTER_HIGH_ID2", self.letter_return_high_id2,
                 (HIGH[1],)),
                ("RETURN_LETTER_HIGH_ID6", self.letter_return_high_id6,
                 (HIGH[2],)),
            ])
        self.actions = deque(actions)
        self.stage = "platform_actions"

    def tick(self, detections=(), shape=None, fresh=False):
        now = self.clock()
        if self.stage == "platform_preselect" and now >= self.timeout:
            self._fail("SECONDARY_PAIR_TIMEOUT")
        elif self.stage == "platform_raise" and now >= self.deadline:
            self.high_ready = True
            self.stage = "platform_entry_hold"
            self.done = "PRESELECT_DONE:" + ":".join(self.selected)
            self.status = "PLATFORM_PICK ARM_HIGH_READY; release H7 approach"
        elif self.stage == "platform_detect":
            if not self.target_seen and now >= self.timeout:
                self.skip("MAIN_TARGET_OR_DEPTH_TIMEOUT")
            elif now >= self.deadline and fresh:
                self._detect(detections, shape)
        elif self.stage == "platform_actions" and now >= self.deadline:
            if self.actions:
                label, callback, args = self.actions.popleft()
                self._command(label, callback, *args)
            else:
                self.high_ready = True
                self.stage = "platform_high_hold"
                self.done = self.finish_reason
                self.status = f"PLATFORM_PICK ARM_HIGH_READY; {self.finish_reason}; release H7 next slot"
        return self.status
