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
RING_PLACE = (535, 330, 120)
POST_OPEN_ID2_RETREAT_TICKS = 100
CENTER_ID6_STEP_TICKS = 5
CENTER_ID2_STEP_TICKS = 7
CENTER_ID6_RANGE = (0, 700)
PLATFORM_NO_TARGET_TIMEOUT_S = 1.5


class PlatformTask:
    def __init__(self, pose, gripper, center, *, ring_place_pair=None,
                 ring_place_id1=None, clock=time.monotonic):
        self.pose = pose
        self.gripper = gripper
        self.center = center
        self.ring_place_pair = ring_place_pair or pose
        self.ring_place_id1 = ring_place_id1 or pose
        self.clock = clock
        self.reset()

    def reset(self):
        self.stage = None
        self.selected = ()
        self.done = self.error = None
        self.deadline = 0.0
        self.actions = deque()
        self.high_ready = False
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
        self.center_id1 = HIGH[0]
        self.center_id2, self.center_id6 = HIGH[1:]
        self.status = "PLATFORM_PICK main camera: waiting target"

    @staticmethod
    def _key(target):
        return (target.get("kind"), target.get("letter") if target.get("kind") == "letter"
                else target.get("color"))

    def _allowed(self, target):
        kind, label = self._key(target)
        return ((kind == "letter" and label in self.selected)
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
                      and d.get("center") is not None]
        allowed = [d for d in candidates if self._allowed(d)]
        if allowed:
            candidates = allowed
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
        self.votes.append((key, point))
        if sum(v is not None and v[0] == key and math.dist(v[1], point) <= 140
               for v in self.votes) < 3:
            return
        if not self._allowed(target):
            self.skip("TARGET_NOT_SELECTED")
            return
        self.target_key, self.target_center = key, point
        dx, dy = point[0] - width / 2, point[1] - height / 2
        if abs(dx) > 45 or abs(dy) > 45:
            id2 = max(450, min(700, self.center_id2 + (
                -CENTER_ID2_STEP_TICKS if dy > 45
                else CENTER_ID2_STEP_TICKS if dy < -45 else 0
            )))
            id6 = max(
                CENTER_ID6_RANGE[0],
                min(CENTER_ID6_RANGE[1], self.center_id6 + (
                    -CENTER_ID6_STEP_TICKS if dx > 45
                    else CENTER_ID6_STEP_TICKS if dx < -45 else 0
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
            id1, id2 = calibrated_grasp_ticks(depth)
        except (ValueError, TypeError):
            self.status = "PLATFORM_PICK waiting valid measured depth 7..30cm"
            return
        self.high_ready = False
        placement = LETTER_PLACE if key[0] == "letter" else RING_PLACE
        print(f"PLATFORM_PICK TARGET kind={key[0]} label={key[1]} depth_cm={depth:.2f} "
              f"down={id1}/{id2}/{self.center_id6} model=measured_7_30cm", flush=True)
        self.finish_reason = f"PICKED_{key[0].upper()}"
        retreat_id2 = max(450, self.center_id2 - POST_OPEN_ID2_RETREAT_TICKS)
        actions = [
            ("GRIPPER_OPEN", self.gripper, (1650,)),
            ("POST_OPEN_ID2_RETREAT", self.pose,
             ((self.center_id1, retreat_id2, self.center_id6), False)),
            ("DESCEND", self.pose, ((id1, id2, self.center_id6), False)),
            ("GRIPPER_CLOSE", self.gripper, (1300,)),
            ("LIFT_HIGH", self.pose, (HIGH, True)),
        ]
        if key[0] == "ring":
            # Ring placement is deliberately sequenced: yaw/pitch first,
            # then ID1, so the arm does not swing all three joints together.
            actions.extend([
                ("PLACE_RING_ID2_ID6", self.ring_place_pair,
                 (RING_PLACE[1], RING_PLACE[2])),
                ("PLACE_RING_ID1", self.ring_place_id1,
                 (RING_PLACE[0],)),
            ])
        else:
            actions.append(("PLACE_LETTER", self.pose, (placement, False)))
        actions.extend([
            ("PLACE_OPEN", self.gripper, (1650,)),
            ("PLACE_CLOSE", self.gripper, (1300,)),
            ("RETURN_HIGH", self.pose, (HIGH, True)),
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
            if now >= self.timeout:
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
