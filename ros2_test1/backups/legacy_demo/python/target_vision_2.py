#!/usr/bin/env python3
"""Independent entry point for the direct 3D grasp workflow."""

from __future__ import annotations

import sys
import time

from . import target_vision as vision1


_vision1_draw = vision1.TargetDetector.draw
DEPTH_MODEL_COEFFICIENTS = (
    -1.173014829734954,
    1.0749625606320283,
    0.3194742643291999,
    0.7040244348005884,
    1.9961971262732805,
)
DEPTH_MODEL_X_SCALE_PX = 200.0
DEPTH_MODEL_Y_SCALE_PX = 200.0
DEPTH_MODEL_OUTPUT_LIMITS_CM = (10.0, 30.0)
SERVO_CALIBRATION_DEPTH_LIMITS_CM = (14.0, 30.0)
SERVO_CALIBRATION_DX_LIMITS_PX = (-283.0, 159.0)
SERVO_CALIBRATION_DY_LIMITS_PX = (-115.0, 219.0)
ID1_MODEL_COEFFICIENTS = (
    286.8817112024376,
    -103.71211151413848,
    1.5375656255013803,
    7.093950296772093,
    -2.4168865880081873,
    4.949762372336365,
    -1.2588212564910017,
    16.939305355437977,
    -21.102113585734653,
    0.24849983671764098,
    43.07131042068869,
    10.623160090087769,
    14.66352189130521,
    -41.67433439041083,
)
ID2_MODEL_COEFFICIENTS = (
    305.3609793943877,
    69.57574421864574,
    -19.520933606011777,
    -5.292690960042246,
    -94.87540947846601,
)
ID6_MODEL_COEFFICIENTS = (
    506.76209247682635,
    -2.7795193879229383,
    27.67798977433311,
    -86.58873991558843,
    1.8400190712294606,
)


def _correct_depth_cm(raw_distance_cm, dx_px, dy_px):
    if raw_distance_cm is None:
        return None
    normalized_x = float(dx_px) / DEPTH_MODEL_X_SCALE_PX
    normalized_y = float(dy_px) / DEPTH_MODEL_Y_SCALE_PX
    intercept, raw_gain, x_gain, y_gain, radial_gain = DEPTH_MODEL_COEFFICIENTS
    corrected = (
        intercept
        + raw_gain * float(raw_distance_cm)
        + x_gain * normalized_x
        + y_gain * normalized_y
        + radial_gain * (normalized_x * normalized_x + normalized_y * normalized_y)
    )
    return max(
        DEPTH_MODEL_OUTPUT_LIMITS_CM[0],
        min(DEPTH_MODEL_OUTPUT_LIMITS_CM[1], corrected),
    )


def _target_with_corrected_depth(target, frame_shape):
    if target is None:
        return None
    corrected_target = dict(target)
    raw_distance_cm = target.get("distance_cm")
    height, width = frame_shape[:2]
    center_x, center_y = target.get("center", (width / 2.0, height / 2.0))
    dx_px = float(center_x) - width / 2.0
    dy_px = float(center_y) - height / 2.0
    corrected_target["raw_distance_cm"] = raw_distance_cm
    corrected_target["distance_cm"] = _correct_depth_cm(
        raw_distance_cm,
        dx_px,
        dy_px,
    )
    return corrected_target


def _calibrated_servo_ticks(corrected_depth_cm, raw_depth_cm, dx_px, dy_px):
    if corrected_depth_cm is None or raw_depth_cm is None:
        return None
    if not (
        SERVO_CALIBRATION_DEPTH_LIMITS_CM[0]
        <= corrected_depth_cm
        <= SERVO_CALIBRATION_DEPTH_LIMITS_CM[1]
        and SERVO_CALIBRATION_DX_LIMITS_PX[0]
        <= dx_px
        <= SERVO_CALIBRATION_DX_LIMITS_PX[1]
        and SERVO_CALIBRATION_DY_LIMITS_PX[0]
        <= dy_px
        <= SERVO_CALIBRATION_DY_LIMITS_PX[1]
    ):
        return None

    depth = (float(corrected_depth_cm) - 22.0) / 8.0
    raw_error = (float(raw_depth_cm) - float(corrected_depth_cm)) / 4.0
    normalized_x = float(dx_px) / 200.0
    normalized_y = float(dy_px) / 200.0
    id1_features = (
        1.0,
        depth,
        raw_error,
        normalized_x,
        normalized_y,
        depth * depth,
        raw_error * raw_error,
        normalized_x * normalized_x,
        raw_error * normalized_x,
        raw_error * normalized_y,
        normalized_x * normalized_y,
        normalized_y * normalized_y,
        depth * normalized_x,
        depth * normalized_y,
    )
    linear_features = (1.0, depth, raw_error, normalized_x, normalized_y)
    id1 = round(sum(a * b for a, b in zip(ID1_MODEL_COEFFICIENTS, id1_features)))
    id2 = round(sum(a * b for a, b in zip(ID2_MODEL_COEFFICIENTS, linear_features)))
    id6 = round(sum(a * b for a, b in zip(ID6_MODEL_COEFFICIENTS, linear_features)))
    return int(id1), int(id2), int(id6)


def _apply_servo_calibration(controller, plan, target, frame_shape):
    height, width = frame_shape[:2]
    center_x, center_y = target.get("center", (width / 2.0, height / 2.0))
    dx_px = float(center_x) - width / 2.0
    dy_px = float(center_y) - height / 2.0
    corrected_depth_cm = target.get("distance_cm")
    raw_depth_cm = target.get("raw_distance_cm", corrected_depth_cm)
    ticks = _calibrated_servo_ticks(
        corrected_depth_cm,
        raw_depth_cm,
        dx_px,
        dy_px,
    )
    if ticks is None:
        return plan, "servo calibration out of range; geometric IK retained"

    id1 = max(controller.id1_limits[0], min(controller.id1_limits[1], ticks[0]))
    id2 = max(controller.id2_limits[0], min(controller.id2_limits[1], ticks[1]))
    id6 = max(vision1.ID6_SAFE_LIMITS[0], min(vision1.ID6_SAFE_LIMITS[1], ticks[2]))
    id1, id2 = vision1.enforce_angle_gap(
        id1,
        id2,
        controller.id2_limits,
        controller.angle_gap_degrees,
    )
    final_x_mm, final_z_mm = vision1.gripper_position_mm(id1, id2)
    overhead_id1, overhead_id2, overhead_error_mm = vision1.solve_gripper_position(
        final_x_mm,
        final_z_mm + vision1.DIRECT_2_OVERHEAD_CLEARANCE_MM,
        controller.id1,
        controller.id2,
        controller.id1_limits,
        controller.id2_limits,
        controller.angle_gap_degrees,
    )
    calibrated_plan = dict(plan)
    calibrated_plan.update(
        {
            "id1": id1,
            "id2": id2,
            "id6": id6,
            "overhead_id1": overhead_id1,
            "overhead_id2": overhead_id2,
            "overhead_error_mm": overhead_error_mm,
            "ik_error_mm": 0.0,
            "servo_calibrated": True,
        }
    )
    return calibrated_plan, (
        f"servo calibration final={id1}/{id2}/{id6} "
        f"overhead={overhead_id1}/{overhead_id2}"
    )


def _draw_vision2(self, frame, detections):
    output, info = _vision1_draw(self, frame, detections)
    height, width = output.shape[:2]
    frame_center = (width // 2, height // 2)
    red_square = max(
        (
            detection
            for detection in detections
            if detection.get("color") == "red"
            and detection.get("kind") == "square"
            and detection.get("center") is not None
        ),
        key=lambda detection: detection.get("bbox", (0, 0, 0, 0))[2]
        * detection.get("bbox", (0, 0, 0, 0))[3],
        default=None,
    )
    if red_square is None:
        offset_text = "RED offset: waiting (center = 0, 0)"
        text_color = (0, 200, 255)
    else:
        target_center = tuple(int(value) for value in red_square["center"])
        dx = target_center[0] - frame_center[0]
        dy = target_center[1] - frame_center[1]
        raw_distance_cm = red_square.get("distance_cm")
        corrected_distance_cm = _correct_depth_cm(raw_distance_cm, dx, dy)
        vision1.cv2.circle(output, target_center, 7, (0, 0, 255), -1)
        vision1.cv2.line(output, frame_center, target_center, (0, 200, 255), 2)
        if corrected_distance_cm is None:
            depth_text = "depth unavailable"
        else:
            depth_text = (
                f"raw={raw_distance_cm:.1f}cm corrected={corrected_distance_cm:.1f}cm"
            )
        offset_text = (
            f"RED offset: dx={dx:+d}px dy={dy:+d}px "
            f"{depth_text} (center = 0, 0)"
        )
        text_color = (0, 255, 255)
    vision1.cv2.putText(
        output,
        offset_text,
        (10, height - 18),
        vision1.cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        text_color,
        2,
    )
    return output, f"{info} | {offset_text}"


def _update_direct_2(self, target, frame_shape):
    now = time.monotonic()
    if self.direct_stage == "complete":
        self.state = "direct_2_complete"
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        return "direct-2 grasp complete; holding object"

    if self.direct_stage == "detect":
        target = _target_with_corrected_depth(target, frame_shape)
        ready, reason = self._target_ready(target)
        if not ready:
            self.state = "direct_2_detect"
            return reason
        plan, plan_text = self._direct_2_plan(target, frame_shape)
        if plan is None:
            self.state = "direct_2_plan_wait"
            return plan_text
        plan, calibration_text = _apply_servo_calibration(
            self,
            plan,
            target,
            frame_shape,
        )
        plan_text = f"{plan_text} | {calibration_text}"
        if max(plan["overhead_error_mm"], plan["ik_error_mm"]) > self.post_center_ik_error_mm:
            self.state = "direct_2_unreachable"
            return f"direct-2 IK rejected: {plan_text}"

        self.direct_target = self._copy_target(target)
        self.direct_plan = plan
        self.arm_preview.publish_plan_marker(plan, plan_text)
        print(
            "DIRECT2 LOCK "
            f"target={vision1.json.dumps(self.direct_target, ensure_ascii=True, default=str)} "
            f"plan={plan_text}",
            flush=True,
        )
        self._set_arm_targets(plan["overhead_id1"], plan["overhead_id2"])
        self.id4 = self.id4_open
        self.id6 = vision1.BASE_YAW_READY_TICK
        if not self.servo_bridge.write_enabled:
            self.direct_stage = "complete"
            self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
            return f"preview direct-2 {plan_text}"

        self.direct_attempts = 1
        self.direct_stage = "overhead_wait"
        self.algorithm_stage = "direct_2_overhead"
        self.claw_settle_deadline = now + vision1.CLAW_SETTLE_S
        return self._send(
            f"direct-2 overhead+open; hold ID6 center | {plan_text}",
            require_feedback=True,
            send_id4=True,
            send_id6=True,
        )

    plan = self.direct_plan
    if plan is None:
        self.direct_stage = "detect"
        return "direct-2 plan missing; detecting again"

    if self.direct_stage == "overhead_wait":
        target_id1 = plan["overhead_id1"]
        target_id2 = plan["overhead_id2"]
        if not self._arm_target_reached(target_id1, target_id2):
            if self.direct_attempts >= vision1.DIRECT_2_MAX_ATTEMPTS:
                self.state = "fault"
                self.algorithm_stage = "fault"
                self.status = (
                    f"direct-2 overhead not reached ID1={self.id1}/{target_id1} "
                    f"ID2={self.id2}/{target_id2}"
                )
                return self.status
            self._set_arm_targets(target_id1, target_id2)
            self.direct_attempts += 1
            return self._send(
                "direct-2 overhead retry "
                f"{self.direct_attempts}/{vision1.DIRECT_2_MAX_ATTEMPTS}",
                require_feedback=True,
            )
        if now < self.claw_settle_deadline:
            return (
                "direct-2 overhead reached; claw opening "
                f"{self.claw_settle_deadline - now:.1f}s"
            )

        self._set_arm_targets(plan["id1"], plan["id2"])
        self.id6 = plan["id6"]
        self.direct_attempts = 1
        self.direct_stage = "final_wait"
        self.algorithm_stage = "direct_2_descend"
        return self._send(
            "direct-2 descend with calculated ID6",
            require_feedback=True,
            send_id6=True,
        )

    if self.direct_stage == "final_wait":
        target_id1 = plan["id1"]
        target_id2 = plan["id2"]
        if not self._arm_target_reached(target_id1, target_id2):
            if self.direct_attempts >= vision1.DIRECT_2_MAX_ATTEMPTS:
                self.state = "fault"
                self.algorithm_stage = "fault"
                self.status = (
                    f"direct-2 final not reached ID1={self.id1}/{target_id1} "
                    f"ID2={self.id2}/{target_id2}"
                )
                return self.status
            self._set_arm_targets(target_id1, target_id2)
            self.direct_attempts += 1
            return self._send(
                "direct-2 final retry "
                f"{self.direct_attempts}/{vision1.DIRECT_2_MAX_ATTEMPTS}",
                require_feedback=True,
            )

        self.id4 = self.id4_closed
        self.direct_stage = "close_wait"
        self.algorithm_stage = "direct_2_close"
        self.claw_settle_deadline = now + vision1.CLAW_SETTLE_S
        return self._send_claw("direct-2 final reached; close claw")

    if self.direct_stage == "close_wait":
        if now < self.claw_settle_deadline:
            return f"direct-2 closing claw {self.claw_settle_deadline - now:.1f}s"
        self.id6 = vision1.BASE_YAW_READY_TICK
        self.direct_stage = "complete"
        self.algorithm_stage = "complete"
        self.state = "direct_2_complete"
        self.arm_preview.set_targets(self.id1, self.id2, self.id4, self.id6)
        result = self._send(
            "direct-2 grasp complete; return ID6 center",
            require_feedback=False,
            send_id6=True,
        )
        self.arm_preview.publish("direct-2 grasp complete; ID6 centered")
        return result

    return f"direct-2 state {self.direct_stage}"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--grasp-direct-2" not in arguments:
        arguments.insert(0, "--grasp-direct-2")
    vision1.TargetDetector.draw = _draw_vision2
    vision1.RedSquareGraspController._update_direct_2 = _update_direct_2
    return vision1.main(arguments)


if __name__ == "__main__":
    main()
