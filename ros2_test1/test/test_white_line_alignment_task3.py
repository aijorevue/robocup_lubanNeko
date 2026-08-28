import cv2
import numpy as np

from ros2_test1.white_line_alignment import WhiteLineAlignmentDetector


def _frame_with_strip(color, start=(100, 285), end=(520, 315)):
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(frame, start, end, color, cv2.FILLED)
    return frame


def test_task3_accepts_true_white_strip():
    result = WhiteLineAlignmentDetector().detect_task3(
        _frame_with_strip((245, 245, 245))
    )
    assert result is not None
    assert abs(result["y_at_center"] - 300.0) < 2.0
    assert abs(result["right_edge_x"] - 520.0) < 3.0


def test_task3_rejects_gray_vehicle_highlight_at_reference_y():
    result = WhiteLineAlignmentDetector().detect_task3(
        _frame_with_strip((150, 150, 150), start=(610, 285), end=(790, 315))
    )
    assert result is None


def test_task3_rejects_gray_vehicle_highlight_below_reference():
    result = WhiteLineAlignmentDetector().detect_task3(
        _frame_with_strip((175, 175, 175), start=(560, 330), end=(790, 345))
    )
    assert result is None
