import cv2
import numpy as np

from ros2_test1.white_line_alignment import WhiteLineAlignmentDetector


def _frame_with_strip():
    frame = np.full((600, 800, 3), 28, dtype=np.uint8)
    strip = np.array(
        [[(90, 250), (710, 240), (710, 260), (90, 270)]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, strip, (245, 245, 245))
    return frame


def test_wide_strip_reports_smoothed_two_edge_centerline():
    detector = WhiteLineAlignmentDetector()
    frame = _frame_with_strip()

    measurements = [detector.detect(frame) for _ in range(3)]
    measurement = measurements[-1]

    assert all(item is not None for item in measurements)
    assert measurement["centerline_source"] == "BAND_EDGE_MIDPOINT"
    assert measurement["edge_support"] >= 400
    assert abs(measurement["y_at_center"] - 255.0) <= 1.5
    assert abs(measurement["angle_deg"] - (-0.92)) <= 0.4

    (x0, y0), (x1, y1) = measurement["center_line"]
    assert x0 < 100
    assert x1 > 700
    center_y = y0 + (y1 - y0) * (400.0 - x0) / (x1 - x0)
    assert abs(center_y - measurement["y_at_center"]) <= 0.1


def test_overhead_box_is_not_accepted_as_task1_line():
    frame = np.full((600, 800, 3), 28, dtype=np.uint8)
    cv2.rectangle(frame, (220, 90), (580, 164), (245, 245, 245), -1)

    measurement = WhiteLineAlignmentDetector().detect(frame)

    assert measurement is None


def test_contour_fit_fallback_still_reports_centerline(monkeypatch):
    detector = WhiteLineAlignmentDetector()
    monkeypatch.setattr(detector, "_fit_task2_band_edges", lambda *_: None)

    measurement = detector.detect(_frame_with_strip())

    assert measurement is not None
    assert measurement["centerline_source"] == "CONTOUR_FIT_FALLBACK"
    (x0, y0), (x1, y1) = measurement["center_line"]
    assert x0 < 100
    assert x1 > 700
    center_y = y0 + (y1 - y0) * (400.0 - x0) / (x1 - x0)
    assert abs(center_y - measurement["y_at_center"]) <= 0.1


def test_empty_frame_does_not_produce_a_centerline():
    frame = np.full((600, 800, 3), 28, dtype=np.uint8)

    measurement = WhiteLineAlignmentDetector().detect(frame)

    assert measurement is None
