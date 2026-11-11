"""
Unit tests for bounding box geometry, IoU, clipping, and NMS suppression.
"""

import pytest
import numpy as np

from src.detection.nms import (
    calculate_iou_single,
    calculate_intersection_ratio,
    nms_boxes,
    filter_duplicate_detections,
)
from src.preprocessing.image_transforms import clip_box_to_image
from src.occupancy.slot_matcher import is_center_inside, compute_overlap_metrics


class TestGeometryAndIoU:
    """Tests for geometric computations and intersection metrics."""

    def test_iou_identical_boxes(self):
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        assert calculate_iou_single(box1, box2) == pytest.approx(1.0, rel=1e-3)

    def test_iou_non_overlapping_boxes(self):
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        assert calculate_iou_single(box1, box2) == 0.0

    def test_iou_partial_overlap(self):
        # 10x10 boxes overlapping by 5x10 (50 sq px)
        # area1 = 100, area2 = 100, inter = 50, union = 150 -> IoU = 50/150 = 0.3333
        box1 = [0, 0, 10, 10]
        box2 = [5, 0, 15, 10]
        assert calculate_iou_single(box1, box2) == pytest.approx(50.0 / 150.0, rel=1e-3)

    def test_iou_inverted_coordinates(self):
        # Handles cases where x2 < x1 or y2 < y1
        box1 = [50, 50, 10, 10]
        box2 = [10, 10, 50, 50]
        assert calculate_iou_single(box1, box2) == pytest.approx(1.0, rel=1e-3)

    def test_intersection_ratio_coverage(self):
        # Slot is [0, 0, 50, 50] (area 2500)
        # Vehicle is [0, 0, 100, 100] (area 10000)
        # Vehicle completely covers slot -> slot coverage should be 1.0
        slot = [0, 0, 50, 50]
        vehicle = [0, 0, 100, 100]
        assert calculate_intersection_ratio(slot, vehicle) == pytest.approx(1.0, rel=1e-3)
        # Vehicle coverage by slot is 2500 / 10000 = 0.25
        assert calculate_intersection_ratio(vehicle, slot) == pytest.approx(0.25, rel=1e-3)

    def test_center_inside_containment(self):
        slot = [10, 10, 50, 50]
        # Vehicle center is (30, 30) -> Inside
        car_inside = [20, 20, 40, 40]
        assert is_center_inside(car_inside, slot) is True

        # Vehicle center is (60, 30) -> Outside
        car_outside = [50, 20, 70, 40]
        assert is_center_inside(car_outside, slot) is False

    def test_clip_box_to_image_bounds(self):
        img_shape = (100, 200, 3)  # h=100, w=200
        out_of_bounds = [-20, -10, 250, 150]
        x1, y1, x2, y2 = clip_box_to_image(out_of_bounds, img_shape)

        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 200
        assert y2 <= 100
        assert x2 > x1
        assert y2 > y1


class TestNMSAndSuppression:
    """Tests for non-maximum suppression and duplicate detection filtering."""

    def test_nms_suppression_duplicates(self):
        boxes = [
            [10, 10, 50, 50],
            [12, 11, 51, 49],  # Highly overlapping duplicate
            [100, 100, 150, 150]  # Separate object
        ]
        scores = [0.90, 0.80, 0.85]
        kept = nms_boxes(boxes, scores, iou_threshold=0.5)

        # Should keep box 0 and box 2, suppressing box 1
        assert 0 in kept
        assert 2 in kept
        assert 1 not in kept

    def test_filter_duplicate_detections_dict(self):
        detections = [
            {"box": [10, 10, 50, 50], "confidence": 0.95, "class_name": "car"},
            {"box": [11, 10, 49, 50], "confidence": 0.60, "class_name": "car"},
        ]
        filtered = filter_duplicate_detections(detections, iou_threshold=0.5)
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.95

    def test_empty_boxes_input(self):
        assert nms_boxes([], []) == []
        assert filter_duplicate_detections([]) == []
