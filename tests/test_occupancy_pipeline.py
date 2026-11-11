"""
Unit and integration tests for SmartPark AI occupancy engine and slot matching.
"""

import pytest
import numpy as np

from src.detection.yolo_detector import DetectedVehicle
from src.occupancy.slot_matcher import SlotMatcher
from src.occupancy.occupancy_engine import OccupancyEngine
from src.analytics.metrics import calculate_classification_metrics


@pytest.fixture
def sample_slots():
    """Returns 4 mock parking slots."""
    return [
        [0, 0, 50, 50],       # Slot 1
        [60, 0, 110, 50],     # Slot 2
        [120, 0, 170, 50],    # Slot 3
        [180, 0, 230, 50],    # Slot 4
    ]


@pytest.fixture
def mock_image():
    """Generates a synthetic 240x300x3 test image."""
    return np.zeros((240, 300, 3), dtype=np.uint8)


class TestSlotMatchingLogic:
    """Tests for spatial vehicle-to-slot allocation."""

    def test_empty_lot_all_free(self, sample_slots):
        matcher = SlotMatcher()
        results = matcher.match(sample_slots, vehicles=[])

        assert len(results) == len(sample_slots)
        assert all(r.status == "FREE" for r in results)
        assert all(r.assigned_vehicle is None for r in results)

    def test_one_to_one_vehicle_assignment(self, sample_slots):
        matcher = SlotMatcher()
        # Single vehicle perfectly inside Slot 2
        vehicle = DetectedVehicle(
            box=[60, 0, 110, 50],
            confidence=0.92,
            class_name="car",
            class_id=2
        )
        results = matcher.match(sample_slots, [vehicle])

        assert results[1].status == "OCCUPIED"
        assert results[1].assigned_vehicle is not None
        assert results[1].assigned_vehicle.confidence == 0.92

        # Verify other slots remain FREE
        assert results[0].status == "FREE"
        assert results[2].status == "FREE"
        assert results[3].status == "FREE"

    def test_prevent_single_vehicle_occupying_multiple_slots(self, sample_slots):
        """
        Crucial test: If a large bounding box covers both Slot 1 and Slot 2,
        1-to-1 bipartite matching MUST allocate it to only one slot (the one with higher affinity).
        """
        matcher = SlotMatcher()
        # Oversized truck spanning across Slot 1 and partially Slot 2
        truck = DetectedVehicle(
            box=[10, 0, 80, 50],  # Overlaps Slot 1 (40px) and Slot 2 (20px)
            confidence=0.88,
            class_name="truck",
            class_id=7
        )
        results = matcher.match(sample_slots, [truck])

        occupied_count = sum(1 for r in results if r.status == "OCCUPIED")
        # Exactly one slot should be occupied by this one vehicle
        assert occupied_count == 1
        assert results[0].status == "OCCUPIED"  # Slot 1 has greater coverage

    def test_fully_occupied_lot(self, sample_slots):
        matcher = SlotMatcher()
        vehicles = [
            DetectedVehicle(box=[0, 0, 50, 50], confidence=0.9, class_name="car", class_id=2),
            DetectedVehicle(box=[60, 0, 110, 50], confidence=0.9, class_name="car", class_id=2),
            DetectedVehicle(box=[120, 0, 170, 50], confidence=0.9, class_name="car", class_id=2),
            DetectedVehicle(box=[180, 0, 230, 50], confidence=0.9, class_name="car", class_id=2),
        ]
        results = matcher.match(sample_slots, vehicles)
        assert all(r.status == "OCCUPIED" for r in results)

    def test_uncertain_boundary_coverage(self, sample_slots):
        matcher = SlotMatcher()
        # Vehicle clipping slot by just 8 pixels (marginal overlap)
        marginal_car = DetectedVehicle(
            box=[45, 0, 95, 50],  # covers 5px of slot 1 [0..50] -> 10%
            confidence=0.75,
            class_name="car",
            class_id=2
        )
        results = matcher.match(sample_slots, [marginal_car])
        # Slot 1 should be FREE or UNCERTAIN, definitely not full OCCUPIED
        assert results[0].status in ("FREE", "UNCERTAIN")


class TestOccupancyEngineEdgeCases:
    """Tests for edge cases, error handling, and invalid inputs."""

    def test_invalid_image_raises_error(self, sample_slots):
        engine = OccupancyEngine(load_resnet=False, load_yolo=False)
        with pytest.raises(ValueError):
            engine.process_image(None, sample_slots)

        with pytest.raises(ValueError):
            engine.process_image(np.array([]), sample_slots)

    def test_empty_slot_list_raises_error(self, mock_image):
        engine = OccupancyEngine(load_resnet=False, load_yolo=False)
        with pytest.raises(ValueError):
            engine.process_image(mock_image, [])


class TestMetricsCalculation:
    """Tests for classification and occupancy metrics computation."""

    def test_perfect_classification_metrics(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 1, 0]
        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_imbalanced_classification_metrics(self):
        y_true = [0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 0]  # Missed the single positive
        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.80
        assert metrics.recall < 1.0
