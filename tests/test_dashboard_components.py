"""
Unit and integration test suite verifying dashboard components and pages in app.py.
Tests model loading, inference execution, export generation, and analytics.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from config.config import OccupancyConfig
from src.occupancy.occupancy_engine import OccupancyEngine
from src.utils.visualizer import ParkingVisualizer


@pytest.fixture
def sample_parking_image():
    """Returns the default parking.jpg test image or a dummy image if missing."""
    img_path = Path("parking.jpg")
    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            return img
    # Dummy fallback image
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    return dummy


@pytest.fixture
def slots_data():
    """Loads parking_slots.json."""
    with open("parking_slots.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_overview_metrics_logic(sample_parking_image, slots_data):
    """Verifies that the occupancy engine produces valid metrics for Page 1 & 2."""
    config = OccupancyConfig(mode="yolo_only")
    engine = OccupancyEngine(config=config)
    result = engine.process_image(sample_parking_image, slots=slots_data)

    assert result is not None
    assert result.total_slots == len(slots_data)
    assert result.total_slots == result.free_slots + result.occupied_slots + result.uncertain_slots
    assert 0.0 <= result.occupancy_rate <= 100.0


def test_visualizer_rendering(sample_parking_image, slots_data):
    """Verifies that ParkingVisualizer creates a visual overlay without error."""
    config = OccupancyConfig(mode="yolo_only")
    engine = OccupancyEngine(config=config)
    result = engine.process_image(sample_parking_image, slots=slots_data)

    visualizer = ParkingVisualizer()
    annotated = visualizer.annotate(sample_parking_image, result, draw_vehicles=True, draw_slots=True)

    assert annotated is not None
    assert annotated.shape == sample_parking_image.shape
    assert annotated.dtype == np.uint8


def test_export_dataframe_and_json(sample_parking_image, slots_data):
    """Verifies CSV and JSON export logic as used in the dashboard."""
    config = OccupancyConfig(mode="yolo_only")
    engine = OccupancyEngine(config=config)
    result = engine.process_image(sample_parking_image, slots=slots_data)

    # DataFrame creation for export
    records = []
    for slot in result.slots:
        records.append({
            "slot_id": slot.slot_id,
            "status": slot.status,
            "confidence": slot.confidence,
            "mode": slot.decision_mode
        })
    df = pd.DataFrame(records)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0

    json_str = json.dumps(result.to_dict(), indent=2)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert "slots" in parsed
    assert "occupancy_rate" in parsed


def test_model_performance_benchmarks_exist():
    """Verifies that outputs/benchmarks.json exists and contains all three modes."""
    benchmarks_path = Path("outputs/benchmarks.json")
    assert benchmarks_path.exists(), "benchmarks.json must exist for Page 5"
    with open(benchmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "yolo_only" in data
    assert "resnet_only" in data
    assert "hybrid" in data
