"""
Benchmarking and Performance Evaluation Suite for SmartPark AI.
Measures:
- Latency and throughput (FPS) for YOLOv8, ResNet18, and Hybrid modes
- Memory utilization via psutil
- Occupancy agreement between detection pipelines
- Writes quantitative benchmark reports to BENCHMARKS.md and outputs/benchmarks.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import psutil

from config.config import OccupancyConfig
from src.occupancy.occupancy_engine import OccupancyEngine
from src.utils.logger import get_logger

logger = get_logger("BenchmarkModes")


def benchmark_pipeline(
    iterations: int = 5,
    image_path: str = "parking.jpg",
    slots_path: str = "parking_slots.json"
) -> Dict[str, Any]:
    """Runs systematic benchmarks across YOLO, ResNet18, and Hybrid modes."""
    img = cv2.imread(image_path)
    if img is None:
        img_p = PROJECT_ROOT / "data" / "sample" / "parking.jpg"
        img = cv2.imread(str(img_p))

    with open(slots_path, "r", encoding="utf-8") as f:
        slots = json.load(f)

    cfg = OccupancyConfig()
    engine = OccupancyEngine(config=cfg, load_resnet=True, load_yolo=True)

    results = {}
    modes = ["yolo_only", "resnet_only", "hybrid"]

    process = psutil.Process()

    for mode in modes:
        logger.info("Executing benchmark for mode: %s (%d iterations)...", mode, iterations)
        latencies = []
        detailed_timings = {"yolo": [], "resnet": [], "matching": []}
        mem_usages = []

        # Warm-up pass
        engine.process_image(img, slots, mode=mode)

        for _ in range(iterations):
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            t0 = time.perf_counter()
            out = engine.process_image(img, slots, mode=mode)
            dur = time.perf_counter() - t0
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB

            latencies.append(dur * 1000)  # ms
            detailed_timings["yolo"].append(out.timings.get("yolo_time", 0.0) * 1000)
            detailed_timings["resnet"].append(out.timings.get("resnet_time", 0.0) * 1000)
            detailed_timings["matching"].append(out.timings.get("matching_time", 0.0) * 1000)
            mem_usages.append(mem_after)

        avg_lat = float(np.mean(latencies))
        fps = float(1000.0 / avg_lat) if avg_lat > 0 else 0.0

        results[mode] = {
            "mode": mode,
            "average_latency_ms": round(avg_lat, 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "min_latency_ms": round(float(np.min(latencies)), 2),
            "max_latency_ms": round(float(np.max(latencies)), 2),
            "fps": round(fps, 2),
            "yolo_time_ms": round(float(np.mean(detailed_timings["yolo"])), 2),
            "resnet_time_ms": round(float(np.mean(detailed_timings["resnet"])), 2),
            "matching_time_ms": round(float(np.mean(detailed_timings["matching"])), 2),
            "memory_usage_mb": round(float(np.mean(mem_usages)), 2),
            "occupied_detected": out.occupied_slots,
            "free_detected": out.free_slots,
            "uncertain_detected": out.uncertain_slots,
            "occupancy_rate_pct": out.occupancy_rate
        }

    # Save benchmark JSON
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "benchmarks.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info("Benchmark complete. Results saved to %s", out_json)
    return results


if __name__ == "__main__":
    benchmark_pipeline(iterations=5)
