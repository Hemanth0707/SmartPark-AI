"""
YOLO Vehicle Detector module for SmartPark AI.
Wraps Ultralytics YOLO with custom NMS, multi-tile processing, and vehicle filtering.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Set, Union, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config.config import DetectionConfig
from src.detection.nms import filter_duplicate_detections
from src.utils.logger import get_logger

logger = get_logger("YOLODetector")


@dataclass
class DetectedVehicle:
    """Represents a single detected vehicle instance."""
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    class_id: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class YOLODetector:
    """Production wrapper around YOLO object detection for parking vehicle monitoring."""

    DEFAULT_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, config: Optional[DetectionConfig] = None, model_path: Optional[str] = None):
        self.config = config or DetectionConfig()
        target_path = model_path or self.config.model_path

        # If designated path doesn't exist, try fallback
        if not Path(target_path).exists() and Path(self.config.fallback_model_path).exists():
            logger.warning("Primary weights not found at %s. Falling back to %s", target_path, self.config.fallback_model_path)
            target_path = self.config.fallback_model_path

        if not Path(target_path).exists():
            raise FileNotFoundError(f"YOLO model weights not found at: {target_path}")

        logger.info("Initializing YOLO model from: %s", target_path)
        self.model = YOLO(target_path)
        self.target_classes: Set[str] = set(self.config.target_classes or self.DEFAULT_CLASSES)

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        use_tiling: Optional[bool] = None,
        imgsz: Optional[int] = None
    ) -> Tuple[List[DetectedVehicle], float]:
        """
        Runs vehicle detection on an input image.

        Args:
            image: BGR numpy image.
            conf_threshold: Optional override for confidence threshold.
            use_tiling: Optional override to run sliding-tile inference.
            imgsz: Optional inference resolution.

        Returns:
            Tuple of (List[DetectedVehicle], inference_duration_seconds).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        conf = conf_threshold if conf_threshold is not None else self.config.confidence_threshold
        tiling = use_tiling if use_tiling is not None else self.config.use_tiling
        sz = imgsz or self.config.imgsz

        start_time = time.perf_counter()

        if tiling:
            raw_detections = self._detect_tiled(image, conf=conf, imgsz=sz)
        else:
            raw_detections = self._detect_full_image(image, conf=conf, imgsz=sz)

        # Apply duplicate suppression / NMS across detections
        filtered_dicts = filter_duplicate_detections(
            raw_detections,
            iou_threshold=self.config.nms_iou_threshold
        )

        vehicles = [
            DetectedVehicle(
                box=[int(c) for c in d["box"]],
                confidence=float(d["confidence"]),
                class_name=str(d["class_name"]),
                class_id=int(d["class_id"])
            )
            for d in filtered_dicts
        ]

        duration = time.perf_counter() - start_time
        logger.debug("Detected %d vehicles in %.3f seconds (tiling=%s).", len(vehicles), duration, tiling)
        return vehicles, duration

    def _detect_full_image(self, image: np.ndarray, conf: float, imgsz: int) -> List[Dict[str, Any]]:
        """Single-pass inference on the full image."""
        results = self.model(
            image,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )
        return self._extract_vehicle_boxes(results, offset_x=0, offset_y=0)

    def _detect_tiled(self, image: np.ndarray, conf: float, imgsz: int) -> List[Dict[str, Any]]:
        """Sliding-tile inference for high resolution aerial/panoramic imagery."""
        height, width = image.shape[:2]
        tile_w = min(self.config.tile_width, width)
        tile_h = min(self.config.tile_height, height)
        overlap = self.config.tile_overlap

        step_x = max(1, tile_w - overlap)
        step_y = max(1, tile_h - overlap)

        all_detections: List[Dict[str, Any]] = []

        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                x2 = min(x + tile_w, width)
                y2 = min(y + tile_h, height)
                tile = image[y:y2, x:x2]

                results = self.model(
                    tile,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False
                )

                tile_dets = self._extract_vehicle_boxes(results, offset_x=x, offset_y=y)
                all_detections.extend(tile_dets)

        return all_detections

    def _extract_vehicle_boxes(
        self,
        results: Any,
        offset_x: int = 0,
        offset_y: int = 0
    ) -> List[Dict[str, Any]]:
        """Filters results to vehicle categories and remaps coordinate offsets."""
        detections: List[Dict[str, Any]] = []

        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                class_name = result.names[cls_id]

                if class_name.lower() in self.target_classes:
                    conf = float(box.conf[0].item())
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1 = coords[0] + offset_x
                    y1 = coords[1] + offset_y
                    x2 = coords[2] + offset_x
                    y2 = coords[3] + offset_y

                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_name": class_name,
                        "class_id": cls_id
                    })

        return detections
