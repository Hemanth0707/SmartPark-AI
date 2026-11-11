"""
Unified Parking Occupancy Engine for SmartPark AI.
Integrates:
- YOLO Vehicle Detection
- ResNet18 Slot Patch Classification
- Spatial Slot Matching (IoU, Intersection, 1-to-1 Hungarian)
- Decision Fusion: YOLO-only, ResNet18-only, and Hybrid (YOLO + ResNet18)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

from config.config import OccupancyConfig, PROJECT_ROOT
from src.detection.yolo_detector import YOLODetector, DetectedVehicle
from src.classification.resnet_classifier import ResNetClassifier
from src.occupancy.slot_matcher import SlotMatcher, MatchResult
from src.utils.logger import get_logger

logger = get_logger("OccupancyEngine")


@dataclass
class SlotStatusDetail:
    """Detailed state of an individual parking space."""
    slot_id: int
    box: List[int]
    status: str  # "FREE", "OCCUPIED", "UNCERTAIN"
    confidence: float
    decision_mode: str
    yolo_status: Optional[str] = None
    yolo_score: Optional[float] = None
    resnet_status: Optional[str] = None
    resnet_prob_busy: Optional[float] = None
    assigned_vehicle: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OccupancyResult:
    """Comprehensive result of the occupancy engine inference."""
    total_slots: int
    occupied_slots: int
    free_slots: int
    uncertain_slots: int
    occupancy_rate: float
    mode: str
    slots: List[SlotStatusDetail]
    detected_vehicles: List[Dict[str, Any]]
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["slots"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.slots]
        return d


class OccupancyEngine:
    """Master computer vision engine governing parking occupancy analysis."""

    def __init__(
        self,
        config: Optional[OccupancyConfig] = None,
        load_resnet: bool = True,
        load_yolo: bool = True
    ):
        self.config = config or OccupancyConfig()

        self.yolo_detector: Optional[YOLODetector] = None
        if load_yolo:
            try:
                self.yolo_detector = YOLODetector(config=self.config.detection)
            except Exception as e:
                logger.warning("Failed to initialize YOLODetector: %s", e)

        self.resnet_classifier: Optional[ResNetClassifier] = None
        if load_resnet:
            try:
                self.resnet_classifier = ResNetClassifier(config=self.config.classification)
            except Exception as e:
                logger.warning("Failed to initialize ResNetClassifier: %s", e)

        self.slot_matcher = SlotMatcher(config=self.config.matching)

    def process_image(
        self,
        image: np.ndarray,
        slots: List[List[int]],
        mode: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> OccupancyResult:
        """
        Executes the end-to-end occupancy detection pipeline on an image.

        Args:
            image: BGR numpy image.
            slots: List of 2D bounding boxes [[x1, y1, x2, y2], ...].
            mode: 'yolo_only', 'resnet_only', or 'hybrid'.
            conf_threshold: Optional confidence threshold override.

        Returns:
            OccupancyResult containing statistics, slot statuses, and timings.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is invalid or empty.")
        if not slots:
            raise ValueError("Parking slots list cannot be empty.")

        active_mode = (mode or self.config.mode).lower()
        pipeline_start = time.perf_counter()

        timings = {
            "yolo_time": 0.0,
            "resnet_time": 0.0,
            "matching_time": 0.0,
            "total_time": 0.0
        }

        detected_vehicles: List[DetectedVehicle] = []
        slot_match_results: List[MatchResult] = []
        resnet_results: List[Dict[str, Any]] = []

        # 1. YOLO Vehicle Detection
        if active_mode in ("yolo_only", "hybrid"):
            if self.yolo_detector is None:
                raise RuntimeError("YOLO detector is requested but not loaded.")
            detected_vehicles, yolo_duration = self.yolo_detector.detect(
                image,
                conf_threshold=conf_threshold
            )
            timings["yolo_time"] = round(yolo_duration, 4)

            # Match vehicles to slots
            match_start = time.perf_counter()
            slot_match_results = self.slot_matcher.match(slots, detected_vehicles)
            timings["matching_time"] = round(time.perf_counter() - match_start, 4)

        # 2. ResNet18 Patch Classification
        if active_mode in ("resnet_only", "hybrid"):
            if self.resnet_classifier is None:
                raise RuntimeError("ResNet classifier is requested but not loaded.")
            resnet_results, resnet_duration = self.resnet_classifier.predict_slots(
                image, slots
            )
            timings["resnet_time"] = round(resnet_duration, 4)

        # 3. Decision Fusion Engine
        slot_details: List[SlotStatusDetail] = []

        for i, slot in enumerate(slots):
            y_res = slot_match_results[i] if slot_match_results else None
            r_res = resnet_results[i] if resnet_results else None

            if active_mode == "yolo_only":
                final_status = y_res.status if y_res else "FREE"
                final_conf = y_res.confidence if y_res else 0.0
                detail = SlotStatusDetail(
                    slot_id=i + 1,
                    box=slot,
                    status=final_status,
                    confidence=round(final_conf, 3),
                    decision_mode="yolo_only",
                    yolo_status=y_res.status if y_res else None,
                    yolo_score=round(y_res.slot_coverage, 3) if y_res else 0.0,
                    assigned_vehicle=y_res.assigned_vehicle.to_dict() if y_res and y_res.assigned_vehicle else None
                )

            elif active_mode == "resnet_only":
                final_status = r_res["status"] if r_res else "FREE"
                final_conf = r_res["confidence"] if r_res else 0.0
                detail = SlotStatusDetail(
                    slot_id=i + 1,
                    box=slot,
                    status=final_status,
                    confidence=round(final_conf, 3),
                    decision_mode="resnet_only",
                    resnet_status=r_res["status"] if r_res else None,
                    resnet_prob_busy=round(r_res["prob_busy"], 3) if r_res else 0.0
                )

            else:  # 'hybrid'
                detail = self._fuse_hybrid_decision(i + 1, slot, y_res, r_res)

            slot_details.append(detail)

        # Calculate summary statistics
        total = len(slot_details)
        occupied = sum(1 for s in slot_details if s.status == "OCCUPIED")
        free = sum(1 for s in slot_details if s.status == "FREE")
        uncertain = sum(1 for s in slot_details if s.status == "UNCERTAIN")
        rate = round((occupied / total) * 100, 2) if total > 0 else 0.0

        timings["total_time"] = round(time.perf_counter() - pipeline_start, 4)

        return OccupancyResult(
            total_slots=total,
            occupied_slots=occupied,
            free_slots=free,
            uncertain_slots=uncertain,
            occupancy_rate=rate,
            mode=active_mode,
            slots=slot_details,
            detected_vehicles=[v.to_dict() for v in detected_vehicles],
            timings=timings
        )

    def _fuse_hybrid_decision(
        self,
        slot_id: int,
        slot_box: List[int],
        yolo_match: Optional[MatchResult],
        resnet_pred: Optional[Dict[str, Any]]
    ) -> SlotStatusDetail:
        """
        Intelligent Bayesian / weighted consensus fusion between YOLO spatial detection
        and ResNet18 visual patch classification.
        """
        y_status = yolo_match.status if yolo_match else "FREE"
        y_cov = yolo_match.slot_coverage if yolo_match else 0.0
        y_conf = yolo_match.confidence if yolo_match else 0.0
        assigned_veh = yolo_match.assigned_vehicle.to_dict() if (yolo_match and yolo_match.assigned_vehicle) else None

        r_busy_prob = resnet_pred["prob_busy"] if resnet_pred else 0.0
        r_conf = resnet_pred["confidence"] if resnet_pred else 0.5
        r_status = resnet_pred["status"] if resnet_pred else "FREE"

        # Case 1: Strong Consensus (Both Agree)
        if y_status == "OCCUPIED" and r_busy_prob >= 0.55:
            final_status = "OCCUPIED"
            confidence = min(1.0, (y_conf * 0.5 + r_conf * 0.5) * 1.1)
        elif y_status == "FREE" and r_busy_prob < 0.40:
            final_status = "FREE"
            confidence = min(1.0, (y_conf * 0.5 + (1.0 - r_busy_prob) * 0.5) * 1.1)

        # Case 2: Divergence / Conflicting signals
        elif y_status == "OCCUPIED" and r_busy_prob < 0.40:
            # YOLO detected vehicle, but ResNet sees patch as free (e.g. slight boundary overlap or shadow)
            if y_cov >= 0.45:
                # Substantial vehicle presence confirmed by YOLO
                final_status = "OCCUPIED"
                confidence = y_conf * 0.75
            else:
                final_status = "UNCERTAIN"
                confidence = 0.50

        elif y_status == "FREE" and r_busy_prob >= 0.65:
            # ResNet is highly confident patch is busy (e.g., YOLO missed small occluded vehicle)
            if self.resnet_classifier and self.resnet_classifier.is_trained:
                final_status = "OCCUPIED"
                confidence = r_conf * 0.80
            else:
                final_status = "UNCERTAIN"
                confidence = 0.50

        else:
            # In-between or ambiguous zone
            if y_cov >= 0.25 or r_busy_prob >= 0.50:
                final_status = "OCCUPIED" if (y_cov >= 0.30 and r_busy_prob >= 0.45) else "UNCERTAIN"
            else:
                final_status = "FREE"
            confidence = 0.60

        return SlotStatusDetail(
            slot_id=slot_id,
            box=slot_box,
            status=final_status,
            confidence=round(confidence, 3),
            decision_mode="hybrid",
            yolo_status=y_status,
            yolo_score=round(y_cov, 3),
            resnet_status=r_status,
            resnet_prob_busy=round(r_busy_prob, 3),
            assigned_vehicle=assigned_veh
        )
