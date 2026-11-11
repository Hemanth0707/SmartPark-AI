"""
Central configuration module for SmartPark AI.
Manages all hyperparameters, paths, model thresholds, and runtime settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DetectionConfig:
    """YOLO detector configuration settings."""
    model_path: str = str(PROJECT_ROOT / "yolo26s.pt")
    fallback_model_path: str = str(PROJECT_ROOT / "yolo26n.pt")
    confidence_threshold: float = 0.15
    nms_iou_threshold: float = 0.45
    imgsz: int = 640
    target_classes: List[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle"
    ])
    use_tiling: bool = True
    tile_width: int = 306
    tile_height: int = 208
    tile_overlap: int = 60


@dataclass
class ClassificationConfig:
    """ResNet18 patch classifier configuration settings."""
    model_path: str = str(PROJECT_ROOT / "models" / "parking_classifier.pth")
    legacy_model_path: str = str(PROJECT_ROOT / "parking_classifier.pth")
    image_size: Tuple[int, int] = (224, 224)
    classes: List[str] = field(default_factory=lambda: ["busy", "free"])
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    early_stopping_patience: int = 3


@dataclass
class SlotMatchingConfig:
    """Parking slot matching and spatial assignment configuration."""
    method: str = "hybrid_overlap"  # 'iou', 'intersection', 'center', 'hybrid_overlap'
    min_iou_occupied: float = 0.20
    min_intersection_occupied: float = 0.30
    uncertain_lower_bound: float = 0.15
    uncertain_upper_bound: float = 0.30
    assignment_strategy: str = "bipartite"  # 'bipartite' (Hungarian) or 'greedy'


@dataclass
class OccupancyConfig:
    """Overall occupancy decision engine settings."""
    mode: str = "hybrid"  # 'yolo_only', 'resnet_only', 'hybrid'
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    matching: SlotMatchingConfig = field(default_factory=SlotMatchingConfig)
    uncertain_threshold_resnet: float = 0.15  # Distance from 0.5 where prediction is considered uncertain
    hybrid_yolo_weight: float = 0.60
    hybrid_resnet_weight: float = 0.40


@dataclass
class UIConfig:
    """Dashboard UI styling and color themes."""
    color_free: Tuple[int, int, int] = (46, 204, 113)      # Bright Green (BGR: 113, 204, 46)
    color_occupied: Tuple[int, int, int] = (231, 76, 60)   # Vivid Red (BGR: 60, 76, 231)
    color_uncertain: Tuple[int, int, int] = (241, 196, 15) # Warm Yellow (BGR: 15, 196, 241)
    color_box_vehicle: Tuple[int, int, int] = (52, 152, 219) # Electric Blue
    theme: str = "dark"
