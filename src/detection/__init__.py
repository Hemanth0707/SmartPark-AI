"""Vehicle detection package for SmartPark AI."""
from .yolo_detector import YOLODetector, DetectedVehicle
from .nms import calculate_iou_single, calculate_intersection_ratio, nms_boxes, filter_duplicate_detections

__all__ = [
    "YOLODetector",
    "DetectedVehicle",
    "calculate_iou_single",
    "calculate_intersection_ratio",
    "nms_boxes",
    "filter_duplicate_detections",
]
