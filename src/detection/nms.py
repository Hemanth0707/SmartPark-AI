"""
Non-Maximum Suppression (NMS) and bounding-box geometry utilities.
Provides vectorized IoU calculations and greedy suppression for multi-tile/multi-scale detections.
"""

from typing import List, Dict, Any, Tuple, Union
import numpy as np
import torch


def calculate_iou_single(
    box1: Union[List[float], Tuple[float, ...], np.ndarray],
    box2: Union[List[float], Tuple[float, ...], np.ndarray]
) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(min(ax1, ax2), min(bx1, bx2))
    inter_y1 = max(min(ay1, ay2), min(by1, by2))
    inter_x2 = min(max(ax1, ax2), max(bx1, bx2))
    inter_y2 = min(max(ay1, ay2), max(by1, by2))

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = abs((ax2 - ax1) * (ay2 - ay1))
    area2 = abs((bx2 - bx1) * (by2 - by1))

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def calculate_intersection_ratio(
    box_inner: Union[List[float], Tuple[float, ...], np.ndarray],
    box_outer: Union[List[float], Tuple[float, ...], np.ndarray]
) -> float:
    """
    Computes the fraction of box_inner covered by box_outer:
    Ratio = Area(Intersection) / Area(box_inner)

    Crucial for parking slots where a vehicle may occupy a large fraction of the slot
    even if the vehicle is larger than the slot (IoU alone might penalize size disparity).
    """
    ix1 = max(min(box_inner[0], box_inner[2]), min(box_outer[0], box_outer[2]))
    iy1 = max(min(box_inner[1], box_inner[3]), min(box_outer[1], box_outer[3]))
    ix2 = min(max(box_inner[0], box_inner[2]), max(box_outer[0], box_outer[2]))
    iy2 = min(max(box_inner[1], box_inner[3]), max(box_outer[1], box_outer[3]))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    inner_area = abs((box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1]))

    if inner_area <= 0.0:
        return 0.0

    return float(inter_area / inner_area)


def nms_boxes(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Performs standard Non-Maximum Suppression over a set of bounding boxes.

    Args:
        boxes: List of [x1, y1, x2, y2]
        scores: List of confidence floats
        iou_threshold: Overlap threshold above which lower-confidence boxes are suppressed.

    Returns:
        List of integer indices of kept boxes.
    """
    if not boxes:
        return []

    np_boxes = np.array(boxes, dtype=np.float32)
    np_scores = np.array(scores, dtype=np.float32)

    # If torch is available and CUDA/CPU tensor is desired, we can use torch or numpy
    x1 = np_boxes[:, 0]
    y1 = np_boxes[:, 1]
    x2 = np_boxes[:, 2]
    y2 = np_boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np_scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def filter_duplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Takes a list of detection dicts (each having 'box' and 'confidence')
    and applies NMS to remove duplicates across overlapping sliding tiles.
    """
    if not detections:
        return []

    boxes = [d["box"] for d in detections]
    scores = [d["confidence"] for d in detections]
    kept_indices = nms_boxes(boxes, scores, iou_threshold=iou_threshold)

    return [detections[idx] for idx in kept_indices]
