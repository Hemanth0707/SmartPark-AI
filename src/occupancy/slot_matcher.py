"""
Parking Slot Matching and Spatial Allocation module for SmartPark AI.
Provides:
- Center-point containment
- Bounding box intersection ratio
- Intersection over Union (IoU)
- Globally optimal 1-to-1 bipartite vehicle-to-slot matching (Hungarian algorithm)
- Ambiguity and uncertainty handling
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from config.config import SlotMatchingConfig
from src.detection.nms import calculate_iou_single, calculate_intersection_ratio
from src.detection.yolo_detector import DetectedVehicle
from src.utils.logger import get_logger

logger = get_logger("SlotMatcher")


@dataclass
class MatchResult:
    """Represents the matching result for a single parking slot."""
    slot_index: int
    slot_box: List[int]
    status: str  # "FREE", "OCCUPIED", "UNCERTAIN"
    assigned_vehicle: Optional[DetectedVehicle] = None
    iou: float = 0.0
    slot_coverage: float = 0.0
    vehicle_coverage: float = 0.0
    center_contained: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.assigned_vehicle:
            d["assigned_vehicle"] = self.assigned_vehicle.to_dict()
        return d


def is_center_inside(
    vehicle_box: Union[List[int], Tuple[int, int, int, int]],
    slot_box: Union[List[int], Tuple[int, int, int, int]]
) -> bool:
    """Checks whether the center coordinate of a vehicle bounding box is inside the slot."""
    vx1, vy1, vx2, vy2 = vehicle_box
    sx1, sy1, sx2, sy2 = slot_box

    cx = (vx1 + vx2) / 2.0
    cy = (vy1 + vy2) / 2.0

    s_min_x, s_max_x = min(sx1, sx2), max(sx1, sx2)
    s_min_y, s_max_y = min(sy1, sy2), max(sy1, sy2)

    return (s_min_x <= cx <= s_max_x) and (s_min_y <= cy <= s_max_y)


def compute_overlap_metrics(
    vehicle_box: List[int],
    slot_box: List[int]
) -> Tuple[float, float, float, bool]:
    """
    Computes comprehensive spatial relationship metrics:
    Returns (iou, slot_coverage, vehicle_coverage, center_contained).
    """
    iou = calculate_iou_single(vehicle_box, slot_box)
    slot_cov = calculate_intersection_ratio(slot_box, vehicle_box)
    veh_cov = calculate_intersection_ratio(vehicle_box, slot_box)
    c_inside = is_center_inside(vehicle_box, slot_box)
    return iou, slot_cov, veh_cov, c_inside


class SlotMatcher:
    """
    Solves spatial association between detected vehicle bounding boxes
    and pre-defined parking slot geometries.
    """

    def __init__(self, config: Optional[SlotMatchingConfig] = None):
        self.config = config or SlotMatchingConfig()

    def match(
        self,
        slots: List[List[int]],
        vehicles: List[DetectedVehicle]
    ) -> List[MatchResult]:
        """
        Associates vehicles to parking slots using 1-to-1 optimal assignment.

        Args:
            slots: List of [x1, y1, x2, y2] slot boundaries.
            vehicles: List of DetectedVehicle instances.

        Returns:
            List of MatchResult for each slot in input order.
        """
        num_slots = len(slots)
        num_vehicles = len(vehicles)

        if num_slots == 0:
            return []

        # If no vehicles detected, all slots are FREE
        if num_vehicles == 0:
            return [
                MatchResult(
                    slot_index=i,
                    slot_box=slots[i],
                    status="FREE",
                    confidence=1.0
                )
                for i in range(num_slots)
            ]

        # Compute cost / affinity matrix between slots (rows) and vehicles (cols)
        # Score blends slot coverage and IoU
        affinity_matrix = np.zeros((num_slots, num_vehicles), dtype=np.float32)
        metrics_grid = []

        for s_idx, slot in enumerate(slots):
            row_metrics = []
            for v_idx, veh in enumerate(vehicles):
                iou, slot_cov, veh_cov, c_in = compute_overlap_metrics(veh.box, slot)
                row_metrics.append((iou, slot_cov, veh_cov, c_in))

                if self.config.method == "center":
                    score = 1.0 if c_in else 0.0
                elif self.config.method == "iou":
                    score = iou
                elif self.config.method == "intersection":
                    score = slot_cov
                else:  # 'hybrid_overlap'
                    # Weighted combination: slot coverage is primary for parking slots,
                    # center containment adds a boost, IoU anchors overall alignment
                    score = slot_cov * 0.65 + iou * 0.25 + (0.10 if c_in else 0.0)

                affinity_matrix[s_idx, v_idx] = score
            metrics_grid.append(row_metrics)

        # 1-to-1 vehicle-slot assignment to prevent one vehicle occupying multiple slots
        assigned_pairs = {}
        if self.config.assignment_strategy == "bipartite":
            # Linear Sum Assignment minimizes total cost; use -affinity
            row_ind, col_ind = linear_sum_assignment(-affinity_matrix)
            for r, c in zip(row_ind, col_ind):
                assigned_pairs[r] = c
        else:
            # Greedy matching
            matched_v = set()
            flat_indices = np.argsort(affinity_matrix.ravel())[::-1]
            for idx in flat_indices:
                r = idx // num_vehicles
                c = idx % num_vehicles
                if r not in assigned_pairs and c not in matched_v and affinity_matrix[r, c] > 0:
                    assigned_pairs[r] = c
                    matched_v.add(c)

        # Resolve slot statuses
        results: List[MatchResult] = []
        for s_idx in range(num_slots):
            slot = slots[s_idx]
            v_idx = assigned_pairs.get(s_idx, None)

            if v_idx is not None and affinity_matrix[s_idx, v_idx] > 0.01:
                veh = vehicles[v_idx]
                iou, slot_cov, veh_cov, c_in = metrics_grid[s_idx][v_idx]
                score = affinity_matrix[s_idx, v_idx]

                # Decision thresholds
                min_occ = self.config.min_intersection_occupied
                unc_lower = self.config.uncertain_lower_bound

                if slot_cov >= min_occ or iou >= self.config.min_iou_occupied or (c_in and slot_cov >= 0.18):
                    status = "OCCUPIED"
                    conf = min(1.0, veh.confidence * (0.5 + 0.5 * min(1.0, slot_cov / 0.5)))
                elif slot_cov >= unc_lower or (c_in and slot_cov < 0.18):
                    status = "UNCERTAIN"
                    conf = veh.confidence * 0.5
                else:
                    status = "FREE"
                    conf = 1.0 - score
                    veh = None  # Did not meet minimum occupancy criteria

                results.append(MatchResult(
                    slot_index=s_idx,
                    slot_box=slot,
                    status=status,
                    assigned_vehicle=veh,
                    iou=round(float(iou), 3),
                    slot_coverage=round(float(slot_cov), 3),
                    vehicle_coverage=round(float(veh_cov), 3),
                    center_contained=c_in,
                    confidence=round(float(conf), 3)
                ))
            else:
                results.append(MatchResult(
                    slot_index=s_idx,
                    slot_box=slot,
                    status="FREE",
                    assigned_vehicle=None,
                    iou=0.0,
                    slot_coverage=0.0,
                    vehicle_coverage=0.0,
                    center_contained=False,
                    confidence=1.0
                ))

        return results
