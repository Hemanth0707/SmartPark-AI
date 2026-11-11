"""Occupancy determination and slot matching package for SmartPark AI."""
from .slot_matcher import SlotMatcher, MatchResult, is_center_inside, compute_overlap_metrics
from .occupancy_engine import OccupancyEngine, OccupancyResult, SlotStatusDetail

__all__ = [
    "SlotMatcher",
    "MatchResult",
    "is_center_inside",
    "compute_overlap_metrics",
    "OccupancyEngine",
    "OccupancyResult",
    "SlotStatusDetail",
]
