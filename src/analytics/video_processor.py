"""
Video processing pipeline for SmartPark AI.
Processes video frames with configurable inference sub-sampling intervals,
tracks occupancy variations over time, and generates analytics summaries.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Generator, Tuple
import cv2
import numpy as np

from src.occupancy.occupancy_engine import OccupancyEngine, OccupancyResult
from src.utils.visualizer import ParkingVisualizer
from src.utils.logger import get_logger

logger = get_logger("VideoProcessor")


@dataclass
class FrameAnalysisResult:
    """Stores the evaluation summary for an individual processed video frame."""
    frame_number: int
    timestamp_seconds: float
    total_slots: int
    occupied: int
    free: int
    uncertain: int
    occupancy_rate: float
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoProcessor:
    """Manages video file stream decoding, temporal sub-sampling, and occupancy tracking."""

    def __init__(
        self,
        engine: OccupancyEngine,
        visualizer: Optional[ParkingVisualizer] = None
    ):
        self.engine = engine
        self.visualizer = visualizer or ParkingVisualizer()

    def process_video(
        self,
        video_path: str,
        slots: List[List[int]],
        frame_interval: int = 15,
        max_frames: Optional[int] = None,
        mode: str = "hybrid",
        output_video_path: Optional[str] = None
    ) -> Tuple[List[FrameAnalysisResult], Dict[str, Any]]:
        """
        Processes a video file frame-by-frame using temporal subsampling.

        Args:
            video_path: Path to source video file.
            slots: Parking slot coordinate boxes.
            frame_interval: Process 1 frame every `frame_interval` frames.
            max_frames: Optional ceiling on total frames to read.
            mode: 'yolo_only', 'resnet_only', or 'hybrid'.
            output_video_path: Optional path to encode annotated MP4 output.

        Returns:
            Tuple of (List[FrameAnalysisResult], overall_video_summary_dict).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {video_path}")

        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video loaded: %s | Resolution: %dx%d | FPS: %.2f | Total Frames: %d",
            video_path, width, height, fps, total_input_frames
        )

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps / max(1, frame_interval), (width, height))

        history: List[FrameAnalysisResult] = []
        frame_idx = 0
        processed_count = 0
        start_processing_time = time.perf_counter()

        last_result: Optional[OccupancyResult] = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            # Temporal subsampling: execute inference only on target cadence
            if frame_idx % frame_interval == 0:
                f_start = time.perf_counter()
                last_result = self.engine.process_image(frame, slots, mode=mode)
                f_duration = (time.perf_counter() - f_start) * 1000

                timestamp = frame_idx / fps
                record = FrameAnalysisResult(
                    frame_number=frame_idx,
                    timestamp_seconds=round(timestamp, 2),
                    total_slots=last_result.total_slots,
                    occupied=last_result.occupied_slots,
                    free=last_result.free_slots,
                    uncertain=last_result.uncertain_slots,
                    occupancy_rate=last_result.occupancy_rate,
                    processing_time_ms=round(f_duration, 2)
                )
                history.append(record)
                processed_count += 1

                if writer and last_result is not None:
                    annotated_frame = self.visualizer.annotate(frame, last_result)
                    writer.write(annotated_frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        total_time = time.perf_counter() - start_processing_time
        effective_fps = frame_idx / total_time if total_time > 0 else 0.0

        summary = {
            "total_video_frames": frame_idx,
            "processed_frames": processed_count,
            "frame_interval": frame_interval,
            "duration_seconds": round(frame_idx / fps, 2),
            "total_processing_time_seconds": round(total_time, 2),
            "effective_fps": round(effective_fps, 2),
            "average_occupancy_rate": round(float(np.mean([h.occupancy_rate for h in history])), 2) if history else 0.0,
            "peak_occupancy_rate": round(float(np.max([h.occupancy_rate for h in history])), 2) if history else 0.0,
            "min_occupancy_rate": round(float(np.min([h.occupancy_rate for h in history])), 2) if history else 0.0,
        }

        logger.info("Video processing complete: %d frames processed in %.2fs (%.2f FPS)", processed_count, total_time, effective_fps)
        return history, summary
