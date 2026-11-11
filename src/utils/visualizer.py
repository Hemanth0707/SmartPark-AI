"""
Visualization utility for SmartPark AI.
Renders color-coded slot overlays, vehicle bounding boxes, and HUD statistics.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import cv2
import numpy as np

from config.config import UIConfig

if TYPE_CHECKING:
    from src.occupancy.occupancy_engine import OccupancyResult, SlotStatusDetail


class ParkingVisualizer:
    """Renders professional overlays on parking lot imagery."""

    # Color mappings in BGR
    COLORS = {
        "FREE": (46, 204, 113),       # Emerald Green
        "OCCUPIED": (60, 76, 231),    # Vivid Coral Red
        "UNCERTAIN": (15, 196, 241),  # Amber Yellow
        "VEHICLE": (219, 152, 52),    # Sky Blue
    }

    def __init__(self, ui_config: Optional[UIConfig] = None):
        self.config = ui_config or UIConfig()

    def annotate(
        self,
        image: np.ndarray,
        result: Any,
        draw_vehicles: bool = True,
        draw_slots: bool = True,
        draw_hud: bool = True,
        draw_conf: bool = True
    ) -> np.ndarray:
        """
        Draws parking slots, vehicles, and status HUD onto a copy of the input image.

        Args:
            image: BGR numpy image.
            result: OccupancyResult from OccupancyEngine.
            draw_vehicles: Whether to outline detected vehicles.
            draw_slots: Whether to draw slot boundary rectangles.
            draw_hud: Whether to render the summary stats bar.
            draw_conf: Whether to annotate slot confidence scores.

        Returns:
            Annotated BGR numpy image.
        """
        canvas = image.copy()
        overlay = canvas.copy()

        # 1. Draw vehicle bounding boxes
        if draw_vehicles and result.detected_vehicles:
            for veh in result.detected_vehicles:
                bx1, by1, bx2, by2 = veh["box"]
                cls_name = veh.get("class_name", "vehicle")
                conf = veh.get("confidence", 0.0)

                cv2.rectangle(
                    canvas,
                    (bx1, by1), (bx2, by2),
                    self.COLORS["VEHICLE"],
                    1,
                    lineType=cv2.LINE_AA
                )
                label = f"{cls_name} {int(conf * 100)}%"
                cv2.putText(
                    canvas,
                    label,
                    (bx1, max(12, by1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    self.COLORS["VEHICLE"],
                    1,
                    lineType=cv2.LINE_AA
                )

        # 2. Draw parking slot rectangles with semi-transparent fill
        if draw_slots:
            for slot in result.slots:
                status = slot.status
                color = self.COLORS.get(status, self.COLORS["FREE"])
                sx1, sy1, sx2, sy2 = slot.box

                # Fill slot in overlay
                cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), color, -1)

                # Outline on canvas
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, 2, lineType=cv2.LINE_AA)

                # Label text
                label = f"S{slot.slot_id:02d}"
                if draw_conf and slot.confidence > 0:
                    label += f" ({int(slot.confidence * 100)}%)"

                # Text backdrop
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                text_y = max(th + 2, sy1 - 4)
                cv2.rectangle(
                    canvas,
                    (sx1, text_y - th - 2),
                    (sx1 + tw + 4, text_y + baseline),
                    (20, 20, 20),
                    -1
                )
                cv2.putText(
                    canvas,
                    label,
                    (sx1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color,
                    1,
                    lineType=cv2.LINE_AA
                )

            # Blend semi-transparent fill: 82% canvas + 18% overlay
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

        # 3. Draw HUD summary bar at top
        if draw_hud:
            h, w = canvas.shape[:2]
            hud_height = 36
            hud_bg = np.zeros((hud_height, w, 3), dtype=np.uint8)
            hud_bg[:] = (20, 24, 33)  # Dark navy/slate

            cv2.addWeighted(canvas[:hud_height, :], 0.15, hud_bg, 0.85, 0, canvas[:hud_height, :])

            hud_text = (
                f"SMARTPARK AI  |  TOTAL: {result.total_slots}  |  "
                f"OCCUPIED: {result.occupied_slots}  |  "
                f"FREE: {result.free_slots}  |  "
                f"UNCERTAIN: {result.uncertain_slots}  |  "
                f"RATE: {result.occupancy_rate:.1f}%  |  "
                f"MODE: {result.mode.upper()}  |  "
                f"LATENCY: {result.timings.get('total_time', 0.0) * 1000:.0f}ms"
            )

            cv2.putText(
                canvas,
                hud_text,
                (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                lineType=cv2.LINE_AA
            )

        return canvas
