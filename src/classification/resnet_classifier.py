"""
ResNet18 Parking Slot Classifier module for SmartPark AI.
Performs fine-grained binary classification (BUSY vs FREE) on cropped parking slot patches.
"""

from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

from config.config import ClassificationConfig
from src.preprocessing.image_transforms import extract_slot_patch, get_inference_transform, cv2_to_pil
from src.utils.logger import get_logger

logger = get_logger("ResNetClassifier")


def build_resnet18(num_classes: int = 2, pretrained_weights: bool = True) -> nn.Module:
    """Builds ResNet18 architecture with a modified 2-class linear output head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained_weights else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class ResNetClassifier:
    """Production ResNet18 classifier for parking slot occupancy assessment."""

    def __init__(
        self,
        config: Optional[ClassificationConfig] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.config = config or ClassificationConfig()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = get_inference_transform(
            image_size=self.config.image_size,
            mean=self.config.norm_mean,
            std=self.config.norm_std
        )
        self.classes = self.config.classes  # ['busy', 'free']

        # Determine weights path
        target_path = weights_path or self.config.model_path
        if not Path(target_path).exists() and Path(self.config.legacy_model_path).exists():
            target_path = self.config.legacy_model_path

        self.model = build_resnet18(num_classes=len(self.classes), pretrained_weights=False)

        self.is_trained = False
        if Path(target_path).exists():
            try:
                state_dict = torch.load(target_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.is_trained = True
                logger.info("Loaded trained ResNet18 weights from: %s", target_path)
            except Exception as e:
                logger.error("Failed to load weights from %s: %s. Using default architecture.", target_path, e)
        else:
            logger.warning("No checkpoint found at %s. Model is untrained; predictions will be baseline.", target_path)

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_patch(self, patch: np.ndarray) -> Dict[str, Any]:
        """
        Classifies a single cropped parking slot patch.

        Args:
            patch: BGR numpy image patch.

        Returns:
            Dict containing predicted class ('busy' or 'free'), confidence, and softmax probs.
        """
        pil_img = cv2_to_pil(patch)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        busy_idx = self.classes.index("busy") if "busy" in self.classes else 0
        free_idx = self.classes.index("free") if "free" in self.classes else 1

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "prediction": pred_class,
            "status": "OCCUPIED" if pred_class == "busy" else "FREE",
            "confidence": confidence,
            "prob_busy": float(probs[busy_idx]),
            "prob_free": float(probs[free_idx]),
        }

    def predict_slots(
        self,
        image: np.ndarray,
        slots: List[Union[List[int], Tuple[int, int, int, int]]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Batch-extracts and classifies all parking slots from an image.

        Args:
            image: Full parking lot image (BGR numpy array).
            slots: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            Tuple of (List of slot prediction dicts, inference_duration_seconds).
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for slot classification.")
        if not slots:
            return [], 0.0

        start_time = time.perf_counter()

        # Extract and transform all patches in batch
        patch_tensors = []
        for slot in slots:
            patch = extract_slot_patch(image, slot, target_size=self.config.image_size)
            pil_img = cv2_to_pil(patch)
            tensor = self.transform(pil_img)
            patch_tensors.append(tensor)

        batch_tensor = torch.stack(patch_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        busy_idx = self.classes.index("busy") if "busy" in self.classes else 0
        free_idx = self.classes.index("free") if "free" in self.classes else 1

        results = []
        for i, slot in enumerate(slots):
            p_busy = float(probs[i, busy_idx])
            p_free = float(probs[i, free_idx])
            pred_class = "busy" if p_busy >= p_free else "free"
            status = "OCCUPIED" if pred_class == "busy" else "FREE"
            confidence = max(p_busy, p_free)

            # Check if prediction is within uncertain margin (close to 50/50)
            uncertain = abs(p_busy - p_free) < 0.20

            results.append({
                "slot_index": i,
                "slot_box": [int(c) for c in slot],
                "prediction": pred_class,
                "status": "UNCERTAIN" if uncertain else status,
                "raw_status": status,
                "confidence": confidence,
                "prob_busy": p_busy,
                "prob_free": p_free,
                "uncertain": uncertain
            })

        duration = time.perf_counter() - start_time
        logger.debug("Classified %d slots via ResNet18 in %.3f seconds.", len(slots), duration)
        return results, duration
