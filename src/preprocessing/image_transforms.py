"""
Image preprocessing and transformation pipeline for SmartPark AI.
Handles slot patch extraction, bounding box clipping, and PyTorch image transforms.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def clip_box_to_image(
    box: Union[List[int], Tuple[int, int, int, int]],
    image_shape: Tuple[int, int, ...]
) -> Tuple[int, int, int, int]:
    """
    Clips and orders bounding box coordinates to ensure they reside strictly within image boundaries.

    Args:
        box: [x1, y1, x2, y2]
        image_shape: (height, width, ...)

    Returns:
        (x1, y1, x2, y2) validated and bounded.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box

    x_min = max(0, min(int(min(x1, x2)), w - 1))
    x_max = max(0, min(int(max(x1, x2)), w))
    y_min = max(0, min(int(min(y1, y2)), h - 1))
    y_max = max(0, min(int(max(y1, y2)), h))

    # Guarantee non-zero dimensions
    if x_max <= x_min:
        x_max = min(w, x_min + 1)
    if y_max <= y_min:
        y_max = min(h, y_min + 1)

    return x_min, y_min, x_max, y_max


def extract_slot_patch(
    image: np.ndarray,
    slot_box: Union[List[int], Tuple[int, int, int, int]],
    target_size: Optional[Tuple[int, int]] = (224, 224)
) -> np.ndarray:
    """
    Safely extracts a region of interest (parking slot patch) from an image.

    Args:
        image: Source image as BGR numpy array.
        slot_box: [x1, y1, x2, y2] coordinates.
        target_size: Optional (width, height) to resize patch to.

    Returns:
        Cropped and optionally resized image patch as BGR numpy array.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image supplied to extract_slot_patch.")

    x1, y1, x2, y2 = clip_box_to_image(slot_box, image.shape)
    patch = image[y1:y2, x1:x2]

    if patch.size == 0:
        # Fallback to 1x1 black patch if crop is invalid
        patch = np.zeros((max(1, y2 - y1), max(1, x2 - x1), 3), dtype=np.uint8)

    if target_size is not None and (patch.shape[1], patch.shape[0]) != target_size:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)

    return patch


def get_inference_transform(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """Returns the standard PyTorch normalization transform for ResNet inference."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_training_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data augmented training transforms and deterministic validation transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Converts a BGR OpenCV image to a RGB PIL Image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
