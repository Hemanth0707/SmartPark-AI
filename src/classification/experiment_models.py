"""
Comprehensive Model Experimentation Suite for SmartPark AI.
Evaluates ResNet18, ResNet34, EfficientNet-B0, and MobileNetV3-Large under strict
zero-data-leakage conditions (Slot-Disjoint Camera A Validation -> Untouched Camera B Test Set).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

from src.utils.logger import get_logger

logger = get_logger("ModelExperiments")


class CNRParkDisjointDataset(Dataset):
    """
    Dataset loader for CNRPark-Patches that filters strictly by camera and slot IDs
    to enforce mathematical separation between training and validation without temporal leakage.
    """

    def __init__(
        self,
        root_dir: str,
        camera: str,
        slot_filter_mode: str = "all",  # "train", "val", "all"
        transform: Optional[T.Compose] = None,
        max_samples_per_class: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.camera = camera
        self.transform = transform
        self.classes = ["busy", "free"]
        self.samples: List[Tuple[Path, int]] = []

        for class_idx, class_name in enumerate(self.classes):
            folder = self.root_dir / camera / class_name
            if not folder.exists():
                continue

            all_files = sorted(os.listdir(folder))
            selected = []
            for f in all_files:
                if not f.endswith(".jpg"):
                    continue

                slot_id = int(f.split("_")[2].split(".")[0])

                if slot_filter_mode == "train":
                    # Slots 1..51 where slot_id % 4 != 0 (75% of slots)
                    if slot_id % 4 != 0:
                        selected.append(folder / f)
                elif slot_filter_mode == "val":
                    # Disjoint slots where slot_id % 4 == 0 (25% of slots)
                    if slot_id % 4 == 0:
                        selected.append(folder / f)
                else:  # "all" (for Camera B test set)
                    selected.append(folder / f)

            if max_samples_per_class and len(selected) > max_samples_per_class:
                selected = selected[:max_samples_per_class]

            for img_path in selected:
                self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_advanced_transforms(image_size: Tuple[int, int] = (224, 224)) -> Tuple[T.Compose, T.Compose]:
    """
    Returns domain-invariant training augmentations:
    - Shadow and illumination simulation via wide ColorJitter (compensates for -30 BGR offset between Camera A & B)
    - Mild camera rotation and translation
    - Preserves straight parking stall line geometry without destructive perspective distortion or erasing
    """
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
        T.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def build_model(arch: str = "resnet18", num_classes: int = 2) -> nn.Module:
    """Builds target architecture with frozen early layers and fine-tuned top stage."""
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[7].parameters():
            p.requires_grad = True
        for p in model.features[8].parameters():
            p.requires_grad = True
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)

    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[-2:].parameters():
            p.requires_grad = True
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, num_classes)

    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """Runs deterministic evaluation and returns comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    y_t = np.array(all_labels)
    y_p = np.array(all_preds)

    acc = float(accuracy_score(y_t, y_p))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_t, y_p, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_t, y_p, average="weighted", zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()

    # Per-class breakdown
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_t, y_p, average=None, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weight), 4),
        "weighted_recall": round(float(r_weight), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "confusion_matrix": cm,
        "total_samples": len(y_t),
        "per_class": {
            class_names[0]: {"precision": round(float(p_per[0]), 4), "recall": round(float(r_per[0]), 4), "f1": round(float(f1_per[0]), 4)},
            class_names[1]: {"precision": round(float(p_per[1]), 4), "recall": round(float(r_per[1]), 4), "f1": round(float(f1_per[1]), 4)}
        }
    }


def train_and_evaluate_architecture(
    arch: str,
    epochs: int = 4,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains on Camera A (Slot-Disjoint Train), validates on Camera A (Slot-Disjoint Val),
    and performs the final zero-leakage evaluation on the untouched Camera B Test Set.
    """
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Starting experiment: Architecture=%s | Epochs=%d | LR=%.5f | Device=%s", arch, epochs, learning_rate, device)

    train_tf, val_tf = get_advanced_transforms()

    # 1. Camera A Train Split (39 slots)
    train_ds = CNRParkDisjointDataset("CNRPark-Patches-150x150", camera="A", slot_filter_mode="train", transform=train_tf)
    # 2. Camera A Validation Split (12 disjoint slots - NO LEAKAGE)
    val_ds = CNRParkDisjointDataset("CNRPark-Patches-150x150", camera="A", slot_filter_mode="val", transform=val_tf)
    # 3. Camera B Untouched Final Test Set (All 53 slots - 2000 evaluation sample)
    test_ds = CNRParkDisjointDataset("CNRPark-Patches-150x150", camera="B", slot_filter_mode="all", transform=val_tf, max_samples_per_class=1200)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Split Sizes: Train(A)=%d | Val(A)=%d | Final Test(B)=%d", len(train_ds), len(val_ds), len(test_ds))

    model = build_model(arch=arch, num_classes=2).to(device)

    # Class-weighted loss with label smoothing to prevent overconfidence
    class_counts = [3035, 1684]
    tot = sum(class_counts)
    class_weights = torch.tensor([tot / (2 * c) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_weights = None
    train_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_ds)

        # Evaluate strictly on Camera A Validation Split
        val_metrics = evaluate_dataset(model, val_loader, device, train_ds.classes)
        logger.info(
            "[%s] Epoch %d/%d | Train Loss: %.4f | Val(A) Acc: %.2f%% | Val(A) Macro F1: %.4f",
            arch, epoch + 1, epochs, epoch_loss, val_metrics["accuracy"] * 100, val_metrics["macro_f1"]
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    total_time = round(time.perf_counter() - train_start, 2)
    logger.info("[%s] Training completed in %.1f seconds.", arch, total_time)

    # Load best validation checkpoint
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Save candidate model checkpoint
    model_save_path = PROJECT_ROOT / "models" / f"parking_{arch}_best.pth"
    torch.save(best_weights, str(model_save_path))

    # Evaluate on:
    # 1. Camera A Validation Split (Unseen Slots from Training Scene)
    final_val_metrics = evaluate_dataset(model, val_loader, device, train_ds.classes)

    # 2. Camera B FINAL UNTOUCHED TEST SET (Completely Unseen Camera, Day & Viewpoint)
    final_test_metrics = evaluate_dataset(model, test_loader, device, train_ds.classes)

    logger.info(
        "=== [%s] FINAL RESULTS ===\n"
        "  Camera A Disjoint Val Acc: %.2f%% | Macro F1: %.4f\n"
        "  Camera B Cross-Camera Acc: %.2f%% | Macro F1: %.4f | Weighted F1: %.4f\n"
        "  Camera B Confusion Matrix: %s",
        arch,
        final_val_metrics["accuracy"] * 100, final_val_metrics["macro_f1"],
        final_test_metrics["accuracy"] * 100, final_test_metrics["macro_f1"], final_test_metrics["weighted_f1"],
        final_test_metrics["confusion_matrix"]
    )

    return {
        "architecture": arch,
        "training_time_seconds": total_time,
        "model_path": str(model_save_path),
        "validation_camera_a": final_val_metrics,
        "test_camera_b": final_test_metrics
    }


def run_full_experiment_matrix() -> Dict[str, Any]:
    """Runs all 4 candidate architectures and generates comparative report."""
    architectures = ["resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_large"]
    results = {}

    for arch in architectures:
        res = train_and_evaluate_architecture(arch=arch, epochs=4, batch_size=64, learning_rate=0.0004)
        results[arch] = res

    out_file = PROJECT_ROOT / "outputs" / "model_experiments.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info("All experiments completed! Results saved to %s", out_file)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Experiments for SmartPark AI")
    parser.add_argument("--arch", type=str, default="all", help="Architecture name or 'all'")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs per model")
    args = parser.parse_args()

    if args.arch == "all":
        run_full_experiment_matrix()
    else:
        res = train_and_evaluate_architecture(arch=args.arch, epochs=args.epochs)
        print(json.dumps(res, indent=2))
