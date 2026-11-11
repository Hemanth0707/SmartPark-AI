"""
ResNet18 Training & Evaluation Pipeline for SmartPark AI.
Features:
- Configurable frozen-backbone vs end-to-end fine-tuning
- Data augmentation pipeline
- Early stopping & Learning Rate scheduling
- Comprehensive metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- Checkpoint persistence and training history logging
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models

from config.config import PROJECT_ROOT, ClassificationConfig
from src.preprocessing.image_transforms import get_training_transforms
from src.analytics.metrics import calculate_classification_metrics
from src.utils.logger import get_logger

logger = get_logger("TrainClassifier")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100 if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[list] = None
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates the model on validation/test set and returns loss & metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    total = len(all_labels)
    val_loss = running_loss / total if total > 0 else 0.0
    metrics = calculate_classification_metrics(all_labels, all_preds, class_names=class_names)

    return val_loss, metrics.to_dict()


def train_resnet18(
    data_dir: str = "dataset",
    output_model_path: str = "models/parking_classifier.pth",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    fine_tune: bool = False,
    patience: int = 3,
    device_name: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main training function for ResNet18 parking slot classifier.

    Args:
        data_dir: Root dataset folder containing train/ and val/ subdirectories.
        output_model_path: Destination path for trained weights.
        epochs: Number of training epochs.
        batch_size: DataLoader mini-batch size.
        learning_rate: Initial learning rate.
        fine_tune: If True, trains backbone layers alongside FC head.
        patience: Early stopping patience epochs.
        device_name: Target device ('cpu', 'cuda', etc.).

    Returns:
        Dictionary containing training history and final evaluation metrics.
    """
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Initiating ResNet18 training on device: %s | Fine-tuning: %s", device, fine_tune)

    train_tf, val_tf = get_training_transforms()

    train_path = Path(data_dir) / "train"
    val_path = Path(data_dir) / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Dataset folders not found at: {train_path} or {val_path}")

    train_dataset = datasets.ImageFolder(str(train_path), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(val_path), transform=val_tf)

    if max_train_samples and max_train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:max_train_samples].tolist()
        train_data_to_use = torch.utils.data.Subset(train_dataset, indices)
    else:
        train_data_to_use = train_dataset

    if max_val_samples and max_val_samples < len(val_dataset):
        v_indices = torch.randperm(len(val_dataset))[:max_val_samples].tolist()
        val_data_to_use = torch.utils.data.Subset(val_dataset, v_indices)
    else:
        val_data_to_use = val_dataset

    train_loader = DataLoader(train_data_to_use, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data_to_use, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Classes detected: %s", train_dataset.classes)
    logger.info("Train images: %d | Validation images: %d", len(train_dataset), len(val_dataset))

    # Initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if not fine_tune:
        # Freeze backbone parameters
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        trainable_params = model.fc.parameters()
    else:
        # Fine-tune: unfreeze layer4 and FC head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        trainable_params = [
            {"params": model.layer4.parameters(), "lr": learning_rate * 0.1},
            {"params": model.fc.parameters(), "lr": learning_rate}
        ]

    model = model.to(device)

    # Class weights to handle class imbalance (busy has ~2x samples of free)
    class_counts = [len(list((train_path / c).glob("*.jpg"))) for c in train_dataset.classes]
    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [total_samples / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    out_file = Path(output_model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = evaluate_model(model, val_loader, criterion, device, train_dataset.classes)

        scheduler.step(v_loss)

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 2))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_metrics["accuracy"] * 100, 2))
        history["val_f1"].append(round(v_metrics["f1_score"], 4))

        epoch_duration = time.perf_counter() - epoch_start
        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%, F1: %.4f",
            epoch + 1, epochs, epoch_duration, t_loss, t_acc, v_loss, v_metrics["accuracy"] * 100, v_metrics["f1_score"]
        )

        # Checkpoint if best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = model.state_dict().copy()
            epochs_without_improvement = 0
            # Save checkpoint
            torch.save(best_weights, str(out_file))
            # Also save legacy path for backward compatibility
            torch.save(best_weights, str(PROJECT_ROOT / "parking_classifier.pth"))
            logger.info("Saved new best model checkpoint to %s", out_file)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epochs_without_improvement)
                break

    total_time = time.perf_counter() - start_time
    logger.info("Training completed in %.2f seconds.", total_time)

    # Load best weights for final evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)

    final_val_loss, final_metrics = evaluate_model(model, val_loader, criterion, device, train_dataset.classes)

    summary = {
        "mode": "fine_tuning" if fine_tune else "frozen_backbone",
        "epochs_trained": len(history["train_loss"]),
        "total_training_time_seconds": round(total_time, 2),
        "best_val_loss": round(best_val_loss, 4),
        "final_metrics": final_metrics,
        "history": history
    }

    # Save metrics JSON
    metrics_path = out_file.parent / f"training_metrics_{'finetune' if fine_tune else 'frozen'}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info("Final Validation Metrics: %s", json.dumps(final_metrics, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 Parking Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze layer4 for fine-tuning")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--out", type=str, default="models/parking_classifier.pth", help="Output model path")
    args = parser.parse_args()

    train_resnet18(
        data_dir=args.data_dir,
        output_model_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune=args.fine_tune
    )
