"""
Analytics, performance evaluation, and historical metrics for SmartPark AI.
Calculates Accuracy, Precision, Recall, F1-score, and confusion matrices.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


@dataclass
class EvaluationMetrics:
    """Evaluation summary metrics dataclass."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    total_samples: int
    class_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None
) -> EvaluationMetrics:
    """
    Computes standard classification metrics:
    Accuracy, Precision, Recall, F1-Score (macro & weighted), and Confusion Matrix.
    """
    if class_names is None:
        class_names = ["busy", "free"]

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    if len(y_t) == 0:
        return EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=[[0, 0], [0, 0]],
            total_samples=0,
            class_names=class_names
        )

    acc = float(accuracy_score(y_t, y_p))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_t, y_p, labels=[0, 1] if len(class_names) == 2 else None)

    return EvaluationMetrics(
        accuracy=round(acc, 4),
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1_score=round(float(f1), 4),
        confusion_matrix=cm.tolist(),
        total_samples=int(len(y_t)),
        class_names=class_names
    )


def generate_occupancy_analytics(
    history_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregates temporal occupancy records into peak utilization, average dwell,
    and distribution statistics for the analytics dashboard.
    """
    if not history_records:
        return {
            "total_records": 0,
            "average_occupancy_rate": 0.0,
            "peak_occupancy_rate": 0.0,
            "min_occupancy_rate": 0.0,
            "hourly_trends": {},
        }

    rates = [r.get("occupancy_rate", 0.0) for r in history_records]
    return {
        "total_records": len(history_records),
        "average_occupancy_rate": round(float(np.mean(rates)), 2),
        "peak_occupancy_rate": round(float(np.max(rates)), 2),
        "min_occupancy_rate": round(float(np.min(rates)), 2),
        "std_occupancy_rate": round(float(np.std(rates)), 2),
    }
