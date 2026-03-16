from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute metrics compatible with HuggingFace Trainer's compute_metrics.

    Args:
        eval_pred: Tuple of (logits, labels) from Trainer evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def detailed_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate a detailed evaluation report with per-class metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_scores: Prediction probabilities for the positive class (optional).
    """
    report: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["BENIGN", "INJECTION"], output_dict=True
        ),
    }

    if y_scores is not None:
        try:
            report["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            report["roc_auc"] = None

    return report
