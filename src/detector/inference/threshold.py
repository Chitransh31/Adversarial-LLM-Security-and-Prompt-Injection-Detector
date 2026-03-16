from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class ThresholdManager:
    """Manages and calibrates the classification threshold for optimal F1."""

    def __init__(self, default: float = 0.5):
        self.threshold = default

    def calibrate(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Find the optimal threshold by maximizing F1 on a validation set.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities for the positive class (injection).

        Returns:
            The optimal threshold value.
        """
        best_threshold = self.threshold
        best_f1 = 0.0

        for threshold in np.arange(0.30, 0.96, 0.01):
            predictions = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        self.threshold = best_threshold
        logger.info(f"Calibrated threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
        return best_threshold

    def apply(self, confidence: float) -> bool:
        """Apply the threshold to a confidence score."""
        return confidence >= self.threshold
