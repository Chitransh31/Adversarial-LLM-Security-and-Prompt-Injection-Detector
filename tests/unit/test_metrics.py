"""Tests for evaluation metrics."""

import numpy as np
import pytest

from detector.evaluation.metrics import compute_metrics, detailed_report


class TestComputeMetrics:
    def test_perfect_predictions(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]])
        labels = np.array([0, 1, 0])

        metrics = compute_metrics((logits, labels))
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        logits = np.array([[-5.0, 5.0], [5.0, -5.0]])
        labels = np.array([0, 1])

        metrics = compute_metrics((logits, labels))
        assert metrics["accuracy"] == 0.0

    def test_mixed_predictions(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0]])
        labels = np.array([0, 1, 0, 1])  # 2 correct, 2 wrong

        metrics = compute_metrics((logits, labels))
        assert metrics["accuracy"] == 0.5


class TestDetailedReport:
    def test_report_contains_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])

        report = detailed_report(y_true, y_pred)
        assert "accuracy" in report
        assert "precision" in report
        assert "recall" in report
        assert "f1" in report
        assert "confusion_matrix" in report
        assert "classification_report" in report

    def test_report_with_scores(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8])

        report = detailed_report(y_true, y_pred, y_scores)
        assert "roc_auc" in report
        assert report["roc_auc"] is not None

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        report = detailed_report(y_true, y_pred)
        cm = report["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
