"""Tests for threshold calibration."""

import numpy as np
import pytest

from detector.inference.threshold import ThresholdManager


class TestThresholdManager:
    def test_default_threshold(self):
        tm = ThresholdManager(default=0.5)
        assert tm.threshold == 0.5

    def test_apply_above_threshold(self):
        tm = ThresholdManager(default=0.5)
        assert tm.apply(0.7) is True

    def test_apply_below_threshold(self):
        tm = ThresholdManager(default=0.5)
        assert tm.apply(0.3) is False

    def test_apply_at_threshold(self):
        tm = ThresholdManager(default=0.5)
        assert tm.apply(0.5) is True

    def test_calibrate_finds_optimal(self):
        tm = ThresholdManager(default=0.5)

        # Perfect separation: scores above 0.6 are all positive
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9])

        optimal = tm.calibrate(y_true, y_scores)
        assert 0.3 <= optimal <= 0.7
        assert tm.threshold == optimal

        # With optimal threshold, predictions should be perfect
        predictions = (y_scores >= optimal).astype(int)
        assert np.array_equal(predictions, y_true)

    def test_calibrate_with_noisy_data(self):
        tm = ThresholdManager(default=0.5)

        # Some overlap between classes
        y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        y_scores = np.array([0.2, 0.3, 0.45, 0.4, 0.55, 0.7, 0.8, 0.9])

        optimal = tm.calibrate(y_true, y_scores)
        assert 0.3 <= optimal <= 0.95
