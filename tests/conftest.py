"""Shared test fixtures."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector.config import AppConfig, InferenceConfig, ModelConfig
from detector.data.schema import PredictionResult


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_prompts():
    """Load sample prompts from fixtures."""
    with open(FIXTURES_DIR / "sample_prompts.json") as f:
        return json.load(f)


@pytest.fixture
def app_config():
    """Default test configuration."""
    return AppConfig(
        model=ModelConfig(name="distilbert-base-uncased", max_length=128),
        inference=InferenceConfig(device="cpu", threshold=0.5),
    )


@pytest.fixture
def mock_predictor():
    """Mock predictor that returns configurable results."""
    predictor = MagicMock()

    def predict_side_effect(text):
        # Simple heuristic: if text contains "ignore" or "override", flag as injection
        is_injection = any(
            kw in text.lower() for kw in ["ignore", "override", "forget", "disregard", "system"]
        )
        return PredictionResult(
            label=1 if is_injection else 0,
            confidence=0.95 if is_injection else 0.92,
            is_injection=is_injection,
            raw_logits=[0.05, 0.95] if is_injection else [0.92, 0.08],
        )

    predictor.predict = MagicMock(side_effect=predict_side_effect)
    predictor.predict_batch = MagicMock(
        side_effect=lambda texts: [predict_side_effect(t) for t in texts]
    )
    return predictor
