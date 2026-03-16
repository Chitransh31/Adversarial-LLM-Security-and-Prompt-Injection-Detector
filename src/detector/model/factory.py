from __future__ import annotations

from detector.config import ModelConfig
from detector.model.classifier import PromptInjectionClassifier


def load_model(config: ModelConfig) -> PromptInjectionClassifier:
    """Load a PromptInjectionClassifier based on configuration.

    If checkpoint_path is set, loads from local checkpoint.
    Otherwise, loads fresh pretrained weights from HuggingFace Hub.
    """
    if config.checkpoint_path:
        return PromptInjectionClassifier.from_checkpoint(config.checkpoint_path)
    return PromptInjectionClassifier.from_pretrained(config.name, config.num_labels)
