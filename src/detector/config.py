from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 512
    checkpoint_path: Optional[str] = None


class TrainingConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    output_dir: str = "artifacts/checkpoints"
    logging_steps: int = 10


class InferenceConfig(BaseModel):
    backend: Literal["pytorch", "onnx"] = "pytorch"
    threshold: float = 0.5
    device: str = "cpu"
    batch_size: int = 32


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    server: ServerConfig = ServerConfig()


def load_config(
    config_path: str | Path = "config/base.yaml",
    override_path: str | Path | None = None,
) -> AppConfig:
    """Load configuration from YAML files with optional override."""
    config_path = Path(config_path)
    data: dict = {}

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    if override_path:
        override_path = Path(override_path)
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            data = _deep_merge(data, overrides)

    return AppConfig(**data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
