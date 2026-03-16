from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class MetricsLoggerCallback(TrainerCallback):
    """Logs evaluation metrics to a JSONL file for later analysis."""

    def __init__(self, log_path: str | Path = "artifacts/logs/training_metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ):
        if metrics is None:
            return

        entry = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            **metrics,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Epoch {state.epoch:.0f} metrics: {metrics}")
