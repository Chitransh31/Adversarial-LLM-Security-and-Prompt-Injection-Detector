from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from detector.config import InferenceConfig
from detector.data.schema import PredictionResult


@runtime_checkable
class Predictor(Protocol):
    """Protocol for prompt injection predictors (Strategy Pattern)."""

    def predict(self, text: str) -> PredictionResult: ...

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]: ...


class PyTorchPredictor:
    """Predictor using a PyTorch model for inference."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        """Classify a single text input."""
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Classify a batch of text inputs."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encodings)
        probabilities = F.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(texts)):
            probs = probabilities[i].cpu().tolist()
            injection_prob = probs[1]
            is_injection = injection_prob >= self.config.threshold
            label = 1 if is_injection else 0

            results.append(
                PredictionResult(
                    label=label,
                    confidence=injection_prob if is_injection else probs[0],
                    is_injection=is_injection,
                    raw_logits=outputs.logits[i].cpu().tolist(),
                )
            )

        return results
