from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptSample:
    text: str
    label: int  # 0 = benign, 1 = injection
    source: str = "unknown"


@dataclass
class PredictionResult:
    label: int
    confidence: float
    is_injection: bool
    raw_logits: list[float] = field(default_factory=list)
