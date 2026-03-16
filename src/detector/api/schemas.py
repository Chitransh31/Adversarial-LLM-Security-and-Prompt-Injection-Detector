from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    is_injection: bool
    confidence: float
    label: str  # "INJECTION" or "BENIGN"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
