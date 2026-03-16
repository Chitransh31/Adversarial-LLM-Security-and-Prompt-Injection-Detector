from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from detector.api.dependencies import get_predictor
from detector.api.schemas import HealthResponse, PredictRequest, PredictResponse
from detector.inference.predictor import Predictor

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, predictor: Predictor = Depends(get_predictor)):
    """Classify a text input as benign or prompt injection."""
    start = time.perf_counter()
    result = predictor.predict(request.text)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        is_injection=result.is_injection,
        confidence=result.confidence,
        label="INJECTION" if result.is_injection else "BENIGN",
        latency_ms=round(latency_ms, 2),
    )


@router.get("/health", response_model=HealthResponse)
def health(predictor: Predictor = Depends(get_predictor)):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
    )
