from __future__ import annotations

from fastapi import Request

from detector.inference.predictor import Predictor


def get_predictor(request: Request) -> Predictor:
    """FastAPI dependency to retrieve the predictor singleton from app state."""
    return request.app.state.predictor
