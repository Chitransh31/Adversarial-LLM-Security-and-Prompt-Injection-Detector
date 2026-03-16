from __future__ import annotations

import logging

from fastapi import FastAPI

from detector.api.middleware import PromptFirewallMiddleware
from detector.api.routes import router
from detector.config import AppConfig
from detector.inference.predictor import PyTorchPredictor
from detector.model.factory import load_model

logger = logging.getLogger(__name__)


def create_app(config: AppConfig) -> FastAPI:
    """Create and configure the FastAPI application.

    Loads the model, creates a predictor, and wires up routes + middleware.
    """
    app = FastAPI(
        title="Prompt Injection Detector",
        description="Real-time NLP security middleware for detecting adversarial prompt injections",
        version="1.0.0",
    )

    # Load model and create predictor
    logger.info("Loading model for API serving...")
    classifier = load_model(config.model)
    predictor = PyTorchPredictor(classifier.model, classifier.tokenizer, config.inference)
    app.state.predictor = predictor
    logger.info("Model loaded successfully.")

    # Register routes
    app.include_router(router)

    # Add firewall middleware
    app.add_middleware(PromptFirewallMiddleware, predictor=predictor)

    return app
