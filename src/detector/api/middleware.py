from __future__ import annotations

import json
import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from detector.inference.predictor import Predictor

logger = logging.getLogger(__name__)


class PromptFirewallMiddleware(BaseHTTPMiddleware):
    """Middleware that intercepts requests and blocks prompt injection attempts.

    Acts as a deterministic AI firewall by classifying incoming prompts
    before they reach the downstream LLM.
    """

    def __init__(self, app, predictor: Predictor, prompt_field: str = "text"):
        super().__init__(app)
        self.predictor = predictor
        self.prompt_field = prompt_field

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only intercept POST requests with JSON bodies
        if request.method != "POST":
            return await call_next(request)

        # Skip health and metrics endpoints
        if request.url.path in ("/health", "/metrics", "/docs", "/openapi.json"):
            return await call_next(request)

        try:
            body = await request.body()
            if not body:
                return await call_next(request)

            data = json.loads(body)
            prompt_text = data.get(self.prompt_field)

            if not prompt_text or not isinstance(prompt_text, str):
                return await call_next(request)

        except (json.JSONDecodeError, UnicodeDecodeError):
            return await call_next(request)

        # Classify the prompt
        start = time.perf_counter()
        result = self.predictor.predict(prompt_text)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Firewall check — injection={result.is_injection}, "
            f"confidence={result.confidence:.4f}, latency={latency_ms:.1f}ms, "
            f"path={request.url.path}"
        )

        if result.is_injection:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "prompt_injection_detected",
                    "message": "Your request was blocked by the AI firewall.",
                    "confidence": round(result.confidence, 4),
                },
            )

        return await call_next(request)
