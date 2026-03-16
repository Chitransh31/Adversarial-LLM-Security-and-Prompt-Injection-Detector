from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from detector.data.schema import PredictionResult
from detector.evaluation.metrics import detailed_report
from detector.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def run_benchmark(
    predictor: Predictor,
    test_dataset: Dataset,
    output_path: str | Path = "artifacts/logs/benchmark_report.json",
) -> dict[str, Any]:
    """Run predictor over test set, compute all metrics, measure latency.

    Args:
        predictor: A Predictor instance (PyTorch or ONNX).
        test_dataset: HuggingFace Dataset with 'text' and 'label' columns.
        output_path: Path to save the JSON report.

    Returns:
        Dictionary containing metrics and latency statistics.
    """
    texts = test_dataset["text"]
    labels = np.array(test_dataset["label"])

    # Run predictions and measure latency
    latencies: list[float] = []
    predictions: list[int] = []
    scores: list[float] = []

    for text in texts:
        start = time.perf_counter()
        result: PredictionResult = predictor.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)
        predictions.append(result.label)
        # Score = probability of injection class
        scores.append(result.raw_logits[1] if len(result.raw_logits) > 1 else result.confidence)

    y_pred = np.array(predictions)
    y_scores = np.array(scores)

    # Compute detailed metrics
    report = detailed_report(labels, y_pred, y_scores)

    # Compute latency statistics
    latency_arr = np.array(latencies)
    report["latency"] = {
        "mean_ms": float(np.mean(latency_arr)),
        "median_ms": float(np.median(latency_arr)),
        "p95_ms": float(np.percentile(latency_arr, 95)),
        "p99_ms": float(np.percentile(latency_arr, 99)),
        "min_ms": float(np.min(latency_arr)),
        "max_ms": float(np.max(latency_arr)),
    }
    report["total_samples"] = len(texts)

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Benchmark report saved to {output_path}")
    logger.info(
        f"Results — F1: {report['f1']:.4f}, "
        f"Precision: {report['precision']:.4f}, "
        f"Recall: {report['recall']:.4f}, "
        f"Latency p95: {report['latency']['p95_ms']:.1f}ms"
    )

    return report
