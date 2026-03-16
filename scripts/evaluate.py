"""CLI entry point for evaluating the prompt injection detector."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector.config import load_config
from detector.data.loader import load_prompt_injection_data
from detector.evaluation.benchmark import run_benchmark
from detector.inference.predictor import PyTorchPredictor
from detector.inference.threshold import ThresholdManager
from detector.model.factory import load_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate the prompt injection detector")
    parser.add_argument(
        "--config", default="config/base.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Override checkpoint path"
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Calibrate threshold on validation set"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint

    # Load model
    classifier = load_model(config.model)
    predictor = PyTorchPredictor(classifier.model, classifier.tokenizer, config.inference)

    # Load data
    dataset = load_prompt_injection_data()

    # Optional threshold calibration
    if args.calibrate:
        import numpy as np

        val_texts = dataset["validation"]["text"]
        val_labels = np.array(dataset["validation"]["label"])

        results = predictor.predict_batch(val_texts)
        val_scores = np.array([r.raw_logits[1] if len(r.raw_logits) > 1 else r.confidence for r in results])

        tm = ThresholdManager(config.inference.threshold)
        optimal = tm.calibrate(val_labels, val_scores)
        config.inference.threshold = optimal
        # Recreate predictor with new threshold
        predictor = PyTorchPredictor(classifier.model, classifier.tokenizer, config.inference)

    # Run benchmark on test set
    report = run_benchmark(predictor, dataset["test"])

    print("\n=== Benchmark Results ===")
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1:        {report['f1']:.4f}")
    if report.get("roc_auc") is not None:
        print(f"ROC-AUC:   {report['roc_auc']:.4f}")
    print(f"\nLatency (p95): {report['latency']['p95_ms']:.1f}ms")
    print(f"Latency (p99): {report['latency']['p99_ms']:.1f}ms")


if __name__ == "__main__":
    main()
