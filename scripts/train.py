"""CLI entry point for training the prompt injection detector."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector.config import load_config
from detector.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the prompt injection detector")
    parser.add_argument(
        "--config", default="config/base.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--override", default=None, help="Path to override config YAML file"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config, args.override)
    metrics = train(config)

    # Save final metrics
    output_path = Path("artifacts/logs/final_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete. Metrics saved to {output_path}")
    print(f"Best model saved to {config.training.output_dir}/best")


if __name__ == "__main__":
    main()
