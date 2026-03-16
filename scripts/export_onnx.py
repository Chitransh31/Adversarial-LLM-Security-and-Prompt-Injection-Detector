"""CLI entry point for exporting the model to ONNX format."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector.model.onnx_export import export_to_onnx


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/best",
        help="Path to the PyTorch checkpoint directory",
    )
    parser.add_argument(
        "--output",
        default="artifacts/onnx",
        help="Output directory for the ONNX model",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Apply INT8 dynamic quantization"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output = export_to_onnx(args.checkpoint, args.output, quantize=args.quantize)
    print(f"\nONNX model exported to {output}")


if __name__ == "__main__":
    main()
