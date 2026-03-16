from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    quantize: bool = False,
) -> Path:
    """Export a fine-tuned model to ONNX format for faster inference.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint directory.
        output_dir: Directory to save the ONNX model.
        quantize: Whether to apply INT8 dynamic quantization.

    Returns:
        Path to the saved ONNX model directory.
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        raise ImportError(
            "ONNX export requires the 'optimum' package. "
            "Install with: pip install 'prompt-injection-detector[onnx]'"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    logger.info(f"Exporting model from {checkpoint_path} to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(checkpoint_path), export=True
    )
    ort_model.save_pretrained(str(output_dir))
    logger.info(f"ONNX model saved to {output_dir}")

    # Optional quantization
    if quantize:
        logger.info("Applying INT8 dynamic quantization...")
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=str(output_dir), quantization_config=qconfig)
        logger.info("Quantized ONNX model saved.")

    return output_dir
