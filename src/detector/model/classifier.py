from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class PromptInjectionClassifier:
    """Thin wrapper coupling a sequence classifier with its tokenizer."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int = 2) -> PromptInjectionClassifier:
        """Load a fresh pretrained model from HuggingFace Hub."""
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model, tokenizer)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> PromptInjectionClassifier:
        """Load a fine-tuned model from a local checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        return cls(model, tokenizer)

    def save(self, output_dir: str | Path) -> None:
        """Save model and tokenizer to a directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
