from __future__ import annotations

import random



from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from detector.config import ModelConfig


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: ModelConfig,
) -> DatasetDict:
    """Tokenize all splits of a DatasetDict."""

    def tokenize_fn(examples: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "source"])
    tokenized.set_format("torch")
    return tokenized


def augment_minority_class(dataset: Dataset, target_ratio: float = 1.0, seed: int = 42) -> Dataset:
    """Augment the minority class (injection=1) to balance the dataset.

    Args:
        dataset: Training dataset with 'text' and 'label' columns.
        target_ratio: Target ratio of minority to majority. 1.0 = balanced.
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    benign = dataset.filter(lambda x: x["label"] == 0)
    injections = dataset.filter(lambda x: x["label"] == 1)

    num_benign = len(benign)
    num_injections = len(injections)
    target_count = int(num_benign * target_ratio)
    num_to_generate = max(0, target_count - num_injections)

    if num_to_generate == 0:
        return dataset

    # Generate augmented samples
    augmented_texts: list[str] = []
    augmented_labels: list[int] = []
    augmented_sources: list[str] = []

    augment_fns = [_random_case, _insert_whitespace, _add_unicode_chars, _add_prefix_variant]

    for i in range(num_to_generate):
        idx = i % num_injections
        original_text = injections[idx]["text"]
        fn = rng.choice(augment_fns)
        augmented_texts.append(fn(original_text, rng))
        augmented_labels.append(1)
        augmented_sources.append("augmented")

    augmented_ds = Dataset.from_dict(
        {"text": augmented_texts, "label": augmented_labels, "source": augmented_sources}
    )

    from datasets import concatenate_datasets

    return concatenate_datasets([dataset, augmented_ds]).shuffle(seed=seed)


def _random_case(text: str, rng: random.Random) -> str:
    """Randomly change case of some characters."""
    return "".join(c.swapcase() if rng.random() < 0.3 else c for c in text)


def _insert_whitespace(text: str, rng: random.Random) -> str:
    """Insert random extra whitespace."""
    words = text.split()
    result = []
    for word in words:
        result.append(word)
        if rng.random() < 0.2:
            result.append(" ")
    return " ".join(result)


def _add_unicode_chars(text: str, rng: random.Random) -> str:
    """Insert zero-width or homoglyph characters."""
    zero_width = ["\u200b", "\u200c", "\u200d", "\ufeff"]
    result = list(text)
    num_insertions = rng.randint(1, 3)
    for _ in range(num_insertions):
        pos = rng.randint(0, len(result))
        result.insert(pos, rng.choice(zero_width))
    return "".join(result)


def _add_prefix_variant(text: str, rng: random.Random) -> str:
    """Add common prompt injection prefixes or suffixes."""
    prefixes = [
        "Please note: ",
        "IMPORTANT: ",
        "System update: ",
        "New instruction: ",
        "Override: ",
    ]
    suffixes = [
        " Do it now.",
        " This is urgent.",
        " Comply immediately.",
        " No questions asked.",
    ]
    if rng.random() < 0.5:
        return rng.choice(prefixes) + text
    else:
        return text + rng.choice(suffixes)
