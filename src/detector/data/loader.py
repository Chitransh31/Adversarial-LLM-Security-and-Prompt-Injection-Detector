from __future__ import annotations

import logging

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

# Column mapping: dataset_name -> {original_col: normalized_col}
_COLUMN_MAPS = {
    "deepset/prompt-injections": {"text": "text", "label": "label"},
}


def load_prompt_injection_data(seed: int = 42) -> DatasetDict:
    """Download, merge, and split prompt injection datasets.

    Returns a DatasetDict with 'train', 'validation', and 'test' splits
    using a stratified 70/15/15 split.
    """
    datasets_list: list[Dataset] = []

    # Load deepset/prompt-injections
    ds = _load_and_normalize("deepset/prompt-injections", split="train")
    datasets_list.append(ds)

    # Attempt to load supplementary dataset
    try:
        ds_extra = _load_and_normalize("protectai/prompt-injection-validation", split="train")
        datasets_list.append(ds_extra)
        logger.info("Loaded supplementary dataset: protectai/prompt-injection-validation")
    except Exception:
        logger.warning(
            "Could not load protectai/prompt-injection-validation. "
            "Proceeding with deepset dataset only."
        )

    # Merge all datasets
    merged = concatenate_datasets(datasets_list)
    logger.info(f"Merged dataset size: {len(merged)} samples")

    # Stratified split: first 85/15 (test), then 82/18 of the 85% (train/val) ≈ 70/15/15
    split1 = merged.train_test_split(test_size=0.15, seed=seed, stratify_by_column="label")
    split2 = split1["train"].train_test_split(
        test_size=0.176, seed=seed, stratify_by_column="label"  # 0.176 * 0.85 ≈ 0.15
    )

    return DatasetDict(
        {
            "train": split2["train"],
            "validation": split2["test"],
            "test": split1["test"],
        }
    )


def _load_and_normalize(dataset_name: str, split: str = "train") -> Dataset:
    """Load a dataset from HuggingFace Hub and normalize columns."""
    ds = load_dataset(dataset_name, split=split)

    # Normalize column names if mapping exists
    col_map = _COLUMN_MAPS.get(dataset_name)
    if col_map:
        for original, normalized in col_map.items():
            if original != normalized and original in ds.column_names:
                ds = ds.rename_column(original, normalized)

    # Verify required columns exist
    required = {"text", "label"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset {dataset_name} missing required columns: {missing}")

    # Add source column
    ds = ds.map(lambda _: {"source": dataset_name})

    # Keep only required columns + source
    keep_cols = ["text", "label", "source"]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    return ds
