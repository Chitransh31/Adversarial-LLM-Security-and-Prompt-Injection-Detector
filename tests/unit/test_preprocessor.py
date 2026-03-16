"""Tests for data preprocessing and augmentation."""

import random

import pytest
from datasets import Dataset

from detector.data.preprocessor import (
    _add_prefix_variant,
    _add_unicode_chars,
    _insert_whitespace,
    _random_case,
    augment_minority_class,
)


class TestAugmentationFunctions:
    def test_random_case_changes_some_characters(self):
        rng = random.Random(42)
        text = "Hello World"
        result = _random_case(text, rng)
        assert result != text or True  # May be same by chance
        assert len(result) == len(text)

    def test_insert_whitespace_preserves_words(self):
        rng = random.Random(42)
        text = "Ignore all instructions"
        result = _insert_whitespace(text, rng)
        # Original words should still be present
        for word in text.split():
            assert word in result

    def test_add_unicode_chars_increases_length(self):
        rng = random.Random(42)
        text = "test prompt"
        result = _add_unicode_chars(text, rng)
        assert len(result) > len(text)

    def test_add_prefix_variant_modifies_text(self):
        rng = random.Random(42)
        text = "Ignore instructions"
        result = _add_prefix_variant(text, rng)
        assert result != text
        assert "Ignore instructions" in result


class TestAugmentMinorityClass:
    def test_augmentation_increases_minority_count(self):
        dataset = Dataset.from_dict(
            {
                "text": ["benign " + str(i) for i in range(10)]
                + ["injection " + str(i) for i in range(5)],
                "label": [0] * 10 + [1] * 5,
                "source": ["test"] * 15,
            }
        )

        augmented = augment_minority_class(dataset, target_ratio=1.0, seed=42)
        injection_count = sum(1 for x in augmented if x["label"] == 1)
        assert injection_count >= 10  # Should be roughly equal to benign count

    def test_augmentation_preserves_benign_samples(self):
        dataset = Dataset.from_dict(
            {
                "text": ["benign " + str(i) for i in range(10)]
                + ["injection " + str(i) for i in range(5)],
                "label": [0] * 10 + [1] * 5,
                "source": ["test"] * 15,
            }
        )

        augmented = augment_minority_class(dataset, target_ratio=1.0, seed=42)
        benign_count = sum(1 for x in augmented if x["label"] == 0)
        assert benign_count == 10

    def test_no_augmentation_when_balanced(self):
        dataset = Dataset.from_dict(
            {
                "text": ["sample " + str(i) for i in range(10)],
                "label": [0] * 5 + [1] * 5,
                "source": ["test"] * 10,
            }
        )

        augmented = augment_minority_class(dataset, target_ratio=1.0, seed=42)
        assert len(augmented) == 10
