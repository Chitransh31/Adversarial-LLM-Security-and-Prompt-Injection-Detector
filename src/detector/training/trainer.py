from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from detector.config import AppConfig
from detector.data.loader import load_prompt_injection_data
from detector.data.preprocessor import augment_minority_class, tokenize_dataset
from detector.evaluation.metrics import compute_metrics
from detector.model.factory import load_model
from detector.training.callbacks import MetricsLoggerCallback

logger = logging.getLogger(__name__)


def train(config: AppConfig) -> dict:
    """Run the full training pipeline.

    1. Load and preprocess data
    2. Load model
    3. Train with HF Trainer
    4. Save best checkpoint
    5. Return training metrics

    Returns:
        Dictionary of final training metrics.
    """
    # Load data
    logger.info("Loading datasets...")
    raw_dataset = load_prompt_injection_data()
    logger.info(
        f"Dataset splits — train: {len(raw_dataset['train'])}, "
        f"val: {len(raw_dataset['validation'])}, test: {len(raw_dataset['test'])}"
    )

    # Augment training set
    logger.info("Augmenting minority class...")
    raw_dataset["train"] = augment_minority_class(raw_dataset["train"], target_ratio=1.0)
    logger.info(f"Augmented train size: {len(raw_dataset['train'])}")

    # Load model and tokenizer
    logger.info(f"Loading model: {config.model.name}")
    classifier = load_model(config.model)

    # Tokenize
    logger.info("Tokenizing datasets...")
    tokenized = tokenize_dataset(raw_dataset, classifier.tokenizer, config.model)

    # Training arguments
    output_dir = Path(config.training.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.inference.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=config.training.logging_steps,
        report_to="none",
        save_total_limit=2,
    )

    # Create trainer
    trainer = Trainer(
        model=classifier.model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[MetricsLoggerCallback()],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # Save best model
    best_dir = output_dir / "best"
    classifier.model = trainer.model
    classifier.save(best_dir)
    logger.info(f"Best model saved to {best_dir}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")

    return {**train_result.metrics, **test_metrics}
