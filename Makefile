.PHONY: install train evaluate serve test lint format clean

install:
	pip install -e ".[dev]"

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

serve:
	python scripts/serve.py

test:
	pytest tests/ -v --cov=src/detector

test-fast:
	pytest tests/ -v -m "not slow" --cov=src/detector

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf artifacts/checkpoints/* artifacts/onnx/* artifacts/logs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
