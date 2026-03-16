"""CLI entry point for serving the prompt injection detector API."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn

from detector.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Serve the prompt injection detector API")
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

    # Import here so config is available via module-level setup if needed
    from detector.api.app import create_app

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
    )


if __name__ == "__main__":
    main()
