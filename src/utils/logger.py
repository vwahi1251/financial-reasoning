"""Logger factory shared across modules."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str = __name__) -> logging.Logger:
    """Create a logger based on LOG_LEVEL environment variable."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        log_format = os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(log_format)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
