"""Prompt loading utilities for prompt text files."""

from __future__ import annotations

import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str, fallback: str) -> str:
    """Load a prompt text file from src/prompts with fallback content."""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        LOGGER.warning("Prompt file not found, using fallback: %s", prompt_path)
        return fallback

    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except OSError as err:
        LOGGER.warning("Failed to read prompt file %s: %s", prompt_path, err)
        return fallback
