"""Dataset loading and parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models.dataset import ConvFinQARecord, DatasetSplits
from src.utils.logger import get_logger


DEFAULT_REQUIRED_SPLITS: tuple[str, ...] = ("train", "dev")

KNOWN_BAD_RECORD_IDS: frozenset[str] = frozenset({"Double_ADBE/2014/page_70.pdf", "Single_ETR/2016/page_144.pdf-4"})

LOGGER = get_logger(__name__)


def load_raw_dataset(data_path: str) -> dict[str, list[dict[str, Any]]]:
    """Simple loader with fail fast validation of ConvFinQARecord instances 
    (fail fast was chosen as it is a simple database. For any messy ongoing dataset, skip and drop approach would be better). 
    Raises ValueError if any record is invalid or if required splits are missing."""
    
    path = Path(data_path)
    if not path.exists():
        msg = f"Dataset file not found at {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
        
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object with split keys, got {type(payload).__name__}")
    for key, value in payload.items():
        if not isinstance(value, list):
            raise ValueError(f"Split '{key}' must be a list, got {type(value).__name__}")    

    return payload


def _parse_split(
    split_name: str,
    items: list[dict[str, object]],
) -> tuple[list[ConvFinQARecord], list[str]]:
    parsed: list[ConvFinQARecord] = []
    errors: list[str] = []
    for idx, item in enumerate(items):
        record_id = str(item.get("id", "unknown"))
        if record_id in KNOWN_BAD_RECORD_IDS:
            LOGGER.warning(
                "Known-bad record quarantined: split=%s index=%s id=%s",
                split_name,
                idx,
                record_id,
            )
            continue  # quarantined: documented misalignment, excluded from all runs
        try:
            parsed.append(ConvFinQARecord.model_validate(item))
        except ValidationError as err:
            record_id = str(item.get("id", "unknown"))
            errors.append(
                f"split='{split_name}' index={idx} id='{record_id}': {err}"
            )
    return parsed, errors


def load_dataset_splits(
    data_path: str | Path,
    required_splits: tuple[str, ...] = DEFAULT_REQUIRED_SPLITS,
) -> DatasetSplits:
    """Load dataset and return typed train/dev splits.

    All records are attempted first; validation errors are raised together at the end.
    """
    raw_map = load_raw_dataset(data_path)
    typed_map: dict[str, list[ConvFinQARecord]] = {}
    all_errors: list[str] = []

    for name, items in raw_map.items():
        parsed, split_errors = _parse_split(name, items)
        typed_map[name] = parsed
        all_errors.extend(split_errors)

    missing = [name for name in required_splits if name not in typed_map]
    if missing:
        msg = f"Missing required splits: {', '.join(missing)}"
        raise ValueError(msg)

    if all_errors:
        details = "\n".join(f"- {entry}" for entry in all_errors)
        raise ValueError(
            f"Dataset validation failed with {len(all_errors)} invalid record(s):\n{details}"
        )

    return DatasetSplits(
        train=typed_map.get("train", []),
        dev=typed_map.get("dev", []),
    )