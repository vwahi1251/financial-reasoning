"""Dataset profiling utilities for ConvFinQA."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from src.data.loader import load_dataset_splits
from src.models.dataset import ConvFinQARecord

UNKNOWN_MARKER = "unknown"


def _is_numeric_like(text: str) -> bool:
    normalized = text.strip().replace(",", "").replace("$", "").replace("%", "")
    if not normalized:
        return False
    try:
        float(normalized)
        return True
    except ValueError:
        return False


def _classify_answer(answer: str) -> str:
    stripped = answer.strip()
    if not stripped:
        return "empty"
    return "numeric_string" if _is_numeric_like(stripped) else "text"


def _profile_dialogue(record: ConvFinQARecord) -> dict:
    questions = record.dialogue.conv_questions
    answers = record.dialogue.conv_answers
    format_counts = Counter(_classify_answer(a) for a in answers)
    return {
        "turn_depth": len(questions),
        "turn_aligned": len(questions) == len(answers),
        "question_lengths": [len(q) for q in questions],
        "answer_lengths": [len(a) for a in answers],
        "format_counts": format_counts,
        "has_empty_answer": format_counts["empty"] > 0,
    }


def _profile_table(record: ConvFinQARecord) -> dict:
    table = record.doc.table
    row_lengths = [len(col) for col in table.values()]
    unknown_count = sum(
        1
        for col in table.values()
        for cell in col.values()
        if isinstance(cell, str) and cell == UNKNOWN_MARKER
    )
    return {
        "columns": len(table),
        "rows_avg": round(mean(row_lengths), 3) if row_lengths else 0,
        "unknown_cells": unknown_count,
    }


def _profile_features(record: ConvFinQARecord) -> dict:
    f = record.features
    if f is None:
        return {"type2": False, "dup_cols": False, "non_numeric": False}
    return {
        "type2": bool(f.has_type2_question),
        "dup_cols": bool(f.has_duplicate_columns),
        "non_numeric": bool(f.has_non_numeric_values),
    }


_EMPTY_PROFILE: dict[str, object] = {
    "records": 0,
    "turn_depth": {"avg": 0.0, "min": 0, "max": 0},
    "question_length_chars": {"avg": 0.0},
    "answer_length_chars": {"avg": 0.0},
    "table": {"columns_avg": 0.0, "rows_avg": 0.0},
    "data_quality": {
        "turn_alignment_issues": 0,
        "unknown_cells": 0,
        "unknown_cells_ids": [],
        "empty_answers": 0,
        "empty_answers_ids": [],
    },
    "answer_format": {"numeric_string": 0, "text": 0, "empty": 0},
    "features": {
        "has_type2_question_true": 0,
        "has_duplicate_columns_true": 0,
        "has_non_numeric_values_true": 0,
    },
}


def _split_profile(records: list[ConvFinQARecord]) -> dict[str, object]:
    if not records:
        return _EMPTY_PROFILE

    turn_depths: list[int] = []
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    table_columns: list[int] = []
    table_rows: list[float] = []

    turn_alignment_issues = 0
    unknown_cells_total = 0
    unknown_cells_ids: list[str] = []
    empty_answers = 0
    empty_answers_ids: list[str] = []
    answer_format: Counter[str] = Counter()

    feat_type2 = 0
    feat_dup_cols = 0
    feat_non_numeric = 0

    for record in records:
        dlg = _profile_dialogue(record)
        turn_depths.append(dlg["turn_depth"])
        question_lengths.extend(dlg["question_lengths"])
        answer_lengths.extend(dlg["answer_lengths"])
        answer_format.update(dlg["format_counts"])
        if not dlg["turn_aligned"]:
            turn_alignment_issues += 1
        if dlg["has_empty_answer"]:
            empty_answers += dlg["format_counts"]["empty"]
            empty_answers_ids.append(record.id)

        tbl = _profile_table(record)
        table_columns.append(tbl["columns"])
        table_rows.append(tbl["rows_avg"])
        if tbl["unknown_cells"] > 0:
            unknown_cells_total += tbl["unknown_cells"]
            unknown_cells_ids.append(record.id)

        feat = _profile_features(record)
        feat_type2 += int(feat["type2"])
        feat_dup_cols += int(feat["dup_cols"])
        feat_non_numeric += int(feat["non_numeric"])

    return {
        "records": len(records),
        "turn_depth": {
            "avg": round(mean(turn_depths), 3),
            "min": min(turn_depths),
            "max": max(turn_depths),
        },
        "question_length_chars": {"avg": round(mean(question_lengths), 3) if question_lengths else 0.0},
        "answer_length_chars": {"avg": round(mean(answer_lengths), 3) if answer_lengths else 0.0},
        "table": {
            "columns_avg": round(mean(table_columns), 3),
            "rows_avg": round(mean(table_rows), 3),
        },
        "data_quality": {
            "turn_alignment_issues": turn_alignment_issues,
            "unknown_cells": unknown_cells_total,
            "unknown_cells_ids": unknown_cells_ids,
            "empty_answers": empty_answers,
            "empty_answers_ids": empty_answers_ids,
        },
        "answer_format": {
            "numeric_string": int(answer_format["numeric_string"]),
            "text": int(answer_format["text"]),
            "empty": int(answer_format["empty"]),
        },
        "features": {
            "has_type2_question_true": feat_type2,
            "has_duplicate_columns_true": feat_dup_cols,
            "has_non_numeric_values_true": feat_non_numeric,
        },
    }


def profile_dataset(data_path: str) -> dict[str, object]:
    """Build a dataset profile with split-level statistics."""
    dataset = load_dataset_splits(data_path=data_path)

    train_profile = _split_profile(dataset.train)
    dev_profile = _split_profile(dataset.dev)

    return {
        "summary": {
            "total_records": len(dataset.train) + len(dataset.dev),
            "train_records": len(dataset.train),
            "dev_records": len(dataset.dev),
        },
        "splits": {
            "train": train_profile,
            "dev": dev_profile,
        },
    }