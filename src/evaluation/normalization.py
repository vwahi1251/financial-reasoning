"""Deterministic answer normalization helpers for numeric comparisons."""

from __future__ import annotations

import re

_NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_BILLION_PATTERN = re.compile(r"\b(billion|billions|bn)\b")
_MILLION_PATTERN = re.compile(r"\b(million|millions|mm|mn)\b")
_THOUSAND_PATTERN = re.compile(r"\b(thousand|thousands)\b")
_CANONICAL_UNIT_SCALES = (1e3, 1e6, 1e9)


def _unit_scales_in_text(text: object) -> set[float]:
    lowered = str(text).lower()
    scales: set[float] = set()
    if _BILLION_PATTERN.search(lowered):
        scales.add(1e9)
    if _MILLION_PATTERN.search(lowered):
        scales.add(1e6)
    if _THOUSAND_PATTERN.search(lowered):
        scales.add(1e3)
    return scales


def _numeric_candidates(text: object, value: float, *, allow_unlabeled_scale_fallback: bool) -> set[float]:
    candidates = {value}
    scales = _unit_scales_in_text(text)

    # Some model outputs provide an absolute currency number without repeating
    # the unit word (e.g., 73,000,000 for a gold stored as 73 in millions).
    if allow_unlabeled_scale_fallback and not scales and abs(value) >= 1e5:
        scales = set(_CANONICAL_UNIT_SCALES)

    for scale in scales:
        # Keep both directions because outputs can be either scaled points
        # (e.g., 2.3 with "billion" context) or absolute values (2.3e9).
        candidates.add(value / scale)
        candidates.add(value * scale)
    return candidates


def parse_numeric_answer(text: object) -> tuple[float, bool] | None:
    """Parse first numeric token and whether the answer is explicitly a percent.

    Accepts any object defensively (floats from executed_answers, strings from
    conv_answers or model output) — a boundary-normalization lesson learned the
    hard way in this project.
    """
    stripped = str(text).strip().replace(",", "")
    if not stripped:
        return None

    match = _NUMERIC_PATTERN.search(stripped)
    if match is None:
        return None

    value = float(match.group(0))
    is_percent = "%" in stripped
    return value, is_percent


def normalized_numeric_match(
    gold: str,
    predicted: str,
    rel_tolerance: float = 1e-3,
    abs_tolerance: float = 1e-3,
) -> bool:
    """Check answer equivalence.

    Numeric golds: compare in ratio space, allowing percent<->ratio scaling on
    either side (gold '14.1%' matches predicted '0.141'; gold '13.4' stored as
    a bare percent in executed_answers matches predicted '13.40%'). The x100
    scale leniency mirrors the FinQA benchmark's own execution-accuracy
    convention, since the dataset stores percent answers inconsistently
    (ratios like 0.528 and bare percents like 13.4).

    Non-numeric golds (~0.3% of turns are yes/no): case-insensitive string
    equality on the first token of the prediction.
    """
    parsed_gold = parse_numeric_answer(gold)
    parsed_pred = parse_numeric_answer(predicted)

    if parsed_gold is None:
        # Non-numeric gold (e.g. "yes"/"no"): compare normalized strings.
        gold_text = str(gold).strip().lower()
        pred_first = str(predicted).strip().lower().splitlines()[0] if str(predicted).strip() else ""
        return bool(gold_text) and (pred_first == gold_text or pred_first.startswith(gold_text))

    if parsed_pred is None:
        return False

    gold_value, gold_is_percent = parsed_gold
    pred_value, pred_is_percent = parsed_pred

    gold_values = _numeric_candidates(gold, gold_value, allow_unlabeled_scale_fallback=False)
    pred_values = _numeric_candidates(predicted, pred_value, allow_unlabeled_scale_fallback=True)

    gold_ratios = {v / 100.0 if gold_is_percent else v for v in gold_values}
    pred_ratios = {v / 100.0 if pred_is_percent else v for v in pred_values}

    def _close(a: float, b: float) -> bool:
        tolerance = max(abs_tolerance, rel_tolerance * max(abs(a), abs(b), 1.0))
        return abs(a - b) <= tolerance

    for gold_ratio in gold_ratios:
        for pred_ratio in pred_ratios:
            if _close(gold_ratio, pred_ratio):
                return True

    def _close_x100(a: float, b: float) -> bool:
        # The x100 fallback is intentionally lenient for ConvFinQA-style scale
        # ambiguity (ratio vs percent points). Keep this only slightly looser
        # than base matching so near-rounding cases (e.g., 1.6383 vs 1.64)
        # pass without broadly widening all numeric checks.
        tolerance = max(abs_tolerance * 2.0, rel_tolerance * max(abs(a), abs(b), 1.0))
        return abs(a - b) <= tolerance

    # Percent-scale leniency: when either side lacks an explicit % marker its
    # scale is ambiguous in this dataset, so accept a x100 factor either way.
    if not gold_is_percent or not pred_is_percent:
        for gold_ratio in gold_ratios:
            for pred_ratio in pred_ratios:
                if _close_x100(gold_ratio, pred_ratio * 100.0) or _close_x100(gold_ratio * 100.0, pred_ratio):
                    return True

    return False