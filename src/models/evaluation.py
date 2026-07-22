"""Shared evaluation models for answer benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.dataset import ConvFinQARecord


@dataclass
class EvalTurn:
    question: str
    gold_answer: str


@dataclass
class EvalExample:
    record: ConvFinQARecord
    turns: list[EvalTurn]
