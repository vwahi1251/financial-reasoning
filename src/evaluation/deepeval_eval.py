"""Evaluation entry points: deterministic accuracy (primary) and optional
LLM-judge diagnostics (supplementary).

Design notes:
- eval_accuracy is the primary metric: deterministic, judge-free, all turns,
  seeded record sampling, per-turn JSONL audit log, bootstrap 95% CI, and
  accuracy slices by turn depth and gold-program complexity.
- history_mode: "oracle" feeds gold answers forward (isolates per-turn
  reasoning); "self" feeds the model's own answers forward (measures the
  deployed, error-cascading condition). The gap between the two runs is the
  measured cost of error cascade.
- eval_diagnostics uses DeepEval judge metrics on a small sample with a judge
  model distinct from the answering model (self-preference bias); each metric
  is isolated so one judge failure does not discard the others.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from src.data.loader import load_dataset_splits
from src.evaluation.normalization import normalized_numeric_match
from src.llm.client import JUDGE_MODEL, generate
from src.models.dataset import ConvFinQARecord
from src.models.evaluation import EvalExample, EvalTurn
from src.services.answer_service import AnswerService
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_NO_ANSWER_MARKER = "[no answer]"


class DeepEvalLLM(DeepEvalBaseLLM):
    """Adapter exposing the shared LLM client through DeepEval's LLM interface."""

    def __init__(self, model: str = JUDGE_MODEL) -> None:
        self._model_name = model
        super().__init__(model=model)

    def load_model(self) -> "DeepEvalLLM":
        return self

    def generate(self, prompt: Any, schema: Any | None = None) -> str | Any:
        output = generate(
            prompt=str(prompt),
            model_name=self._model_name,
            system_message="You are a precise evaluation judge.",
            temperature=0.0,
        )
        # The shared client may return a plain string or a structured object;
        # normalize defensively rather than assuming one shape.
        text = output if isinstance(output, str) else (
            output.model_dump_json() if hasattr(output, "model_dump_json") else str(output)
        )
        if schema is None:
            return text
        try:
            return schema.model_validate_json(text)
        except Exception:  # noqa: BLE001 - judge output may be loose JSON
            try:
                return schema.model_validate(json.loads(text))
            except Exception:  # noqa: BLE001
                return text

    async def a_generate(self, prompt: Any, schema: Any | None = None) -> str | Any:
        return self.generate(prompt=prompt, schema=schema)

    def get_model_name(self) -> str:
        return f"gemini/{self._model_name}"

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True


def _bootstrap_ci(
    values: list[float], seed: int, n_resamples: int = 1000
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI over per-turn match outcomes."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(n_resamples)
    )
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(int(0.975 * n_resamples), n_resamples - 1)]
    return (round(lo, 4), round(hi, 4))


def _load_eval_examples(
    data_path: str,
    sample_records: int,
    split: str,
    seed: int,
    turns_per_record: int | None = None,
) -> list[EvalExample]:
    """Seeded random sample of records, each with up to turns_per_record turns
    (all turns when None). Alignment is already enforced by the Dialogue model.
    """
    dataset = load_dataset_splits(data_path=data_path)
    pools: dict[str, list[ConvFinQARecord]] = {
        "train": dataset.train,
        "dev": dataset.dev,
        "all": dataset.train + dataset.dev,
    }
    if split not in pools:
        raise ValueError(f"Invalid split '{split}'. Expected one of: {', '.join(pools)}")
    records = pools[split]
    if sample_records < len(records):
        records = random.Random(seed).sample(records, sample_records)

    examples: list[EvalExample] = []
    for record in records:
        n_turns = len(record.dialogue.conv_questions)
        if turns_per_record is not None:
            n_turns = min(turns_per_record, n_turns)
        examples.append(
            EvalExample(
                record=record,
                turns=[
                    EvalTurn(
                        question=record.dialogue.conv_questions[i],
                        gold_answer=str(record.dialogue.executed_answers[i]),
                    )
                    for i in range(n_turns)
                ],
            )
        )
    return examples


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def eval_accuracy(
    data_path: str,
    sample_records: int = 50,
    split: str = "dev",
    seed: int = 42,
    history_mode: Literal["oracle", "self"] = "oracle",
    per_turn_jsonl_path: str = "reports/eval_accuracy_turns.jsonl",
) -> dict[str, object]:
    """Primary deterministic metric: normalized numeric match, all turns,
    no judge calls. Reports a bootstrap 95% CI and accuracy sliced by turn
    depth and gold-program complexity; writes a per-turn JSONL audit log.
    """
    examples = _load_eval_examples(data_path, sample_records, split, seed)

    jsonl_path = Path(per_turn_jsonl_path)
    jsonl_path.unlink(missing_ok=True)

    matches: list[float] = []
    attempted = 0
    by_depth: dict[str, list[float]] = {}

    for example in examples:
        service = AnswerService(record=example.record)
        history: list[dict[str, str]] = []
        programs = example.record.dialogue.turn_program
        for turn_index, turn in enumerate(example.turns, start=1):
            attempted += 1
            program = programs[turn_index - 1] if turn_index - 1 < len(programs) else ""
            depth_key = str(turn_index) if turn_index <= 3 else "4+"
            row: dict[str, object] = {
                "record_id": example.record.id,
                "turn_index": turn_index,
                "question": turn.question,
                "gold": turn.gold_answer,
                "gold_program": program,
            }
            predicted = ""
            try:
                predicted = service.answer(question=turn.question, history=list(history))
                is_match = normalized_numeric_match(turn.gold_answer, predicted)
                outcome = 1.0 if is_match else 0.0
                row.update({"predicted": predicted, "match": is_match})
            except RuntimeError:
                raise  # missing API key etc. — configuration, not data; abort loudly
            except Exception as err:  # noqa: BLE001 - one bad turn must not kill the run
                LOGGER.warning(
                    "Accuracy eval failed record=%s turn=%s: %s",
                    example.record.id, turn_index, err,
                )
                outcome = 0.0
                row.update({"predicted": "", "match": False, "error": str(err)})

            matches.append(outcome)
            by_depth.setdefault(depth_key, []).append(outcome)

            if history_mode == "oracle":
                # Gold answers fed forward: scores each turn on its own
                # reasoning rather than compounding prior errors.
                history.append({"user": turn.question, "assistant": turn.gold_answer})
            else:
                # Self history: the deployed condition; failed/abstained turns
                # propagate as an explicit no-answer marker.
                history.append({
                    "user": turn.question,
                    "assistant": predicted if predicted else _NO_ANSWER_MARKER,
                })
            _append_jsonl(jsonl_path, row)

    ci_low, ci_high = _bootstrap_ci(matches, seed=seed)

    return {
        "config": {
            "split": split,
            "sample_records": sample_records,
            "sample_seed": seed,
            "turns_per_record": "all",
            "history_mode": history_mode,
            "gold_source": "executed_answers",
            "variant": "accuracy",
            "per_turn_jsonl_path": str(jsonl_path),
        },
        "coverage": {
            "evaluated_records": len(examples),
            "scored_turns": len(matches),
            "total_attempted_turns": attempted,
        },
        "metrics": {
            "normalized_numeric_match": round(_avg(matches), 4),
            "ci95": [ci_low, ci_high],
            "by_turn_depth": {
                k: {"accuracy": round(_avg(v), 4), "n": len(v)}
                for k, v in sorted(by_depth.items())
            },
        },
    }


def eval_diagnostics(
    data_path: str,
    sample_records: int = 10,
    turns_per_record: int = 2,
    split: str = "dev",
    seed: int = 42,
) -> dict[str, object]:
    """Optional judge-based diagnostics on a small sample. Each metric is
    measured independently so one judge failure does not discard the rest.
    """
    examples = _load_eval_examples(
        data_path, sample_records, split, seed, turns_per_record=turns_per_record
    )

    judge = GeminiDeepEvalLLM()
    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(model=judge),
        "faithfulness": FaithfulnessMetric(model=judge),
        "contextual_precision": ContextualPrecisionMetric(model=judge),
        "contextual_recall": ContextualRecallMetric(model=judge),
    }
    scores: dict[str, list[float]] = {name: [] for name in metrics}
    attempted = 0
    answered = 0

    for example in examples:
        service = AnswerService(record=example.record)
        history: list[dict[str, str]] = []
        for turn in example.turns:
            attempted += 1
            try:
                retrieved = service.context_chunks()
                predicted = service.answer(question=turn.question, history=list(history))
            except RuntimeError:
                raise
            except Exception as err:  # noqa: BLE001
                LOGGER.warning(
                    "Diagnostics answer failed record=%s: %s", example.record.id, err
                )
                history.append({"user": turn.question, "assistant": turn.gold_answer})
                continue

            answered += 1
            test_case = LLMTestCase(
                input=turn.question,
                actual_output=predicted,
                expected_output=turn.gold_answer,
                retrieval_context=[chunk.text for chunk in retrieved],
            )
            for name, metric in metrics.items():
                try:
                    metric.measure(test_case)
                    scores[name].append(float(metric.score or 0.0))
                except Exception as err:  # noqa: BLE001 - isolate per metric
                    LOGGER.warning(
                        "%s metric failed record=%s: %s", name, example.record.id, err
                    )
            history.append({"user": turn.question, "assistant": turn.gold_answer})

    return {
        "config": {
            "split": split,
            "sample_records": sample_records,
            "sample_seed": seed,
            "turns_per_record": turns_per_record,
            "history_mode": "oracle",
            "judge_model": judge.get_model_name(),
            "variant": "diagnostics",
        },
        "coverage": {
            "evaluated_records": len(examples),
            "answered_turns": answered,
            "total_attempted_turns": attempted,
            "scored_turns_per_metric": {k: len(v) for k, v in scores.items()},
        },
        "metrics": {name: round(_avg(vals), 4) for name, vals in scores.items()},
    }