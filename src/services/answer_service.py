"""Answer service: builds prompts and orchestrates structured generation."""

from __future__ import annotations

import os
import re

from src.llm.client import DEFAULT_MODEL, generate
from src.models.answering import RetrievedChunk, StructuredAnswerPlan
from src.models.dataset import ConvFinQARecord
from src.prompts.loader import load_prompt
from src.utils.arithmetic import ArithmeticEngine, OperationStep
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
HISTORY_MAX_TURNS = int(os.getenv("RAG_HISTORY_MAX_TURNS", "6"))
HISTORY_SEMANTIC_HINT_MAX_CHARS = int(os.getenv("RAG_HISTORY_SEMANTIC_HINT_MAX_CHARS", "120"))
_NO_ANSWER_HISTORY_MARKER = "[no answer - insufficient context]"
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
_ELLIPSIS = "..."
_RETRY_SUFFIX = (
        "Your previous output did not satisfy the required schema. "
        "Return strictly valid JSON for the requested schema only."
)
_PLANNING_GUIDANCE = """Dataset semantics and planning policy:
- This is ConvFinQA-style table QA: column headers are dimensions (often years/periods), row labels are metrics, and cell values are the numeric evidence.
- Use conversation history for references like 'this', 'that', 'the total', or follow-up turns.
- Lookup-first policy: if the question asks for a single metric at a single period/entity (e.g., 'operating income in 2017'), choose one operand and use no operations.
- Only use arithmetic when the question explicitly asks for composition or transformation:
    total/sum/combined -> add
    in relation to/share/ratio/of total -> divide
    change/increase/decrease (in units, e.g. 'in millions') -> subtract only
    percentage change / relative change -> subtract then divide (or percent_change)
    variance/decline gap between max and min in same section/year -> subtract(max, min)
- Do not add values merely because multiple nearby rows are present in the same column.
- Follow-up lookup rule: if a question asks for a single metric in a specific year (e.g., 'and what was it in 2016?'), resolve the metric from history and perform direct lookup with no arithmetic.
- Section disambiguation rule: phrases like "first section" and "second section" map to header suffixes (1)/(2); do not use unsuffixed or different-section columns.

Few-shot examples (from the train split):
Example 0 (direct lookup -> no arithmetic):
Question: what is the net cash from operating activities in 2009?
Output: {"status":"ok","operands":{"net_cash_2009":206588},"operations":[],"final_step":"net_cash_2009","final_unit":"number","final_text":"Read the 2009 net cash cell directly; no arithmetic required."}

Example 1 (total -> add):
Question: how many total shares are subject to outstanding awards?
Output: {"status":"ok","operands":{"o1":2530454,"o2":5923147},"operations":[{"step_id":"s0","op":"add","args":["o1","o2"]}],"final_step":"s0","final_unit":"number","final_text":"Sum the two outstanding award share counts; no combined total is given."}

Example 2 (proportion of total -> divide):
Question: what proportion does this represent?
Output: {"status":"ok","operands":{"o1":2530454,"o2":5923147},"operations":[{"step_id":"s0","op":"add","args":["o1","o2"]},{"step_id":"s1","op":"divide","args":["o2","s0"]}],"final_step":"s1","final_unit":"percent","final_text":"The share count divided by the combined total."}

Example 3 (percentage change -> subtract then divide):
Question: what percentage change does this represent?
Output: {"status":"ok","operands":{"new":206588,"old":181001},"operations":[{"step_id":"s0","op":"subtract","args":["new","old"]},{"step_id":"s1","op":"divide","args":["s0","old"]}],"final_step":"s1","final_unit":"percent","final_text":"Percent change is delta divided by the baseline."}"""


def serialize_record_context(record: ConvFinQARecord) -> list[str]:
    """Serialize a record into ordered context blocks (pre-text, table columns, post-text).

    Each table column becomes one block with one row per line. Newline-separated
    rows (rather than pipe-delimited runs) keep each label:value pair visually
    distinct — flat delimiter-joined rows were observed to invite cross-row
    aggregation errors (e.g. summing 'operating income' with adjacent cost rows).
    """
    chunks: list[str] = []

    if pre_text := record.doc.pre_text.strip():
        chunks.append(f"Pre-text: {pre_text}")

    for col_name, row_map in record.doc.table.items():
        rows = "\n".join(f"  - {row_key}: {cell_value}" for row_key, cell_value in row_map.items())
        chunks.append(f"Column: {col_name}\n{rows}")

    if post_text := record.doc.post_text.strip():
        chunks.append(f"Post-text: {post_text}")

    return chunks or ["No contextual data available for this record."]


def _history_block(history: list[dict[str, str]]) -> str:
    """Render conversation history for retrieval and generation."""
    recent = history[-HISTORY_MAX_TURNS:] if HISTORY_MAX_TURNS > 0 else history
    return "\n".join(
        f"user: {turn['user']}\nassistant: {_normalize_history_assistant(turn.get('assistant', ''))}"
        for turn in recent
    )


def _normalize_history_assistant(assistant_text: str) -> str:
    """Make assistant history concise and machine-usable for follow-up turns."""
    text = assistant_text.strip()
    if not text:
        return _NO_ANSWER_HISTORY_MARKER

    insufficient = _insufficient_context_message().strip().lower()
    if text.lower() == insufficient or "do not have sufficient information" in text.lower():
        return _NO_ANSWER_HISTORY_MARKER

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    summary_line = lines[1] if len(lines) > 1 else ""

    # Preserve some semantic anchor from explanation text so follow-up
    # references like "that" retain meaning (metric/year/unit relationship).
    semantic_hint = re.sub(r"\s+", " ", summary_line).strip() if summary_line else ""
    semantic_hint = semantic_hint[:HISTORY_SEMANTIC_HINT_MAX_CHARS-3].rstrip() + "..." if len(semantic_hint) > HISTORY_SEMANTIC_HINT_MAX_CHARS else semantic_hint

    match = _NUMERIC_TOKEN_RE.search(first_line)
    if not match:
        return f"{first_line} | {semantic_hint}" if semantic_hint else first_line

    token = match.group(0)
    is_percent = token.endswith("%")
    numeric_token = token.replace("$", "").replace("%", "").replace(",", "")
    try:
        value = float(numeric_token)
    except ValueError:
        return first_line

    if is_percent:
        value_text = f"{value / 100.0:g} [percent]"
    elif value.is_integer():
        value_text = f"{int(value)} [number]"
    else:
        value_text = f"{value:g} [number]"
    return f"{value_text} | {semantic_hint}" if semantic_hint else value_text


def _insufficient_context_message() -> str:
    return load_prompt(
        "insufficient_context_message.txt",
        "I do not have sufficient information in the retrieved context to answer confidently. "
        "Please ask a more specific question about this record's table, pre-text, or post-text.",
    )


def _answer_system_prompt() -> str:
    base = load_prompt(
        "answer_system_prompt.txt",
        "Return a structured operation plan with operands and operations only.",
    )
    return f"{base}\n\n{_PLANNING_GUIDANCE}"


def _full_context_chunks(record: ConvFinQARecord) -> list[RetrievedChunk]:
    sections = serialize_record_context(record)
    return [RetrievedChunk(text=section) for section in sections]


def _build_prompt(
    record_id: str,
    question: str,
    history: list[dict[str, str]],
    retrieved: list[RetrievedChunk],
) -> str:
    context_block = "\n\n".join(
        f"[{idx + 1}]\n{item.text}"
        for idx, item in enumerate(retrieved)
    )
    history_block = _history_block(history)
    return (
        f"Record ID: {record_id}\n"
        f"Conversation History:\n{history_block or 'None'}\n\n"
        f"Retrieved Context:\n{context_block}\n\n"
        f"User Question: {question}\n\n"
        "Return JSON only with keys: status, operands, operations, final_step, final_unit, final_text. "
        "status must be one of: ok, insufficient_context. "
        "operations must be ordered steps with: step_id, op(add|subtract|multiply|divide|percent_change), args. "
        "subtract, divide, and percent_change take exactly 2 args; add and multiply take 2 or more args. "
        "Always use exact numeric values from table cells when a table provides the needed values; do not use rounded or approximate prose values like 'approximately'. "
        "Use only retrieved context values, and put arithmetic in operations. "
        "Do not compute or include the final numeric result in the JSON."
    )


def _format_numeric(value: float, unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized == "percent":
        # Dataset plans use both ratio outputs (e.g., 0.52847 from divide)
        # and percentage-point outputs (e.g., 13.4 from lookup/percent_change).
        percent_value = value * 100.0 if abs(value) <= 1.0 else value
        return f"{percent_value:.2f}%"
    if normalized == "currency":
        return f"${value:,.2f}"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _has_computable_plan(plan: StructuredAnswerPlan) -> bool:
    """Return True when the plan provides a path to a final value."""
    if not plan.final_step:
        return False
    if plan.final_step in plan.operands:
        return True
    operation_ids = {step.step_id for step in plan.operations}
    return plan.final_step in operation_ids


def _render_answer(value: float, unit: str, final_text: str) -> str:
    formatted_value = _format_numeric(value, unit)
    if final_text:
        return f"{formatted_value}\n{final_text}"
    return formatted_value


def _compute_final_value(plan: StructuredAnswerPlan) -> float:
    if not _has_computable_plan(plan):
        raise ValueError("missing_computable_plan")

    if plan.final_step in plan.operands:
        return plan.operands[plan.final_step]

    engine = ArithmeticEngine(operands=plan.operands)
    for step in plan.operations:
        engine.run_step(OperationStep(step_id=step.step_id, op=step.op, args=step.args))

    if plan.final_step not in engine.values:
        raise ValueError("final_step_not_computed")
    return engine.values[plan.final_step]


class AnswerService:
    """Orchestrates reasoning-style answer generation for a single record."""

    def __init__(
        self,
        record: ConvFinQARecord,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.record = record
        self.model_name = model_name
        self._all_context = _full_context_chunks(record=record)

    def context_chunks(self) -> list[RetrievedChunk]:
        """Return the deterministic context used for all reasoning."""
        return self._all_context

    def answer(self, question: str, history: list[dict[str, str]]) -> str:
        prompt = _build_prompt(
            record_id=self.record.id,
            question=question,
            history=history,
            retrieved=self._all_context,
        )
        system_prompt = _answer_system_prompt()

        try:
            plan = generate(
                prompt=prompt,
                model_name=self.model_name,
                system_message=system_prompt,
                temperature=0.0,
            )

            if plan.status == "insufficient_context":
                return _insufficient_context_message()

            final_value = _compute_final_value(plan)
            return _render_answer(final_value, plan.final_unit, plan.final_text)
        except Exception as err:  # noqa: BLE001
            LOGGER.warning(
                "Answer failure_type=execution_error record=%s error_class=%s detail=%s",
                self.record.id,
                err.__class__.__name__,
                err,
            )
            try:
                retry_plan = generate(
                    prompt=prompt,
                    model_name=self.model_name,
                    system_message=f"{system_prompt}\n\n{_RETRY_SUFFIX}",
                    temperature=0.0,
                )
                LOGGER.debug("Structured plan retry: %s", retry_plan.model_dump_json(indent=2))
                if retry_plan.status == "insufficient_context":
                    return _insufficient_context_message()

                retry_value = _compute_final_value(retry_plan)
                return _render_answer(retry_value, retry_plan.final_unit, retry_plan.final_text)
            except Exception as retry_err:  
                LOGGER.warning(
                    "Answer retry failure record=%s error_class=%s detail=%s",
                    self.record.id,
                    retry_err.__class__.__name__,
                    retry_err,
                )
            return _insufficient_context_message()


def build_answer_service(data_path: str, record_id: str) -> AnswerService:
    """Convenience factory: load dataset and return an AnswerService for a record ID."""
    from src.data.loader import load_dataset_splits

    dataset = load_dataset_splits(data_path=data_path)
    all_records = dataset.train + dataset.dev
    record = next((r for r in all_records if r.id == record_id), None)
    if record is None:
        msg = f"Record '{record_id}' not found in dataset."
        raise ValueError(msg)
    return AnswerService(record=record)