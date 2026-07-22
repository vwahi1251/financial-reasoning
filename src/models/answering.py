"""Structured answering and retrieval payload models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


@dataclass
class RetrievedChunk:
    text: str


def _coerce_arg_to_str(value: object) -> object:
    # LLMs can emit numeric literals as JSON numbers (e.g., ["s0", 100]).
    # Args are token-like identifiers/literals, so normalize numerics to strings.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return value


def _coerce_id_to_str(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return value


ArgToken = Annotated[str, BeforeValidator(_coerce_arg_to_str)]
PlanId = Annotated[str, BeforeValidator(_coerce_id_to_str)]


class OperationPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: PlanId = Field(description="Unique identifier of this operation step.")
    op: Literal["add", "subtract", "multiply", "divide", "percent_change"] = Field(description="Arithmetic operation to perform.")
    args: list[ArgToken] = Field(default_factory=list, description="Ordered argument references for this operation.")


class StructuredAnswerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "insufficient_context"] = Field(description="Indicates whether the plan can be executed to produce a final value.")
    operands: dict[str, float] = Field(default_factory=dict, description="Mapping of operand identifiers to numeric values.")
    operations: list[OperationPlanStep] = Field(default_factory=list, description="Ordered list of operation steps to compute the final value.")
    final_step: PlanId = Field(default="", description="Identifier of the final result, either an operand key or operation step id.")
    final_unit: Literal["number", "currency", "percent"] = Field(default="number", description="Formatting unit for the final value.")
    final_text: str = Field(default="", description="Optional short natural-language summary.")
