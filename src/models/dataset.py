"""Domain models for split-based conversational financial QA datasets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

NumericValue = float | int
CellValue = NumericValue | str


class Document(BaseModel):
    """Document content associated with a conversation record."""

    pre_text: str = ""
    post_text: str = ""
    table: dict[str, dict[str, CellValue]] = Field(default_factory=dict)

    @field_validator("table", mode="before")
    @classmethod
    def replace_empty_cells(cls, v: dict) -> dict:
        """Replace empty-string cell values with 'unknown'."""
        if not isinstance(v, dict):
            return v
        for row in v.values():
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, str) and not value.strip():
                    row[key] = "unknown"
        return v


class Dialogue(BaseModel):
    """Conversation-level turn data."""

    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str] = Field(default_factory=list)
    executed_answers: list[CellValue] = Field(default_factory=list)
    qa_split: list[bool] = Field(default_factory=list)


class Features(BaseModel):
    """Precomputed descriptive features for one record."""

    num_dialogue_turns: int | None = None
    has_type2_question: bool | None = None
    has_duplicate_columns: bool | None = None
    has_non_numeric_values: bool | None = None


class ConvFinQARecord(BaseModel):
    """Single conversational financial QA record."""

    id: str
    doc: Document
    dialogue: Dialogue
    features: Features | None = None

    @model_validator(mode="after")
    def check_dialogue_alignment(self) -> "ConvFinQARecord":
        n = len(self.dialogue.conv_questions)
        if n == 0:
            raise ValueError(f"dialogue has no turns, id={self.id}")
        for name in ("conv_answers", "turn_program", "executed_answers"):
            actual_len = len(getattr(self.dialogue, name))
            if actual_len != n:
                raise ValueError(
                    f"turn misalignment: {n} questions but {actual_len} {name}, id={self.id}"
                )
        return self


class DatasetSplits(BaseModel):
    """Container for train/dev split records."""

    train: list[ConvFinQARecord] = Field(default_factory=list)
    dev: list[ConvFinQARecord] = Field(default_factory=list)

    def split_names(self) -> tuple[str, ...]:
        """Return available split names in stable order."""
        return ("train", "dev")

    def to_raw_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Export typed records back to raw dictionaries."""
        return {
            "train": [item.model_dump(mode="python") for item in self.train],
            "dev": [item.model_dump(mode="python") for item in self.dev],
        }
