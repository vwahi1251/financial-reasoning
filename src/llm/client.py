"""This client is specialized to answer-plan generation; judge diagnostics would require a text-mode path"""

from __future__ import annotations

import os
from time import sleep

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models.answering import StructuredAnswerPlan

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
load_dotenv()

DEFAULT_MODEL = os.getenv("RAG_MODEL", "gemini-3.5-flash")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", DEFAULT_MODEL)


def _resolve_api_key() -> str:
    # Support standard Gemini env vars and a legacy generic API_KEY.
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "API_KEY environment variable is not set. Please set it to your API key"
        )
    return api_key



def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(
        term in msg
        for term in ("429", "rate limit", "too many requests", "timeout", "timed out",
                     "temporarily unavailable", "service unavailable", "connection")
    )


def generate(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_retries: int | None = None,
    base_delay_s: float | None = None,
    max_completion_tokens: int | None = None,
    system_message: str = "You are a grounded assistant for financial QA.",
    temperature: float = 0.0,
) -> StructuredAnswerPlan:
    """Call Gemini with retries and return a StructuredAnswerPlan."""

    max_retries = max_retries if max_retries is not None else int(os.getenv("RAG_MAX_RETRIES", "3"))
    base_delay_s = base_delay_s if base_delay_s is not None else float(os.getenv("RAG_RETRY_BASE_DELAY_SECONDS", "0.5"))
    max_completion_tokens = (
        max_completion_tokens
        if max_completion_tokens is not None
        else int(os.getenv("RAG_MAX_COMPLETION_TOKENS", "2048"))
    )

    llm_kwargs: dict[str, object] = {
        "model": model_name,
        "temperature": temperature,
        "max_output_tokens": max_completion_tokens,
        "api_key": _resolve_api_key(),
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            llm = ChatGoogleGenerativeAI(**llm_kwargs).with_structured_output(
                StructuredAnswerPlan,
            )

            response = llm.invoke(
                [
                    ("system", system_message),
                    ("human", prompt),
                ]
            )
            if not isinstance(response, StructuredAnswerPlan):
                return StructuredAnswerPlan.model_validate(response)
            return response
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < max_retries and _is_retryable(err):
                delay_s = base_delay_s * (2 ** attempt)
                LOGGER.warning("LLM retry %s/%s after network error: %s", attempt + 1, max_retries, err)
                sleep(delay_s)
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError("No response generated.")
