"""
Main typer app for ConvFinQA
"""

import json
import os

import typer
from dotenv import load_dotenv
from rich import print as rich_print

from src.data.loader import load_dataset_splits
from src.data.profile import profile_dataset
from src.evaluation.deepeval_eval import eval_accuracy, eval_diagnostics
from src.services.answer_service import build_answer_service, serialize_record_context

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)

load_dotenv()


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    data_path: str = typer.Option(
        os.getenv("CONVFINQA_DATA_PATH", "data/convfinqa_dataset.json"),
        help="Path to dataset JSON file.",
    ),
) -> None:
    """Ask questions about a specific record."""
    answer_service = build_answer_service(data_path=data_path, record_id=record_id)
    history = []
    while True:
        message = input(">>> ")
        if message.strip().lower() in {"exit", "quit"}:
            break
        response = answer_service.answer(question=message, history=history)
        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
        history.append({"user": message, "assistant": response})

@app.command()
def load_data(
    data_path: str = typer.Option(
        os.getenv("CONVFINQA_DATA_PATH", "data/convfinqa_dataset.json"),
        help="Path to dataset JSON file.",
    ),
) -> None:
    """Load dataset and print basic split stats."""
    dataset = load_dataset_splits(data_path=data_path)
    rich_print("[bold]Dataset loaded successfully[/bold]")
    rich_print(f"train records: {len(dataset.train)}")
    rich_print(f"dev records: {len(dataset.dev)}")
    if dataset.train:
        rich_print(f"sample train id: {dataset.train[0].id}")


@app.command("profile-data")
def profile_data(
    data_path: str = typer.Option(
        os.getenv("CONVFINQA_DATA_PATH", "data/convfinqa_dataset.json"),
        help="Path to dataset JSON file.",
    ),
) -> None:
    """Profile dataset quality and split-level statistics."""
    report = profile_dataset(data_path=data_path)
    rich_print(json.dumps(report, indent=2, ensure_ascii=True))


@app.command("eval-accuracy")
def eval_accuracy_cmd(
    data_path: str = typer.Option(
        os.getenv("CONVFINQA_DATA_PATH", "data/convfinqa_dataset.json"),
        help="Path to dataset JSON file.",
    ),
    sample_records: int = typer.Option(50, min=1, help="Number of records to evaluate."),
    split: str = typer.Option("dev", help="Dataset split to evaluate: dev | train | all"),
    seed: int = typer.Option(42, help="Seed for deterministic random record sampling."),
    history_mode: str = typer.Option(
        "oracle",
        help="History mode for turn rollout: oracle | self",
    ),
    per_turn_jsonl_path: str = typer.Option(
        "reports/eval_accuracy_turns.jsonl",
        help="Path to write per-turn JSONL logs.",
    ),
) -> None:
    """Primary accuracy evaluation (no judge calls)."""
    report = eval_accuracy(
        data_path=data_path,
        sample_records=sample_records,
        split=split,
        seed=seed,
        history_mode=history_mode,
        per_turn_jsonl_path=per_turn_jsonl_path,
    )

    rich_print(json.dumps(report, indent=2, ensure_ascii=True))


@app.command("eval-diagnostics")
def eval_diagnostics_cmd(
    data_path: str = typer.Option(
        os.getenv("CONVFINQA_DATA_PATH", "data/convfinqa_dataset.json"),
        help="Path to dataset JSON file.",
    ),
    sample_records: int = typer.Option(10, min=1, help="Number of records to evaluate."),
    turns_per_record: int = typer.Option(2, min=1, help="Max turns evaluated per record."),
    split: str = typer.Option(
        "dev",
        help="Dataset split to evaluate: dev | train | all",
    ),
    seed: int = typer.Option(42, help="Seed for deterministic random record sampling."),
) -> None:
    """Optional DeepEval diagnostics on a smaller sample."""
    report = eval_diagnostics(
        data_path=data_path,
        sample_records=sample_records,
        turns_per_record=turns_per_record,
        split=split,
        seed=seed,
    )

    rich_print(json.dumps(report, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    app()
