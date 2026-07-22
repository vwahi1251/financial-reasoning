"""Deterministic arithmetic execution for operation plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass
class OperationStep:
    step_id: str
    op: str
    args: list[str]


class ArithmeticEngine:
    """Executes arithmetic steps using referenced operands and prior step results."""

    def __init__(self, operands: dict[str, float]) -> None:
        self._values: dict[str, float] = dict(operands)

    @property
    def values(self) -> dict[str, float]:
        return dict(self._values)

    def _resolve(self, token: str) -> float:
        key = token.strip()
        if key in self._values:
            return self._values[key]
        return float(key)

    def run_step(self, step: OperationStep) -> float:
        op = step.op.lower().strip()
        args = [self._resolve(arg) for arg in step.args]

        if op == "add":
            if len(args) < 2:
                raise ValueError("add requires at least 2 args")
            result = sum(args)
        elif op == "subtract":
            if len(args) != 2:
                raise ValueError("subtract requires 2 args")
            result = args[0] - args[1]
        elif op == "multiply":
            if len(args) < 2:
                raise ValueError("multiply requires at least 2 args")
            result = prod(args)
        elif op == "divide":
            if len(args) != 2:
                raise ValueError("divide requires 2 args")
            if args[1] == 0:
                raise ValueError("division by zero")
            result = args[0] / args[1]
        elif op == "percent_change":
            if len(args) != 2:
                raise ValueError("percent_change requires 2 args")
            old_value = args[1]
            if old_value == 0:
                raise ValueError("percent_change baseline cannot be zero")
            result = ((args[0] - old_value) / abs(old_value)) * 100.0
        else:
            raise ValueError(f"unsupported op: {step.op}")

        self._values[step.step_id] = result
        return result
