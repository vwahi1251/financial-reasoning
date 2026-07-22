from __future__ import annotations

import unittest

from src.models.answering import StructuredAnswerPlan
from src.services.answer_service import _format_numeric, _normalize_history_assistant
from src.utils.arithmetic import ArithmeticEngine, OperationStep


class TestAnswerServiceDeterministic(unittest.TestCase):
    def test_arithmetic_engine_executes_steps(self) -> None:
        engine = ArithmeticEngine(operands={"a": 273.0, "b": 643.0})
        result = engine.run_step(OperationStep(step_id="r0", op="divide", args=["a", "b"]))
        self.assertAlmostEqual(result, 0.42457, places=5)
        self.assertIn("r0", engine.values)

    def test_arithmetic_engine_percent_change(self) -> None:
        engine = ArithmeticEngine(operands={"new": 239.0, "old": 208.3})
        result = engine.run_step(
            OperationStep(step_id="pct", op="percent_change", args=["new", "old"])
        )
        self.assertAlmostEqual(result, 14.7383, places=3)

    def test_format_numeric(self) -> None:
        self.assertEqual(_format_numeric(14.1, "percent"), "14.10%")
        self.assertEqual(_format_numeric(0.52847, "percent"), "52.85%")
        self.assertEqual(_format_numeric(531.0, "number"), "531")

    def test_plan_accepts_numeric_json_args(self) -> None:
        plan = StructuredAnswerPlan.model_validate(
            {
                "status": "ok",
                "operands": {"o1": 500.0},
                "operations": [{"step_id": "s0", "op": "divide", "args": ["o1", 100]}],
                "final_step": "s0",
            }
        )
        self.assertEqual(plan.operations[0].args, ["o1", "100"])

    def test_plan_coerces_numeric_ids_to_strings(self) -> None:
        plan = StructuredAnswerPlan.model_validate(
            {
                "status": "ok",
                "operands": {"o1": 500.0},
                "operations": [{"step_id": 1, "op": "divide", "args": ["o1", 100]}],
                "final_step": 1,
            }
        )
        self.assertEqual(plan.operations[0].step_id, "1")
        self.assertEqual(plan.final_step, "1")

    def test_plan_coerces_null_final_step_to_empty_string(self) -> None:
        plan = StructuredAnswerPlan.model_validate(
            {
                "status": "insufficient_context",
                "operands": {},
                "operations": [],
                "final_step": None,
            }
        )
        self.assertEqual(plan.final_step, "")

    def test_normalize_history_assistant_keeps_semantic_hint(self) -> None:
        normalized = _normalize_history_assistant(
            "932\nThe decline in net earnings is explicitly stated as $932 million in the post-text."
        )
        self.assertIn("932 [number]", normalized)
        self.assertIn("decline in net earnings", normalized.lower())

    def test_normalize_history_assistant_percent_to_ratio_with_hint(self) -> None:
        normalized = _normalize_history_assistant(
            "13.40%\nThat is the decline as a percentage of 2015 net earnings."
        )
        self.assertIn("0.134 [percent]", normalized)
        self.assertIn("2015 net earnings", normalized.lower())


if __name__ == "__main__":
    unittest.main()
