from __future__ import annotations

import unittest

from src.evaluation.normalization import normalized_numeric_match, parse_numeric_answer


class TestEvaluationDeterministic(unittest.TestCase):
    def test_parse_numeric_answer(self) -> None:
        self.assertEqual(parse_numeric_answer("14.1%"), (14.1, True))
        self.assertEqual(parse_numeric_answer("0.14136"), (0.14136, False))
        self.assertEqual(parse_numeric_answer("$1,234.50"), (1234.5, False))
        self.assertIsNone(parse_numeric_answer("n/a"))

    def test_normalized_numeric_match_percent_ratio_equivalence(self) -> None:
        self.assertTrue(normalized_numeric_match("14.1%", "0.141"))
        self.assertTrue(normalized_numeric_match("0.42457", "42.457%"))
        self.assertTrue(normalized_numeric_match("1.6383", "1.64%"))
        self.assertTrue(normalized_numeric_match("1234.5", "$1,234.50"))
        self.assertFalse(normalized_numeric_match("14.1%", "0.12"))

    def test_normalized_numeric_match_with_unit_scale_words(self) -> None:
        self.assertTrue(
            normalized_numeric_match(
                "2.3",
                "$2,300,000,000.00\nThe total expense for repairs and maintenance incurred in 2013 was $2.3 billion.",
            )
        )
        self.assertTrue(
            normalized_numeric_match(
                "2.1",
                "2100000000\nThe total expense for repairs and maintenance incurred in 2012 was $2.1 billion.",
            )
        )
        self.assertTrue(
            normalized_numeric_match(
                "0.2",
                "200000000\nThe difference between the repairs and maintenance expenses for 2013 and 2012, in billions.",
            )
        )

    def test_normalized_numeric_match_with_unit_scale_without_unit_word(self) -> None:
        self.assertTrue(
            normalized_numeric_match(
                "73.0",
                "$73,000,000.00\nThe net change is the difference between the 2002 and 2001 earnings for non-utility nuclear.",
            )
        )

if __name__ == "__main__":
    unittest.main()
