from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_history_window_stability_audit import (
    _actual_turnover,
    _style_l1,
    jaccard,
)


class HistoryWindowStabilityAuditTests(unittest.TestCase):
    def test_jaccard_handles_overlap_and_two_empty_sets(self) -> None:
        self.assertAlmostEqual(jaccard({"A", "B"}, {"B", "C"}), 1 / 3)
        self.assertEqual(jaccard(set(), set()), 1.0)

    def test_style_l1_aligns_missing_factors_as_zero(self) -> None:
        left = pd.Series({"VALUE": 0.7, "QUALITY": 0.3})
        right = pd.Series({"VALUE": 0.4, "MOMENTUM": 0.6})
        self.assertAlmostEqual(_style_l1(left, right), 1.2)

    def test_actual_turnover_uses_target_weight_changes(self) -> None:
        log = [
            {"date": "2024-01-05", "picks": ["A", "B"], "weights": [0.6, 0.4]},
            {"date": "2024-01-12", "picks": ["A", "C"], "weights": [0.5, 0.5]},
        ]
        self.assertAlmostEqual(_actual_turnover(log), 2.0)


if __name__ == "__main__":
    unittest.main()
