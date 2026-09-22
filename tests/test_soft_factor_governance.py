from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.build_soft_factor_governance_backtest import (
    build_governance_log,
    classify_governance_state,
)


class SoftFactorGovernanceTests(unittest.TestCase):
    def test_old_reject_becomes_watch_when_data_is_valid(self) -> None:
        row = pd.Series(
            {
                "as_of_date": pd.Timestamp("2025-02-07"),
                "history_end": pd.Timestamp("2025-02-06"),
                "coverage": 0.95,
                "valid_symbols": 280,
                "decision": "REJECT",
            }
        )
        state, reason = classify_governance_state(
            row,
            minimum_coverage=0.80,
            minimum_valid_symbols=100,
        )
        self.assertEqual(state, "WATCH")
        self.assertIn("reject", reason)

    def test_data_failure_quarantines_even_an_old_pass(self) -> None:
        row = pd.Series(
            {
                "as_of_date": pd.Timestamp("2025-02-07"),
                "history_end": pd.Timestamp("2025-02-06"),
                "coverage": 0.50,
                "valid_symbols": 280,
                "decision": "PASS",
            }
        )
        state, _ = classify_governance_state(
            row,
            minimum_coverage=0.80,
            minimum_valid_symbols=100,
        )
        self.assertEqual(state, "QUARANTINE")

    def test_main_keeps_watch_full_and_shadow_only_halves_multiplier(self) -> None:
        date = pd.Timestamp("2025-02-07")
        decisions = pd.DataFrame(
            {
                "factor": ["A", "B"],
                "decision": ["PASS", "REJECT"],
                "reasons": ["", "weak"],
                "as_of_date": [date, date],
                "history_start": [pd.Timestamp("2024-01-01")] * 2,
                "history_end": [pd.Timestamp("2025-02-06")] * 2,
                "coverage": [0.99, 0.99],
                "valid_symbols": [300, 300],
            }
        )
        variants = {
            "SOFT_GOVERNANCE_MAIN": {
                "active_multiplier": 1.0,
                "watch_multiplier": 1.0,
            },
            "SOFT_WATCH_HALF_SHADOW": {
                "active_multiplier": 1.0,
                "watch_multiplier": 0.5,
            },
        }
        out = build_governance_log(
            decisions,
            {"TEST": ["A", "B"]},
            variants,
            evaluation_start=date,
            evaluation_end=date,
            minimum_coverage=0.80,
            minimum_valid_symbols=100,
        )
        main = out[out["variant"] == "SOFT_GOVERNANCE_MAIN"].set_index("factor")
        half = out[out["variant"] == "SOFT_WATCH_HALF_SHADOW"].set_index("factor")
        self.assertAlmostEqual(main.loc["A", "within_family_weight"], 0.5)
        self.assertAlmostEqual(main.loc["B", "within_family_weight"], 0.5)
        self.assertAlmostEqual(half.loc["A", "within_family_weight"], 2.0 / 3.0)
        self.assertAlmostEqual(half.loc["B", "within_family_weight"], 1.0 / 3.0)
        self.assertTrue(np.allclose(out.groupby(["variant", "date"])["total_factor_weight"].sum(), 1.0))


if __name__ == "__main__":
    unittest.main()
