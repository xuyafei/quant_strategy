from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.build_no_admission_rolling_ic_weight_backtest import (
    _known_ic_window,
    _within_family_target_weights,
    build_point_in_time_weight_log,
)


class NoAdmissionRollingIcWeightTests(unittest.TestCase):
    def test_known_ic_window_requires_complete_forward_return(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=20)
        rebalance = dates[15]
        window, signal_end, outcome_end = _known_ic_window(
            dates,
            rebalance,
            forward_days=5,
            lookback_days=4,
        )
        self.assertEqual(len(window), 4)
        self.assertEqual(signal_end, dates[9])
        self.assertEqual(outcome_end, dates[14])
        self.assertLess(outcome_end, rebalance)

    def test_positive_ic_evidence_changes_weights_without_dropping_factors(self) -> None:
        metrics = pd.DataFrame(
            {
                "factor": ["A", "B", "C"],
                "mean_ic": [0.06, 0.02, -0.01],
                "ic_ir": [0.50, 0.10, -0.10],
            }
        )
        weights, reason = _within_family_target_weights(
            metrics,
            ["A", "B", "C"],
            minimum_weight=0.10,
        )
        self.assertEqual(reason, "positive_mean_ic_icir_blend")
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue((weights > 0.0).all())
        self.assertGreater(weights["A"], weights["B"])
        self.assertGreater(weights["B"], weights["C"])
        self.assertAlmostEqual(weights["C"], 0.10)

    def test_weight_log_is_point_in_time_and_keeps_every_factor(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=180)
        rebalances = dates[130::5]
        ic = pd.Series(np.linspace(-0.1, 0.1, len(dates)), index=dates)
        log = build_point_in_time_weight_log(
            {"A": ic, "B": -ic},
            dates,
            rebalances,
            {"TEST": ["A", "B"]},
            forward_days=5,
            lookback_days=120,
            minimum_valid_days=60,
            minimum_within_family_weight=0.10,
            smoothing=0.50,
        )
        self.assertEqual(set(log["factor"]), {"A", "B"})
        self.assertTrue((log["within_family_weight"] > 0.0).all())
        totals = log.groupby("date")["within_family_weight"].sum()
        self.assertTrue(np.allclose(totals.to_numpy(float), 1.0))
        self.assertTrue(
            (pd.to_datetime(log["ic_outcome_end"]) < pd.to_datetime(log["date"])).all()
        )


if __name__ == "__main__":
    unittest.main()
