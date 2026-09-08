"""机器学习打分因子测试。"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from config import get_settings
from factors.factor_ml import ML_SCORE_NAME, build_ml_score_factor, forward_return_label


class MLScoreFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2024-01-01", periods=90)
        self.symbols = ["AAA", "BBB", "CCC"]
        base = np.arange(len(self.dates), dtype=float)
        self.prices = pd.DataFrame(
            {
                "AAA": 10.0 + base * 0.10,
                "BBB": 20.0 + base * 0.03,
                "CCC": 30.0 - base * 0.02,
            },
            index=self.dates,
        )
        idx = pd.MultiIndex.from_product([self.dates, self.symbols], names=["date", "symbol"])
        rows = []
        for dt in self.dates:
            for sym in self.symbols:
                rows.append(
                    {
                        "MOMENTUM": {"AAA": 1.0, "BBB": 0.2, "CCC": -0.5}[sym],
                        "VOLATILITY": {"AAA": 0.1, "BBB": 0.2, "CCC": 0.3}[sym],
                    }
                )
        self.panel = pd.DataFrame(rows, index=idx)

    def test_forward_return_label(self) -> None:
        label = forward_return_label(self.prices, forward_days=2)
        expected = self.prices.loc[self.dates[2], "AAA"] / self.prices.loc[self.dates[0], "AAA"] - 1.0
        self.assertAlmostEqual(float(label.loc[(self.dates[0], "AAA")]), float(expected))
        self.assertTrue(pd.isna(label.loc[(self.dates[-1], "AAA")]))

    def test_build_ml_score_factor_uses_past_training_window(self) -> None:
        settings = replace(
            get_settings(),
            ml_score_model="hist_gradient_boosting",
            ml_score_forward_days=5,
            ml_score_train_lookback_days=40,
            ml_score_min_train_days=10,
            ml_score_min_train_rows=20,
            ml_score_refit_every_days=10,
        )

        score, log = build_ml_score_factor(self.panel, self.prices, settings)

        self.assertEqual(score.name, ML_SCORE_NAME)
        self.assertGreater(int(score.notna().sum()), 0)
        self.assertFalse(log.empty)
        for rec in log.to_dict("records"):
            self.assertLess(pd.Timestamp(rec["train_end"]), pd.Timestamp(rec["prediction_date"]))
            self.assertGreaterEqual(int(rec["n_train_days"]), settings.ml_score_min_train_days)
            self.assertEqual(int(rec["n_features"]), 2)

    def test_does_not_predict_rows_with_all_features_missing(self) -> None:
        settings = replace(
            get_settings(),
            ml_score_forward_days=5,
            ml_score_train_lookback_days=40,
            ml_score_min_train_days=10,
            ml_score_min_train_rows=20,
            ml_score_refit_every_days=10,
        )
        masked = self.panel.copy()
        inactive_index = (self.dates[-1], "CCC")
        masked.loc[inactive_index, :] = np.nan

        score, _ = build_ml_score_factor(masked, self.prices, settings)

        self.assertTrue(pd.isna(score.loc[inactive_index]))


if __name__ == "__main__":
    unittest.main()
