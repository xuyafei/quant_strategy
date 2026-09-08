import unittest

import pandas as pd

from analysis.factor_diagnostics import (
    batch_factor_group_returns,
    batch_factor_long_excess,
    factor_group_return_detail,
    factor_long_excess_summary,
    factor_long_only_nav,
)


class FactorDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=8)
        self.prices = pd.DataFrame(
            {
                "AAA": [10, 11, 12, 13, 14, 15, 16, 17],
                "BBB": [10, 10, 10, 10, 10, 10, 10, 10],
                "CCC": [10, 9, 8, 7, 6, 5, 4, 3],
            },
            index=self.dates,
        )
        idx = pd.MultiIndex.from_product([self.dates, self.prices.columns], names=["date", "symbol"])
        self.factor = pd.Series(
            [3.0, 2.0, 1.0] * len(self.dates),
            index=idx,
            name="QUALITY",
        )

    def test_factor_long_only_nav_picks_top_symbol(self):
        nav, log = factor_long_only_nav(
            self.factor,
            self.prices,
            top_k=1,
            rebalance_freq="D",
            name="QUALITY",
        )
        self.assertFalse(nav.empty)
        self.assertGreater(float(nav.iloc[-1]), 1.0)
        self.assertTrue(log)
        self.assertTrue(all(rec["picks"] == ["AAA"] for rec in log if rec["picks"]))

    def test_batch_factor_long_excess_returns_summary(self):
        panel = self.factor.to_frame()
        summary, navs = batch_factor_long_excess(
            panel,
            self.prices,
            top_k=1,
            rebalance_freq="D",
        )
        self.assertEqual(list(summary["factor"]), ["QUALITY"])
        self.assertIn("excess_ann_return", summary.columns)
        self.assertIn("information_ratio", summary.columns)
        self.assertIn("QUALITY", navs)

    def test_long_excess_uses_explicit_benchmark_universe(self):
        _, default_stats = factor_long_excess_summary(
            self.factor,
            self.prices,
            factor_name="QUALITY",
            top_k=1,
            rebalance_freq="D",
        )
        _, matched_stats = factor_long_excess_summary(
            self.factor,
            self.prices,
            factor_name="QUALITY",
            top_k=1,
            rebalance_freq="D",
            benchmark_prices=self.prices[["AAA"]],
        )
        self.assertGreater(
            float(default_stats["excess_ann_return"]),
            float(matched_stats["excess_ann_return"]),
        )

    def test_factor_group_returns_show_top_group_outperforming(self):
        detail = factor_group_return_detail(
            self.factor,
            self.prices,
            factor_name="QUALITY",
            group_count=3,
            rebalance_freq="D",
        )
        self.assertFalse(detail.empty)
        first_day = detail[detail["date"] == self.dates[0]]
        low = float(first_day[first_day["group"] == 1]["period_return"].iloc[0])
        high = float(first_day[first_day["group"] == 3]["period_return"].iloc[0])
        self.assertGreater(high, low)

    def test_batch_factor_group_returns_returns_summary(self):
        panel = self.factor.to_frame()
        detail, summary = batch_factor_group_returns(
            panel,
            self.prices,
            group_count=3,
            rebalance_freq="D",
        )
        self.assertFalse(detail.empty)
        self.assertFalse(summary.empty)
        self.assertIn("top_minus_bottom_mean", summary.columns)
        self.assertIn("monotonicity_score", summary.columns)
        quality_top = summary[(summary["factor"] == "QUALITY") & (summary["group"] == 3)].iloc[0]
        self.assertGreater(float(quality_top["top_minus_bottom_mean"]), 0.0)


if __name__ == "__main__":
    unittest.main()
