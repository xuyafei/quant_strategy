import unittest
from dataclasses import replace

import pandas as pd

from analysis.factor_validation import (
    build_factor_decay_monitor,
    build_multi_horizon_out_of_sample_validation,
    build_out_of_sample_validation,
    build_rolling_out_of_sample_validation,
    split_train_validation_dates,
    summarize_multi_horizon_validation,
    summarize_rolling_out_of_sample_validation,
)
from config import get_settings


class FactorValidationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-01-01", periods=80)
        self.prices = pd.DataFrame(
            {
                "AAA": [10.0 + i * 0.20 for i in range(len(dates))],
                "BBB": [10.0 + i * 0.05 for i in range(len(dates))],
                "CCC": [20.0 - i * 0.05 for i in range(len(dates))],
                "DDD": [15.0 - i * 0.10 for i in range(len(dates))],
            },
            index=dates,
        )
        idx = pd.MultiIndex.from_product([dates, self.prices.columns], names=["date", "symbol"])
        good_vals = [4.0, 3.0, 2.0, 1.0] * len(dates)
        weak_vals = [1.0, 2.0, 3.0, 4.0] * len(dates)
        self.panel = pd.DataFrame({"GOOD": good_vals, "WEAK": weak_vals}, index=idx)

    def test_split_train_validation_dates(self):
        train, validation = split_train_validation_dates(self.panel, self.prices, train_ratio=0.5)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(validation), 0)
        self.assertLess(train[-1], validation[0])

    def test_build_out_of_sample_validation_and_monitor(self):
        settings = replace(get_settings(), rebalance_freq="D", top_k=1)
        validation = build_out_of_sample_validation(
            self.panel,
            self.prices,
            settings,
            factors=["GOOD", "WEAK"],
            train_ratio=0.5,
        )
        self.assertEqual(set(validation["factor"]), {"GOOD", "WEAK"})
        self.assertIn("validation_excess_ann_return", validation.columns)
        good = validation[validation["factor"] == "GOOD"].iloc[0]
        weak = validation[validation["factor"] == "WEAK"].iloc[0]
        self.assertGreater(float(good["validation_excess_ann_return"]), 0.0)
        self.assertLess(float(weak["validation_excess_ann_return"]), 0.0)

        monitor = build_factor_decay_monitor(validation)
        self.assertEqual(set(monitor["factor"]), {"GOOD", "WEAK"})
        weak_status = str(monitor[monitor["factor"] == "WEAK"]["status"].iloc[0])
        self.assertIn(weak_status, {"WATCH", "DEGRADED", "FAILED"})

    def test_build_rolling_out_of_sample_validation_and_summary(self):
        settings = replace(
            get_settings(),
            rebalance_freq="D",
            top_k=1,
            rolling_oos_train_days=30,
            rolling_oos_validation_days=10,
            rolling_oos_step_days=10,
            rolling_oos_min_validation_days=8,
        )
        rolling = build_rolling_out_of_sample_validation(
            self.panel,
            self.prices,
            settings,
            factors=["GOOD", "WEAK"],
        )
        self.assertGreaterEqual(int(rolling["window_id"].nunique()), 4)
        self.assertEqual(set(rolling["factor"]), {"GOOD", "WEAK"})
        self.assertIn("validation_excess_ann_return", rolling.columns)

        summary = summarize_rolling_out_of_sample_validation(rolling)
        self.assertEqual(set(summary["factor"]), {"GOOD", "WEAK"})
        good = summary[summary["factor"] == "GOOD"].iloc[0]
        weak = summary[summary["factor"] == "WEAK"].iloc[0]
        self.assertGreater(float(good["excess_positive_window_rate"]), 0.5)
        self.assertLess(float(weak["excess_positive_window_rate"]), 0.5)

    def test_multi_horizon_validation_includes_next_rebalance(self):
        settings = replace(get_settings(), rebalance_freq="ME", top_k=1)

        validation = build_multi_horizon_out_of_sample_validation(
            self.panel,
            self.prices,
            settings,
            factors=["GOOD", "WEAK"],
            horizons=(1, 5, 20),
        )
        self.assertEqual(
            set(validation["horizon"]),
            {"1D", "5D", "20D", "NEXT_REBALANCE"},
        )
        summary = summarize_multi_horizon_validation(validation)
        self.assertEqual(set(summary["factor"]), {"GOOD", "WEAK"})
        good = summary[summary["factor"] == "GOOD"].iloc[0]
        self.assertGreaterEqual(int(good["supportive_horizons"]), 3)


if __name__ == "__main__":
    unittest.main()
