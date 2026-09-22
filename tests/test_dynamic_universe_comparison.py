from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_dynamic_universe_comparison import (
    _parse_tushare_dates,
    build_weekly_raw_factor_panel,
)


class DynamicUniverseComparisonTests(unittest.TestCase):
    def test_compact_integer_dates_do_not_become_1970(self) -> None:
        parsed = _parse_tushare_dates(pd.Series([20250103, "2025-01-06"]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2025-01-03"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2025-01-06"))

    def test_finance_merge_uses_only_announced_record(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=65)
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": "000001.SZ",
                "close": 10.0,
                "adj_close": range(10, 75),
            }
        )
        finance = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": [20240103, 20250103],
                "end_date": [20231231, 20241231],
                "eps": [1.0, 100.0],
                "ocfps": [2.0, 200.0],
            }
        )
        decision = pd.DatetimeIndex([dates[-1]])
        panel = build_weekly_raw_factor_panel(prices, finance, decision)
        row = panel.loc[(dates[-1], "000001.SZ")]
        self.assertAlmostEqual(float(row["PE"]), -10.0)
        self.assertAlmostEqual(float(row["FREE_CASH_FLOW_YIELD"]), 0.2)


if __name__ == "__main__":
    unittest.main()
