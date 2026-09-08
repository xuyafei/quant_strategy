"""data_quality：价格、因子与调仓日覆盖率自检。"""
from __future__ import annotations

import unittest

import pandas as pd

from analysis.data_quality import (
    factor_coverage,
    factor_daily_coverage,
    price_coverage,
    rebalance_coverage,
)


class TestDataQuality(unittest.TestCase):
    def test_price_and_factor_coverage(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {"AAA": [1.0, 2.0, 3.0], "BBB": [1.0, None, 2.0]},
            index=days,
        )
        pc = price_coverage(prices)
        self.assertEqual(list(pc["symbol"]), ["AAA", "BBB"])
        self.assertAlmostEqual(float(pc.loc[pc["symbol"] == "AAA", "coverage"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(pc.loc[pc["symbol"] == "BBB", "coverage"].iloc[0]), 2 / 3)

        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        panel = pd.DataFrame(
            {
                "MOMENTUM": [1.0, 2.0, 3.0, None, 5.0, 6.0],
                "VALUE": [None, None, 1.0, 2.0, 3.0, 4.0],
            },
            index=idx,
        )
        fc = factor_coverage(panel)
        self.assertEqual(set(fc["factor"]), {"MOMENTUM", "VALUE"})
        self.assertAlmostEqual(float(fc.loc[fc["factor"] == "MOMENTUM", "coverage"].iloc[0]), 5 / 6)

        daily = factor_daily_coverage(panel)
        self.assertEqual(set(daily["factor"]), {"MOMENTUM", "VALUE"})
        first_value = daily[(daily["date"] == days[0]) & (daily["factor"] == "VALUE")]
        self.assertEqual(int(first_value["valid_symbols"].iloc[0]), 0)

        eligible = pd.Series([True, False, True, True, True, False], index=idx)
        active_fc = factor_coverage(panel, eligible_mask=eligible)
        self.assertAlmostEqual(
            float(active_fc.loc[active_fc["factor"] == "MOMENTUM", "coverage"].iloc[0]),
            3 / 4,
        )
        active_daily = factor_daily_coverage(panel, eligible_mask=eligible)
        first_active = active_daily[
            (active_daily["date"] == days[0]) & (active_daily["factor"] == "MOMENTUM")
        ]
        self.assertEqual(int(first_active["total_rows"].iloc[0]), 1)

    def test_rebalance_coverage(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=2)
        prices = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [1.0, None]}, index=days)
        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        panel = pd.DataFrame(
            {"MOMENTUM": [1.0, 2.0, 3.0, 4.0], "VALUE": [1.0, None, 3.0, 4.0]},
            index=idx,
        )
        rc = rebalance_coverage(panel, prices, days, factors=["MOMENTUM", "VALUE"])
        self.assertEqual(int(rc.loc[0, "price_valid_symbols"]), 2)
        self.assertEqual(int(rc.loc[1, "price_valid_symbols"]), 1)
        self.assertEqual(int(rc.loc[1, "MOMENTUM_tradable_symbols"]), 1)
        self.assertEqual(int(rc.loc[1, "all_factor_tradable_symbols"]), 1)


if __name__ == "__main__":
    unittest.main()
