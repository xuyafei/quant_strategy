import unittest

import pandas as pd

from live.universe_history import (
    build_membership_intervals,
    filter_prices_by_membership,
    mask_factor_panel_by_membership,
    mask_wide_prices_by_membership,
)


class UniverseHistoryTest(unittest.TestCase):
    def test_reconstructs_and_filters_point_in_time_membership(self) -> None:
        changes = pd.DataFrame(
            [
                {"effective_date": "2024-01-03", "action": "ENTER", "ts_code": "000003.SZ"},
                {"effective_date": "2024-01-03", "action": "EXIT", "ts_code": "000001.SZ"},
            ]
        )
        membership = build_membership_intervals(
            ["000002.SZ", "000003.SZ"],
            changes,
            start="2024-01-01",
            end="2024-01-05",
            as_of="2024-01-05",
            expected_size=2,
        )
        prices = pd.DataFrame(
            [
                {"trade_date": date, "ts_code": symbol, "close": 1.0}
                for date in pd.date_range("2024-01-01", "2024-01-05")
                for symbol in ["000001.SZ", "000002.SZ", "000003.SZ"]
            ]
        )
        filtered = filter_prices_by_membership(prices, membership)
        before = set(filtered.loc[filtered["trade_date"] == pd.Timestamp("2024-01-02"), "ts_code"])
        after = set(filtered.loc[filtered["trade_date"] == pd.Timestamp("2024-01-03"), "ts_code"])
        self.assertEqual(before, {"000001.SZ", "000002.SZ"})
        self.assertEqual(after, {"000002.SZ", "000003.SZ"})
        self.assertEqual(filtered.groupby("trade_date")["ts_code"].nunique().unique().tolist(), [2])

    def test_membership_masks_selection_but_preserves_source_valuation_prices(self) -> None:
        days = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        membership = pd.DataFrame(
            [
                {"ts_code": "AAA", "effective_from": days[0], "effective_to": days[1]},
                {"ts_code": "BBB", "effective_from": days[2], "effective_to": days[2]},
            ]
        )
        prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.0], "BBB": [20.0, 21.0, 22.0]}, index=days)
        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        panel = pd.DataFrame({"factor": 1.0}, index=idx)

        masked_panel = mask_factor_panel_by_membership(panel, membership)
        benchmark_prices = mask_wide_prices_by_membership(prices, membership)

        self.assertTrue(pd.isna(masked_panel.loc[(days[2], "AAA"), "factor"]))
        self.assertEqual(masked_panel.loc[(days[2], "BBB"), "factor"], 1.0)
        self.assertTrue(pd.isna(benchmark_prices.loc[days[2], "AAA"]))
        self.assertEqual(prices.loc[days[2], "AAA"], 12.0)


if __name__ == "__main__":
    unittest.main()
