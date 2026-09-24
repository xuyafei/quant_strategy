import unittest

import numpy as np
import pandas as pd

from strategies.style import build_fixed_family_score, build_focused_style_score
from universe.strategy import (
    StrategyUniverseProfile,
    build_strategy_universe_report,
    daily_basic_for_index,
    eligibility_for_profile,
    point_in_time_industry,
    profiles_from_config,
)


class StrategyUniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.to_datetime(["2025-01-03", "2025-01-10"])
        self.symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
        rows = []
        for date in self.dates:
            for symbol in self.symbols:
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "eligible": symbol != "000004.SZ",
                        "exclude_reason": "" if symbol != "000004.SZ" else "st_or_star_st",
                    }
                )
        self.base = pd.DataFrame(rows)
        daily_rows = []
        for date in self.dates:
            for number, symbol in enumerate(self.symbols, start=1):
                daily_rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "ts_code": symbol,
                        "pe_ttm": [10.0, -5.0, 30.0, 8.0][number - 1],
                        "pb": [1.0, np.nan, 3.0, 0.8][number - 1],
                        "ps_ttm": 2.0,
                        "total_mv": number * 200.0,
                        "circ_mv": number * 100.0,
                    }
                )
        self.daily = pd.DataFrame(daily_rows)
        self.index = pd.MultiIndex.from_product(
            [self.dates, self.symbols], names=["date", "symbol"]
        )
        self.factors = pd.DataFrame(
            {
                "FREE_CASH_FLOW_YIELD": [0.1, np.nan, 0.3, 0.4] * 2,
                "REVENUE_GROWTH": [5.0, 10.0, 20.0, 30.0] * 2,
                "PROFIT_GROWTH": [1.0, np.nan, 12.0, 15.0] * 2,
            },
            index=self.index,
        )
        self.membership = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "l1_code": "801080.SI",
                    "l1_name": "电子",
                    "in_date": "20240101",
                    "out_date": "20250103",
                },
                {
                    "ts_code": "000001.SZ",
                    "l1_code": "801780.SI",
                    "l1_name": "银行",
                    "in_date": "20250110",
                    "out_date": None,
                },
                {
                    "ts_code": "000002.SZ",
                    "l1_code": "801750.SI",
                    "l1_name": "计算机",
                    "in_date": "20240101",
                    "out_date": None,
                },
                {
                    "ts_code": "000003.SZ",
                    "l1_code": "801780.SI",
                    "l1_name": "银行",
                    "in_date": "20240101",
                    "out_date": None,
                },
            ]
        )
        self.profiles = profiles_from_config(
            [
                {"name": "LARGE_CAP", "kind": "size_large", "parameters": {"minimum_percentile": 0.70}},
                {"name": "MID_SMALL_CAP", "kind": "size_mid_small", "parameters": {"maximum_percentile": 0.70}},
                {"name": "VALUE_READY", "kind": "value", "parameters": {"minimum_components": 2}},
                {"name": "GROWTH_READY", "kind": "growth", "parameters": {"minimum_components": 2}},
                {
                    "name": "TECH_THEME",
                    "kind": "industry_theme",
                    "parameters": {"l1_names": ["电子", "计算机"]},
                },
            ]
        )

    def test_profiles_are_derived_from_base_and_point_in_time_inputs(self) -> None:
        report = build_strategy_universe_report(
            self.base, self.daily, self.factors, self.membership, self.profiles
        )
        first = report[report["date"].eq(self.dates[0])].set_index("symbol")
        second = report[report["date"].eq(self.dates[1])].set_index("symbol")

        self.assertEqual(set(first.index[first["eligible_LARGE_CAP"]]), {"000003.SZ"})
        self.assertEqual(
            set(first.index[first["eligible_MID_SMALL_CAP"]]),
            {"000001.SZ", "000002.SZ"},
        )
        self.assertTrue(first.loc["000001.SZ", "eligible_VALUE_READY"])
        self.assertFalse(first.loc["000002.SZ", "eligible_VALUE_READY"])
        self.assertTrue(first.loc["000003.SZ", "eligible_GROWTH_READY"])
        self.assertFalse(first.loc["000002.SZ", "eligible_GROWTH_READY"])
        self.assertTrue(first.loc["000001.SZ", "eligible_TECH_THEME"])
        self.assertFalse(second.loc["000001.SZ", "eligible_TECH_THEME"])
        self.assertTrue(second.loc["000002.SZ", "eligible_TECH_THEME"])
        for profile in self.profiles:
            self.assertFalse(first.loc["000004.SZ", profile.eligibility_column])

    def test_daily_basic_asof_never_uses_future_record(self) -> None:
        one = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-03"), "000001.SZ")], names=["date", "symbol"]
        )
        future = self.daily[
            (self.daily["trade_date"] == "20250110")
            & (self.daily["ts_code"] == "000001.SZ")
        ]
        attached = daily_basic_for_index(one, future)
        self.assertTrue(pd.isna(attached.iloc[0]["daily_basic_date"]))
        self.assertTrue(pd.isna(attached.iloc[0]["circ_mv_yuan"]))

    def test_daily_basic_accepts_microsecond_csv_dates(self) -> None:
        one = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-03"), "000001.SZ")], names=["date", "symbol"]
        )
        same_day = self.daily[
            (self.daily["trade_date"] == "20250103")
            & (self.daily["ts_code"] == "000001.SZ")
        ].copy()
        attached = daily_basic_for_index(one, same_day)
        self.assertEqual(attached.iloc[0]["circ_mv_yuan"], 1_000_000.0)

    def test_industry_intervals_are_inclusive_and_historical(self) -> None:
        mapped = point_in_time_industry(self.index, self.membership)
        self.assertEqual(mapped.loc[(self.dates[0], "000001.SZ")], "电子")
        self.assertEqual(mapped.loc[(self.dates[1], "000001.SZ")], "银行")

    def test_eligibility_helper_returns_named_mask(self) -> None:
        report = build_strategy_universe_report(
            self.base, self.daily, self.factors, self.membership, self.profiles
        )
        mask = eligibility_for_profile(report, "LARGE_CAP")
        self.assertEqual(mask.name, "LARGE_CAP")
        self.assertEqual(int(mask.sum()), 2)


class StyleStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.to_datetime(["2025-01-03"])
        symbols = ["A", "B", "C", "D"]
        self.index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        self.raw = pd.DataFrame(
            {
                "V1": [1.0, 2.0, 3.0, 4.0],
                "V2": [1.0, 2.0, 3.0, 4.0],
                "G1": [4.0, 3.0, 2.0, 1.0],
            },
            index=self.index,
        )
        self.eligible = pd.Series(True, index=self.index)
        self.industry = pd.Series("X", index=self.index)

    def test_focused_style_score_prefers_stronger_equal_weight_components(self) -> None:
        score, components = build_focused_style_score(
            self.raw,
            self.eligible,
            self.industry,
            ["V1", "V2"],
            minimum_components=2,
        )
        self.assertEqual(score.groupby(level="date").idxmax().iloc[0][1], "D")
        self.assertEqual(list(components.columns), ["V1", "V2"])

    def test_fixed_family_score_requires_configured_families(self) -> None:
        score, family = build_fixed_family_score(
            self.raw,
            self.eligible,
            self.industry,
            {"VALUE": ["V1", "V2"], "GROWTH": ["G1"]},
        )
        self.assertEqual(set(family.columns), {"VALUE", "GROWTH"})
        self.assertEqual(len(score), 4)


if __name__ == "__main__":
    unittest.main()
