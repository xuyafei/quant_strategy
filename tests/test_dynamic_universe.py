from __future__ import annotations

import unittest

import pandas as pd

from universe.dynamic import build_dynamic_universe_report, eligibility_from_report


def _calendar() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=12)
    return pd.DataFrame({"cal_date": dates, "is_open": 1})


class DynamicUniverseTests(unittest.TestCase):
    def test_dynamic_universe_applies_point_in_time_age_st_and_liquidity(self) -> None:
        calendar = _calendar()
        dates = pd.DatetimeIndex(calendar["cal_date"])
        symbols = ["000001.SZ", "000002.SZ", "600001.SH"]
        rows = []
        for date in dates:
            for symbol in symbols:
                rows.append(
                    {
                        "trade_date": date,
                        "ts_code": symbol,
                        "close": 10.0,
                        "amount_yuan": 80_000_000.0 if symbol != "600001.SH" else 5_000_000.0,
                    }
                )
        prices = pd.DataFrame(rows)
        basic = pd.DataFrame(
            {
                "ts_code": symbols,
                "name": ["甲", "乙", "丙"],
                "industry": ["银行", "电子", "医药"],
                "exchange": ["SZSE", "SZSE", "SSE"],
                "market": ["主板", "主板", "主板"],
                "list_status": ["L", "L", "L"],
                "list_date": ["20200101", "20240108", "20200101"],
                "delist_date": ["", "", ""],
            }
        )
        names = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["ST甲"],
                "start_date": ["20240110"],
                "end_date": ["20240112"],
                "ann_date": ["20240109"],
            }
        )
        report = build_dynamic_universe_report(
            prices,
            basic,
            names,
            calendar,
            [dates[7], dates[10], dates[11]],
            min_listing_sessions=5,
            liquidity_window=5,
            min_valid_days=4,
            min_adv_yuan=50_000_000.0,
        )
        mask = eligibility_from_report(report)
        self.assertFalse(bool(mask.loc[(dates[7], "000001.SZ")]))
        self.assertTrue(bool(mask.loc[(dates[10], "000001.SZ")]))
        self.assertFalse(bool(mask.loc[(dates[7], "000002.SZ")]))
        self.assertTrue(bool(mask.loc[(dates[11], "000002.SZ")]))
        self.assertFalse(report.loc[report["symbol"].eq("600001.SH"), "eligible"].any())
        self.assertEqual(
            report.loc[
                (report["date"].eq(dates[11])) & report["symbol"].eq("600001.SH"),
                "exclude_reason",
            ].iloc[0],
            "adv20_below_min",
        )

    def test_missing_decision_day_price_blocks_new_eligibility(self) -> None:
        calendar = _calendar()
        dates = pd.DatetimeIndex(calendar["cal_date"])
        prices = pd.DataFrame(
            {
                "trade_date": dates[:-1],
                "ts_code": "000001.SZ",
                "close": 10.0,
                "amount_yuan": 100_000_000.0,
            }
        )
        basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "list_date": ["20200101"],
                "delist_date": [""],
            }
        )
        report = build_dynamic_universe_report(
            prices,
            basic,
            None,
            calendar,
            [dates[-1]],
            min_listing_sessions=2,
            liquidity_window=5,
            min_valid_days=4,
            min_adv_yuan=1.0,
        )
        self.assertFalse(bool(report.iloc[0]["eligible"]))
        self.assertIn("no_price_on_decision_date", str(report.iloc[0]["exclude_reason"]))


if __name__ == "__main__":
    unittest.main()
