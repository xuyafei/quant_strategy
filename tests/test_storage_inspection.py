"""SQLite 数据库巡检日报测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from storage.inspection import (
    build_database_quality_report,
    database_quality_report_markdown,
    save_database_quality_report,
)
from storage.warehouse import upsert_factor_panel_daily, upsert_fina_indicator, upsert_prices_daily


class StorageInspectionTests(unittest.TestCase):
    def test_database_quality_report_detects_price_and_factor_health(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "quant.db"
            cache = root / "cache"
            cache.mkdir()
            pd.DataFrame({"trade_date": ["2026-08-14"]}).to_csv(cache / "prices_long.csv", index=False)
            pd.DataFrame({"date": ["2026-08-14"], "600519.SH": [10.0]}).to_csv(
                cache / "prices_wide_close.csv",
                index=False,
            )
            pd.DataFrame({"date": ["2026-08-14"], "symbol": ["600519.SH"], "MOMENTUM": [1.0]}).to_csv(
                cache / "factor_panel.csv",
                index=False,
            )

            upsert_prices_daily(
                db,
                pd.DataFrame(
                    {
                        "trade_date": ["2026-08-14", "2026-08-14"],
                        "ts_code": ["600519.SH", "000001.SZ"],
                        "open": [10, 20],
                        "high": [11, 21],
                        "low": [9, 19],
                        "close": [10.5, 20.5],
                        "volume": [1000, 2000],
                    }
                ),
            )
            upsert_fina_indicator(
                db,
                pd.DataFrame(
                    {
                        "ts_code": ["600519.SH"],
                        "ann_date": ["2026-04-30"],
                        "end_date": ["2025-12-31"],
                        "roe": [20.0],
                        "fcff_ps": [2.0],
                    }
                ),
            )
            idx = pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-08-14"), "600519.SH")],
                names=["date", "symbol"],
            )
            upsert_factor_panel_daily(db, pd.DataFrame({"MOMENTUM": [0.1]}, index=idx))

            parts = build_database_quality_report(
                db,
                expected_symbols=["600519.SH", "000001.SZ", "000002.SZ"],
                as_of_date="2026-08-16",
                cache_dir=cache,
            )
            self.assertIn("summary", parts)
            self.assertEqual(parts["price_health"].loc[0, "missing_symbols"], 1)
            self.assertIn("MOMENTUM", set(parts["factor_health"]["factor"]))
            self.assertTrue((parts["cache_file_health"]["exists"] == True).all())

            md = database_quality_report_markdown(parts)
            self.assertIn("数据库巡检日报", md)
            self.assertIn("overall_status", md)

    def test_save_database_quality_report_writes_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "quant.db"
            upsert_prices_daily(
                db,
                pd.DataFrame(
                    {
                        "trade_date": ["2026-08-14"],
                        "ts_code": ["600519.SH"],
                        "open": [10],
                        "high": [11],
                        "low": [9],
                        "close": [10.5],
                    }
                ),
            )
            paths = save_database_quality_report(db, root / "report", as_of_date="2026-08-14")
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
