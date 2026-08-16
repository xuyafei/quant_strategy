"""SQLite 数据仓库读写与缓存导出测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from storage.warehouse import (
    export_factor_panel_cache,
    export_price_cache,
    load_factor_panel_daily,
    load_fina_indicator,
    load_prices_daily,
    upsert_factor_panel_daily,
    upsert_fina_indicator,
    upsert_prices_daily,
)


class StorageWarehouseTests(unittest.TestCase):
    def test_upsert_prices_and_export_main_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "quant.db"
            prices = pd.DataFrame(
                {
                    "trade_date": ["2026-08-13", "2026-08-14", "2026-08-14"],
                    "ts_code": ["600519.SH", "600519.SH", "000001.SZ"],
                    "open": [10.0, 11.0, 20.0],
                    "high": [10.5, 11.5, 21.0],
                    "low": [9.8, 10.8, 19.5],
                    "close": [10.2, 11.2, 20.5],
                    "volume": [1000, 1100, 2000],
                    "amount": [10200, 12320, 41000],
                }
            )
            self.assertEqual(upsert_prices_daily(db, prices, source="unit"), 3)

            updated = prices.copy()
            updated.loc[updated["ts_code"] == "600519.SH", "close"] = [10.3, 11.3]
            self.assertEqual(upsert_prices_daily(db, updated, source="unit2"), 3)

            loaded = load_prices_daily(db, start="2026-08-14")
            self.assertEqual(len(loaded), 2)
            close = loaded.set_index("ts_code").loc["600519.SH", "close"]
            self.assertAlmostEqual(float(close), 11.3)

            paths = export_price_cache(db, Path(td) / "cache", start="2026-08-14")
            self.assertTrue(paths["prices_long"].is_file())
            self.assertTrue(paths["prices_wide_close"].is_file())
            wide = pd.read_csv(paths["prices_wide_close"], index_col=0)
            self.assertIn("600519.SH", wide.columns)
            self.assertAlmostEqual(float(wide.loc["2026-08-14", "600519.SH"]), 11.3)

    def test_upsert_fina_indicator_keeps_cash_flow_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "quant.db"
            fina = pd.DataFrame(
                {
                    "ts_code": ["600519.SH"],
                    "ann_date": ["2026-04-30"],
                    "end_date": ["2025-12-31"],
                    "roe": [25.0],
                    "fcff_ps": [3.2],
                    "ocf_to_profit": [1.1],
                }
            )
            self.assertEqual(upsert_fina_indicator(db, fina, source="unit"), 1)
            loaded = load_fina_indicator(db)
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded.loc[0, "fcff_ps"]), 3.2)
            self.assertAlmostEqual(float(loaded.loc[0, "ocf_to_profit"]), 1.1)

    def test_upsert_factor_panel_and_export_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "quant.db"
            idx = pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2026-08-13"), "600519.SH"),
                    (pd.Timestamp("2026-08-14"), "600519.SH"),
                ],
                names=["date", "symbol"],
            )
            panel = pd.DataFrame({"MOMENTUM": [0.1, 0.2], "ROE": [1.0, 1.1]}, index=idx)
            self.assertEqual(upsert_factor_panel_daily(db, panel, source="unit"), 4)
            loaded = load_factor_panel_daily(db, start="2026-08-14")
            self.assertEqual(list(loaded.columns), ["MOMENTUM", "ROE"])
            self.assertAlmostEqual(float(loaded.iloc[0]["MOMENTUM"]), 0.2)

            path = export_factor_panel_cache(db, Path(td) / "cache", start="2026-08-14")
            exported = pd.read_csv(path)
            self.assertEqual(list(exported.columns), ["date", "symbol", "MOMENTUM", "ROE"])
            self.assertEqual(exported.loc[0, "symbol"], "600519.SH")


if __name__ == "__main__":
    unittest.main()
