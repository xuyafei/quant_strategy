"""小资金人工确认实盘单。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.manual_confirmation import (
    build_manual_confirmation_report,
    build_manual_confirmation_sheet,
    save_manual_confirmation,
)
from live.paper_runner import run_daily_paper_trade


class ManualConfirmationTests(unittest.TestCase):
    def test_build_and_save_manual_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="LIVE",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["freeze_manifest"] = {
                "as_of_date": "2024-01-30",
                "manifest_path": str(Path(td) / "freeze_manifest.json"),
                "live_policy": {"strategy": "LIVE"},
                "stock_pool": {"sha256": "abc123"},
                "git": {"commit": "deadbeef", "is_dirty": False},
            }
            monitor = pd.DataFrame(
                [
                    {"factor": "ROE", "status": "WATCH", "reasons": "ic_deteriorated"},
                    {"factor": "MOMENTUM", "status": "OK", "reasons": ""},
                ]
            )

            sheet = build_manual_confirmation_sheet(result, factor_monitor=monitor)
            report = build_manual_confirmation_report(result, sheet, factor_monitor=monitor)
            paths = save_manual_confirmation(settings, result, factor_monitor=monitor)

            self.assertEqual(list(sheet["manual_action"]), ["CONFIRM_WITH_CAUTION"])
            self.assertEqual(str(sheet["factor_health_status"].iloc[0]), "WATCH")
            self.assertEqual(str(sheet["freeze_as_of_date"].iloc[0]), "2024-01-30")
            self.assertEqual(str(sheet["freeze_stock_pool_sha256"].iloc[0]), "abc123")
            self.assertIn("小资金人工确认实盘单", report)
            self.assertIn("版本冻结", report)
            self.assertIn("2024-01-30", report)
            self.assertIn("ROE:WATCH:ic_deteriorated", report)
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
