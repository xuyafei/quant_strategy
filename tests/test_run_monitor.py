"""Live run monitor tests."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.run_monitor import build_live_run_monitor, save_live_run_monitor, summarize_live_run_monitor


class LiveRunMonitorTests(unittest.TestCase):
    def test_monitor_detects_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "output"
            settings = replace(get_settings(), output_dir=output, data_dir=root / "data")
            strategy = "LIVE"
            trade_date = "2026-08-24"
            safe = strategy

            (output / "live_freeze" / trade_date).mkdir(parents=True)
            freeze = output / "live_freeze" / trade_date / "freeze_manifest.json"
            freeze.write_text("{}", encoding="utf-8")

            (output / "rebalance_logs").mkdir(parents=True)
            pd.DataFrame(
                {
                    "date": [trade_date, trade_date],
                    "symbol": ["AAA", "BBB"],
                    "weight": [0.5, 0.5],
                    "selected": [True, True],
                }
            ).to_csv(output / "rebalance_logs" / ("%s.csv" % safe), index=False)

            (output / "cache").mkdir(parents=True)
            pd.DataFrame({"date": [trade_date], "AAA": [10.0], "BBB": [20.0]}).to_csv(
                output / "cache" / "prices_wide_close.csv",
                index=False,
            )

            (output / "live_orders" / safe).mkdir(parents=True)
            pd.DataFrame({"symbol": ["AAA"]}).to_csv(
                output / "live_orders" / safe / ("%s_manual_confirm.csv" % trade_date),
                index=False,
            )

            (output / "paper_account" / safe).mkdir(parents=True)
            pd.DataFrame(
                {"date": [trade_date], "cash": [1000.0], "market_value": [0.0], "total_asset": [1000.0]}
            ).to_csv(output / "paper_account" / safe / "snapshots.csv", index=False)

            (output / "risk_control_reports" / safe).mkdir(parents=True)
            pd.DataFrame({"status": ["PASS"], "module": ["运行检查"]}).to_csv(
                output / "risk_control_reports" / safe / "daily_risk_control_report_20260824.csv",
                index=False,
            )

            (output / "paper_reports" / safe).mkdir(parents=True)
            (output / "paper_reports" / safe / ("%s.md" % trade_date)).write_text("ok", encoding="utf-8")

            monitor = build_live_run_monitor(
                settings,
                strategy=strategy,
                trade_date=trade_date,
                freeze_manifest_path=freeze,
            )
            status, detail = summarize_live_run_monitor(monitor)
            self.assertEqual(status, "NA")
            self.assertIn("execution.execution_feedback=NA", detail)
            self.assertEqual(int((monitor["status"] == "BLOCK").sum()), 0)

            paths = save_live_run_monitor(settings, monitor, strategy=strategy, trade_date=trade_date)
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["markdown"].is_file())

    def test_missing_core_files_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output", data_dir=Path(td) / "data")
            monitor = build_live_run_monitor(settings, strategy="LIVE", trade_date="2026-08-24")
            status, detail = summarize_live_run_monitor(monitor)
            self.assertEqual(status, "BLOCK")
            self.assertIn("version.freeze_manifest=BLOCK", detail)


if __name__ == "__main__":
    unittest.main()
