"""Daily live SOP tests."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from config import get_settings
from live.live_sop import build_live_daily_sop, build_live_sop_report, save_live_daily_sop


class LiveSOPTests(unittest.TestCase):
    def test_sop_contains_core_daily_commands(self) -> None:
        settings = get_settings()
        sop = build_live_daily_sop(
            settings,
            strategy="FUSED_ROLLING_SCORE_WEIGHTED",
            trade_date="2026-08-27",
        )
        command_text = "\n".join(sop["command"].astype(str).tolist())
        self.assertIn("python main.py", command_text)
        self.assertIn("scripts/run_daily_paper.py", command_text)
        self.assertIn("scripts/build_live_run_monitor.py", command_text)
        self.assertIn("scripts/build_semi_auto_checklist.py", command_text)
        self.assertIn("scripts/build_execution_feedback.py", command_text)
        self.assertIn("scripts/build_live_performance_attribution.py", command_text)
        self.assertIn("scripts/build_live_deviation_analysis.py", command_text)
        self.assertEqual(str(sop.loc[0, "phase"]), "pre_market")
        self.assertIn("next_day_review", set(sop["phase"]))

    def test_sop_can_include_broker_reconcile_step(self) -> None:
        settings = get_settings()
        sop = build_live_daily_sop(
            settings,
            strategy="LIVE",
            trade_date="2026-08-27",
            broker_positions_path=Path("broker_positions.csv"),
            include_broker_reconcile=True,
        )
        self.assertIn("纸面账户与真实账户只读对账", set(sop["step"]))
        command_text = "\n".join(sop["command"].astype(str).tolist())
        self.assertIn("scripts/reconcile_paper_broker.py", command_text)
        self.assertIn("broker_positions.csv", command_text)

    def test_save_sop_outputs_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            sop = build_live_daily_sop(settings, strategy="LIVE", trade_date="2026-08-27")
            paths = save_live_daily_sop(settings, strategy="LIVE", trade_date="2026-08-27", sop=sop)
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["markdown"].is_file())
            text = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("小资金实盘每日 SOP", text)
            self.assertIn("READY_FOR_MANUAL_ORDER", text)
            self.assertIn("output/live_sop", build_live_sop_report(sop))


if __name__ == "__main__":
    unittest.main()
