"""Semi-auto checklist tests."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.semi_auto_checklist import (
    build_semi_auto_checklist,
    save_semi_auto_checklist,
)


class SemiAutoChecklistTests(unittest.TestCase):
    def test_ready_when_required_items_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            freeze = Path(td) / "freeze_manifest.json"
            report = Path(td) / "paper.md"
            freeze.write_text("{}", encoding="utf-8")
            report.write_text("ok", encoding="utf-8")
            run_monitor = pd.DataFrame({"status": ["PASS"], "category": ["data"], "item": ["target_weights"]})
            risk = pd.DataFrame({"status": ["PASS"], "module": ["订单预检查"]})
            manual = pd.DataFrame({"symbol": ["AAA"], "check_status": ["PASS"], "manual_action": ["EXECUTE"]})
            deviation = pd.DataFrame(
                {
                    "module": ["target_tracking"],
                    "status": ["PASS"],
                    "metric": ["max_abs_weight_diff"],
                }
            )

            checklist, decision = build_semi_auto_checklist(
                strategy="LIVE",
                trade_date="2026-08-26",
                freeze_manifest_path=freeze,
                paper_report_path=report,
                run_monitor=run_monitor,
                risk_control_report=risk,
                manual_confirmation=manual,
                deviation_summary=deviation,
            )

            self.assertEqual(str(decision.loc[0, "decision"]), "READY_FOR_MANUAL_ORDER")
            self.assertEqual(str(decision.loc[0, "status"]), "PASS")
            self.assertEqual(int((checklist["required_before_order"]).sum()), 6)

    def test_watch_decision_when_manual_review_needed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            freeze = Path(td) / "freeze_manifest.json"
            report = Path(td) / "paper.md"
            freeze.write_text("{}", encoding="utf-8")
            report.write_text("ok", encoding="utf-8")
            run_monitor = pd.DataFrame({"status": ["PASS"], "category": ["data"], "item": ["target_weights"]})
            risk = pd.DataFrame({"status": ["WATCH"], "module": ["容量与冲击成本"]})
            manual = pd.DataFrame({"symbol": ["AAA"], "check_status": ["PASS"]})
            deviation = pd.DataFrame({"module": ["target_tracking"], "status": ["PASS"], "metric": ["x"]})

            _checklist, decision = build_semi_auto_checklist(
                strategy="LIVE",
                trade_date="2026-08-26",
                freeze_manifest_path=freeze,
                paper_report_path=report,
                run_monitor=run_monitor,
                risk_control_report=risk,
                manual_confirmation=manual,
                deviation_summary=deviation,
            )
            self.assertEqual(str(decision.loc[0, "decision"]), "MANUAL_REVIEW")
            self.assertEqual(str(decision.loc[0, "status"]), "WATCH")

    def test_blocks_when_required_item_missing_and_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            checklist, decision = build_semi_auto_checklist(
                strategy="LIVE",
                trade_date="2026-08-26",
            )
            self.assertEqual(str(decision.loc[0, "decision"]), "DO_NOT_TRADE")
            self.assertEqual(str(decision.loc[0, "status"]), "BLOCK")
            paths = save_semi_auto_checklist(
                settings,
                strategy="LIVE",
                trade_date="2026-08-26",
                checklist=checklist,
                decision=decision,
            )
            self.assertTrue(paths["checklist"].is_file())
            self.assertTrue(paths["decision"].is_file())
            self.assertTrue(paths["markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
