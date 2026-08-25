"""真实成交回填与执行偏差分析。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.execution_feedback import (
    build_execution_feedback,
    build_execution_feedback_report,
    build_next_day_execution_review,
    build_next_day_review_report,
    save_execution_feedback,
)


class ExecutionFeedbackTests(unittest.TestCase):
    def test_build_execution_feedback_status_and_summary(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "strategy": "LIVE",
                    "symbol": "AAA",
                    "side": "BUY",
                    "delta_shares": 500,
                    "price": 10.0,
                    "estimated_amount": 5000.0,
                    "check_status": "PASS",
                    "manual_action": "CONFIRM_MANUALLY",
                    "executed_qty": 500,
                    "executed_price": 10.1,
                    "operator": "me",
                    "confirmed_at": "2024-01-31 14:50",
                    "execution_note": "filled",
                },
                {
                    "date": "2024-01-31",
                    "strategy": "LIVE",
                    "symbol": "BBB",
                    "side": "SELL",
                    "delta_shares": -400,
                    "price": 20.0,
                    "estimated_amount": 8000.0,
                    "check_status": "PASS",
                    "manual_action": "CONFIRM_MANUALLY",
                    "executed_qty": 200,
                    "executed_price": 19.8,
                    "operator": "me",
                    "confirmed_at": "2024-01-31 14:51",
                    "execution_note": "partial",
                },
                {
                    "date": "2024-01-31",
                    "strategy": "LIVE",
                    "symbol": "CCC",
                    "side": "BUY",
                    "delta_shares": 100,
                    "price": 30.0,
                    "estimated_amount": 3000.0,
                    "check_status": "BLOCK",
                    "manual_action": "DO_NOT_EXECUTE",
                    "executed_qty": "",
                    "executed_price": "",
                    "operator": "",
                    "confirmed_at": "",
                    "execution_note": "blocked",
                },
            ]
        )

        detail, summary = build_execution_feedback(frame)

        self.assertEqual(list(detail["execution_status"]), ["FILLED", "PARTIAL", "BLOCKED"])
        self.assertAlmostEqual(float(detail.loc[0, "price_slippage_pct"]), 0.01)
        self.assertEqual(int(summary.loc[0, "n_filled"]), 1)
        self.assertEqual(int(summary.loc[0, "n_partial"]), 1)
        self.assertEqual(int(summary.loc[0, "n_blocked"]), 1)
        self.assertAlmostEqual(float(summary.loc[0, "executed_buy_amount"]), 5050.0)
        self.assertAlmostEqual(float(summary.loc[0, "executed_sell_amount"]), 3960.0)

        report = build_execution_feedback_report(detail, summary)
        self.assertIn("真实成交回填与执行偏差分析", report)
        self.assertIn("PARTIAL", report)

        prices = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-02-01"],
                "AAA": [10.0, 10.4],
                "BBB": [20.0, 19.5],
                "CCC": [30.0, 30.2],
            }
        )
        review, review_summary = build_next_day_execution_review(detail, prices)
        self.assertEqual(str(review_summary.loc[0, "review_date"]), "2024-02-01")
        self.assertEqual(int(review_summary.loc[0, "n_reviewed"]), 2)
        self.assertAlmostEqual(float(review_summary.loc[0, "buy_next_day_pnl"]), 150.0)
        self.assertAlmostEqual(float(review_summary.loc[0, "sell_avoidance_pnl"]), 60.0)
        review_report = build_next_day_review_report(review, review_summary)
        self.assertIn("真实成交次日复盘", review_report)

    def test_save_execution_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
            )
            frame = pd.DataFrame(
                [
                    {
                        "date": "2024-01-31",
                        "strategy": "LIVE",
                        "symbol": "AAA",
                        "side": "BUY",
                        "delta_shares": 100,
                        "price": 10.0,
                        "estimated_amount": 1000.0,
                        "check_status": "PASS",
                        "manual_action": "CONFIRM_MANUALLY",
                        "executed_qty": 100,
                        "executed_price": 10.0,
                    }
                ]
            )
            detail, summary = build_execution_feedback(frame)
            prices = pd.DataFrame({"date": ["2024-02-01"], "AAA": [10.2]})
            review, review_summary = build_next_day_execution_review(detail, prices, review_date="2024-02-01")
            paths = save_execution_feedback(
                settings,
                detail,
                summary,
                next_day_review=review,
                next_day_summary=review_summary,
            )

            self.assertTrue(paths["detail"].is_file())
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["report"].is_file())
            self.assertTrue(paths["next_day_review"].is_file())
            self.assertTrue(paths["next_day_summary"].is_file())
            self.assertTrue(paths["next_day_report"].is_file())


if __name__ == "__main__":
    unittest.main()
