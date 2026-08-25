"""Live deviation analysis tests."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.deviation_analysis import (
    build_live_deviation_analysis,
    save_live_deviation_analysis,
    summarize_deviation_status,
)


class DeviationAnalysisTests(unittest.TestCase):
    def test_pass_when_paper_tracks_target_and_broker_matches(self) -> None:
        snapshots = pd.DataFrame(
            {
                "date": ["2026-08-25"],
                "cash": [0.0],
                "market_value": [10000.0],
                "total_asset": [10000.0],
                "n_positions": [2],
            }
        )
        target = pd.Series({"AAA": 0.5, "BBB": 0.5})
        positions = pd.DataFrame({"symbol": ["AAA", "BBB"], "shares": [500, 250]})
        prices = pd.DataFrame({"date": ["2026-08-25"], "AAA": [10.0], "BBB": [20.0]})
        summary, position = build_live_deviation_analysis(
            strategy="LIVE",
            trade_date="2026-08-25",
            snapshots=snapshots,
            target_weights=target,
            paper_positions=positions,
            broker_positions=positions,
            prices=prices,
        )

        status, detail = summarize_deviation_status(summary)
        self.assertEqual(status, "PASS")
        self.assertEqual(detail, "ok")
        self.assertEqual(set(position["status"]), {"PASS"})

    def test_detects_target_and_broker_deviation(self) -> None:
        snapshots = pd.DataFrame(
            {
                "date": ["2026-08-25"],
                "cash": [1000.0],
                "market_value": [9000.0],
                "total_asset": [10000.0],
                "n_positions": [1],
            }
        )
        target = pd.Series({"AAA": 0.5, "BBB": 0.5})
        paper_positions = pd.DataFrame({"symbol": ["AAA"], "shares": [300]})
        broker_positions = pd.DataFrame({"symbol": ["AAA", "BBB"], "shares": [280, 100]})
        prices = pd.DataFrame({"date": ["2026-08-25"], "AAA": [10.0], "BBB": [20.0]})
        feedback = pd.DataFrame(
            {
                "execution_status": ["FILLED", "NOT_EXECUTED"],
                "price_slippage_pct": [0.0, 0.0],
            }
        )

        summary, position = build_live_deviation_analysis(
            strategy="LIVE",
            trade_date="2026-08-25",
            snapshots=snapshots,
            target_weights=target,
            paper_positions=paper_positions,
            broker_positions=broker_positions,
            prices=prices,
            execution_feedback=feedback,
            weight_watch_threshold=0.02,
            weight_block_threshold=0.05,
            max_unfilled_ratio=0.2,
        )

        status, detail = summarize_deviation_status(summary)
        self.assertEqual(status, "BLOCK")
        self.assertIn("target_tracking.max_abs_weight_diff=BLOCK", detail)
        self.assertIn("WATCH", set(summary["status"]))
        self.assertIn("BLOCK", set(position["status"]))

    def test_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            snapshots = pd.DataFrame(
                {
                    "date": ["2026-08-25"],
                    "cash": [0.0],
                    "market_value": [10000.0],
                    "total_asset": [10000.0],
                    "n_positions": [1],
                }
            )
            target = {"AAA": 1.0}
            positions = pd.DataFrame({"symbol": ["AAA"], "shares": [1000]})
            prices = pd.DataFrame({"date": ["2026-08-25"], "AAA": [10.0]})
            summary, position = build_live_deviation_analysis(
                strategy="LIVE",
                trade_date="2026-08-25",
                snapshots=snapshots,
                target_weights=target,
                paper_positions=positions,
                prices=prices,
            )
            paths = save_live_deviation_analysis(
                settings,
                strategy="LIVE",
                trade_date="2026-08-25",
                summary=summary,
                position=position,
            )
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["position_deviation"].is_file())
            self.assertTrue(paths["markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
