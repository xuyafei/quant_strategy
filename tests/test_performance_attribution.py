"""Performance attribution tests."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.performance_attribution import (
    build_performance_attribution,
    save_performance_attribution,
)


class PerformanceAttributionTests(unittest.TestCase):
    def test_builds_summary_and_stock_contribution(self) -> None:
        snapshots = pd.DataFrame(
            {
                "date": ["2026-08-24", "2026-08-25"],
                "cash": [1000.0, 900.0],
                "market_value": [9000.0, 9400.0],
                "total_asset": [10000.0, 10300.0],
                "n_positions": [2, 2],
            }
        )
        prices = pd.DataFrame(
            {
                "date": ["2026-08-24", "2026-08-25"],
                "AAA": [10.0, 11.0],
                "BBB": [20.0, 19.0],
            }
        )
        positions = pd.DataFrame({"symbol": ["AAA", "BBB"], "shares": [500, 200]})
        feedback = pd.DataFrame(
            {
                "side": ["BUY", "SELL"],
                "suggested_price": [10.0, 20.0],
                "executed_price": [10.1, 19.8],
                "executed_qty": [100, 50],
            }
        )

        summary, stock = build_performance_attribution(
            snapshots,
            strategy="LIVE",
            trade_date="2026-08-25",
            prices=prices,
            positions=positions,
            execution_feedback=feedback,
        )

        rec = summary.iloc[0]
        self.assertEqual(rec["status"], "PASS")
        self.assertAlmostEqual(float(rec["account_return"]), 0.03)
        self.assertAlmostEqual(float(rec["benchmark_return"]), 0.025)
        self.assertAlmostEqual(float(rec["active_return"]), 0.005)
        self.assertAlmostEqual(float(rec["stock_contribution_return"]), 0.03)
        self.assertAlmostEqual(float(rec["execution_slippage_cost"]), 20.0)
        self.assertEqual(list(stock["symbol"]), ["AAA", "BBB"])
        self.assertAlmostEqual(float(stock.loc[stock["symbol"] == "AAA", "contribution_return"].iloc[0]), 0.05)

    def test_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            snapshots = pd.DataFrame(
                {
                    "date": ["2026-08-24", "2026-08-25"],
                    "cash": [1000.0, 1000.0],
                    "market_value": [9000.0, 9100.0],
                    "total_asset": [10000.0, 10100.0],
                    "n_positions": [1, 1],
                }
            )
            prices = pd.DataFrame({"date": ["2026-08-24", "2026-08-25"], "AAA": [10.0, 10.2]})
            positions = pd.DataFrame({"symbol": ["AAA"], "shares": [500]})
            summary, stock = build_performance_attribution(
                snapshots,
                strategy="LIVE",
                trade_date="2026-08-25",
                prices=prices,
                positions=positions,
            )
            paths = save_performance_attribution(
                settings,
                strategy="LIVE",
                trade_date="2026-08-25",
                summary=summary,
                stock=stock,
            )
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["stock_contribution"].is_file())
            self.assertTrue(paths["markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
