import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.daily_paper_cli import load_latest_target_weights
from live.strategy_universe import (
    equal_weight_rebalance_log,
    latest_universe_snapshot,
    save_runtime_exports,
)


class LiveStrategyUniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.to_datetime(["2025-01-06", "2025-01-13"])
        symbols = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        self.score = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=index)
        self.report = pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": symbol,
                    "eligible": True,
                    "eligible_LARGE_CAP": symbol != "C",
                    "name": symbol,
                    "industry_l1": "电子",
                    "circ_mv_yuan": 1.0,
                    "adv20_yuan": 2.0,
                }
                for date in dates
                for symbol in symbols
            ]
        )

    def test_rebalance_log_is_paper_trade_compatible(self) -> None:
        log = equal_weight_rebalance_log(
            self.score, strategy="TEST", universe="LARGE_CAP", top_k=2
        )
        self.assertTrue((log.groupby("date")["weight"].sum().round(12) == 1.0).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            log.to_csv(path, index=False)
            date, weights = load_latest_target_weights(path, trade_date="2025-01-13")
            self.assertEqual(date, pd.Timestamp("2025-01-13"))
            self.assertAlmostEqual(float(weights.sum()), 1.0)
            self.assertEqual(set(weights.index), {"B", "C"})

    def test_latest_snapshot_uses_profile_eligibility(self) -> None:
        snapshot = latest_universe_snapshot(self.report, universe="LARGE_CAP")
        self.assertEqual(set(snapshot["symbol"]), {"A", "B"})
        self.assertEqual(snapshot["date"].nunique(), 1)
        self.assertEqual(snapshot["date"].iloc[0], pd.Timestamp("2025-01-13"))

    def test_save_runtime_exports_writes_manifest_logs_and_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_runtime_exports(
                Path(tmp),
                report=self.report,
                shifted_scores={"TEST": self.score},
                universe_by_strategy={"TEST": "LARGE_CAP"},
                top_k_by_strategy={"TEST": 2},
                protocol_sha256="abc",
            )
            self.assertTrue(paths["manifest"].is_file())
            self.assertTrue(paths["rebalance_log_TEST"].is_file())
            self.assertTrue(paths["universe_snapshot_LARGE_CAP"].is_file())


if __name__ == "__main__":
    unittest.main()
