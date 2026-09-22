from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.build_multi_memory_consensus_backtest import (
    SOURCE_ORDER,
    _consensus_support_diagnostic,
    _state_forward_return_diagnostic,
    build_consensus_score,
    build_disagreement_monitor,
    cross_sectional_source_ranks,
)


class MultiMemoryConsensusTests(unittest.TestCase):
    def test_source_ranks_are_computed_within_each_date(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-05"), "A"),
                (pd.Timestamp("2024-01-05"), "B"),
                (pd.Timestamp("2024-01-12"), "A"),
                (pd.Timestamp("2024-01-12"), "B"),
            ],
            names=["date", "symbol"],
        )
        scores = {
            source: pd.Series([1.0, 2.0, 10.0, 5.0], index=index)
            for source in SOURCE_ORDER
        }
        ranks = cross_sectional_source_ranks(scores)
        self.assertEqual(ranks.loc[(pd.Timestamp("2024-01-05"), "B"), "EXPANDING"], 1.0)
        self.assertEqual(ranks.loc[(pd.Timestamp("2024-01-12"), "A"), "EXPANDING"], 1.0)

    def test_consensus_requires_minimum_valid_sources(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-05"), "A"), (pd.Timestamp("2024-01-05"), "B")],
            names=["date", "symbol"],
        )
        ranks = pd.DataFrame(
            {
                "ROLLING_260D": [0.2, np.nan],
                "ROLLING_504D": [0.6, np.nan],
                "EXPANDING": [0.8, 1.0],
            },
            index=index,
        )
        score = build_consensus_score(ranks, SOURCE_ORDER, minimum_valid_sources=2)
        self.assertAlmostEqual(score.iloc[0], 0.6)
        self.assertTrue(np.isnan(score.iloc[1]))

    def test_disagreement_monitor_maps_green_yellow_and_red(self) -> None:
        dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "symbol"])
        ranks = pd.DataFrame(index=index, columns=SOURCE_ORDER, dtype=float)
        ranks.loc[dates[0], :] = [
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
        ]
        ranks.loc[dates[1], :] = [
            [1.0, 1.0, 0.1],
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 1.0],
        ]
        ranks.loc[dates[2], :] = [
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.5],
            [0.1, 0.1, 1.0],
        ]
        monitor = build_disagreement_monitor(ranks, threshold=0.40, top_k=1)
        self.assertEqual(monitor["risk_state"].tolist(), ["GREEN", "YELLOW", "RED"])
        self.assertEqual(monitor["gross_exposure_multiplier"].tolist(), [1.0, 0.75, 0.5])

    def test_posthoc_support_counts_source_top10_votes(self) -> None:
        date = pd.Timestamp("2024-01-05")
        index = pd.MultiIndex.from_product(
            [[date], ["A", "B", "C"]], names=["date", "symbol"]
        )
        ranks = pd.DataFrame(
            {
                "ROLLING_260D": [1.0, 0.5, 0.1],
                "ROLLING_504D": [1.0, 0.5, 0.1],
                "EXPANDING": [1.0, 0.5, 0.1],
            },
            index=index,
        )
        meta = {
            "rebalance_log": [
                {"date": date, "selected_picks": ["A", "B", "C"]}
            ]
        }
        detail, summary = _consensus_support_diagnostic(ranks, meta, date)
        self.assertEqual(detail.loc[0, "selected_with_3_source_votes"], 3)
        self.assertEqual(
            summary["FULL_ACTIVE"]["average_share_with_at_least_2_source_votes"],
            1.0,
        )

    def test_posthoc_state_return_uses_next_rebalance_interval(self) -> None:
        dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
        nav = pd.Series([1.0, 1.1, 0.99], index=dates)
        benchmark = pd.Series([1.0, 1.05, 1.05], index=dates)
        monitor = pd.DataFrame(
            {"date": dates, "risk_state": ["GREEN", "RED", "YELLOW"]}
        )
        meta = {"rebalance_log": [{"date": date} for date in dates]}
        detail, summary = _state_forward_return_diagnostic(
            nav, benchmark, monitor, meta, dates[0]
        )
        self.assertEqual(len(detail), 2)
        self.assertAlmostEqual(
            summary["GREEN"]["mean_full_risk_consensus_next_return"], 0.10
        )
        self.assertAlmostEqual(
            summary["RED"]["mean_full_risk_consensus_next_return"], -0.10
        )


if __name__ == "__main__":
    unittest.main()
