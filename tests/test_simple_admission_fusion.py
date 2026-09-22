from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_simple_admission_fusion_backtest import (
    build_fixed_family_fusion,
    select_factors_within_families,
)
from scripts.build_no_admission_fusion_backtest import _period_rows


class SimpleAdmissionFusionTests(unittest.TestCase):
    def test_selection_keeps_pass_only_and_prunes_within_family(self) -> None:
        selection = pd.DataFrame(
            {
                "factor": ["A", "B", "C", "D"],
                "decision": ["PASS", "PASS", "PASS", "WATCH"],
            }
        )
        redundancy = pd.DataFrame(
            [
                {
                    "factor_a": "A",
                    "factor_b": "B",
                    "recommended_keep": "A",
                    "recommended_drop": "B",
                }
            ]
        )
        selected, audit = select_factors_within_families(
            selection,
            redundancy,
            {"VALUE": ["A", "B"], "QUALITY": ["C", "D"]},
        )
        self.assertEqual(selected, {"VALUE": ["A"], "QUALITY": ["C"]})
        retained = set(
            audit.loc[
                audit["retained_after_within_family_redundancy"], "factor"
            ].astype(str)
        )
        self.assertEqual(retained, {"A", "C"})

    def test_fixed_fusion_respects_cutoff_and_point_in_time_mask(self) -> None:
        dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C"]], names=["date", "symbol"]
        )
        panel = pd.DataFrame(
            {
                "F1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "F2": [3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
            },
            index=index,
        )
        eligible = pd.Series(True, index=index)
        eligible.loc[(pd.Timestamp("2025-01-03"), "C")] = False
        fused, families = build_fixed_family_fusion(
            panel,
            {"VALUE": ["F1"], "QUALITY": ["F2"]},
            eligible,
            pd.Timestamp("2025-01-03"),
        )
        self.assertTrue(fused.xs(pd.Timestamp("2025-01-02"), level="date").isna().all())
        self.assertTrue(pd.isna(fused.loc[(pd.Timestamp("2025-01-03"), "C")]))
        self.assertEqual(list(families.columns), ["VALUE", "QUALITY"])

    def test_no_admission_period_rows_compare_every_strategy_to_same_benchmark(self) -> None:
        dates = pd.date_range("2025-01-03", periods=4, freq="D")
        benchmark = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
        strategies = {
            "A": pd.Series([1.0, 1.1, 1.1, 1.2], index=dates),
            "B": pd.Series([1.0, 0.9, 1.0, 1.1], index=dates),
        }
        rows = _period_rows(
            strategies,
            benchmark,
            {"FULL": (dates[0], dates[-1])},
        )
        self.assertEqual(set(rows["strategy"]), {"A", "B"})
        self.assertTrue((rows["benchmark_total_return"] == 0.0).all())
        self.assertGreater(
            float(rows.set_index("strategy").loc["A", "return_gap_vs_benchmark"]),
            float(rows.set_index("strategy").loc["B", "return_gap_vs_benchmark"]),
        )


if __name__ == "__main__":
    unittest.main()
