import unittest

import pandas as pd

from analysis.factor_selection import build_factor_selection_table, selected_factors_for_fusion


class FactorSelectionTests(unittest.TestCase):
    def test_build_factor_selection_table(self):
        factors = ["GOOD", "WEAK", "LOW_COVERAGE"]
        coverage = pd.DataFrame(
            [
                {"factor": "GOOD", "coverage": 0.95, "valid_dates": 120, "valid_symbols": 20},
                {"factor": "WEAK", "coverage": 0.90, "valid_dates": 120, "valid_symbols": 20},
                {"factor": "LOW_COVERAGE", "coverage": 0.20, "valid_dates": 120, "valid_symbols": 20},
            ]
        )
        weights = pd.DataFrame(
            [
                {
                    "factor": "GOOD",
                    "factor_score": 0.80,
                    "fusion_weight": 0.70,
                    "mean_ic": 0.03,
                    "ic_ir": 0.50,
                    "positive_rate": 0.60,
                    "top_minus_bottom_ann": 0.10,
                    "monotonicity_score": 0.80,
                },
                {
                    "factor": "WEAK",
                    "factor_score": 0.10,
                    "fusion_weight": 0.20,
                    "mean_ic": -0.01,
                    "ic_ir": -0.10,
                    "positive_rate": 0.45,
                    "top_minus_bottom_ann": -0.05,
                    "monotonicity_score": 0.30,
                },
                {
                    "factor": "LOW_COVERAGE",
                    "factor_score": 0.90,
                    "fusion_weight": 0.10,
                    "mean_ic": 0.05,
                    "ic_ir": 0.70,
                    "positive_rate": 0.65,
                    "top_minus_bottom_ann": 0.20,
                    "monotonicity_score": 0.90,
                },
            ]
        )
        monitor = pd.DataFrame(
            [
                {
                    "factor": "GOOD",
                    "status": "OK",
                    "severity": 0,
                    "validation_ic_mean": 0.02,
                    "validation_positive_rate": 0.58,
                    "validation_excess_ann_return": 0.05,
                    "validation_top_minus_bottom_ann": 0.08,
                },
                {
                    "factor": "WEAK",
                    "status": "WATCH",
                    "severity": 1,
                    "validation_ic_mean": -0.01,
                    "validation_positive_rate": 0.45,
                    "validation_excess_ann_return": -0.02,
                    "validation_top_minus_bottom_ann": -0.03,
                },
                {
                    "factor": "LOW_COVERAGE",
                    "status": "OK",
                    "severity": 0,
                    "validation_ic_mean": 0.03,
                    "validation_positive_rate": 0.60,
                    "validation_excess_ann_return": 0.10,
                    "validation_top_minus_bottom_ann": 0.10,
                },
            ]
        )

        table = build_factor_selection_table(
            factors=factors,
            factor_coverage=coverage,
            factor_weight_summary=weights,
            factor_decay_monitor=monitor,
        )

        decision = dict(zip(table["factor"], table["decision"]))
        self.assertEqual(decision["GOOD"], "PASS")
        self.assertEqual(decision["WEAK"], "WATCH")
        self.assertEqual(decision["LOW_COVERAGE"], "REJECT")
        low_reasons = str(table.loc[table["factor"] == "LOW_COVERAGE", "reasons"].iloc[0])
        self.assertIn("coverage_below_threshold", low_reasons)

    def test_selected_factors_for_fusion_prefers_pass_then_watch_then_fallback(self):
        table = pd.DataFrame(
            {
                "factor": ["GOOD", "WEAK"],
                "decision": ["PASS", "WATCH"],
            }
        )
        self.assertEqual(selected_factors_for_fusion(table, ["GOOD", "WEAK"]), ["GOOD"])

        table["decision"] = ["REJECT", "WATCH"]
        self.assertEqual(selected_factors_for_fusion(table, ["GOOD", "WEAK"]), ["WEAK"])

        table["decision"] = ["REJECT", "REJECT"]
        self.assertEqual(selected_factors_for_fusion(table, ["GOOD", "WEAK"]), ["GOOD", "WEAK"])

    def test_constant_factor_without_valid_ic_is_rejected(self):
        table = build_factor_selection_table(
            factors=["NO_SIGNAL"],
            factor_coverage=pd.DataFrame(
                [{"factor": "NO_SIGNAL", "coverage": 1.0, "valid_dates": 120, "valid_symbols": 50}]
            ),
            factor_weight_summary=pd.DataFrame(
                [
                    {
                        "factor": "NO_SIGNAL",
                        "factor_score": 0.8,
                        "fusion_weight": 1.0,
                        "mean_ic": float("nan"),
                        "ic_ir": float("nan"),
                        "positive_rate": float("nan"),
                        "top_minus_bottom_ann": 0.1,
                        "monotonicity_score": 0.8,
                    }
                ]
            ),
            factor_decay_monitor=pd.DataFrame(
                [{"factor": "NO_SIGNAL", "status": "WATCH", "severity": 1}]
            ),
        )

        self.assertEqual(table.loc[0, "decision"], "REJECT")
        self.assertIn("ic_unavailable", table.loc[0, "reasons"])

    def test_enhanced_validation_can_override_one_day_fixed_split_failure(self):
        table = build_factor_selection_table(
            factors=["SLOW_FACTOR"],
            factor_coverage=pd.DataFrame(
                [{"factor": "SLOW_FACTOR", "coverage": 0.8, "valid_dates": 120}]
            ),
            factor_weight_summary=pd.DataFrame(
                [
                    {
                        "factor": "SLOW_FACTOR",
                        "factor_score": 0.4,
                        "mean_ic": 0.01,
                        "positive_rate": 0.55,
                        "top_minus_bottom_ann": 0.05,
                        "monotonicity_score": 0.6,
                    }
                ]
            ),
            factor_decay_monitor=pd.DataFrame(
                [{"factor": "SLOW_FACTOR", "status": "DEGRADED"}]
            ),
            multi_horizon_summary=pd.DataFrame(
                [
                    {
                        "factor": "SLOW_FACTOR",
                        "status": "ROBUST",
                        "supportive_horizon_rate": 0.75,
                    }
                ]
            ),
            rolling_out_of_sample_summary=pd.DataFrame(
                [
                    {
                        "factor": "SLOW_FACTOR",
                        "status": "WATCH",
                        "supportive_window_rate": 0.5,
                    }
                ]
            ),
        )

        self.assertEqual(table.loc[0, "decision"], "PASS")
        self.assertIn("fixed_split_degraded", table.loc[0, "reasons"])


if __name__ == "__main__":
    unittest.main()
