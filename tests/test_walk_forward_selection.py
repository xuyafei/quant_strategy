from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.walk_forward_selection import (
    build_walk_forward_factor_fusion,
    validate_walk_forward_audit,
)
from config import get_settings


class WalkForwardSelectionTests(unittest.TestCase):
    def test_audit_validator_rejects_same_day_history(self) -> None:
        decisions = pd.DataFrame(
            {
                "as_of_date": [pd.Timestamp("2024-02-01")],
                "history_end": [pd.Timestamp("2024-02-01")],
                "factor": ["MOMENTUM"],
            }
        )
        summary = pd.DataFrame(
            {"date": [pd.Timestamp("2024-02-01")], "signal_status": ["ACTIVE"]}
        )
        with self.assertRaisesRegex(ValueError, "调仓日或未来数据"):
            validate_walk_forward_audit(decisions, summary, pd.DataFrame())

    def setUp(self) -> None:
        self.dates = pd.bdate_range("2024-01-01", "2024-05-31")
        self.symbols = ["AAA", "BBB", "CCC"]
        self.prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 15.0, len(self.dates)),
                "BBB": np.linspace(12.0, 13.0, len(self.dates)),
                "CCC": np.linspace(14.0, 11.0, len(self.dates)),
            },
            index=self.dates,
        )
        index = pd.MultiIndex.from_product(
            [self.dates, self.symbols], names=["date", "symbol"]
        )
        values = np.tile([2.0, 0.0, -2.0], len(self.dates))
        self.panel = pd.DataFrame({"MOMENTUM": values}, index=index)
        self.settings = replace(
            get_settings(),
            rebalance_freq="ME",
            force_final_rebalance=False,
            top_k=1,
            walk_forward_min_history_days=20,
            walk_forward_min_rolling_windows=0,
            rolling_factor_weight_lookback_days=20,
            rolling_factor_weight_min_days=5,
            rolling_factor_weight_min_weight=0.0,
            rolling_factor_weight_max_weight=1.0,
            rolling_factor_weight_smoothing=1.0,
        )

    @staticmethod
    def _passing_gate(*args, **kwargs):
        selection = pd.DataFrame(
            {
                "factor": ["MOMENTUM"],
                "decision": ["PASS"],
                "selected_for_fusion": [True],
                "reasons": [""],
            }
        )
        rolling = pd.DataFrame({"factor": ["MOMENTUM"], "n_windows": [2]})
        return selection, rolling, pd.DataFrame(), pd.DataFrame()

    def test_gate_history_ends_before_every_rebalance(self) -> None:
        with (
            patch(
                "analysis.walk_forward_selection.evaluate_factor_gate_asof",
                side_effect=self._passing_gate,
            ),
            patch(
                "analysis.walk_forward_selection._style_weights",
                return_value=(pd.Series({"PRICE_VOLUME_STYLE": 1.0}), "test"),
            ),
        ):
            _, decisions, summary, _ = build_walk_forward_factor_fusion(
                self.panel,
                self.prices,
                self.settings,
                factors=["MOMENTUM"],
            )
        active = summary[summary["signal_status"] == "ACTIVE"]
        self.assertGreater(len(active), 0)
        self.assertTrue((pd.to_datetime(active["history_end"]) < pd.to_datetime(active["date"])).all())
        audited = decisions[decisions["decision"] == "PASS"]
        self.assertTrue(
            (pd.to_datetime(audited["history_end"]) < pd.to_datetime(audited["as_of_date"])).all()
        )

    def test_future_changes_do_not_change_prior_scores(self) -> None:
        cutoff = pd.Timestamp("2024-04-30")
        future_panel = self.panel.copy()
        future_prices = self.prices.copy()
        future_mask = future_panel.index.get_level_values("date") > cutoff
        future_panel.loc[future_mask, "MOMENTUM"] *= -100.0
        future_prices.loc[future_prices.index > cutoff, "AAA"] *= 5.0

        def build(panel, prices):
            with (
                patch(
                    "analysis.walk_forward_selection.evaluate_factor_gate_asof",
                    side_effect=self._passing_gate,
                ),
                patch(
                    "analysis.walk_forward_selection._style_weights",
                    return_value=(pd.Series({"PRICE_VOLUME_STYLE": 1.0}), "test"),
                ),
            ):
                return build_walk_forward_factor_fusion(
                    panel,
                    prices,
                    self.settings,
                    factors=["MOMENTUM"],
                )[0]

        original = build(self.panel, self.prices)
        changed = build(future_panel, future_prices)
        prior = original.index.get_level_values("date") <= cutoff
        pd.testing.assert_series_equal(original.loc[prior], changed.loc[prior])

    def test_history_lookback_caps_gate_input_to_trading_days(self) -> None:
        observed_days: list[int] = []

        def capture_gate(panel_history, *_args, **_kwargs):
            observed_days.append(
                panel_history.index.get_level_values("date").nunique()
            )
            return self._passing_gate()

        settings = replace(
            self.settings,
            walk_forward_history_lookback_days=20,
        )
        with (
            patch(
                "analysis.walk_forward_selection.evaluate_factor_gate_asof",
                side_effect=capture_gate,
            ),
            patch(
                "analysis.walk_forward_selection._style_weights",
                return_value=(pd.Series({"PRICE_VOLUME_STYLE": 1.0}), "test"),
            ),
        ):
            build_walk_forward_factor_fusion(
                self.panel,
                self.prices,
                settings,
                factors=["MOMENTUM"],
            )

        self.assertGreater(len(observed_days), 0)
        self.assertTrue(all(days == 20 for days in observed_days))

    def test_parallel_gate_evaluation_matches_serial_result(self) -> None:
        def build(n_jobs: int):
            with (
                patch(
                    "analysis.walk_forward_selection.evaluate_factor_gate_asof",
                    side_effect=self._passing_gate,
                ),
                patch(
                    "analysis.walk_forward_selection._style_weights",
                    return_value=(pd.Series({"PRICE_VOLUME_STYLE": 1.0}), "test"),
                ),
            ):
                return build_walk_forward_factor_fusion(
                    self.panel,
                    self.prices,
                    self.settings,
                    factors=["MOMENTUM"],
                    n_jobs=n_jobs,
                )

        serial = build(1)
        parallel = build(2)
        for left, right in zip(serial, parallel):
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                pd.testing.assert_frame_equal(left, right)

    def test_warmup_dates_have_explicit_empty_scores(self) -> None:
        settings = replace(self.settings, walk_forward_min_history_days=10_000)
        scores, decisions, summary, _ = build_walk_forward_factor_fusion(
            self.panel,
            self.prices,
            settings,
            factors=["MOMENTUM"],
        )
        self.assertTrue(scores.isna().all())
        self.assertEqual(set(summary["signal_status"]), {"WARMUP"})
        self.assertEqual(set(decisions["decision"]), {"WARMUP"})


if __name__ == "__main__":
    unittest.main()
