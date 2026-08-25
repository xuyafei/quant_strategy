"""cache_io：实验记录落盘自检。"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.cache_io import (
    save_data_quality_reports,
    save_decision_logs,
    save_order_checks,
    save_order_plans,
    save_paper_trades,
    save_performance_summary,
    save_rebalance_logs,
    save_risk_exposure_logs,
    save_risk_exposure_summary,
    save_run_cache,
    save_run_config,
    save_turnover_logs,
)


class TestExperimentOutputs(unittest.TestCase):
    def test_run_config_performance_and_rebalance_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            settings = replace(
                get_settings(),
                output_dir=root / "output",
                data_dir=root / "data",
            )

            cfg_path = save_run_config(settings)
            self.assertTrue(cfg_path.is_file())
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["output_dir"], str(settings.output_dir))
            self.assertEqual(cfg["data_dir"], str(settings.data_dir))
            self.assertIn("written_utc", cfg)

            days = pd.bdate_range("2024-01-01", periods=2)
            long_df = pd.DataFrame(
                {
                    "trade_date": [days[0], days[1]],
                    "ts_code": ["AAA", "AAA"],
                    "close": [1.0, 1.1],
                    "adj_close": [0.9, 1.1],
                    "volume": [100.0, 120.0],
                }
            )
            prices = pd.DataFrame({"AAA": [1.0, 1.1]}, index=days)
            idx = pd.MultiIndex.from_product([days, ["AAA"]], names=["date", "symbol"])
            panel = pd.DataFrame({"MOMENTUM": [0.0, 0.1]}, index=idx)
            cache_paths = save_run_cache(settings, long_df, prices, panel, panel_zscore=panel)
            self.assertTrue(cache_paths["factor_panel"].is_file())
            self.assertTrue(cache_paths["factor_panel_zscore"].is_file())
            self.assertTrue(cache_paths["prices_wide_close"].is_file())
            self.assertTrue(cache_paths["prices_wide_adj_close"].is_file())
            close_wide = pd.read_csv(cache_paths["prices_wide_close"], index_col=0)
            adj_wide = pd.read_csv(cache_paths["prices_wide_adj_close"], index_col=0)
            self.assertAlmostEqual(float(close_wide.iloc[0]["AAA"]), 1.0)
            self.assertAlmostEqual(float(adj_wide.iloc[0]["AAA"]), 0.9)

            dq_paths = save_data_quality_reports(
                settings,
                {
                    "factor_coverage": pd.DataFrame(
                        {"factor": ["MOMENTUM"], "coverage": [0.9]}
                    )
                },
            )
            self.assertTrue(dq_paths["factor_coverage"].is_file())

            perf_path = save_performance_summary(
                settings,
                {
                    "MOMENTUM": {
                        "ann_return": 0.12,
                        "ann_vol": 0.20,
                        "sharpe": 0.60,
                        "max_drawdown": -0.08,
                    }
                },
            )
            perf = pd.read_csv(perf_path)
            self.assertEqual(list(perf["strategy"]), ["MOMENTUM"])
            self.assertAlmostEqual(float(perf.loc[0, "sharpe"]), 0.60)

            logs = save_rebalance_logs(
                settings,
                {
                    "MOMENTUM": {
                        "rebalance_log": [
                            {
                                "date": pd.Timestamp("2024-01-31"),
                                "picks": ["AAA", "BBB"],
                                "weights": [0.6, 0.4],
                                "weighting": "max_sharpe",
                            }
                        ]
                    }
                },
            )
            log_path = logs["MOMENTUM"]
            self.assertTrue(log_path.is_file())
            log_df = pd.read_csv(log_path)
            self.assertEqual(
                list(log_df.columns),
                [
                    "date",
                    "symbol",
                    "weight",
                    "weighting",
                    "rank",
                    "selected",
                    "selected_rank",
                    "target_turnover",
                    "turnover_capped",
                    "turnover_scale",
                    "n_candidates_before_liquidity",
                    "n_candidates_after_liquidity",
                    "liquidity_filter_enabled",
                    "liquidity_lookback_days",
                    "min_avg_volume",
                    "min_avg_amount",
                    "liquidity_missing_data",
                    "n_trade_blocked",
                    "trade_status_filter_enabled",
                    "trade_status_missing_data",
                    "industry_cap_enabled",
                    "max_industry_weight",
                    "industry_missing_data",
                    "industry_cap_applied",
                    "max_industry_exposure",
                    "n_industries",
                    "volatility_target_enabled",
                    "target_volatility",
                    "portfolio_estimated_volatility",
                    "volatility_target_scale",
                    "cash_target_weight",
                    "volatility_target_applied",
                    "volatility_target_missing_data",
                    "min_positions_enabled",
                    "min_positions",
                    "min_positions_actual",
                    "min_positions_exposure",
                    "min_positions_applied",
                ],
            )
            self.assertEqual(list(log_df["symbol"]), ["AAA", "BBB"])
            self.assertEqual(list(log_df["rank"]), [1, 2])

            decision_logs = save_decision_logs(
                settings,
                {
                    "MOMENTUM": {
                        "decision_log": [
                            {
                                "date": pd.Timestamp("2024-01-31"),
                                "symbol": "AAA",
                                "factor_score": 1.2,
                                "factor_rank": 1,
                                "passed_liquidity_filter": True,
                                "selected_by_signal": True,
                                "selected_rank": 1,
                                "previous_weight": 0.0,
                                "raw_target_weight": 0.6,
                                "final_target_weight": 0.6,
                                "weighting": "max_sharpe",
                                "turnover_capped": False,
                                "action": "buy",
                                "decision_reason": "selected_topk",
                            }
                        ]
                    }
                },
            )
            decision_path = decision_logs["MOMENTUM"]
            self.assertTrue(decision_path.is_file())
            decision_df = pd.read_csv(decision_path)
            self.assertEqual(
                list(decision_df.columns),
                [
                    "date",
                    "symbol",
                    "factor_score",
                    "factor_rank",
                    "passed_liquidity_filter",
                    "selected_by_signal",
                    "selected_rank",
                    "previous_weight",
                    "raw_target_weight",
                    "final_target_weight",
                    "weighting",
                    "turnover_capped",
                    "is_suspended",
                    "is_limit_up",
                    "is_limit_down",
                    "trade_blocked",
                    "trade_block_reason",
                    "industry",
                    "industry_cap_applied",
                    "action",
                    "decision_reason",
                    "n_candidates_before_liquidity",
                    "n_candidates_after_liquidity",
                    "liquidity_filter_enabled",
                    "liquidity_lookback_days",
                    "min_avg_volume",
                    "min_avg_amount",
                    "liquidity_missing_data",
                    "trade_status_filter_enabled",
                    "trade_status_missing_data",
                    "industry_cap_enabled",
                    "max_industry_weight",
                    "industry_missing_data",
                    "volatility_target_enabled",
                    "target_volatility",
                    "portfolio_estimated_volatility",
                    "volatility_target_scale",
                    "cash_target_weight",
                    "volatility_target_applied",
                    "volatility_target_missing_data",
                    "min_positions_enabled",
                    "min_positions",
                    "min_positions_actual",
                    "min_positions_exposure",
                    "min_positions_applied",
                ],
            )
            self.assertEqual(str(decision_df.loc[0, "decision_reason"]), "selected_topk")

            turnover_paths = save_turnover_logs(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": [pd.Timestamp("2024-01-31")],
                            "turnover": [1.0],
                            "estimated_cost": [0.001],
                        }
                    )
                },
            )
            turnover_path = turnover_paths["MOMENTUM"]
            self.assertTrue(turnover_path.is_file())
            turnover_df = pd.read_csv(turnover_path)
            self.assertAlmostEqual(float(turnover_df.loc[0, "turnover"]), 1.0)

            order_paths = save_order_plans(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": ["2024-01-31"],
                            "symbol": ["AAA"],
                            "side": ["BUY"],
                            "current_shares": [0],
                            "target_shares": [100],
                            "delta_shares": [100],
                            "price": [10.0],
                            "estimated_amount": [1000.0],
                            "current_value": [0.0],
                            "target_value": [1000.0],
                            "current_weight": [0.0],
                            "target_weight": [0.1],
                            "trade_reason": ["increase_to_target_weight"],
                        }
                    )
                },
            )
            order_path = order_paths["MOMENTUM"]
            self.assertTrue(order_path.is_file())
            order_df = pd.read_csv(order_path)
            self.assertEqual(list(order_df["symbol"]), ["AAA"])
            self.assertEqual(list(order_df["side"]), ["BUY"])

            order_check_paths = save_order_checks(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": ["2024-01-31"],
                            "symbol": ["AAA"],
                            "side": ["BUY"],
                            "delta_shares": [100],
                            "price": [10.0],
                            "estimated_amount": [1000.0],
                            "check_status": ["PASS"],
                            "check_reason": ["pass"],
                        }
                    )
                },
            )
            order_check_path = order_check_paths["MOMENTUM"]
            self.assertTrue(order_check_path.is_file())
            order_check_df = pd.read_csv(order_check_path)
            self.assertEqual(list(order_check_df["check_status"]), ["PASS"])

            paper_trade_paths = save_paper_trades(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": ["2024-01-31"],
                            "symbol": ["AAA"],
                            "side": ["BUY"],
                            "qty": [100],
                            "price": [10.0],
                            "gross_amount": [1000.0],
                            "commission": [0.3],
                            "net_cash_flow": [-1000.3],
                            "cash_before": [10_000.0],
                            "cash_after": [8_999.7],
                            "position_before": [0],
                            "position_after": [100],
                            "fill_status": ["FILLED"],
                            "fill_reason": ["filled"],
                        }
                    )
                },
            )
            paper_trade_path = paper_trade_paths["MOMENTUM"]
            self.assertTrue(paper_trade_path.is_file())
            paper_trade_df = pd.read_csv(paper_trade_path)
            self.assertEqual(list(paper_trade_df["fill_status"]), ["FILLED"])

            risk_paths = save_risk_exposure_logs(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": [pd.Timestamp("2024-01-31")],
                            "hhi": [0.52],
                            "effective_n": [1.0 / 0.52],
                            "top1_weight": [0.6],
                        }
                    )
                },
            )
            risk_path = risk_paths["MOMENTUM"]
            self.assertTrue(risk_path.is_file())
            risk_df = pd.read_csv(risk_path)
            self.assertAlmostEqual(float(risk_df.loc[0, "hhi"]), 0.52)

            risk_summary_path = save_risk_exposure_summary(
                settings,
                {"MOMENTUM": {"avg_effective_n": 1.0 / 0.52, "max_hhi": 0.52}},
            )
            self.assertTrue(risk_summary_path.is_file())
            risk_summary = pd.read_csv(risk_summary_path)
            self.assertEqual(list(risk_summary["strategy"]), ["MOMENTUM"])


if __name__ == "__main__":
    unittest.main()
