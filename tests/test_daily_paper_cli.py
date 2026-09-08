"""每日纸面交易命令行辅助逻辑。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.daily_paper_cli import (
    format_daily_paper_summary,
    load_latest_prices,
    load_latest_target_weights,
    run_daily_paper_from_outputs,
)
from live.paper_guard import DailyPaperGuardError
from live.paper_run_control import DailyPaperRunControlError


class TestDailyPaperCli(unittest.TestCase):
    def test_load_latest_inputs_respects_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rebalance = root / "rebalance.csv"
            prices = root / "prices.csv"
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.4, "selected": True},
                    {"date": "2024-01-31", "symbol": "BBB", "weight": 0.6, "selected": True},
                    {"date": "2024-02-29", "symbol": "CCC", "weight": 1.0, "selected": True},
                ]
            ).to_csv(rebalance, index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
                    {"date": "2024-02-29", "AAA": 11.0, "BBB": 21.0, "CCC": 31.0},
                ]
            ).to_csv(prices, index=False)

            target_date, weights = load_latest_target_weights(rebalance, trade_date="2024-02-01")
            price_date, latest_prices = load_latest_prices(prices, trade_date="2024-02-01")

            self.assertEqual(target_date.strftime("%Y-%m-%d"), "2024-01-31")
            self.assertEqual(weights.to_dict(), {"AAA": 0.4, "BBB": 0.6})
            self.assertEqual(price_date.strftime("%Y-%m-%d"), "2024-01-31")
            self.assertAlmostEqual(float(latest_prices["AAA"]), 10.0)

    def test_load_latest_target_keeps_turnover_retained_positions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rebalance = Path(td) / "rebalance.csv"
            pd.DataFrame(
                [
                    {"date": "2024-02-29", "symbol": "NEW", "weight": 0.6, "selected": True},
                    {"date": "2024-02-29", "symbol": "OLD", "weight": 0.4, "selected": False},
                ]
            ).to_csv(rebalance, index=False)

            _, weights = load_latest_target_weights(rebalance, trade_date="2024-02-29")

            self.assertEqual(weights.to_dict(), {"NEW": 0.6, "OLD": 0.4})

    def test_run_from_outputs_writes_daily_account_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)

            self.assertIn("strategy=TEST", summary)
            self.assertEqual(list(result["orders"]["symbol"]), ["AAA"])
            self.assertAlmostEqual(result["cash"], 5_000.0)
            self.assertTrue((settings.output_dir / "paper_account" / "TEST" / "snapshots.csv").is_file())
            self.assertTrue((settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md").is_file())
            self.assertTrue((settings.output_dir / "live_orders" / "TEST" / "2024-01-31_manual_confirm.csv").is_file())
            self.assertIn("paper_report", result["paths"])
            self.assertIn("manual_confirmation", result["paths"])

    def test_run_from_outputs_adds_factor_health_to_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            monitor_path = settings.output_dir / "factor_validation" / "factor_decay_monitor.csv"
            monitor_path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "factor": "ML_SCORE",
                        "status": "WATCH",
                        "reasons": "validation_positive_rate_below_threshold",
                        "validation_ic_mean": 0.01,
                        "validation_positive_rate": 0.45,
                    },
                ]
            ).to_csv(monitor_path, index=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                factor_decay_monitor_path=monitor_path,
            )
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("factor_health=WATCH", summary)
            self.assertIn("risky_factors=1", summary)
            self.assertIn("## 因子健康与失效监控", report)
            self.assertIn("| ML_SCORE | WATCH | validation_positive_rate_below_threshold |", report)

    def test_run_from_outputs_adds_style_exposure_to_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            style_path = settings.output_dir / "factor_diagnostics" / "style_exposure.csv"
            style_path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-31",
                        "strategy": "TEST",
                        "style": "PRICE_VOLUME_STYLE",
                        "weighted_exposure": 1.25,
                        "abs_weighted_exposure": 1.25,
                        "score_coverage": 1.0,
                        "n_positions": 5,
                        "n_scored_positions": 5,
                    }
                ]
            ).to_csv(style_path, index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("style_exposure=PRICE_VOLUME_STYLE", summary)
            self.assertIn("2024-01-31:PRICE_VOLUME_STYLE:1.2500:positive", summary)
            self.assertIn("## 组合风格暴露", report)
            self.assertIn("| 2024-01-31 | PRICE_VOLUME_STYLE | 1.2500 |", report)

    def test_run_from_outputs_adds_risk_blacklist_to_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            settings.data_dir.mkdir(parents=True)
            blacklist_path = settings.data_dir / "risk_blacklist.csv"
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "name": "测试股票",
                        "severity": "HIGH",
                        "reason": "manual_risk_review",
                        "active": True,
                    }
                ]
            ).to_csv(blacklist_path, index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("risk_blacklist=WATCH", summary)
            self.assertIn("block=1", summary)
            self.assertIn("risk_blacklist", str(result["order_checks"].loc[0, "check_reason"]))
            self.assertIn("## 风险预警与黑名单", report)
            self.assertIn("| AAA | 测试股票 | HIGH | manual_risk_review |", report)

    def test_run_from_outputs_adds_risk_gate_to_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            gate_path = Path(td) / "risk_gate.csv"
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "symbol": "AAA",
                        "name": "测试股票",
                        "gate_status": "WATCH",
                        "severity": "WATCH",
                        "risk_count": 1,
                        "block_count": 0,
                        "watch_count": 1,
                        "sources": "negative_sentiment:unit",
                        "reason": "negative_sentiment_watch",
                        "latest_triggered_at": "2024-01-30",
                        "expires_at": "2024-02-04",
                    }
                ]
            ).to_csv(gate_path, index=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                risk_gate_path=gate_path,
            )
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("risk_gate=WATCH", summary)
            self.assertIn("WATCH=1", summary)
            self.assertIn("## 统一风险门禁", report)
            self.assertIn("| AAA | 测试股票 | WATCH | 1 |", report)

    def test_run_from_outputs_adds_portfolio_risk_limits_to_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")
            risk_path = settings.output_dir / "portfolio_risk_limits" / "TEST" / "daily_risk_limit_checks_20240131.csv"
            control_path = settings.output_dir / "risk_control_reports" / "TEST" / "daily_risk_control_report_20240131.csv"

            self.assertIn("risk_limits=BLOCK", summary)
            self.assertIn("stress_tests=BLOCK", summary)
            self.assertIn("risk_control_report=BLOCK", summary)
            self.assertIn("## 风险总控日报", report)
            self.assertIn("max_single_position_weight", report)
            self.assertIn("## 组合压力测试", report)
            self.assertTrue(risk_path.is_file())
            self.assertTrue(control_path.is_file())
            self.assertTrue((settings.output_dir / "stress_tests" / "TEST" / "daily_stress_tests_20240131.csv").is_file())
            self.assertEqual(str(result["risk_limit_checks"].set_index("limit_id").loc["max_single_position_weight", "status"]), "BLOCK")
            self.assertEqual(str(result["risk_control_report"].iloc[0]["status"]), "BLOCK")

    def test_run_from_outputs_adds_capacity_impact_to_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)
            pd.DataFrame(
                [
                    {"trade_date": "2024-01-30", "ts_code": "AAA", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100_000, "amount": 1_000_000.0},
                    {"trade_date": "2024-01-31", "ts_code": "AAA", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100_000, "amount": 1_000_000.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_long.csv", index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")
            capacity_dir = settings.output_dir / "capacity_impact" / "TEST"

            self.assertIn("capacity_impact=PASS", summary)
            self.assertAlmostEqual(float(result["capacity_impact"].loc[0, "participation_rate"]), 0.005)
            self.assertIn("## 容量与冲击成本", report)
            self.assertTrue((capacity_dir / "daily_capacity_impact_detail_20240131.csv").is_file())
            self.assertTrue((capacity_dir / "daily_capacity_impact_summary_20240131.csv").is_file())

    def test_run_from_outputs_applies_drawdown_control_before_orders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            account_dir = settings.output_dir / "paper_account" / "TEST"
            account_dir.mkdir(parents=True)
            pd.DataFrame([{"cash": 1_000.0, "updated_at": "2024-02-29"}]).to_csv(account_dir / "account.csv", index=False)
            pd.DataFrame(
                [
                    {"symbol": "AAA", "shares": 1000, "available_shares": 1000, "updated_at": "2024-02-29"},
                ]
            ).to_csv(account_dir / "positions.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "cash": 0.0, "market_value": 10_000.0, "total_asset": 10_000.0, "n_positions": 1},
                    {"date": "2024-02-29", "cash": 1_000.0, "market_value": 8_000.0, "total_asset": 9_000.0, "n_positions": 1},
                ]
            ).to_csv(account_dir / "snapshots.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-03-29", "symbol": "AAA", "weight": 0.8, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-03-29", "AAA": 8.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-03-29.md"
            report = report_path.read_text(encoding="utf-8")
            control_path = settings.output_dir / "drawdown_control" / "TEST" / "daily_drawdown_control_20240329.csv"

            self.assertIn("drawdown_control=WATCH", summary)
            self.assertAlmostEqual(float(result["drawdown_control"].loc[0, "target_weight_scale"]), 0.5)
            self.assertAlmostEqual(float(result["target_weights_after_drawdown"]["AAA"]), 0.4)
            self.assertEqual(str(result["orders"].loc[0, "side"]), "SELL")
            self.assertIn("## 回撤止损与降仓控制", report)
            self.assertTrue(control_path.is_file())

    def test_run_from_outputs_allows_drawdown_stop_to_cash_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            account_dir = settings.output_dir / "paper_account" / "TEST"
            account_dir.mkdir(parents=True)
            pd.DataFrame([{"cash": 0.0, "updated_at": "2024-02-29"}]).to_csv(account_dir / "account.csv", index=False)
            pd.DataFrame(
                [
                    {"symbol": "AAA", "shares": 1000, "available_shares": 1000, "updated_at": "2024-02-29"},
                ]
            ).to_csv(account_dir / "positions.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "cash": 0.0, "market_value": 10_000.0, "total_asset": 10_000.0, "n_positions": 1},
                ]
            ).to_csv(account_dir / "snapshots.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-03-29", "symbol": "AAA", "weight": 0.8, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-03-29", "AAA": 8.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")

            self.assertEqual(str(result["drawdown_control"].loc[0, "status"]), "BLOCK")
            self.assertTrue(result["target_weights_after_drawdown"].empty)
            self.assertEqual(str(result["orders"].loc[0, "side"]), "SELL")
            self.assertAlmostEqual(float(result["cash"]), 8000.0)

    def test_run_from_outputs_can_use_simulated_broker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                execution_mode="simulated_broker",
            )
            summary = format_daily_paper_summary(result)

            self.assertIn("execution_mode=simulated_broker", summary)
            self.assertEqual(list(result["broker_orders"]["status"]), ["FILLED"])
            self.assertAlmostEqual(result["cash"], 5_000.0)

    def test_run_control_blocks_non_trading_day_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-26", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-26", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            with self.assertRaises(DailyPaperRunControlError):
                run_daily_paper_from_outputs(settings, strategy="TEST", trade_date="2024-01-27")

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                trade_date="2024-01-27",
                allow_non_trading_day=True,
            )
            self.assertEqual(pd.Timestamp(result["trade_date"]).strftime("%Y-%m-%d"), "2024-01-27")

    def test_run_control_blocks_duplicate_snapshot_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            run_daily_paper_from_outputs(settings, strategy="TEST", generate_report=False)
            with self.assertRaises(DailyPaperRunControlError):
                run_daily_paper_from_outputs(settings, strategy="TEST", generate_report=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                generate_report=False,
                allow_rerun=True,
            )
            self.assertEqual(pd.Timestamp(result["trade_date"]).strftime("%Y-%m-%d"), "2024-01-31")

    def test_guard_blocks_missing_target_price(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "BBB": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            with self.assertRaises(DailyPaperGuardError):
                run_daily_paper_from_outputs(settings, strategy="TEST")

    def test_guard_warning_is_returned_for_stale_price(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            (settings.output_dir / "rebalance_logs").mkdir(parents=True)
            (settings.output_dir / "cache").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                trade_date="2024-02-20",
                max_price_age_days=3,
                allow_non_trading_day=True,
            )
            summary = format_daily_paper_summary(result)

            self.assertIn("stale_price_date", summary)
            self.assertEqual(result["guard_issues"][0].severity, "WARNING")


if __name__ == "__main__":
    unittest.main()
