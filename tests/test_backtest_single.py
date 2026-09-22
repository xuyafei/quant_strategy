"""backtest_single：配权模式与 _weights_for_rebalance 自检。"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.backtest_single import (
    _apply_aggregate_exposure_cap,
    _apply_gross_exposure_multiplier,
    _apply_industry_weight_cap,
    _apply_max_position_cap,
    _apply_rebalance_turnover_cap,
    _rebalance_dates,
    _target_weight_lists,
    _weights_for_rebalance,
    run_single_backtest,
)
from config import get_settings


def _price_wide_for_weights() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=80)
    syms = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(2)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.012, size=(len(days), len(syms))), axis=0)
    return pd.DataFrame(px, index=days, columns=syms)


class TestWeightsForRebalance(unittest.TestCase):
    def test_gross_exposure_multiplier_leaves_cash(self) -> None:
        weights, meta = _apply_gross_exposure_multiplier(
            {"AAA": 0.6, "BBB": 0.4},
            0.75,
        )
        self.assertAlmostEqual(sum(weights.values()), 0.75)
        self.assertAlmostEqual(meta["cash_target_weight_after_gross"], 0.25)
        self.assertTrue(meta["gross_exposure_scaled"])

    def test_apply_max_position_cap(self) -> None:
        w, capped = _apply_max_position_cap([0.8, 0.1, 0.1], 0.5)
        self.assertTrue(capped)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertLessEqual(max(w), 0.5 + 1e-9)
        self.assertAlmostEqual(w[0], 0.5)

    def test_apply_max_position_cap_infeasible_keeps_normalized_weights(self) -> None:
        w, capped = _apply_max_position_cap([0.5, 0.5], 0.4)
        self.assertFalse(capped)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertEqual(w, [0.5, 0.5])

    def test_industry_cap_does_not_reinflate_single_position(self) -> None:
        target = {
            "BANK_A": 0.2,
            "BANK_B": 0.2,
            "TECH": 0.2,
            "ENERGY": 0.2,
            "HEALTH": 0.2,
        }
        industries = {
            "BANK_A": "Bank",
            "BANK_B": "Bank",
            "TECH": "Tech",
            "ENERGY": "Energy",
            "HEALTH": "Health",
        }
        weights, changed, exposure = _apply_industry_weight_cap(
            target,
            industries,
            max_industry_weight=0.35,
            max_position_weight=0.20,
        )
        self.assertTrue(changed)
        self.assertLessEqual(max(weights.values()), 0.20 + 1e-12)
        self.assertLessEqual(exposure["Bank"], 0.35 + 1e-12)
        self.assertAlmostEqual(sum(weights.values()), 0.95)

    def test_aggregate_exposure_cap_redistributes_to_non_group(self) -> None:
        target = {
            "CHIP": 0.30,
            "TELECOM": 0.25,
            "BANK": 0.25,
            "ENERGY": 0.20,
        }
        in_group = {"CHIP": True, "TELECOM": True, "BANK": False, "ENERGY": False}
        industries = {
            "CHIP": "Semiconductor",
            "TELECOM": "Communication",
            "BANK": "Bank",
            "ENERGY": "Energy",
        }
        weights, changed, exposure = _apply_aggregate_exposure_cap(
            target,
            in_group,
            0.25,
            max_position_weight=0.40,
            industry_by_symbol=industries,
            max_industry_weight=0.40,
        )
        self.assertTrue(changed)
        self.assertLessEqual(exposure, 0.25 + 1e-12)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertLessEqual(max(weights.values()), 0.40 + 1e-12)

    def test_aggregate_exposure_cap_keeps_cash_when_redistribution_is_infeasible(self) -> None:
        weights, changed, exposure = _apply_aggregate_exposure_cap(
            {"CHIP": 0.60, "BANK": 0.40},
            {"CHIP": True, "BANK": False},
            0.25,
            max_position_weight=0.40,
            industry_by_symbol={"CHIP": "Tech", "BANK": "Bank"},
            max_industry_weight=0.40,
        )
        self.assertTrue(changed)
        self.assertAlmostEqual(exposure, 0.25)
        self.assertAlmostEqual(sum(weights.values()), 0.65)

    def test_apply_rebalance_turnover_cap(self) -> None:
        prev = {"AAA": 0.5, "BBB": 0.5}
        target = {"CCC": 0.5, "DDD": 0.5}
        capped, did_cap, turnover, scale = _apply_rebalance_turnover_cap(prev, target, 1.0)
        self.assertTrue(did_cap)
        self.assertAlmostEqual(turnover, 2.0)
        self.assertAlmostEqual(scale, 0.5)
        self.assertAlmostEqual(sum(capped.values()), 1.0)
        self.assertAlmostEqual(capped["AAA"], 0.25)
        self.assertAlmostEqual(capped["BBB"], 0.25)
        self.assertAlmostEqual(capped["CCC"], 0.25)
        self.assertAlmostEqual(capped["DDD"], 0.25)

    def test_apply_rebalance_turnover_cap_skips_initial_build(self) -> None:
        capped, did_cap, turnover, scale = _apply_rebalance_turnover_cap(
            {},
            {"AAA": 0.5, "BBB": 0.5},
            0.5,
        )
        self.assertFalse(did_cap)
        self.assertAlmostEqual(turnover, 1.0)
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual(capped, {"AAA": 0.5, "BBB": 0.5})

    def test_rebalance_dates_can_force_final_trading_day(self) -> None:
        prices = pd.DataFrame(
            {"AAA": np.linspace(1.0, 2.0, 13)},
            index=pd.bdate_range("2026-06-01", "2026-06-17"),
        )
        settings = replace(get_settings(), force_final_rebalance=True)
        dates = _rebalance_dates(prices, settings)
        self.assertIn(pd.Timestamp("2026-06-17"), dates)

    def test_rebalance_dates_uses_last_trading_day_of_month(self) -> None:
        prices = pd.DataFrame(
            {"AAA": np.linspace(1.0, 2.0, 22)},
            index=pd.bdate_range("2025-05-01", "2025-05-30"),
        )
        dates = _rebalance_dates(prices, get_settings())
        self.assertEqual(list(dates), [pd.Timestamp("2025-05-30")])

    def test_target_weight_lists_prefers_selected_then_previous(self) -> None:
        picks, weights = _target_weight_lists(
            {"OLD": 0.25, "NEW": 0.75},
            ["NEW", "OLD"],
        )
        self.assertEqual(picks, ["NEW", "OLD"])
        self.assertEqual(weights, [0.75, 0.25])

    def test_missing_quote_uses_last_price_for_valuation(self) -> None:
        days = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])
        prices = pd.DataFrame(
            {"AAA": [10.0, np.nan, 11.0], "BBB": [10.0, 10.0, 10.0]},
            index=days,
        )
        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(0.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 2.0
        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=1,
            commission_rate=0.0,
            max_position_weight=1.0,
            max_industry_weight=0.0,
            target_volatility=0.0,
            min_positions=0,
            max_rebalance_turnover=0.0,
        )

        nav, _ = run_single_backtest(
            "TEST_STALE_VALUATION",
            factor_values=factor,
            prices=prices,
            settings=settings,
        )

        self.assertAlmostEqual(float(nav.loc[days[1]]), 1.0)
        self.assertAlmostEqual(float(nav.loc[days[2]]), 1.1)

    def test_equal_mode(self) -> None:
        px = _price_wide_for_weights()
        s = get_settings()
        s2 = replace(s, portfolio_weighting="equal")
        dt = px.index[50]
        w, lab = _weights_for_rebalance(px, ["AAA", "BBB"], dt, s2)
        self.assertEqual(lab, "equal")
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertEqual(len(w), 2)
        self.assertTrue(all(abs(x - 0.5) < 1e-9 for x in w))

    def test_risk_parity_sums_to_one(self) -> None:
        px = _price_wide_for_weights()
        s = replace(get_settings(), portfolio_weighting="risk_parity")
        dt = px.index[50]
        w, lab = _weights_for_rebalance(px, list(px.columns), dt, s)
        self.assertEqual(lab, "risk_parity")
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(x >= -1e-12 for x in w))

    def test_risk_parity_fallback_short_history(self) -> None:
        px = _price_wide_for_weights()
        s = replace(
            get_settings(),
            portfolio_weighting="risk_parity",
            optimizer_return_window=200,
            optimizer_min_obs=200,
        )
        dt = px.index[10]
        w, lab = _weights_for_rebalance(px, list(px.columns), dt, s)
        self.assertEqual(lab, "risk_parity_fallback")
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(abs(x - 1.0 / 3.0) < 1e-9 for x in w))


class TestRunSingleRiskParity(unittest.TestCase):
    def test_end_to_end_runs(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=120)
        syms = ["X", "Y"]
        rng = np.random.default_rng(3)
        px = pd.DataFrame(
            50.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=(len(days), len(syms))), axis=0),
            index=days,
            columns=syms,
        )
        idx = pd.MultiIndex.from_product([days, syms], names=["date", "symbol"])
        factor = pd.Series(rng.standard_normal(len(idx)), index=idx)
        settings = replace(get_settings(), portfolio_weighting="risk_parity", top_k=2)
        nav, meta = run_single_backtest(
            "TEST",
            factor_values=factor,
            prices=px,
            settings=settings,
            top_k=2,
        )
        self.assertGreater(len(nav), 0)
        self.assertEqual(meta.get("portfolio_weighting"), "risk_parity")
        log = meta.get("rebalance_log") or []
        self.assertGreater(len(log), 0)
        labs = {rec.get("weighting") for rec in log}
        allowed = {
            "risk_parity",
            "risk_parity_capped",
            "risk_parity_turnover_capped",
            "risk_parity_capped_turnover_capped",
            "risk_parity_fallback",
            "risk_parity_fallback_turnover_capped",
            "equal",
            "equal_turnover_capped",
        }
        self.assertTrue(labs <= allowed)
        self.assertIn("max_rebalance_turnover", meta)


class TestLiquidityFilter(unittest.TestCase):
    def test_low_volume_candidate_is_filtered_before_topk(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=45)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 11.0, len(days)),
                "BBB": np.linspace(10.0, 12.0, len(days)),
            },
            index=days,
        )
        long_rows = []
        for dt in days:
            long_rows.append({"trade_date": dt, "ts_code": "AAA", "close": prices.loc[dt, "AAA"], "volume": 5000.0})
            long_rows.append({"trade_date": dt, "ts_code": "BBB", "close": prices.loc[dt, "BBB"], "volume": 10.0})
        long_df = pd.DataFrame(long_rows)

        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(0.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 1.0
        factor.loc[(slice(None), "BBB")] = 2.0

        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=1,
            min_avg_volume=1000.0,
            liquidity_lookback_days=5,
        )
        _, meta = run_single_backtest(
            "TEST_LIQ",
            factor_values=factor,
            prices=prices,
            settings=settings,
            long_prices=long_df,
        )
        log = [rec for rec in (meta.get("rebalance_log") or []) if rec.get("picks")]
        self.assertGreater(len(log), 0)
        first = log[0]
        self.assertEqual(first["selected_picks"], ["AAA"])
        self.assertEqual(first["n_candidates_before_liquidity"], 2)
        self.assertEqual(first["n_candidates_after_liquidity"], 1)
        self.assertTrue(first["liquidity_filter_enabled"])
        decision_log = meta.get("decision_log") or []
        self.assertGreater(len(decision_log), 0)
        filtered = [
            rec
            for rec in decision_log
            if rec.get("symbol") == "BBB" and rec.get("decision_reason") == "filtered_by_liquidity"
        ]
        self.assertGreater(len(filtered), 0)
        selected = [
            rec
            for rec in decision_log
            if rec.get("symbol") == "AAA" and rec.get("selected_by_signal")
        ]
        self.assertGreater(len(selected), 0)
        self.assertEqual(selected[0]["action"], "buy")


class TestTradeStatusFilter(unittest.TestCase):
    def test_limit_up_blocks_new_buy(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=45)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 11.0, len(days)),
                "BBB": np.linspace(10.0, 10.5, len(days)),
            },
            index=days,
        )
        long_rows = []
        for dt in days:
            is_rebalance = dt == prices.resample("ME").last().index.intersection(prices.index)[0]
            long_rows.append(
                {
                    "trade_date": dt,
                    "ts_code": "AAA",
                    "close": prices.loc[dt, "AAA"],
                    "volume": 5000.0,
                    "is_limit_up": bool(is_rebalance),
                    "is_limit_down": False,
                    "is_suspended": False,
                }
            )
            long_rows.append(
                {
                    "trade_date": dt,
                    "ts_code": "BBB",
                    "close": prices.loc[dt, "BBB"],
                    "volume": 5000.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                }
            )
        long_df = pd.DataFrame(long_rows)

        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(0.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 2.0
        factor.loc[(slice(None), "BBB")] = 1.0

        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=1,
            enable_trade_status_filter=True,
            max_rebalance_turnover=0.0,
        )
        _, meta = run_single_backtest(
            "TEST_STATUS",
            factor_values=factor,
            prices=prices,
            settings=settings,
            long_prices=long_df,
        )
        decision_log = meta.get("decision_log") or []
        blocked = [
            rec
            for rec in decision_log
            if rec.get("symbol") == "AAA" and rec.get("trade_block_reason") == "blocked_by_limit_up"
        ]
        self.assertGreater(len(blocked), 0)
        self.assertTrue(blocked[0]["trade_blocked"])
        self.assertEqual(blocked[0]["final_target_weight"], 0.0)
        self.assertIn("blocked_by_limit_up", blocked[0]["decision_reason"])


class TestIndustryCap(unittest.TestCase):
    def test_industry_cap_limits_group_exposure(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=45)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 11.0, len(days)),
                "BBB": np.linspace(10.0, 10.8, len(days)),
                "CCC": np.linspace(10.0, 10.5, len(days)),
            },
            index=days,
        )
        long_rows = []
        for dt in days:
            long_rows.append({"trade_date": dt, "ts_code": "AAA", "close": prices.loc[dt, "AAA"], "volume": 5000.0, "industry": "Tech"})
            long_rows.append({"trade_date": dt, "ts_code": "BBB", "close": prices.loc[dt, "BBB"], "volume": 5000.0, "industry": "Tech"})
            long_rows.append({"trade_date": dt, "ts_code": "CCC", "close": prices.loc[dt, "CCC"], "volume": 5000.0, "industry": "Bank"})
        long_df = pd.DataFrame(long_rows)

        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB", "CCC"]], names=["date", "symbol"])
        factor = pd.Series(0.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 3.0
        factor.loc[(slice(None), "BBB")] = 2.0
        factor.loc[(slice(None), "CCC")] = 1.0

        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=3,
            max_industry_weight=0.6,
            max_rebalance_turnover=0.0,
        )
        _, meta = run_single_backtest(
            "TEST_INDUSTRY",
            factor_values=factor,
            prices=prices,
            settings=settings,
            long_prices=long_df,
        )
        log = [rec for rec in (meta.get("rebalance_log") or []) if rec.get("picks")]
        self.assertGreater(len(log), 0)
        first = log[0]
        weights = dict(zip(first["picks"], first["weights"]))
        self.assertTrue(first["industry_cap_applied"])
        self.assertLessEqual(weights["AAA"] + weights["BBB"], 0.6 + 1e-9)
        self.assertAlmostEqual(weights["CCC"], 0.4)

        decision_log = meta.get("decision_log") or []
        selected = [rec for rec in decision_log if rec.get("symbol") == "AAA" and rec.get("selected_by_signal")]
        self.assertGreater(len(selected), 0)
        self.assertEqual(selected[0]["industry"], "Tech")
        self.assertTrue(selected[0]["industry_cap_applied"])
        self.assertIn("industry_cap_adjusted", selected[0]["decision_reason"])


class TestVolatilityTarget(unittest.TestCase):
    def test_volatility_target_scales_exposure_to_cash(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=80)
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0005, 0.035, size=(len(days), 2))
        px = 100.0 * np.cumprod(1.0 + returns, axis=0)
        prices = pd.DataFrame(px, index=days, columns=["AAA", "BBB"])

        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(1.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 2.0

        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=2,
            target_volatility=0.02,
            volatility_target_lookback_days=30,
            volatility_target_min_obs=5,
            max_rebalance_turnover=0.0,
        )
        _, meta = run_single_backtest(
            "TEST_VOL_TARGET",
            factor_values=factor,
            prices=prices,
            settings=settings,
        )
        log = [rec for rec in (meta.get("rebalance_log") or []) if rec.get("picks")]
        self.assertGreater(len(log), 0)
        first = log[0]
        self.assertTrue(first["volatility_target_applied"])
        self.assertLess(first["volatility_target_scale"], 1.0)
        self.assertLess(sum(first["weights"]), 1.0)
        self.assertGreater(first["cash_target_weight"], 0.0)

        decision_log = meta.get("decision_log") or []
        selected = [rec for rec in decision_log if rec.get("symbol") == "AAA" and rec.get("selected_by_signal")]
        self.assertGreater(len(selected), 0)
        self.assertIn("volatility_target_scaled", selected[0]["decision_reason"])


class TestMinPositionsRule(unittest.TestCase):
    def test_rebalance_override_scales_gross_exposure(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=45)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 11.0, len(days)),
                "BBB": np.linspace(10.0, 10.5, len(days)),
            },
            index=days,
        )
        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(1.0, index=idx)
        first_rebalance = prices.groupby(pd.Grouper(freq="ME")).apply(lambda x: x.index[-1]).iloc[0]
        overrides = pd.DataFrame(
            {
                "date": [first_rebalance],
                "risk_state": ["RED"],
                "gross_exposure_multiplier": [0.5],
                "max_tech_growth_weight": [0.0],
            }
        )
        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=2,
            commission_rate=0.0,
            max_position_weight=1.0,
            max_industry_weight=0.0,
            target_volatility=0.0,
            min_positions=0,
            max_rebalance_turnover=0.0,
        )
        _, meta = run_single_backtest(
            "TEST_DYNAMIC_GROSS",
            factor_values=factor,
            prices=prices,
            settings=settings,
            rebalance_overrides=overrides,
        )
        first = [rec for rec in meta["rebalance_log"] if rec.get("picks")][0]
        self.assertEqual(first["risk_state"], "RED")
        self.assertAlmostEqual(sum(first["weights"]), 0.5)
        self.assertAlmostEqual(first["cash_target_weight"], 0.5)

    def test_min_positions_scales_exposure_to_cash(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=45)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 11.0, len(days)),
                "BBB": np.linspace(10.0, 10.5, len(days)),
            },
            index=days,
        )
        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(1.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 2.0

        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=2,
            min_positions=3,
            min_positions_exposure=0.5,
            max_rebalance_turnover=0.0,
        )
        _, meta = run_single_backtest(
            "TEST_MIN_POSITIONS",
            factor_values=factor,
            prices=prices,
            settings=settings,
        )
        log = [rec for rec in (meta.get("rebalance_log") or []) if rec.get("picks")]
        self.assertGreater(len(log), 0)
        first = log[0]
        self.assertTrue(first["min_positions_applied"])
        self.assertEqual(first["min_positions_actual"], 2)
        self.assertAlmostEqual(sum(first["weights"]), 0.5)
        self.assertAlmostEqual(first["cash_target_weight"], 0.5)

        decision_log = meta.get("decision_log") or []
        selected = [rec for rec in decision_log if rec.get("symbol") == "AAA" and rec.get("selected_by_signal")]
        self.assertGreater(len(selected), 0)
        self.assertIn("min_positions_scaled", selected[0]["decision_reason"])

    def test_trade_status_keeps_cash_target_weight(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=45)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 11.0, len(days)),
                "BBB": np.linspace(10.0, 10.5, len(days)),
            },
            index=days,
        )
        first_rebalance = prices.resample("ME").last().index.intersection(prices.index)[0]
        long_rows = []
        for dt in days:
            for sym in ["AAA", "BBB"]:
                long_rows.append(
                    {
                        "trade_date": dt,
                        "ts_code": sym,
                        "close": prices.loc[dt, sym],
                        "volume": 5000.0,
                        "is_limit_up": bool(dt == first_rebalance and sym == "AAA"),
                        "is_limit_down": False,
                        "is_suspended": False,
                    }
                )
        long_df = pd.DataFrame(long_rows)
        idx = pd.MultiIndex.from_product([days, ["AAA", "BBB"]], names=["date", "symbol"])
        factor = pd.Series(1.0, index=idx)
        factor.loc[(slice(None), "AAA")] = 2.0

        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=2,
            min_positions=3,
            min_positions_exposure=0.5,
            enable_trade_status_filter=True,
            max_rebalance_turnover=0.0,
        )
        _, meta = run_single_backtest(
            "TEST_STATUS_WITH_CASH",
            factor_values=factor,
            prices=prices,
            settings=settings,
            long_prices=long_df,
        )
        log = [rec for rec in (meta.get("rebalance_log") or []) if rec.get("picks")]
        self.assertGreater(len(log), 0)
        first = log[0]
        self.assertEqual(first["picks"], ["BBB"])
        self.assertAlmostEqual(sum(first["weights"]), 0.5)
        self.assertAlmostEqual(first["cash_target_weight"], 0.5)

    def test_empty_signal_cash_policy_liquidates_previous_positions(self) -> None:
        days = pd.bdate_range("2024-01-01", "2024-03-01")
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(10.0, 12.0, len(days)),
                "BBB": np.linspace(10.0, 10.5, len(days)),
            },
            index=days,
        )
        index = pd.MultiIndex.from_product([days, prices.columns], names=["date", "symbol"])
        factor = pd.Series(np.nan, index=index)
        factor.loc[(pd.Timestamp("2024-01-31"), "AAA")] = 2.0
        factor.loc[(pd.Timestamp("2024-01-31"), "BBB")] = 1.0
        settings = replace(
            get_settings(),
            portfolio_weighting="equal",
            top_k=1,
            commission_rate=0.0,
            max_position_weight=1.0,
            max_rebalance_turnover=0.0,
            force_final_rebalance=False,
        )
        _, meta = run_single_backtest(
            "EMPTY_TO_CASH",
            factor_values=factor,
            prices=prices,
            settings=settings,
            empty_signal_policy="cash",
        )
        logs = meta["rebalance_log"]
        self.assertEqual(logs[-1]["weighting"], "empty_signal_cash")
        self.assertEqual(logs[-1]["picks"], [])
        self.assertEqual(logs[-1]["cash_target_weight"], 1.0)
        self.assertEqual(meta["empty_signal_policy"], "cash")


if __name__ == "__main__":
    unittest.main()
