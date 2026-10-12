import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from live.board_rules import target_shares, quantity_error
from live.forward_ledger import (initialize, prepare, advance, latest_day, seal, verify,
                                  session_config, required_symbols, sha, calendar_dates, next_session)
from live.order_builder import build_order_plan
from live.order_precheck import precheck_order_plan
from live.paper_runner import run_daily_paper_trade
from scripts.fetch_forward_market import build_market
from scripts.run_forward_daily import run as daily_run


class ForwardLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.root = self.base / "session"
        self.data.mkdir()
        self.signal = "2026-09-24"
        self.execution = "2026-09-28"
        dates = pd.date_range("2026-09-23", "2026-10-12")
        opens = {"20260923", "20260924", "20260928", "20260929", "20260930", "20261008", "20261009", "20261012"}
        pd.DataFrame({"cal_date": dates.strftime("%Y%m%d"), "is_open": [int(x.strftime("%Y%m%d") in opens) for x in dates]}).to_csv(self.data / "trade_calendar.csv", index=False)
        self.protocol = self.data / "strategy.json"
        self.protocol.write_text(json.dumps({"strategies": [{"name": "ONE"}, {"name": "TWO"}]}))
        pd.DataFrame({"ts_code": ["000001.SZ", "688001.SH"], "trade_date": ["20260924"] * 2, "adj_factor": [1., 1.]}).to_csv(self.data / "adj_factor_all_a.csv.gz", index=False, compression="gzip")
        target = pd.DataFrame({"symbol": ["000001.SZ", "688001.SH"], "rank": [1, 2], "weight": [.5, .5], "score": [1., .8]})
        self.bundle = {"targets": {"ONE": target.assign(strategy="ONE"), "TWO": target.assign(strategy="TWO")},
                       "prices": pd.DataFrame({"ts_code": ["000001.SZ", "688001.SH"], "trade_date": [pd.Timestamp(self.signal)] * 2, "close": [10., 20.]}),
                       "report": pd.DataFrame({"symbol": ["000001.SZ", "688001.SH"], "name": ["A", "B"], "industry_l1": ["bank", "tech"]})}
        self.now = patch("live.forward_ledger.utc_now", return_value=pd.Timestamp("2026-09-27T12:00:00Z"))
        self.clock = self.now.start()
        self.addCleanup(self.now.stop)
        self.build = patch("live.forward_ledger.build_forward_signals", return_value=self.bundle)
        self.build.start()
        self.addCleanup(self.build.stop)
        self.receipt = patch("live.forward_ledger._receipt", return_value={"through_date": self.signal})
        self.receipt.start()
        self.addCleanup(self.receipt.stop)

    def init(self):
        return initialize(self.root, self.data, self.protocol, self.signal, self.execution, 10000.)

    def market(self, date="2026-09-28", close=10., factor=1., missing=False):
        out = self.base / ("market_" + date)
        out.mkdir()
        df = pd.DataFrame({"date": [date, date], "symbol": ["000001.SZ", "688001.SH"], "close": [close, 20.],
                           "adj_factor": [factor, 1.], "price_date": [date, date],
                           "is_suspended": [False, False], "is_limit_up": [False, False], "is_limit_down": [False, False]})
        if self.root.exists():
            df = df[df.symbol.isin(required_symbols(self.root, date))]
        if missing:
            df = df.iloc[:1]
        df.to_csv(out / "market.csv", index=False)
        pd.DataFrame(columns=["symbol", "reason"]).to_csv(out / "risk_blacklist.csv", index=False)
        seal(out, {"date": date, "retrieved_at_utc": date + "T08:00:00Z"})
        return out

    def open_clock(self, date="2026-09-28"):
        self.clock.return_value = pd.Timestamp(date + "T09:00:00Z")

    def test_init_seals_plans_and_keeps_cash_only(self):
        self.init()
        plan = self.root / "plans/20260928"
        verify(plan)
        self.assertFalse((self.root / "days").exists())
        accounts = pd.read_csv(self.root / "initial/paper_account/ONE/account.csv")
        self.assertEqual(accounts.cash.iloc[0], 10000.)
        summary = pd.read_csv(plan / "summary.csv")
        self.assertTrue(summary.status.eq("FROZEN_TARGETS_NOT_FILLS").all())

    def test_duplicate_init_and_prepare_do_not_change_plan(self):
        self.init()
        path = self.root / "plans/20260928/manifest.json"
        digest = sha(path)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.init()
        with self.assertRaises(RuntimeError):
            prepare(self.root, self.data, self.signal, self.execution)
        self.assertEqual(digest, sha(path))

    def test_late_freeze_is_rejected_without_session(self):
        self.clock.return_value = pd.Timestamp("2026-09-28T01:30:00Z")
        with self.assertRaisesRegex(RuntimeError, "too late"):
            self.init()
        self.assertFalse(self.root.exists())

    def test_holiday_execution_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "consecutive"):
            initialize(self.root, self.data, self.protocol, self.signal, "2026-09-25")

    def test_first_day_then_mark_inherits_positions_and_cash(self):
        self.init()
        self.open_clock()
        day1 = advance(self.root, self.market(), self.execution)
        s1 = pd.read_csv(day1 / "summary.csv")
        self.assertTrue(s1.action.eq("REBALANCE").all())
        self.assertTrue(s1.cash.ge(0).all())
        pos1 = pd.read_csv(day1 / "paper_account/ONE/positions.csv")
        self.assertTrue(pos1.available_shares.eq(0).all())
        self.open_clock("2026-09-29")
        day2 = advance(self.root, self.market("2026-09-29", close=11.), "2026-09-29")
        s2 = pd.read_csv(day2 / "summary.csv")
        self.assertTrue(s2.action.eq("HOLD_MARK").all())
        self.assertEqual(s1.cash.tolist(), s2.cash.tolist())
        pos2 = pd.read_csv(day2 / "paper_account/ONE/positions.csv")
        self.assertEqual(pos1.shares.tolist(), pos2.shares.tolist())
        self.assertTrue(pos2.available_shares.eq(pos2.shares).all())
        self.assertGreater(s2.total_asset.iloc[0], s1.total_asset.iloc[0])
        self.assertEqual(latest_day(self.root)[1]["date"], "2026-09-29")

    def test_future_duplicate_and_skipped_days_blocked(self):
        self.init()
        market = self.market()
        with self.assertRaisesRegex(RuntimeError, "not closed"):
            advance(self.root, market, self.execution)
        self.open_clock("2026-09-29")
        with self.assertRaisesRegex(RuntimeError, "next open"):
            advance(self.root, market, "2026-09-29")
        advance(self.root, market, self.execution)
        with self.assertRaisesRegex(RuntimeError, "next open"):
            advance(self.root, market, self.execution)

    def test_missing_market_coverage_does_not_mutate_accounts(self):
        self.init()
        self.open_clock()
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            advance(self.root, self.market(missing=True), self.execution)
        self.assertEqual(latest_day(self.root)[1]["kind"], "INITIAL_CASH_ONLY")

    def test_second_account_failure_is_atomic_and_retryable(self):
        self.init()
        self.open_clock()
        market = self.market()
        calls = []
        def sometimes_fail(*args, **kwargs):
            calls.append(kwargs["strategy"])
            if len(calls) == 2:
                raise RuntimeError("injected second-account failure")
            return run_daily_paper_trade(*args, **kwargs)
        with patch("live.forward_ledger.run_daily_paper_trade", side_effect=sometimes_fail):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                advance(self.root, market, self.execution)
        self.assertEqual(latest_day(self.root)[1]["kind"], "INITIAL_CASH_ONLY")
        self.assertFalse((self.root / "days/20260928").exists())
        self.assertFalse(list((self.root / "days").glob(".staging-*")))
        advance(self.root, market, self.execution)
        self.assertEqual(latest_day(self.root)[1]["date"], self.execution)

    def test_tampered_target_blocked(self):
        self.init()
        path = self.root / "plans/20260928/ONE.csv"
        path.write_text(path.read_text() + "\n")
        self.open_clock()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            advance(self.root, self.market(), self.execution)

    def test_corporate_action_stops_mark(self):
        self.init()
        self.open_clock()
        advance(self.root, self.market(), self.execution)
        self.open_clock("2026-09-29")
        with self.assertRaisesRegex(RuntimeError, "corporate action"):
            advance(self.root, self.market("2026-09-29", factor=1.1), "2026-09-29")
        self.assertEqual(latest_day(self.root)[1]["date"], self.execution)

    def test_bad_cash_and_lock_rejected(self):
        for cash in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                initialize(self.root, self.data, self.protocol, self.signal, self.execution, cash)
        lock = self.base / "session.lock"
        lock.mkdir()
        with self.assertRaisesRegex(RuntimeError, "lock"):
            self.init()

    def test_unknown_status_and_stale_active_price_rejected(self):
        self.init()
        self.open_clock()
        for name, value in [("is_suspended", "unknown"), ("price_date", self.signal)]:
            with self.subTest(name=name):
                path = self.base / ("bad_" + name)
                path.mkdir()
                good = self.market() if not (self.base / "market_2026-09-28").exists() else self.base / "market_2026-09-28"
                df = pd.read_csv(good / "market.csv")
                df[name] = value
                df.to_csv(path / "market.csv", index=False)
                seal(path, {"date": self.execution, "retrieved_at_utc": self.execution + "T08:00:00Z"})
                with self.assertRaises(RuntimeError):
                    advance(self.root, path, self.execution)

    def test_blacklist_blocks_orders(self):
        self.init()
        self.open_clock()
        m = self.market()
        pd.DataFrame([{"symbol": "000001.SZ", "reason": "test risk"}]).to_csv(m / "risk_blacklist.csv", index=False)
        seal(m, {"date": self.execution, "retrieved_at_utc": self.execution + "T08:00:00Z"})
        day = advance(self.root, m, self.execution)
        checks = pd.read_csv(day / "order_checks/ONE.csv").set_index("symbol")
        self.assertEqual(checks.loc["000001.SZ", "check_status"], "BLOCK")

    def test_closed_day_daily_entry_does_not_fetch_or_mutate(self):
        self.init()
        with patch("scripts.run_forward_daily.fetch_market") as fetch:
            result = daily_run(self.root, self.data, "2026-09-27")
        self.assertEqual(result["status"], "MARKET_CLOSED_NO_MUTATION")
        fetch.assert_not_called()

    def test_calendar_holidays_and_missing_days(self):
        sessions = calendar_dates(self.data / "trade_calendar.csv")
        self.assertEqual(str(next_session(sessions, self.signal).date()), self.execution)
        self.assertEqual(str(next_session(sessions, "2026-09-30").date()), "2026-10-08")
        df = pd.read_csv(self.data / "trade_calendar.csv").iloc[1:]
        df = df.drop(df.index[2])
        df.to_csv(self.data / "bad_calendar.csv", index=False)
        with self.assertRaisesRegex(RuntimeError, "gaps"):
            calendar_dates(self.data / "bad_calendar.csv")

    def test_five_sessions_with_weekly_rebalance_across_holiday(self):
        self.init()
        for date in ["2026-09-28", "2026-09-29", "2026-09-30"]:
            self.open_clock(date)
            advance(self.root, self.market(date), date)
        # Cannot silently skip the weekly plan and pretend that holding was intended.
        self.open_clock("2026-10-08")
        with self.assertRaisesRegex(RuntimeError, "rebalance plan is missing"):
            advance(self.root, self.base / "not_fetched", "2026-10-08")
        self.open_clock("2026-09-30")
        self.bundle["prices"]["trade_date"] = pd.Timestamp("2026-09-30")
        pd.DataFrame({"ts_code": ["000001.SZ", "688001.SH"], "trade_date": ["20260930"] * 2,
                      "adj_factor": [1., 1.]}).to_csv(self.data / "adj_factor_all_a.csv.gz", index=False, compression="gzip")
        prepare(self.root, self.data, "2026-09-30", "2026-10-08")
        self.assertTrue((self.root / "plans/20260928").exists())
        for date in ["2026-10-08", "2026-10-09"]:
            self.open_clock(date)
            advance(self.root, self.market(date), date)
        latest, _ = latest_day(self.root)
        snapshots = pd.read_csv(latest / "paper_account/ONE/snapshots.csv")
        self.assertEqual(len(snapshots), 6)  # initial cash + five synthetic sessions
        self.assertEqual(len(list((self.root / "days").glob("????????"))), 5)
        self.assertEqual(pd.read_csv(self.root / "days/20261008/summary.csv").action.iloc[0], "REBALANCE")

    def test_daily_retry_does_not_refetch_or_trade(self):
        self.init()
        self.open_clock()
        day = advance(self.root, self.market(), self.execution)
        before = sha(day / "manifest.json")
        with patch("scripts.run_forward_daily.utc_now", return_value=self.clock.return_value), patch("scripts.run_forward_daily.fetch_market") as fetch:
            result = daily_run(self.root, self.data, self.execution)
        self.assertEqual(result["next_action"], "HOLD_MARK")
        fetch.assert_not_called()
        self.assertEqual(before, sha(day / "manifest.json"))


class BoardAndMarketTest(unittest.TestCase):
    def test_star_and_main_quantities(self):
        self.assertEqual(target_shares("688001.SH", 1990, 10), 0)
        self.assertEqual(target_shares("688001.SH", 2050, 10), 205)
        self.assertEqual(target_shares("000001.SZ", 2050, 10), 200)
        self.assertEqual(quantity_error("688001.SH", "BUY", 205, 0), "")
        self.assertEqual(quantity_error("688001.SH", "BUY", 100, 0), "below_star_minimum_200")
        self.assertEqual(quantity_error("688001.SH", "SELL", 50, 50), "")
        self.assertNotEqual(quantity_error("688001.SH", "SELL", 50, 250), "")
        self.assertEqual(quantity_error("000001.SZ", "SELL", 150, 150), "")
        with self.assertRaises(ValueError):
            target_shares("920001.BJ", 10000, 10)

    def test_fee_inclusive_precheck(self):
        plan = build_order_plan({"000001.SZ": 1.}, {}, {"000001.SZ": 10.}, cash=1000., board_rules=True)
        checks = precheck_order_plan(plan, cash=1000., board_rules=True, commission_rate=.0003)
        self.assertEqual(checks.check_status.iloc[0], "BLOCK")

    def test_missing_quote_is_not_suspension_and_exact_dates_required(self):
        daily = pd.DataFrame(columns=["ts_code", "trade_date", "close"])
        limits = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260928", "up_limit": 11., "down_limit": 9.}])
        adj = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260928", "adj_factor": 1.}])
        old = pd.DataFrame([{"symbol": "000001.SZ", "close": 10., "price_date": "2026-09-24"}])
        with self.assertRaisesRegex(RuntimeError, "without suspension"):
            build_market(daily, limits, adj, pd.DataFrame(), {"000001.SZ"}, "2026-09-28", old)
        suspend = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260928", "suspend_type": "S"}])
        result = build_market(daily, limits, adj, suspend, {"000001.SZ"}, "2026-09-28", old)
        self.assertTrue(result.is_suspended.iloc[0])
        self.assertEqual(result.price_date.iloc[0], "2026-09-24")
        bad = adj.assign(trade_date="20260924")
        with self.assertRaisesRegex(RuntimeError, "wrong-date"):
            build_market(daily, limits, bad, suspend, {"000001.SZ"}, "2026-09-28", old)


if __name__ == "__main__":
    unittest.main()
