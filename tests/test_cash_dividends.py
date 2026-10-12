import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from live.cash_dividends import FIELDS, POLICY, normalize_events, payments_for
from live.forward_ledger import (ROOT, advance, latest_day, prepare, sha, verify, session_config)
from live.forward_revision import EXTRA_CODE, REVISION_DIR, upgrade
from scripts.fetch_forward_actions import run as fetch_actions
from scripts.run_forward_daily import run as daily_run
from tests import test_forward_ledger as fixtures


def vendor_event(**changes):
    row = dict.fromkeys(FIELDS.split(","), None)
    row.update(ts_code="000001.SZ", end_date="20260630", ann_date="20260920",
               imp_ann_date="20260920", div_proc="实施", stk_div=0., cash_div_tax=.1,
               record_date="20260928", ex_date="20260929", pay_date="20260929")
    return pd.DataFrame([{**row, **changes}])


class CashDividendLedgerTest(unittest.TestCase):
    def setUp(self):
        fixtures.ForwardLedgerTest.setUp(self)
        token = patch("scripts.fetch_forward_actions.get_tushare_token", return_value="TEST_ONLY")
        token.start()
        self.addCleanup(token.stop)

    init = fixtures.ForwardLedgerTest.init
    market = fixtures.ForwardLedgerTest.market
    open_clock = fixtures.ForwardLedgerTest.open_clock

    def ready(self):
        self.init()
        self.open_clock()
        self.day1 = advance(self.root, self.market(), self.execution)
        self.old_hash = sha(self.day1 / "manifest.json")
        self.open_clock("2026-09-29")
        old = json.loads((self.root / "initial/session.json").read_text())["code_hashes"]
        self.review = {"policy": POLICY, "anchor_date": "2026-09-28", "effective_date": "2026-09-29",
                       "anchor_manifest_sha256": self.old_hash, "old_code_hashes": old,
                       "new_code_hashes": {name: sha(ROOT / name) for name in set(old) | set(EXTRA_CODE)},
                       "reason": "unit-test authorized review"}
        self.review_path = self.base / "review.json"
        self.review_path.write_text(json.dumps(self.review))
        upgrade(self.root, self.review_path)
        return self.market("2026-09-29", close=9.9, factor=10/9.9)

    def fetch(self, market, raw=None):
        with patch("scripts.fetch_forward_actions._pro") as pro, patch("scripts.fetch_forward_actions.utc_now", return_value=self.clock.return_value):
            pro.return_value.dividend.return_value = vendor_event() if raw is None else raw
            return fetch_actions(self.root, market, "2026-09-29")

    def test_dividend_reconciles_cash_nav_shares_and_history(self):
        market = self.ready()
        self.fetch(market)
        day2 = advance(self.root, market, "2026-09-29")
        before = pd.read_csv(self.day1 / "summary.csv").set_index("strategy")
        after = pd.read_csv(day2 / "summary.csv").set_index("strategy")
        for strategy in before.index:
            pos = pd.read_csv(self.day1 / f"paper_account/{strategy}/positions.csv")
            qty = pos.set_index("symbol").loc["000001.SZ", "shares"]
            self.assertAlmostEqual(after.loc[strategy, "cash"] - before.loc[strategy, "cash"], qty * .1)
            self.assertAlmostEqual(after.loc[strategy, "total_asset"], before.loc[strategy, "total_asset"])
            self.assertEqual(after.loc[strategy, "filled_orders"], 0)
            pos2 = pd.read_csv(day2 / f"paper_account/{strategy}/positions.csv")
            self.assertEqual(pos.shares.tolist(), pos2.shares.tolist())
        history = pd.read_csv(day2 / "dividends.csv")
        self.assertEqual(len(history), 2)
        self.assertEqual(sha(self.day1 / "manifest.json"), self.old_hash)
        verify(self.root / "initial")
        verify(day2)
        # A normal retry neither downloads nor applies dividends twice.
        digest = sha(day2 / "manifest.json")
        with patch("scripts.run_forward_daily.utc_now", return_value=self.clock.return_value), patch("scripts.run_forward_daily.fetch_market") as fetch:
            daily_run(self.root, self.data, "2026-09-29")
            fetch.assert_not_called()
        self.assertEqual(sha(day2 / "manifest.json"), digest)
        with self.assertRaisesRegex(RuntimeError, "next open"):
            advance(self.root, market, "2026-09-29")
        # Next day carries the cash and event history, without a second credit.
        self.open_clock("2026-09-30")
        m3 = self.market("2026-09-30", close=9.9, factor=10/9.9)
        with patch("scripts.fetch_forward_actions._pro") as pro, patch("scripts.fetch_forward_actions.utc_now", return_value=self.clock.return_value):
            pro.return_value.dividend.return_value = pd.DataFrame(columns=FIELDS.split(","))
            fetch_actions(self.root, m3, "2026-09-30")
        d3 = advance(self.root, m3, "2026-09-30")
        self.assertEqual(pd.read_csv(d3 / "summary.csv").dividend_cash_gross.sum(), 0)
        self.assertEqual(len(pd.read_csv(d3 / "dividends.csv")), 2)

    def test_bad_events_reject_without_publishing(self):
        market = self.ready()
        for change in [{"stk_div": .1}, {"stk_div": None}, {"stk_co_rate": .1},
                       {"record_date": "20260924"}, {"pay_date": "20260930"},
                       {"cash_div_tax": float("nan")}, {"cash_div_tax": -.1},
                       {"cash_div_tax": 1.}, {"ann_date": "20260930"},
                       {"div_proc": "预案"}]:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.fetch(market, vendor_event(**change))
            self.assertFalse((self.root / "corporate_actions/20260929").exists())
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.fetch(market, pd.concat([vendor_event(), vendor_event()]))
        self.assertEqual(latest_day(self.root)[1]["date"], "2026-09-28")

    def test_absent_evidence_cannot_bypass_guard(self):
        market = self.ready()
        with self.assertRaises(FileNotFoundError):
            advance(self.root, market, "2026-09-29")
        with self.assertRaisesRegex(RuntimeError, "unexplained"):
            self.fetch(market, pd.DataFrame(columns=FIELDS.split(",")))
        self.assertFalse((self.root / "days/20260929").exists())

    def test_action_tampering_blocked(self):
        market = self.ready()
        source = self.fetch(market)
        p = source / "events.csv"
        p.write_text(p.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            advance(self.root, market, "2026-09-29")

    def test_second_account_failure_does_not_double_dividend_on_retry(self):
        market = self.ready()
        self.fetch(market)
        from live.forward_ledger import save_account_state
        count = []
        def fail(*args, **kwargs):
            count.append(1)
            if len(count) == 2:
                raise RuntimeError("injected dividend account failure")
            return save_account_state(*args, **kwargs)
        with patch("live.forward_ledger.save_account_state", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                advance(self.root, market, "2026-09-29")
        self.assertFalse((self.root / "days/20260929").exists())
        self.assertEqual(sha(self.day1 / "manifest.json"), self.old_hash)
        day2 = advance(self.root, market, "2026-09-29")
        self.assertEqual(len(pd.read_csv(day2 / "dividends.csv")), 2)

    def test_revision_is_explicit_and_immutable(self):
        self.ready()
        self.assertEqual(session_config(self.root)["corporate_actions"], POLICY)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            upgrade(self.root, self.review_path)
        p = self.root / REVISION_DIR / "review.json"
        p.write_text(p.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            session_config(self.root)

    def test_changed_execution_code_still_blocks_after_upgrade(self):
        self.ready()
        def changed(path):
            return "changed" if path == ROOT / "live/cash_dividends.py" else sha(path)
        with patch("live.forward_ledger.sha", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "code changed"):
                session_config(self.root)

    def test_cash_dividend_on_rebalance_is_received_before_order_flow(self):
        market = self.ready()
        self.fetch(market)
        advance(self.root, market, "2026-09-29")
        self.open_clock("2026-09-30")
        m30 = self.market("2026-09-30", close=9.9, factor=10/9.9)
        with patch("scripts.fetch_forward_actions._pro") as pro, patch("scripts.fetch_forward_actions.utc_now", return_value=self.clock.return_value):
            pro.return_value.dividend.return_value = pd.DataFrame(columns=FIELDS.split(","))
            fetch_actions(self.root, m30, "2026-09-30")
        d30 = advance(self.root, m30, "2026-09-30")
        self.bundle["prices"]["trade_date"] = pd.Timestamp("2026-09-30")
        self.bundle["prices"].loc[self.bundle["prices"].ts_code.eq("000001.SZ"), "close"] = 9.9
        pd.DataFrame({"ts_code":["000001.SZ","688001.SH"], "trade_date":["20260930"]*2,
                      "adj_factor":[10/9.9,1.]}).to_csv(self.data/"adj_factor_all_a.csv.gz", index=False, compression="gzip")
        prepare(self.root, self.data, "2026-09-30", "2026-10-08")
        self.open_clock("2026-10-08")
        m8 = self.market("2026-10-08", close=9.8, factor=10/9.8)
        with patch("scripts.fetch_forward_actions._pro") as pro, patch("scripts.fetch_forward_actions.utc_now", return_value=self.clock.return_value):
            pro.return_value.dividend.return_value = vendor_event(record_date="20260930",ex_date="20261008",pay_date="20261008")
            fetch_actions(self.root, m8, "2026-10-08")
        d8 = advance(self.root, m8, "2026-10-08")
        summary = pd.read_csv(d8/"summary.csv").set_index("strategy")
        prior = pd.read_csv(d30/"summary.csv").set_index("strategy")
        for strategy in prior.index:
            flows = pd.read_csv(d8/f"paper_trades/{strategy}.csv").net_cash_flow.sum()
            self.assertEqual(summary.loc[strategy,"action"], "REBALANCE")
            self.assertAlmostEqual(summary.loc[strategy,"cash"], prior.loc[strategy,"cash"] + summary.loc[strategy,"dividend_cash_gross"] + flows)
        self.assertEqual(len(pd.read_csv(d8/"dividends.csv")), 4)


class CashPaymentUnitTest(unittest.TestCase):
    def test_only_record_date_shares_and_explicit_duplicate_protection(self):
        event = pd.DataFrame([dict(event_id="event",symbol="000001.SZ",record_date="2026-09-28",
                                   ex_date="2026-09-29",pay_date="2026-09-29",cash_per_share_gross=.04)])
        positions = pd.DataFrame([dict(symbol="000001.SZ", shares=1000, available_shares=0)])
        paid = payments_for(event, positions, "ONE", set())
        self.assertEqual(paid.cash_gross.item(), 40)
        with self.assertRaisesRegex(RuntimeError, "already credited"):
            payments_for(event, positions, "ONE", {"event"})
        self.assertTrue(payments_for(event, positions.assign(symbol="600001.SH"), "ONE", set()).empty)

    def test_vendor_schema_failure_is_not_empty_success(self):
        with self.assertRaisesRegex(RuntimeError, "schema"):
            normalize_events(pd.DataFrame(), set(), pd.DataFrame(), pd.DataFrame(), "2026-09-28", "2026-09-29")


if __name__ == "__main__":
    unittest.main()
