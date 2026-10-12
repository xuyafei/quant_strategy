import unittest
import pandas as pd

from scripts.build_forward_holding_day_report import build, check_close
from live.forward_ledger import advance, seal, sha, verify
from tests import test_cash_dividends as fixtures


class HoldingReportTest(unittest.TestCase):
    setUp = fixtures.CashDividendLedgerTest.setUp
    init = fixtures.CashDividendLedgerTest.init
    market = fixtures.CashDividendLedgerTest.market
    open_clock = fixtures.CashDividendLedgerTest.open_clock
    ready = fixtures.CashDividendLedgerTest.ready
    fetch = fixtures.CashDividendLedgerTest.fetch

    def test_report_reconciles_and_does_not_mutate_ledger(self):
        market = self.ready()
        meta = verify(market)
        seal(market, {k:v for k,v in {**meta,"required_symbols":2}.items() if k!="files"})
        self.fetch(market)
        day = advance(self.root,market,"2026-09-29")
        digest = sha(day/"manifest.json")
        out = self.root/"reports/test"
        facts = build(self.root,"2026-09-29",out)
        self.assertAlmostEqual(facts["sum_daily_pnl"],0.)
        self.assertGreater(facts["sum_dividends_gross"],0.)
        self.assertTrue((out/"daily_vs_cumulative.png").exists())
        self.assertTrue((out/"dividend_price_pnl.png").exists())
        self.assertEqual(sha(day/"manifest.json"),digest)
        verify(day)
        with self.assertRaises(FileExistsError):
            build(self.root,"2026-09-29",out)

    def test_source_destination_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"source"):
            build(self.root,"2026-09-29",self.root/"days/20260929")

    def test_invalid_accounting_values_fail(self):
        for number in [float("nan"),float("inf"),1.1]:
            with self.assertRaisesRegex(ValueError,"reconciliation"):
                check_close(number,1.,"test")


if __name__ == "__main__":
    unittest.main()
