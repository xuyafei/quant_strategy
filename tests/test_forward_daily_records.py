import unittest
from unittest.mock import patch

from live.forward_ledger import advance,sha,verify
from scripts.export_forward_daily_records import export,table
from scripts.run_forward_daily_with_records import run
from tests import test_forward_ledger as fixtures


class DailyRecordsTest(unittest.TestCase):
    setUp=fixtures.ForwardLedgerTest.setUp
    init=fixtures.ForwardLedgerTest.init
    market=fixtures.ForwardLedgerTest.market
    open_clock=fixtures.ForwardLedgerTest.open_clock

    def first(self):
        self.init();self.open_clock()
        return advance(self.root,self.market(),self.execution)

    def test_first_and_holding_days_with_no_source_mutation(self):
        day=self.first();before=sha(day/'manifest.json')
        result=export(self.root)
        self.assertEqual(result['days'],1)
        folder=self.root/'reports/daily_records'
        first=folder/'2026-09-28.md'
        self.assertIn('000001.SZ',first.read_text())
        self.assertIn('当日模拟成交',first.read_text())
        digest=sha(first)
        self.assertEqual(export(self.root),result)
        self.assertEqual(sha(first),digest)
        self.open_clock('2026-09-29')
        advance(self.root,self.market('2026-09-29',close=11.),'2026-09-29')
        self.assertEqual(export(self.root)['days'],2)
        holding=(folder/'2026-09-29.md').read_text()
        self.assertIn('当日无成交',holding)
        self.assertIn('查看 2026-09-29',(folder/'README.md').read_text())
        self.assertFalse((folder/'2026-09-30.md').exists())
        self.assertEqual(sha(day/'manifest.json'),before)
        verify(day)

    def test_manual_edits_are_not_overwritten(self):
        self.first();export(self.root)
        file=self.root/'reports/daily_records/2026-09-28.md'
        file.write_text(file.read_text()+'user note')
        with self.assertRaises(FileExistsError):
            export(self.root)
        self.assertTrue(file.read_text().endswith('user note'))

    def test_modified_ledger_is_rejected(self):
        day=self.first()
        file=day/'summary.csv';file.write_text(file.read_text()+'\n')
        with self.assertRaises(RuntimeError):
            export(self.root)
        self.assertFalse((self.root/'reports/daily_records').exists())

    def test_empty_session_does_not_create_future_record(self):
        self.init()
        self.assertEqual(export(self.root)['status'],'NO_COMMITTED_DAYS')
        self.assertFalse((self.root/'reports/daily_records').exists())

    def test_daily_retry_exports_without_trading(self):
        day=self.first();before=sha(day/'manifest.json')
        with patch('scripts.run_forward_daily.utc_now',return_value=self.clock.return_value),patch('scripts.run_forward_daily.fetch_market') as fetch:
            result=run(self.root,self.data,self.execution)
        fetch.assert_not_called()
        self.assertEqual(result['daily_records']['days'],1)
        self.assertEqual(sha(day/'manifest.json'),before)

    def test_markdown_cells_escaped(self):
        self.assertIn('a\\|b c',table(['x'],[['a|b\nc']]))

    def test_post_commit_failure_still_exports_and_remains_failure(self):
        self.first()
        with patch('scripts.run_forward_daily_with_records.frozen_daily_run',side_effect=RuntimeError('weekly preparation failed')):
            with self.assertRaisesRegex(RuntimeError,'weekly preparation failed'):
                run(self.root,self.data,self.execution)
        self.assertTrue((self.root/'reports/daily_records/2026-09-28.md').exists())

    def test_closed_day_wrapper_does_not_export(self):
        self.init()
        with patch('scripts.run_forward_daily_with_records.export') as exporter:
            result=run(self.root,self.data,'2026-09-27')
        self.assertEqual(result['status'],'MARKET_CLOSED_NO_MUTATION')
        exporter.assert_not_called()


if __name__=='__main__':
    unittest.main()
