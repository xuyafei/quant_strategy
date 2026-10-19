import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.fetch_long_history_dataset import csv_write, json_write, sha
from scripts.validate_long_history_market import validate, validate_aggregates, summarize_source_quality


class MarketValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        csv_write(self.root/'trade_cal.csv.gz', pd.DataFrame({'cal_date':['20100104'], 'is_open':[1]}))
        csv_write(self.root/'stock_basic.csv.gz', pd.DataFrame({'ts_code':['600000.SH'], 'list_status':['D']}))
        self.raw = pd.DataFrame({'ts_code':['600000.SH'], 'trade_date':['20100104'],
                                 'open':[10.], 'high':[11.], 'low':[9.], 'close':[10.]})
        self.adj = pd.DataFrame({'ts_code':['600000.SH'], 'trade_date':['20100104'], 'adj_factor':[1.]})

    def run_check(self, aliases_path=None):
        for endpoint, frame in [('daily', self.raw), ('adj_factor', self.adj)]:
            path = self.root/'parts'/endpoint/'20100104.csv.gz'
            path.parent.mkdir(parents=True, exist_ok=True)
            csv_write(path, frame)
        return validate(self.root, aliases_path)

    def test_valid_delisted_stock(self):
        self.assertTrue(self.run_check())

    def test_empty_partition_rejected(self):
        self.raw = self.raw.iloc[:0]
        self.assertFalse(self.run_check())

    def test_infinite_price_rejected(self):
        self.raw.loc[0,'close'] = float('inf')
        self.assertFalse(self.run_check())

    def test_impossible_ohlc_rejected(self):
        self.raw.loc[0,'high'] = 8.
        self.assertFalse(self.run_check())

    def test_missing_matching_adjustment_rejected(self):
        self.adj.loc[0,'ts_code'] = '000001.SZ'
        self.assertFalse(self.run_check())

    def test_wrong_date_rejected(self):
        self.raw.loc[0,'trade_date'] = '20100105'
        self.assertFalse(self.run_check())

    def test_duplicate_keys_rejected(self):
        self.raw = pd.concat([self.raw, self.raw])
        self.assertFalse(self.run_check())

    def test_unknown_identity_rejected(self):
        self.raw.loc[0,'ts_code'] = self.adj.loc[0,'ts_code'] = '000001.SZ'
        self.assertFalse(self.run_check())

    def alias_fixture(self, effective='20100105', new_code='600000.SH'):
        path = self.root/'aliases.json'
        json_write(path, {'aliases':[{'old_code':'000022.SZ', 'new_code':new_code,
                                    'effective_date':effective, 'source_url':'https://example.com/disclosure'}]})
        self.raw.loc[0,'ts_code'] = self.adj.loc[0,'ts_code'] = '000022.SZ'
        return path

    def test_documented_old_identity_before_change(self):
        self.assertTrue(self.run_check(self.alias_fixture()))

    def test_old_identity_after_change_rejected(self):
        self.assertFalse(self.run_check(self.alias_fixture(effective='20100104')))

    def test_unknown_alias_target_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid historical identity alias'):
            self.run_check(self.alias_fixture(new_code='000001.SZ'))


class AggregateValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        csv_write(self.root/'daily.csv.gz', pd.DataFrame({'ts_code':['600000.SH'], 'close':[10.]}))
        json_write(self.root/'parts/daily/20100104.csv.json', {'columns':['ts_code','close']})
        for stage in ['market','finance','industry','audit_opinions']:
            json_write(self.root/f'{stage}_failures.json', [])
        self.coverage = {'core_partition_download_complete':True, 'datasets':{'daily':{'rows':1}}}

    def run_check(self):
        with patch('scripts.validate_long_history_market.audit', return_value=self.coverage):
            return validate_aggregates(self.root)

    def test_valid_aggregate(self):
        self.assertTrue(self.run_check())

    def test_truncated_aggregate_rejected(self):
        self.coverage['datasets']['daily']['rows'] = 2
        self.assertFalse(self.run_check())

    def test_missing_column_rejected(self):
        csv_write(self.root/'daily.csv.gz', pd.DataFrame({'ts_code':['600000.SH']}))
        self.assertFalse(self.run_check())

    def test_missing_failure_manifest_rejected(self):
        (self.root/'market_failures.json').unlink()
        self.assertFalse(self.run_check())

    def test_missing_partitions_rejected(self):
        self.coverage['core_partition_download_complete'] = False
        self.assertFalse(self.run_check())


class SourceQualityTests(unittest.TestCase):
    def test_b_shares_and_unknown_codes_are_reported_not_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            csv_write(root/'stock_basic.csv.gz', pd.DataFrame({'ts_code':['001872.SZ']}))
            json_write(root/'identity_aliases.json', {'aliases':[{'old_code':'000022.SZ','new_code':'001872.SZ'}]})
            path = root/'daily_basic.csv.gz'
            csv_write(path, pd.DataFrame({'ts_code':['001872.SZ','000022.SZ','200001.SZ','900901.SH','601001.SH']}))
            before = sha(path)
            report = summarize_source_quality(root)['datasets']['daily_basic.csv.gz']
            self.assertEqual(report['b_share_rows'], 2)
            self.assertEqual(report['non_b_codes_outside_master'], ['601001.SH'])
            self.assertEqual(sha(path), before)


if __name__ == '__main__':
    unittest.main()
