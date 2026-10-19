import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collections import namedtuple

import pandas as pd

from scripts.fetch_long_history_dataset import paginated, quarter_ends, csv_write, merge_parts, sh_sz, Downloader, audit_symbols, json_write


class LongHistoryTests(unittest.TestCase):
    def test_pagination_continues_below_requested_limit(self):
        source = pd.DataFrame({'x': range(5)})
        def call(endpoint, **kw):
            return source.iloc[kw['offset']:kw['offset']+2]
        result, pages = paginated(call, 'test', {}, page_size=10)
        self.assertEqual(len(result), 5)
        self.assertEqual([p['rows'] for p in pages], [2, 2, 1, 0])

    def test_repeated_pages_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'repeated'):
            paginated(lambda *a, **k: pd.DataFrame({'x': [1]}), 'test', {})

    def test_known_daily_cap_stops_on_short_page(self):
        def call(*a, **kw):
            self.assertEqual(kw['offset'], 0)
            return pd.DataFrame({'x':[1,2]})
        frame, pages = paginated(call, 'daily', {}, page_size=6000, known_page_cap=True)
        self.assertEqual(len(pages), 1)
        self.assertEqual(len(frame), 2)

    def test_source_duplicates_retained_for_audit(self):
        data = iter([pd.DataFrame({'x': [1, 2]}), pd.DataFrame({'x': [2, 3]}), pd.DataFrame()])
        result, _ = paginated(lambda *a, **k: next(data), 'test', {})
        self.assertEqual(len(result), 4)
        self.assertEqual(int(result.duplicated().sum()), 1)

    def test_unfinished_quarter_excluded(self):
        self.assertEqual(quarter_ends('2026-01-01', '2026-10-15'), ['20260331','20260630','20260930'])

    def test_exchange_filter(self):
        frame = sh_sz(pd.DataFrame({'ts_code': ['600000.SH','000001.SZ','920001.BJ']}))
        self.assertEqual(len(frame), 2)

    def test_audit_opinions_include_documented_old_codes(self):
        with tempfile.TemporaryDirectory() as d:
            basic = pd.DataFrame({'ts_code':['001872.SZ'], 'list_date':['19930505']})
            json_write(Path(d)/'identity_aliases.json', {'aliases':[{'old_code':'000022.SZ','new_code':'001872.SZ'}]})
            self.assertEqual(audit_symbols(d, basic, '20261015'), ['000022.SZ','001872.SZ'])

    def test_unknown_alias_does_not_expand_download(self):
        with tempfile.TemporaryDirectory() as d:
            basic = pd.DataFrame({'ts_code':['001872.SZ'], 'list_date':['19930505']})
            json_write(Path(d)/'identity_aliases.json', {'aliases':[{'old_code':'000043.SZ','new_code':'001914.SZ'}]})
            with self.assertRaisesRegex(ValueError, 'target absent'):
                audit_symbols(d, basic, '20261015')

    def test_streaming_union_schema(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, out = [Path(d)/n for n in ['a.csv.gz','b.csv.gz','out.csv.gz']]
            csv_write(a, pd.DataFrame({'ts_code':['a'], 'x':[1]}))
            csv_write(b, pd.DataFrame({'ts_code':['b'], 'y':[2]}))
            self.assertEqual(merge_parts([a,b],out),2)
            result = pd.read_csv(out)
            self.assertEqual(list(result), ['ts_code','x','y'])
            self.assertTrue(pd.isna(result.loc[0,'y']))

    def test_resume_verifies_hash_and_request(self):
        with tempfile.TemporaryDirectory() as d:
            dl = Downloader(Path(d), 'not-a-real-token')
            dl.call = lambda *a, **kw: pd.DataFrame({'ts_code':['600000.SH'], 'trade_date':['20100104'], 'close':[10.]})
            path = dl.fetch('daily','20100104',{'trade_date':'20100104'},['close'],True)
            dl.call = lambda *a, **kw: self.fail('cached file should not request network')
            self.assertEqual(dl.fetch('daily','20100104',{'trade_date':'20100104'}), path)
            with self.assertRaisesRegex(ValueError,'request mismatch'):
                dl.fetch('daily','20100104',{'trade_date':'20100105'})
            path.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                dl.fetch('daily','20100104',{'trade_date':'20100104'})

    def test_incorrect_date_is_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            dl = Downloader(Path(d), 'not-a-real-token')
            dl.call = lambda *a, **kw: pd.DataFrame({'ts_code':['600000.SH'], 'trade_date':['20100105'], 'close':[10.]})
            with self.assertRaisesRegex(ValueError,'wrong trade date'):
                dl.fetch('daily','20100104',{'trade_date':'20100104'})
            self.assertFalse(list(Path(d).rglob('*.csv.gz')))

    def test_disk_guard_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            dl = Downloader(Path(d), 'not-a-real-token')
            disk = namedtuple('disk','total used free')(100,99,1)
            with patch('scripts.fetch_long_history_dataset.shutil.disk_usage',return_value=disk):
                with self.assertRaisesRegex(RuntimeError,'Disk safety stop'):
                    dl.fetch('daily','20100104',{'trade_date':'20100104'})


if __name__ == '__main__':
    unittest.main()
