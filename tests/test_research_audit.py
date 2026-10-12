import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from analysis.research_audit import build_research_audit, render_markdown
from analysis.return_attribution import ols_hac
from analysis.selection_diagnostics import cscv
from scripts.run_research_audit import run


class ResearchAuditTest(unittest.TestCase):
    pairs = {'a': 'benchmark', 'b': 'benchmark'}

    def nav(self, rows=101):
        rng = np.random.default_rng(32)
        values = np.vstack([np.zeros((1, 3)), rng.normal(.0003, .01, (rows-1, 3))])
        return pd.DataFrame(np.cumprod(1+values, axis=0),
                            index=pd.bdate_range('2025-01-01', periods=rows),
                            columns=['a', 'b', 'benchmark'])

    def test_costed_nav_readonly_and_no_trading_approval(self):
        nav = self.nav(); before = nav.copy(deep=True)
        report = build_research_audit(nav, self.pairs,
                                      cscv_window=[str(nav.index[1].date()), str(nav.index[-1].date())])
        pd.testing.assert_frame_equal(nav, before)
        self.assertFalse(report['live_approval'])
        self.assertFalse(report['changes_trading_rules'])
        self.assertEqual(report['selection']['splits'], 252)
        self.assertEqual(report['point_in_time_data_check'], 'NOT_ASSESSED_FROM_NAV')
        self.assertIn('不是未来亏损概率', render_markdown(report))
        json.dumps(report, allow_nan=False)

    def test_missing_nav_and_nonmatching_dates_rejected(self):
        nav = self.nav(); nav.iloc[1, 0] = np.nan
        with self.assertRaises(ValueError):
            build_research_audit(nav, self.pairs)
        nav = self.nav()
        factors = nav.pct_change(fill_method=None).iloc[2:][['benchmark']]
        with self.assertRaisesRegex(ValueError, 'align'):
            build_research_audit(nav, self.pairs, factors=factors,
                                 attribution_window=[nav.index[1], nav.index[-1]])

    def test_short_forward_sample_not_interpreted_as_pass(self):
        nav = self.nav(5)
        report = build_research_audit(nav, self.pairs,
                                      cscv_window=[nav.index[1], nav.index[-1]])
        self.assertTrue(all(r['status'] == 'NOT_ASSESSED' for r in report['candidates']))
        self.assertEqual(report['selection']['status'], 'NOT_ASSESSED')
        self.assertNotIn('NaN', json.dumps(report, allow_nan=False))

    def test_degenerate_candidate_does_not_shrink_family(self):
        nav = self.nav(); nav.b = nav.benchmark
        report = build_research_audit(nav, self.pairs)
        a, b = report['candidates']
        self.assertEqual(b['status'], 'NOT_ASSESSED')
        self.assertEqual(report['family_size'], 2)
        self.assertAlmostEqual(a['adjusted_positive_p']['holm'], min(1., 2*a['p_positive']))

    def test_explicit_cscv_window_and_no_silent_truncation(self):
        nav = self.nav(106)
        bad = build_research_audit(nav, self.pairs, cscv_window=[nav.index[1], nav.index[-1]])
        self.assertEqual(bad['selection']['status'], 'NOT_ASSESSED')
        good = build_research_audit(nav, self.pairs, cscv_window=[nav.index[6], nav.index[-1]])
        self.assertEqual(good['selection']['excluded_rows'], 5)

    def test_regression_schema_and_complexity_bounds(self):
        nav = self.nav()
        returns = nav.pct_change(fill_method=None).iloc[1:]
        with self.assertRaises(ValueError):
            ols_hac(returns.a, returns[['benchmark']].rename(columns={'benchmark': 'intercept'}))
        with self.assertRaises(ValueError):
            ols_hac(returns.a, returns[['benchmark']], lags=2.5)
        long = pd.concat([returns]*2, ignore_index=True)
        long.index = pd.bdate_range('2020-01-01', periods=200)
        with self.assertRaisesRegex(ValueError, 'too many'):
            cscv(long, blocks=20)

    def test_style_attribution_reconciles_arithmetic_mean(self):
        nav = self.nav()
        factors = nav.pct_change(fill_method=None).iloc[1:][['benchmark']]
        report = build_research_audit(nav, self.pairs, factors=factors,
                                      attribution_window=[nav.index[1], nav.index[-1]])
        self.assertEqual(report['attribution']['status'], 'ASSESSED')
        for row in report['attribution']['results']:
            expected = nav[row['strategy']].pct_change(fill_method=None).iloc[1:].mean()*252
            self.assertAlmostEqual(sum(row['arithmetic_contributions'].values()), expected)

    def test_cli_reproducible_hash_checked_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'nav.csv'
            self.nav().to_csv(path, index_label='date')
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = dict(nav=dict(path='nav.csv', sha256=checksum), pairs=self.pairs,
                        frequency='daily', return_units='decimal',
                        family_description='two predetermined candidates', cost_convention='synthetic test')
            spec_path = root/'spec.json'; spec_path.write_text(json.dumps(spec))
            first = run(spec_path, root/'first'); second = run(spec_path, root/'second')
            self.assertEqual(first, second)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            self.assertEqual((root/'first/report.json').read_bytes(), (root/'second/report.json').read_bytes())
            with self.assertRaises(FileExistsError):
                run(spec_path, root/'first')
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                run(spec_path, root/'third')
            self.assertFalse((root/'third').exists())

    def test_duplicate_headers_rejected_before_pandas_renames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'nav.csv'
            path.write_text('date,a,a,benchmark\n2025-01-01,1,1,1\n2025-01-02,2,2,2\n')
            spec = dict(nav=dict(path='nav.csv', sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            config = root/'spec.json'; config.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'headers'):
                run(config, root/'out')


if __name__ == '__main__':
    unittest.main()
