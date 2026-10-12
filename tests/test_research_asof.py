import unittest
import numpy as np
import pandas as pd

from analysis.financial_asof import normalize_statements, annual_asof
from analysis.signal_asof import signals_asof


class ResearchAsOfTest(unittest.TestCase):
    def raw(self):
        common = dict(ts_code=['A'], ann_date=['20240401'], f_ann_date=['20240402'],
                      end_date=['20231231'], report_type=[1], comp_type=[1])
        return (pd.DataFrame(dict(**common, revenue=[100.], oper_cost=[70.])),
                pd.DataFrame(dict(**common, total_assets=[300.], total_hldr_eqy_exc_min_int=[60.])))

    def test_strict_availability_and_future_revision(self):
        income, balance = self.raw()
        revision = income.assign(ann_date='20240501', f_ann_date='20240501', revenue=999.)
        i, _ = normalize_statements(pd.concat([income, revision]), 'income')
        b, _ = normalize_statements(balance, 'balance')
        self.assertTrue(annual_asof(i, b, '2024-04-02').empty)
        self.assertEqual(annual_asof(i, b, '2024-04-03').loc['A', 'income_revenue'], 100)
        self.assertEqual(annual_asof(i, b, '2024-05-02').loc['A', 'income_revenue'], 999)

    def test_different_annual_periods_do_not_join(self):
        income, balance = self.raw()
        i, _ = normalize_statements(income, 'income')
        b, _ = normalize_statements(balance.assign(end_date='20221231'), 'balance')
        self.assertTrue(annual_asof(i, b, '2024-04-03').empty)

    def test_conflicting_version_invalidates_values(self):
        income, _ = self.raw()
        i, audit = normalize_statements(pd.concat([income, income.assign(revenue=300.)]), 'income')
        self.assertEqual(audit['ambiguous_versions'], 1)
        self.assertTrue(i.revenue.isna().all())
        i, audit = normalize_statements(pd.concat([income, income]), 'income')
        self.assertEqual(audit['ambiguous_versions'], 0)
        self.assertEqual(len(i), 1)

    def test_malformed_later_date_cannot_fall_back_to_earlier_date(self):
        income, _ = self.raw()
        with self.assertRaisesRegex(ValueError, 'malformed'):
            normalize_statements(income.assign(f_ann_date='bad-date'), 'income')

    def test_quarters_and_other_report_types_filtered(self):
        income, _ = self.raw()
        i, audit = normalize_statements(pd.concat([
            income, income.assign(end_date='20230930'), income.assign(report_type=4)]), 'income')
        self.assertEqual(len(i), 1)
        self.assertEqual(audit['filtered_rows_before_dedup'], 2)

    def prices(self):
        dates = pd.bdate_range('2024-01-01', periods=190)
        return pd.DataFrame(np.exp(np.arange(190)[:, None]*np.array([.001, .002, .003])),
                            index=dates, columns=['A', 'B', 'C'])

    def test_future_prices_cannot_change_prior_signal(self):
        p = self.prices()
        specs = {'momentum': [126, 21, 1], 'reversal': [5, 0, -1]}
        before, _ = signals_asof(p, p.index[160], list(p), specs)
        p.iloc[161:] = np.nan
        after, _ = signals_asof(p, p.index[160], list(p), specs)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(list(before), list(specs))  # no hardcoded BLEND injected
        np.testing.assert_allclose(before.momentum, p.iloc[139]/p.iloc[13]-1)

    def test_common_sample_and_missing_current_quote(self):
        p = self.prices()
        p.iloc[13, 0] = np.nan
        p.iloc[160, 1] = np.nan
        frame, audit = signals_asof(p, p.index[160], list(p), {'M': [126, 21, 1], 'R': [5, 0, -1]})
        self.assertEqual(frame.index.tolist(), ['C'])
        self.assertEqual(audit['excluded_asof'], 2)

    def test_recent_skip_and_unknown_symbol(self):
        p = self.prices()
        specs = {'M': [126, 21, 1]}
        frame, _ = signals_asof(p, p.index[160], list(p), specs)
        p.iloc[140:161] *= 2
        other, _ = signals_asof(p, p.index[160], list(p), specs)
        pd.testing.assert_frame_equal(frame, other)
        with self.assertRaises(ValueError):
            signals_asof(p, p.index[160], ['UNKNOWN'], specs)


if __name__ == '__main__':
    unittest.main()
