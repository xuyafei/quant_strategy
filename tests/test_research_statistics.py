import unittest
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from analysis.research_statistics import adjust_pvalues, hac_mean, paired_returns


class MultipleTestingTest(unittest.TestCase):
    def test_known_adjustments(self):
        result = adjust_pvalues([.01, .04, .03, .002])
        np.testing.assert_allclose(result.holm, [.03, .06, .06, .008])
        np.testing.assert_allclose(result.bh, [.02, .04, .04, .008])
        np.testing.assert_allclose(result.by, np.array([.02, .04, .04, .008])*25/12)

    def test_permutation_invariance_and_bounds(self):
        values = np.random.default_rng(8).uniform(size=30)
        permutation = np.random.default_rng(9).permutation(30)
        np.testing.assert_allclose(adjust_pvalues(values).iloc[permutation], adjust_pvalues(values[permutation]))
        result = adjust_pvalues(values)
        self.assertTrue((result.holm <= result.bonferroni + 1e-15).all())
        self.assertTrue((result.by >= result.bh - 1e-15).all())

    def test_single_and_extreme_pvalues(self):
        np.testing.assert_allclose(adjust_pvalues([.04]), .04)
        result = adjust_pvalues([0, 1])
        np.testing.assert_allclose(result.iloc[0], 0)
        np.testing.assert_allclose(result.iloc[1], 1)

    def test_invalid_pvalues_fail(self):
        for values in ([], [np.nan], [-.1], [1.1], [[.1]]):
            with self.assertRaises(ValueError):
                adjust_pvalues(values)

    def test_lag_zero_matches_mean_standard_error(self):
        values = np.random.default_rng(12).normal(.01, .1, 200)
        result = hac_mean(values, 0)
        self.assertAlmostEqual(result['hac_se'], values.std(ddof=1)/np.sqrt(len(values)))
        self.assertAlmostEqual(result['statistic'], ttest_1samp(values, 0).statistic)

    def test_hac_matches_independent_intercept_sandwich(self):
        values = np.random.default_rng(16).normal(size=90)
        residual = values-values.mean()
        kernel = np.zeros((90, 90))
        for row in range(90):
            for column in range(90):
                distance = abs(row-column)
                kernel[row, column] = max(0, 1-distance/6)
        expected = np.sqrt(residual @ kernel @ residual / (90*89))
        self.assertAlmostEqual(hac_mean(values, 5)['hac_se'], expected)

    def test_negative_mean_is_not_positive_evidence(self):
        values = np.random.default_rng(12).normal(-1, .1, 200)
        result = hac_mean(values)
        self.assertGreater(result['p_positive'], .99)
        self.assertLess(result['p_two_sided'], .001)

    def test_missing_and_constant_returns_fail(self):
        for values in (np.ones(100), np.full(100, np.nan), np.zeros(20)):
            with self.assertRaises(ValueError):
                hac_mean(values)

    def test_nav_no_fill_and_duplicate_rejection(self):
        nav = pd.DataFrame({'a': np.linspace(1, 2, 100)}, index=pd.date_range('2020-01-01', periods=100))
        self.assertEqual(len(paired_returns(nav)), 99)
        nav.iloc[3, 0] = np.nan
        with self.assertRaises(ValueError):
            paired_returns(nav)
        with self.assertRaises(ValueError):
            paired_returns(pd.DataFrame({'a': np.ones(100)}, index=[0]*100))


if __name__ == '__main__':
    unittest.main()
