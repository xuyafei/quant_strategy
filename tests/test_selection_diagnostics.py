import math
import unittest

import numpy as np
import pandas as pd

from analysis.selection_diagnostics import cscv, cscv_summary, information_ratio


class OverfittingTests(unittest.TestCase):
    def frame(self, rows=40, cols=3):
        rng = np.random.default_rng(45)
        return pd.DataFrame(rng.normal(size=(rows, cols)),
                            index=pd.bdate_range('2020-01-01', periods=rows),
                            columns=list('abc')[:cols])

    def test_ir(self):
        x = np.array([1., 2., 4., 3.])
        self.assertAlmostEqual(information_ratio(x), x.mean()/x.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(information_ratio(x), information_ratio(x*7))

    def test_undefined(self):
        for x in ([1, 1, 1], [1, np.nan, 3], [1]):
            with self.assertRaises(ValueError):
                information_ratio(x)

    def test_partition_count_and_complements(self):
        result = cscv(self.frame(), 10)
        self.assertEqual(len(result), math.comb(10, 5))
        groups = {frozenset(map(int, s.split(','))) for s in result.train_blocks}
        self.assertTrue(all(frozenset(set(range(10))-set(s)) in groups for s in groups))
        self.assertTrue(((result.relative_rank > 0) & (result.relative_rank < 1)).all())

    def test_persistent_winner(self):
        frame = self.frame()
        frame['b'] = frame.a + 10
        frame['c'] = frame.a - 10
        result = cscv(frame, 4)
        self.assertEqual(set(result.winner), {'b'})
        self.assertTrue((result.test_rank == 3).all())
        self.assertEqual(cscv_summary(result)['pbo_inclusive'], 0)

    def test_reversing_winner(self):
        frame = self.frame(8, 2)
        frame['a'] = [2, 3, 4, 5, -2, -3, -4, -5]
        frame['b'] = -frame.a
        result = cscv(frame, 2)
        self.assertEqual(cscv_summary(result)['pbo_strict'], 1)

    def test_ties_and_median_mass(self):
        frame = self.frame()
        frame['b'] = frame.a
        frame['c'] = frame.a
        result = cscv(frame[['c', 'b', 'a']], 4)
        self.assertEqual(set(result.winner), {'a'})
        stats = cscv_summary(result)
        self.assertEqual(stats['pbo_strict'], 0)
        self.assertEqual(stats['median_mass'], 1)
        self.assertEqual(stats['pbo_inclusive'], 1)

    def test_bad_partitions_and_inputs(self):
        for blocks in (0, 3, 6, 41):
            with self.assertRaises(ValueError):
                cscv(self.frame(), blocks)
        for frame in (self.frame().iloc[::-1], self.frame().assign(a=np.nan),
                      pd.concat([self.frame(), self.frame()]), self.frame()[['a', 'a']]):
            with self.assertRaises(ValueError):
                cscv(frame, 4)

    def test_column_order_invariance(self):
        frame = self.frame()
        pd.testing.assert_frame_equal(cscv(frame, 4), cscv(frame[['c', 'a', 'b']], 4))


if __name__ == '__main__':
    unittest.main()
