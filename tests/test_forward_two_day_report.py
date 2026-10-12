import unittest
import numpy as np
from scripts.build_forward_two_day_report import period_returns


class TwoDayReturnsTest(unittest.TestCase):
    def test_compounding_not_arithmetic_sum(self):
        result=period_returns(100.,90.,99.)
        self.assertAlmostEqual(result['first_return_pct'],-10.)
        self.assertAlmostEqual(result['second_return_pct'],10.)
        self.assertAlmostEqual(result['period_return_pct'],-1.)
        self.assertAlmostEqual(result['first_pnl']+result['second_pnl'],result['period_pnl'])

    def test_invalid_values(self):
        for end in [0.,-1.,np.nan,np.inf]:
            with self.assertRaises(ValueError):
                period_returns(100.,101.,end)


if __name__=='__main__':
    unittest.main()
