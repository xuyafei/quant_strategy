import unittest
import pandas as pd
from scripts.build_forward_first_day_article import cash_components


class CashAttributionTest(unittest.TestCase):
    def test_budget_rounding_block_and_fee_reconcile(self):
        orders = pd.DataFrame({"target_shares": [0, 100, 100], "target_value": [2000., 2000., 2000.],
                               "estimated_amount": [0., 1900., 1800.]})
        checks = pd.DataFrame({"check_status": ["PASS", "BLOCK"], "estimated_amount": [1900., 1800.]})
        result = cash_components(orders, checks, 1., 4099.)
        self.assertEqual(result["minimum_quantity_budget"], 2000.)
        self.assertEqual(result["rounding_residual"], 300.)
        self.assertEqual(result["blocked_order_notional"], 1800.)
        self.assertEqual(result["reconciliation_error"], 0.)

    def test_wrong_cash_is_not_silently_attributed(self):
        orders = pd.DataFrame({"target_shares": [100], "target_value": [2000.], "estimated_amount": [1900.]})
        checks = pd.DataFrame({"check_status": ["PASS"], "estimated_amount": [1900.]})
        with self.assertRaisesRegex(ValueError, "reconcile"):
            cash_components(orders, checks, 1., 100.)


if __name__ == "__main__":
    unittest.main()
