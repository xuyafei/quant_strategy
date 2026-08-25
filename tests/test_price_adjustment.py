"""复权价格口径测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from storage.price_adjustment import add_adjusted_close


class PriceAdjustmentTests(unittest.TestCase):
    def test_qfq_removes_false_ex_dividend_drop(self) -> None:
        prices = pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-05"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "close": [100.0, 90.0],
                "adj_factor": [0.9, 1.0],
            }
        )
        out = add_adjusted_close(prices, mode="qfq")
        self.assertEqual(list(out["adj_close"].round(6)), [90.0, 90.0])
        raw_return = out["close"].pct_change().iloc[-1]
        adj_return = out["adj_close"].pct_change().iloc[-1]
        self.assertAlmostEqual(float(raw_return), -0.10)
        self.assertAlmostEqual(float(adj_return), 0.0)

    def test_hfq_uses_direct_close_times_factor(self) -> None:
        prices = pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-05"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "close": [10.0, 12.0],
                "adj_factor": [2.0, 2.5],
            }
        )
        out = add_adjusted_close(prices, mode="hfq")
        self.assertEqual(list(out["adj_close"]), [20.0, 30.0])


if __name__ == "__main__":
    unittest.main()
