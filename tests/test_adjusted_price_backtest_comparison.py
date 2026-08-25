"""复权价格回测对比脚本测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_adjusted_price_backtest_comparison import run_monthly_momentum_topk


class AdjustedPriceBacktestComparisonTests(unittest.TestCase):
    def test_momentum_selection_can_change_after_adjustment(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=45)
        a_trend = [100.0 + i * 0.5 for i in range(45)]
        raw = pd.DataFrame(
            {
                "A": a_trend[:25] + [x * 0.7 for x in a_trend[25:]],
                "B": [100.0 + i * 0.1 for i in range(45)],
            },
            index=dates,
        )
        adjusted = raw.copy()
        adjusted["A"] = a_trend
        _, raw_log = run_monthly_momentum_topk(raw, top_k=1, lookback=20)
        _, adj_log = run_monthly_momentum_topk(adjusted, top_k=1, lookback=20)
        self.assertNotEqual(
            raw_log.iloc[-1]["selected_symbols"],
            adj_log.iloc[-1]["selected_symbols"],
        )


if __name__ == "__main__":
    unittest.main()
