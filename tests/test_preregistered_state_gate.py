from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.build_preregistered_state_gate_backtest import build_state_monitor


class TestPreregisteredStateGate(unittest.TestCase):
    def test_state_inputs_exclude_rebalance_day_prices(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=130)
        symbols = ["TECH1", "TECH2", "BANK1", "BANK2"]
        base = np.linspace(100.0, 110.0, len(days))
        prices = pd.DataFrame({symbol: base.copy() for symbol in symbols}, index=days)
        shocked = prices.copy()
        shocked.loc[days[-1], ["TECH1", "TECH2"]] *= 3.0
        membership = pd.DataFrame(
            {
                "ts_code": symbols,
                "effective_from": [days[0]] * len(symbols),
                "effective_to": [days[-1]] * len(symbols),
            }
        )
        long_prices = pd.DataFrame(
            [
                {
                    "trade_date": date,
                    "ts_code": symbol,
                    "industry": "半导体" if symbol.startswith("TECH") else "银行",
                }
                for date in days
                for symbol in symbols
            ]
        )
        kwargs = {
            "long_prices": long_prices,
            "membership": membership,
            "rebalance_dates": pd.DatetimeIndex([days[-1]]),
            "tech_industries": {"半导体"},
        }
        normal = build_state_monitor(prices=prices, **kwargs).iloc[0]
        changed = build_state_monitor(prices=shocked, **kwargs).iloc[0]
        self.assertEqual(changed["history_end"], days[-2])
        for column in [
            "tech_relative_return_20d",
            "tech_relative_return_60d",
            "tech_breadth_above_60d_ma",
            "tech_volatility_ratio_20d_120d",
        ]:
            left, right = normal[column], changed[column]
            if pd.isna(left):
                self.assertTrue(pd.isna(right))
            else:
                self.assertAlmostEqual(float(left), float(right))


if __name__ == "__main__":
    unittest.main()
