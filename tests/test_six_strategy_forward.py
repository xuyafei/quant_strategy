import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from live.six_strategy_forward import refresh_runtime_market_cache
from scripts.fetch_six_strategy_trade_status import run as build_trade_status


class SixStrategyForwardTest(unittest.TestCase):
    def test_runtime_cache_requires_exact_complete_date_and_carries_suspension_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "data"
            output = base / "output"
            data.mkdir()
            pd.DataFrame(
                [
                    {"trade_date": "2026-09-21", "ts_code": "000001.SZ", "close": 10.0, "amount_yuan": 1_000_000.0},
                    {"trade_date": "2026-09-21", "ts_code": "000002.SZ", "close": 20.0, "amount_yuan": 2_000_000.0},
                    {"trade_date": "2026-09-22", "ts_code": "000001.SZ", "close": 10.5, "amount_yuan": 1_200_000.0},
                ]
            ).to_csv(data / "prices_all_a_qfq.csv.gz", index=False, compression="gzip")

            paths = refresh_runtime_market_cache(
                data,
                output,
                through_date="2026-09-22",
            )
            wide = pd.read_csv(paths["prices_wide"])
            self.assertEqual(wide["date"].iloc[-1], "2026-09-22")
            self.assertEqual(float(wide["000002.SZ"].iloc[-1]), 20.0)
            long = pd.read_csv(paths["prices_long"])
            self.assertEqual(set(long.columns), {"date", "symbol", "close", "amount"})

            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                refresh_runtime_market_cache(
                    data,
                    output,
                    through_date="2026-09-23",
                )
            execution_prices = data / "execution_prices.csv"
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-09-23",
                        "ts_code": "000001.SZ",
                        "close": 10.8,
                        "amount_yuan": 1_300_000.0,
                    }
                ]
            ).to_csv(execution_prices, index=False)
            refreshed = refresh_runtime_market_cache(
                data,
                output,
                through_date="2026-09-23",
                execution_prices_path=execution_prices,
            )
            exact = pd.read_csv(refreshed["prices_wide"])
            self.assertEqual(exact["date"].iloc[-1], "2026-09-23")
            self.assertEqual(float(exact["000001.SZ"].iloc[-1]), 10.8)

    def test_trade_status_uses_daily_presence_and_exact_price_limits(self) -> None:
        class FakePro:
            def daily(
                self,
                *,
                ts_code: str,
                start_date: str,
                end_date: str,
            ) -> pd.DataFrame:
                if ts_code != "000001.SZ":
                    return pd.DataFrame()
                return pd.DataFrame(
                    [
                        {
                            "ts_code": ts_code,
                            "trade_date": start_date,
                            "close": 11.0,
                            "amount": 1000.0,
                        }
                    ]
                )

            def stk_limit(self, *, trade_date: str) -> pd.DataFrame:
                self.trade_date = trade_date
                return pd.DataFrame(
                    [
                        {"ts_code": "000001.SZ", "up_limit": 11.0, "down_limit": 9.0},
                        {"ts_code": "000002.SZ", "up_limit": 22.0, "down_limit": 18.0},
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "data"
            output = base / "output"
            data.mkdir(parents=True)
            (output / "rebalance_logs").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2026-09-24", "symbol": "000001.SZ", "weight": 0.5},
                    {"date": "2026-09-24", "symbol": "000002.SZ", "weight": 0.5},
                ]
            ).to_csv(output / "rebalance_logs" / "TEST.csv", index=False)

            with patch(
                "scripts.fetch_six_strategy_trade_status.get_tushare_token",
                return_value="test-token",
            ), patch(
                "tushare.pro_api", return_value=FakePro()
            ):
                paths = build_trade_status(
                    data,
                    output,
                    "2026-09-24",
                    workers=1,
                )
            status = pd.read_csv(paths["trade_status"])
            first = status.set_index("symbol").loc["000001.SZ"]
            second = status.set_index("symbol").loc["000002.SZ"]
            self.assertTrue(bool(first["is_limit_up"]))
            self.assertFalse(bool(first["is_suspended"]))
            self.assertTrue(bool(second["is_suspended"]))
            prices = pd.read_csv(paths["execution_prices"])
            self.assertEqual(list(prices["ts_code"]), ["000001.SZ"])
            self.assertEqual(float(prices["amount_yuan"].iloc[0]), 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
