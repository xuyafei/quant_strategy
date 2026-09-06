"""QMT 只读接入自检。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from live.broker import BrokerReadOnlyError, RealBrokerConfig
from live.qmt_readonly import QmtReadOnlyAdapter, save_qmt_readonly_snapshot


class FakeTrader:
    def query_account_status(self):
        return [SimpleNamespace(account_id="demo", status=0)]

    def query_stock_asset(self, account):
        return SimpleNamespace(cash=8000, market_value=2000, total_asset=10000)

    def query_stock_positions(self, account):
        return [SimpleNamespace(stock_code="600000.SH", volume=100, can_use_volume=80,
                                market_price=20, market_value=2000)]

    def query_stock_orders(self, account, cancelable_only):
        return [SimpleNamespace(order_id=1, order_time="2026-09-05 10:00:00",
                                stock_code="600000.SH", order_type=23, order_volume=100,
                                price=20, order_status=56, status_msg="已成",
                                traded_volume=100, traded_price=20)]

    def query_stock_trades(self, account):
        return [SimpleNamespace(traded_id=2, order_id=1, traded_time="2026-09-05 10:00:01",
                                stock_code="600000.SH", order_type=23,
                                traded_volume=100, traded_price=20)]


class QmtReadOnlyTests(unittest.TestCase):
    def test_sync_maps_all_readonly_snapshots_and_blocks_trade(self) -> None:
        adapter = QmtReadOnlyAdapter(
            RealBrokerConfig(provider="qmt", account_id="demo"),
            trader=FakeTrader(), account_ref=object(),
        )
        adapter.sync()
        self.assertEqual(adapter.get_account().total_asset, 10000)
        self.assertEqual(adapter.get_positions().iloc[0]["available_shares"], 80)
        self.assertEqual(adapter.get_orders().iloc[0]["side"], "BUY")
        self.assertEqual(adapter.get_trades().iloc[0]["amount"], 2000)
        with self.assertRaises(BrokerReadOnlyError):
            adapter.submit_order(symbol="600000.SH", side="BUY", qty=100, price=20)

    def test_snapshot_is_persisted_with_readonly_manifest(self) -> None:
        adapter = QmtReadOnlyAdapter(
            RealBrokerConfig(provider="qmt", account_id="demo"),
            trader=FakeTrader(), account_ref=object(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_qmt_readonly_snapshot(adapter, Path(tmp), trade_date="2026-09-05")
            self.assertTrue(all(path.exists() for path in paths.values()))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["read_only"])
            self.assertEqual(manifest["trade_count"], 1)

    def test_none_list_result_is_recorded_as_ambiguous_warning(self) -> None:
        trader = FakeTrader()
        trader.query_stock_positions = lambda account: None
        adapter = QmtReadOnlyAdapter(
            RealBrokerConfig(provider="qmt", account_id="demo"),
            trader=trader, account_ref=object(),
        )
        adapter.sync()
        self.assertTrue(adapter.get_positions().empty)
        self.assertIn("positions_returned_none: empty_or_query_failed", adapter.query_warnings)


if __name__ == "__main__":
    unittest.main()
