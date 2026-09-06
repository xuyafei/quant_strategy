"""QMT 快照验收测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.qmt_acceptance import audit_qmt_snapshot_history, validate_qmt_snapshot


def write_snapshot(root: Path, date_s: str) -> Path:
    target = root / date_s
    target.mkdir(parents=True)
    pd.DataFrame([{"cash": 8000, "market_value": 2000, "total_asset": 10000,
                   "updated_at": date_s + "T15:01:00"}]).to_csv(target / "account.csv", index=False)
    pd.DataFrame([{"symbol": "600000.SH", "shares": 100, "available_shares": 100,
                   "price": 20, "market_value": 2000, "updated_at": date_s + "T15:01:00"}]).to_csv(
        target / "positions.csv", index=False
    )
    pd.DataFrame(columns=["order_id"]).to_csv(target / "orders.csv", index=False)
    pd.DataFrame(columns=["trade_id"]).to_csv(target / "trades.csv", index=False)
    (target / "manifest.json").write_text(json.dumps({
        "read_only": True, "account_status": "OK", "query_warnings": []
    }), encoding="utf-8")
    return target


class QmtAcceptanceTests(unittest.TestCase):
    def test_snapshot_can_be_reconciled_with_ui_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = write_snapshot(root, "2026-09-05")
            ui_account = root / "ui_account.csv"
            ui_positions = root / "ui_positions.csv"
            pd.DataFrame([{"cash": 8000, "market_value": 2000, "total_asset": 10000}]).to_csv(
                ui_account, index=False
            )
            pd.DataFrame([{"symbol": "600000.SH", "shares": 100,
                           "available_shares": 100}]).to_csv(ui_positions, index=False)
            checks = validate_qmt_snapshot(snapshot, ui_account_path=ui_account,
                                           ui_positions_path=ui_positions)
            self.assertNotIn("BLOCK", set(checks["status"]))
            self.assertEqual(
                checks.loc[checks["check"] == "ui_positions_reconciliation", "status"].iloc[0], "PASS"
            )

    def test_history_requires_configured_number_of_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_snapshot(root, "2026-09-04")
            write_snapshot(root, "2026-09-05")
            _, summary = audit_qmt_snapshot_history(root, min_days=2)
            self.assertEqual(summary["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
