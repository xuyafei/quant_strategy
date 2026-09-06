"""MiniQMT/QMT 真实账户只读适配器。"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd

from live.broker import (
    ORDER_STATUS_CANCELLED,
    ORDER_STATUS_FILLED,
    ORDER_STATUS_NEW,
    ORDER_STATUS_PARTIALLY_FILLED,
    ORDER_STATUS_REJECTED,
    BrokerAccount,
    RealBrokerConfig,
    RealBrokerReadOnlyAdapter,
)


TRADE_COLUMNS = [
    "trade_id", "order_id", "date", "symbol", "side", "qty", "price",
    "amount", "traded_at",
]
POSITION_COLUMNS = ["symbol", "shares", "available_shares", "price", "market_value", "updated_at"]

QMT_ORDER_STATUS = {
    48: ORDER_STATUS_NEW,  # 未报
    49: ORDER_STATUS_NEW,  # 待报
    50: ORDER_STATUS_NEW,  # 已报
    51: ORDER_STATUS_NEW,  # 已报待撤
    52: ORDER_STATUS_PARTIALLY_FILLED,  # 部成待撤
    53: ORDER_STATUS_PARTIALLY_FILLED,  # 部撤
    54: ORDER_STATUS_CANCELLED,
    55: ORDER_STATUS_PARTIALLY_FILLED,
    56: ORDER_STATUS_FILLED,
    57: ORDER_STATUS_REJECTED,
    255: "UNKNOWN",
}
QMT_ACCOUNT_STATUS = {
    -1: "INVALID", 0: "OK", 1: "WAITING_LOGIN", 2: "LOGGING_IN",
    3: "FAILED", 4: "INITIALIZING", 5: "CORRECTING", 6: "CLOSED",
    7: "ASSISTANT_LINK_FAILED", 8: "DISABLED_BY_SYSTEM", 9: "DISABLED_BY_USER",
}


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _side(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"23", "BUY", "买入"}:
        return "BUY"
    if text in {"24", "SELL", "卖出"}:
        return "SELL"
    return text or "UNKNOWN"


def _time_text(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
        except (ValueError, OSError, OverflowError):
            pass
    return str(value)


def _order_status(value: Any) -> str:
    try:
        return QMT_ORDER_STATUS.get(int(value), "UNKNOWN")
    except (TypeError, ValueError):
        text = str(value).strip().upper()
        return text or "UNKNOWN"


def _public_fields(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return sorted(str(key) for key in obj)
    return sorted(name for name in dir(obj) if not name.startswith("_") and not callable(getattr(obj, name, None)))


class QmtReadOnlyAdapter(RealBrokerReadOnlyAdapter):
    """把 QMT 查询结果映射成项目统一结构；不暴露任何交易能力。"""

    def __init__(
        self,
        config: RealBrokerConfig,
        *,
        trader: Any,
        account_ref: Any,
        connection_info: dict[str, Any] | None = None,
    ) -> None:
        if config.provider != "qmt":
            raise ValueError("QmtReadOnlyAdapter 仅支持 provider=qmt")
        super().__init__(config)
        self.trader = trader
        self.account_ref = account_ref
        self._trades = pd.DataFrame(columns=TRADE_COLUMNS)
        self.query_warnings: list[str] = []
        self.account_status: str = "UNKNOWN"
        self.field_audit: dict[str, list[str]] = {}
        self.connection_info = dict(connection_info or {})

    def sync(self) -> None:
        asset = self.trader.query_stock_asset(self.account_ref)
        if asset is None:
            raise RuntimeError("QMT 未返回账户资产，请检查连接、账号和订阅状态")
        self.query_warnings = []
        statuses = self.trader.query_account_status() if hasattr(self.trader, "query_account_status") else []
        account_id = str(_value(self.account_ref, "account_id", default=self.config.account_id))
        for status_item in statuses or []:
            if str(_value(status_item, "account_id", default="")) == account_id:
                status_code = int(_value(status_item, "status", default=-999))
                self.account_status = QMT_ACCOUNT_STATUS.get(status_code, "UNKNOWN_%s" % status_code)
                break
        if self.account_status not in {"OK", "CLOSED", "UNKNOWN"}:
            raise RuntimeError("QMT 账户状态不可用于只读同步: %s" % self.account_status)

        positions_raw = self.trader.query_stock_positions(self.account_ref)
        orders_raw = self.trader.query_stock_orders(self.account_ref, False)
        trades_raw = self.trader.query_stock_trades(self.account_ref)
        # 官方文档说明这三类查询的 None 同时可能表示失败或列表为空。
        # 资产查询成功且账号状态正常时按空表保存，同时在 manifest 留下歧义警告。
        for name, value in (("positions", positions_raw), ("orders", orders_raw), ("trades", trades_raw)):
            if value is None:
                self.query_warnings.append("%s_returned_none: empty_or_query_failed" % name)
        positions = positions_raw or []
        orders = orders_raw or []
        trades = trades_raw or []
        now = datetime.now().isoformat(timespec="seconds")

        self.field_audit = {"asset": _public_fields(asset)}
        for name, values in (("position", positions), ("order", orders), ("trade", trades)):
            self.field_audit[name] = _public_fields(values[0]) if values else []

        account = BrokerAccount(
            cash=float(_value(asset, "cash", default=0.0) or 0.0),
            market_value=float(_value(asset, "market_value", default=0.0) or 0.0),
            total_asset=float(_value(asset, "total_asset", default=0.0) or 0.0),
            updated_at=now,
        )
        position_rows = []
        for item in positions:
            shares = int(_value(item, "volume", "shares", default=0) or 0)
            market_value = float(_value(item, "market_value", default=0.0) or 0.0)
            implied_price = market_value / shares if shares > 0 else 0.0
            position_rows.append({
                "symbol": str(_value(item, "stock_code", "symbol", default="")),
                "shares": shares,
                "available_shares": int(_value(item, "can_use_volume", "available_shares", default=0) or 0),
                "price": implied_price,
                "market_value": market_value,
                "updated_at": now,
            })
        order_rows = []
        for item in orders:
            submitted = _time_text(_value(item, "order_time", "submitted_at", default=""))
            order_rows.append({
                "order_id": str(_value(item, "order_id", default="")),
                "date": submitted[:10],
                "symbol": str(_value(item, "stock_code", "symbol", default="")),
                "side": _side(_value(item, "order_type", "side", default="")),
                "qty": int(_value(item, "order_volume", "qty", default=0) or 0),
                "price": float(_value(item, "price", default=0.0) or 0.0),
                "status": _order_status(_value(item, "order_status", "status", default="UNKNOWN")),
                "reason": str(_value(item, "status_msg", "reason", default="")),
                "filled_qty": int(_value(item, "traded_volume", "filled_qty", default=0) or 0),
                "avg_price": float(_value(item, "traded_price", "avg_price", default=0.0) or 0.0),
                "gross_amount": 0.0,
                "commission": 0.0,
                "cash_after": 0.0,
                "position_after": 0,
                "submitted_at": submitted,
            })
        self.update_snapshot(
            account=account,
            positions=pd.DataFrame(position_rows, columns=POSITION_COLUMNS),
            orders=pd.DataFrame(order_rows),
        )

        trade_rows = []
        for item in trades:
            traded_at = _time_text(_value(item, "traded_time", "trade_time", default=""))
            qty = int(_value(item, "traded_volume", "qty", default=0) or 0)
            price = float(_value(item, "traded_price", "price", default=0.0) or 0.0)
            trade_rows.append({
                "trade_id": str(_value(item, "traded_id", "trade_id", default="")),
                "order_id": str(_value(item, "order_id", default="")),
                "date": traded_at[:10],
                "symbol": str(_value(item, "stock_code", "symbol", default="")),
                "side": _side(_value(item, "order_type", "side", default="")),
                "qty": qty,
                "price": price,
                "amount": float(_value(item, "traded_amount", "amount", default=qty * price) or 0.0),
                "traded_at": traded_at,
            })
        self._trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)

    def get_trades(self) -> pd.DataFrame:
        return self._trades.copy()


def save_qmt_readonly_snapshot(
    adapter: QmtReadOnlyAdapter,
    output_dir: Path | str,
    *,
    trade_date: Any = None,
) -> dict[str, Path]:
    """同步并保存只读快照；不会调用任何委托或撤单接口。"""
    adapter.sync()
    date_s = str(trade_date or datetime.now().date())[:10]
    safe_account = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in adapter.config.account_id
    ).strip("_") or "account"
    target = Path(output_dir) / safe_account / date_s
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "account": target / "account.csv",
        "positions": target / "positions.csv",
        "orders": target / "orders.csv",
        "trades": target / "trades.csv",
        "manifest": target / "manifest.json",
    }
    pd.DataFrame([asdict(adapter.get_account())]).to_csv(paths["account"], index=False)
    adapter.get_positions().to_csv(paths["positions"], index=False)
    adapter.get_orders().to_csv(paths["orders"], index=False)
    adapter.get_trades().to_csv(paths["trades"], index=False)
    manifest = {
        "provider": adapter.config.provider,
        "account_id": adapter.config.account_id,
        "mode": adapter.config.mode,
        "trade_date": date_s,
        "synced_at": datetime.now().isoformat(timespec="seconds"),
        "read_only": True,
        "account_status": adapter.account_status,
        "query_warnings": adapter.query_warnings,
        "connection": adapter.connection_info,
        "position_count": len(adapter.get_positions()),
        "order_count": len(adapter.get_orders()),
        "trade_count": len(adapter.get_trades()),
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    field_audit_path = target / "field_audit.json"
    field_audit_path.write_text(
        json.dumps(adapter.field_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["field_audit"] = field_audit_path
    return paths
