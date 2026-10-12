"""
订单预检查：在纸面交易或真实下单前，检查订单计划是否具备基础可执行性。

本模块只做规则检查，不修改订单、不模拟成交、不连接券商。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from live.risk_blacklist import risk_blacklist_map


CHECK_COLUMNS = [
    "date",
    "symbol",
    "side",
    "delta_shares",
    "price",
    "estimated_amount",
    "check_status",
    "check_reason",
    "cash_before",
    "cash_after",
    "available_shares",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "is_blacklisted",
    "blacklist_severity",
    "blacklist_reason",
]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _available_shares(current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None) -> pd.Series:
    if current_positions is None:
        return pd.Series(dtype=float)
    if isinstance(current_positions, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in current_positions.columns else "ts_code"
        if symbol_col not in current_positions.columns:
            raise ValueError("current_positions DataFrame 须包含 symbol 或 ts_code 列")
        share_col = "available_shares" if "available_shares" in current_positions.columns else "shares"
        if share_col not in current_positions.columns:
            raise ValueError("current_positions DataFrame 须包含 shares 或 available_shares 列")
        out = pd.Series(
            current_positions[share_col].astype(float).to_numpy(),
            index=current_positions[symbol_col].astype(str),
            dtype=float,
        )
    elif isinstance(current_positions, pd.Series):
        out = current_positions.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in current_positions.items()}, dtype=float)
    return out.groupby(level=0).sum().sort_index()


def _status_map(trade_status: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, bool]]:
    if trade_status is None:
        return {}
    if isinstance(trade_status, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in trade_status.columns else "ts_code"
        if symbol_col not in trade_status.columns:
            raise ValueError("trade_status DataFrame 须包含 symbol 或 ts_code 列")
        out: dict[str, dict[str, bool]] = {}
        for rec in trade_status.to_dict("records"):
            symbol = str(rec.get(symbol_col, ""))
            if not symbol:
                continue
            out[symbol] = {
                "is_suspended": _to_bool(rec.get("is_suspended", False)),
                "is_limit_up": _to_bool(rec.get("is_limit_up", False)),
                "is_limit_down": _to_bool(rec.get("is_limit_down", False)),
            }
        return out
    return {
        str(symbol): {
            "is_suspended": _to_bool(flags.get("is_suspended", False)),
            "is_limit_up": _to_bool(flags.get("is_limit_up", False)),
            "is_limit_down": _to_bool(flags.get("is_limit_down", False)),
        }
        for symbol, flags in trade_status.items()
    }


def _order_rank(side: str) -> int:
    if side == "SELL":
        return 0
    if side == "BUY":
        return 1
    return 2


def precheck_order_plan(
    order_plan: pd.DataFrame,
    *,
    cash: float,
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    trade_status: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
    risk_blacklist: pd.DataFrame | Mapping[str, Any] | Iterable[str] | None = None,
    lot_size: int = 100,
    min_order_amount: float = 0.0,
    cash_buffer: float = 0.0,
    block_blacklisted: bool = True,
    board_rules: bool = False,
    commission_rate: float = 0.0,
) -> pd.DataFrame:
    """
    检查订单计划是否满足基础执行约束。

    :param order_plan: `live.order_builder.build_order_plan` 输出的订单计划
    :param cash: 当前可用现金
    :param current_positions: 当前持仓；若含 `available_shares` 列则卖出检查优先使用它
    :param trade_status: 停牌 / 涨跌停状态，含 `is_suspended/is_limit_up/is_limit_down`
    :param risk_blacklist: 风险黑名单，含 `symbol/ts_code` 与可选 `reason/severity`
    :param lot_size: 买入订单应满足的最小交易单位
    :param min_order_amount: 小于该金额的 BUY/SELL 标记为 BLOCK
    :param cash_buffer: 买入后至少保留的现金缓冲
    :param block_blacklisted: 是否直接阻断命中黑名单的 BUY/SELL 订单
    :return: 每笔订单的检查结果，列见 `CHECK_COLUMNS`
    """
    if lot_size <= 0:
        raise ValueError("lot_size 必须为正整数")
    if cash < 0:
        raise ValueError("cash 不能为负")
    if cash_buffer < 0:
        raise ValueError("cash_buffer 不能为负")
    if min_order_amount < 0:
        raise ValueError("min_order_amount 不能为负")

    if order_plan.empty:
        return pd.DataFrame(columns=CHECK_COLUMNS)

    required = {"symbol", "side", "delta_shares", "price", "estimated_amount"}
    missing = required - set(order_plan.columns)
    if missing:
        raise ValueError("order_plan 缺少必要列: %s" % ", ".join(sorted(missing)))

    available = _available_shares(current_positions)
    status_by_symbol = _status_map(trade_status)
    blacklist_by_symbol = risk_blacklist_map(
        risk_blacklist,
        trade_date=order_plan["date"].max() if "date" in order_plan.columns else None,
    )
    orders = order_plan.copy()
    orders["side"] = orders["side"].astype(str).str.upper()
    orders["_order_rank"] = orders["side"].map(_order_rank)
    orders = orders.sort_values(["_order_rank", "symbol"]).reset_index(drop=True)

    cash_now = float(cash)
    rows: list[dict[str, Any]] = []
    for rec in orders.to_dict("records"):
        symbol = str(rec.get("symbol", ""))
        side = str(rec.get("side", "")).upper()
        delta = int(round(float(rec.get("delta_shares", 0))))
        price = float(rec.get("price", float("nan")))
        amount = float(rec.get("estimated_amount", abs(delta) * price))
        cash_before = cash_now
        flags = status_by_symbol.get(symbol, {})
        suspended = bool(flags.get("is_suspended", False))
        limit_up = bool(flags.get("is_limit_up", False))
        limit_down = bool(flags.get("is_limit_down", False))
        available_shares = int(round(float(available.get(symbol, 0.0))))
        blacklist_info = blacklist_by_symbol.get(symbol, {})
        is_blacklisted = bool(blacklist_info)
        blacklist_reason = str(blacklist_info.get("reason", ""))
        blacklist_severity = str(blacklist_info.get("severity", ""))

        reasons: list[str] = []
        if side not in {"BUY", "SELL", "HOLD"}:
            reasons.append("invalid_side")
        if not math.isfinite(price) or price <= 0.0:
            reasons.append("invalid_price")
        if side == "BUY" and delta <= 0:
            reasons.append("invalid_buy_delta")
        if side == "SELL" and delta >= 0:
            reasons.append("invalid_sell_delta")
        if side in {"BUY", "SELL"} and amount <= 0.0:
            reasons.append("invalid_amount")
        if side in {"BUY", "SELL"} and min_order_amount > 0 and amount < min_order_amount:
            reasons.append("below_min_order_amount")
        if board_rules and side in {"BUY", "SELL"}:
            from live.board_rules import quantity_error
            error = quantity_error(symbol, side, abs(delta), available_shares)
            if error:
                reasons.append(error)
        elif side == "BUY" and abs(delta) % int(lot_size) != 0:
            reasons.append("not_lot_size")
        if suspended and side in {"BUY", "SELL"}:
            reasons.append("suspended")
        if side == "BUY" and limit_up:
            reasons.append("limit_up_blocks_buy")
        if side == "SELL" and limit_down:
            reasons.append("limit_down_blocks_sell")
        if side == "SELL" and abs(delta) > available_shares:
            reasons.append("insufficient_available_shares")
        if side == "BUY" and cash_now - amount * (1 + commission_rate) < cash_buffer:
            reasons.append("insufficient_cash")
        if block_blacklisted and is_blacklisted and side in {"BUY", "SELL"}:
            reasons.append("risk_blacklist")

        status = "PASS" if not reasons else "BLOCK"
        if status == "PASS":
            if side == "SELL":
                cash_now += amount * (1 - commission_rate)
            elif side == "BUY":
                cash_now -= amount * (1 + commission_rate)

        rows.append(
            {
                "date": rec.get("date", ""),
                "symbol": symbol,
                "side": side,
                "delta_shares": delta,
                "price": price,
                "estimated_amount": amount,
                "check_status": status,
                "check_reason": "pass" if not reasons else "|".join(reasons),
                "cash_before": cash_before,
                "cash_after": cash_now,
                "available_shares": available_shares,
                "is_suspended": suspended,
                "is_limit_up": limit_up,
                "is_limit_down": limit_down,
                "is_blacklisted": is_blacklisted,
                "blacklist_severity": blacklist_severity,
                "blacklist_reason": blacklist_reason,
            }
        )

    return pd.DataFrame(rows, columns=CHECK_COLUMNS)
