"""
目标持仓到订单计划：把策略输出的目标权重转换为买卖股数。

本模块只生成订单计划，不连接券商、不判断涨跌停/停牌，也不模拟成交。
这些检查会在后续的订单预检查与纸面交易模块继续补齐。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd


ORDER_COLUMNS = [
    "date",
    "symbol",
    "side",
    "current_shares",
    "target_shares",
    "delta_shares",
    "price",
    "estimated_amount",
    "current_value",
    "target_value",
    "current_weight",
    "target_weight",
    "trade_reason",
]


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _target_weights_to_series(target_weights: Mapping[str, float] | pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(target_weights, pd.DataFrame):
        if "symbol" not in target_weights.columns or "target_weight" not in target_weights.columns:
            raise ValueError("target_weights DataFrame 须包含 symbol 与 target_weight 列")
        out = pd.Series(
            target_weights["target_weight"].astype(float).to_numpy(),
            index=target_weights["symbol"].astype(str),
            dtype=float,
        )
    elif isinstance(target_weights, pd.Series):
        out = target_weights.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in target_weights.items()}, dtype=float)

    out = out.groupby(level=0).sum().sort_index()
    out = out[out.abs() > 1e-12]
    if (out < -1e-12).any():
        bad = ", ".join(out[out < -1e-12].index[:5])
        raise ValueError("目标权重不能为负: %s" % bad)
    if float(out.sum()) > 1.0 + 1e-8:
        raise ValueError("目标权重之和不能超过 1.0；当前 %.6f" % float(out.sum()))
    return out.clip(lower=0.0)


def _positions_to_series(current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None) -> pd.Series:
    if current_positions is None:
        return pd.Series(dtype=float)
    if isinstance(current_positions, pd.DataFrame):
        if "symbol" not in current_positions.columns or "shares" not in current_positions.columns:
            raise ValueError("current_positions DataFrame 须包含 symbol 与 shares 列")
        out = pd.Series(
            current_positions["shares"].astype(float).to_numpy(),
            index=current_positions["symbol"].astype(str),
            dtype=float,
        )
    elif isinstance(current_positions, pd.Series):
        out = current_positions.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in current_positions.items()}, dtype=float)
    return out.groupby(level=0).sum().sort_index()


def _prices_to_series(latest_prices: Mapping[str, float] | pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(latest_prices, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in latest_prices.columns else "ts_code"
        price_col = "price"
        if price_col not in latest_prices.columns:
            for candidate in ("close", "last_price", "latest_price"):
                if candidate in latest_prices.columns:
                    price_col = candidate
                    break
        if symbol_col not in latest_prices.columns or price_col not in latest_prices.columns:
            raise ValueError("latest_prices DataFrame 须包含 symbol/ts_code 与 price/close 列")
        out = pd.Series(
            latest_prices[price_col].astype(float).to_numpy(),
            index=latest_prices[symbol_col].astype(str),
            dtype=float,
        )
    elif isinstance(latest_prices, pd.Series):
        out = latest_prices.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in latest_prices.items()}, dtype=float)
    return out.groupby(level=0).last().sort_index()


def _round_target_shares(target_value: float, price: float, lot_size: int) -> int:
    if target_value <= 0.0:
        return 0
    if lot_size <= 1:
        return int(math.floor(target_value / price))
    return int(math.floor(target_value / price / lot_size) * lot_size)


def build_order_plan(
    target_weights: Mapping[str, float] | pd.Series | pd.DataFrame,
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
    latest_prices: Mapping[str, float] | pd.Series | pd.DataFrame,
    *,
    cash: float = 0.0,
    total_asset: float | None = None,
    trade_date: Any = None,
    lot_size: int = 100,
    min_order_amount: float = 0.0,
    include_holds: bool = False,
    board_rules: bool = False,
) -> pd.DataFrame:
    """
    将目标权重转换成订单计划。

    :param target_weights: `{symbol: weight}`、Series，或含 `symbol,target_weight` 的 DataFrame
    :param current_positions: 当前持仓，含 `symbol,shares`，也可传映射；空表示全现金
    :param latest_prices: 最新价格，映射 / Series / 含 `symbol,price` 或 `ts_code,close` 的 DataFrame
    :param cash: 当前现金；`total_asset` 为空时用于估算账户总资产
    :param total_asset: 账户总资产；为空则按 `cash + 当前持仓市值` 估算
    :param trade_date: 订单日期
    :param lot_size: 买入目标按 lot_size 向下取整，A 股通常为 100
    :param min_order_amount: 小于该成交金额的订单会被过滤
    :param include_holds: True 时保留 HOLD 行，便于审计；False 时只输出 BUY/SELL
    :return: DataFrame，列见 `ORDER_COLUMNS`
    """
    if lot_size <= 0:
        raise ValueError("lot_size 必须为正整数")
    if min_order_amount < 0:
        raise ValueError("min_order_amount 不能为负")

    target = _target_weights_to_series(target_weights)
    current = _positions_to_series(current_positions)
    prices = _prices_to_series(latest_prices)

    symbols = sorted(set(target.index) | set(current.index))
    if not symbols:
        return pd.DataFrame(columns=ORDER_COLUMNS)

    missing = [s for s in symbols if s not in prices.index or not math.isfinite(float(prices.loc[s])) or float(prices.loc[s]) <= 0.0]
    if missing:
        raise ValueError("缺少有效价格，无法生成订单: %s" % ", ".join(missing[:10]))

    current_values = current.reindex(symbols).fillna(0.0) * prices.reindex(symbols)
    account_value = float(total_asset) if total_asset is not None else float(cash) + float(current_values.sum())
    if account_value <= 0.0:
        raise ValueError("账户总资产必须大于 0")

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        price = float(prices.loc[symbol])
        current_shares = int(round(float(current.get(symbol, 0.0))))
        target_weight = float(target.get(symbol, 0.0))
        current_value = current_shares * price
        target_value = account_value * target_weight
        target_shares = _round_target_shares(target_value, price, lot_size)
        if board_rules:
            from live.board_rules import target_shares as board_target_shares
            target_shares = board_target_shares(symbol, target_value, price)
        delta_shares = target_shares - current_shares
        estimated_amount = abs(delta_shares) * price

        if delta_shares > 0:
            side = "BUY"
            reason = "increase_to_target_weight"
        elif delta_shares < 0:
            side = "SELL"
            reason = "reduce_to_target_weight"
        else:
            side = "HOLD"
            reason = "already_at_target_lot"

        if side != "HOLD" and estimated_amount < float(min_order_amount):
            side = "HOLD"
            reason = "below_min_order_amount"
            target_shares = current_shares
            delta_shares = 0
            estimated_amount = 0.0

        if side == "HOLD" and target_weight > 0 and target_shares == 0:
            reason = "target_value_below_lot"

        if side != "HOLD" or include_holds:
            rows.append(
                {
                    "date": _date_to_str(trade_date),
                    "symbol": symbol,
                    "side": side,
                    "current_shares": current_shares,
                    "target_shares": int(target_shares),
                    "delta_shares": int(delta_shares),
                    "price": price,
                    "estimated_amount": estimated_amount,
                    "current_value": current_value,
                    "target_value": target_value,
                    "current_weight": current_value / account_value,
                    "target_weight": target_weight,
                    "trade_reason": reason,
                }
            )

    order_rank = {"SELL": 0, "BUY": 1, "HOLD": 2}
    df = pd.DataFrame(rows, columns=ORDER_COLUMNS)
    if df.empty:
        return df
    df["_order_rank"] = df["side"].map(order_rank).fillna(9)
    df = df.sort_values(["_order_rank", "symbol"]).drop(columns=["_order_rank"]).reset_index(drop=True)
    return df


def latest_target_weights_from_rebalance_meta(meta: Mapping[str, Any]) -> tuple[pd.Timestamp | None, pd.Series]:
    """从回测 meta 的最近一期 rebalance_log 提取 `{symbol: target_weight}`。"""
    log = list(meta.get("rebalance_log") or [])
    if not log:
        return None, pd.Series(dtype=float)
    rec = log[-1]
    picks = list(rec.get("picks") or [])
    weights = list(rec.get("weights") or [])
    target = {
        str(symbol): float(weights[i])
        for i, symbol in enumerate(picks)
        if i < len(weights) and float(weights[i]) > 0.0
    }
    dt = rec.get("date")
    return (pd.Timestamp(dt) if dt not in (None, "") else None), pd.Series(target, dtype=float)


def build_order_plan_from_rebalance_meta(
    meta: Mapping[str, Any],
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
    latest_prices: Mapping[str, float] | pd.Series | pd.DataFrame,
    *,
    cash: float = 0.0,
    total_asset: float | None = None,
    lot_size: int = 100,
    min_order_amount: float = 0.0,
    include_holds: bool = False,
) -> pd.DataFrame:
    """从最近一期回测目标持仓直接生成订单计划。"""
    trade_date, target = latest_target_weights_from_rebalance_meta(meta)
    return build_order_plan(
        target,
        current_positions,
        latest_prices,
        cash=cash,
        total_asset=total_asset,
        trade_date=trade_date,
        lot_size=lot_size,
        min_order_amount=min_order_amount,
        include_holds=include_holds,
    )
