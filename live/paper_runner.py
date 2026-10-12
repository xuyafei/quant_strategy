"""
每日纸面交易运行器：串联账户状态、订单计划、预检查、纸面成交与状态保存。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from live.account_state import (
    load_account_state,
    positions_from_trades,
    save_account_state,
)
from live.broker import ORDER_STATUS_FILLED, SimulatedBroker
from live.cache_io import save_order_checks, save_order_plans, save_paper_trades
from live.order_builder import build_order_plan
from live.order_precheck import precheck_order_plan
from live.paper_trading import PAPER_TRADE_COLUMNS, paper_account_snapshot, run_paper_trading


EXECUTION_MODE_PAPER_TRADING = "paper_trading"
EXECUTION_MODE_SIMULATED_BROKER = "simulated_broker"


def _positions_to_series(
    positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
) -> pd.Series:
    if positions is None:
        return pd.Series(dtype=float)
    if isinstance(positions, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in positions.columns else "ts_code"
        if symbol_col not in positions.columns or "shares" not in positions.columns:
            raise ValueError("positions DataFrame 须包含 symbol/ts_code 与 shares 列")
        out = pd.Series(
            positions["shares"].astype(float).to_numpy(),
            index=positions[symbol_col].astype(str),
            dtype=float,
        )
    elif isinstance(positions, pd.Series):
        out = positions.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in positions.items()}, dtype=float)
    return out.groupby(level=0).sum().sort_index()


def _broker_orders_to_paper_trades(
    broker_orders: pd.DataFrame,
    *,
    initial_cash: float,
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
) -> pd.DataFrame:
    """把统一券商订单回报转成旧版纸面成交表，保持日报和异常检查兼容。"""
    if broker_orders.empty:
        return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)

    positions = _positions_to_series(current_positions)
    cash_now = float(initial_cash)
    rows: list[dict[str, Any]] = []
    for rec in broker_orders.to_dict("records"):
        symbol = str(rec.get("symbol", ""))
        side = str(rec.get("side", "")).upper()
        qty = int(round(float(rec.get("qty", 0))))
        status = str(rec.get("status", "")).upper()
        price = float(rec.get("avg_price", 0.0) or rec.get("price", 0.0) or 0.0)
        gross = float(rec.get("gross_amount", 0.0) or 0.0)
        commission = float(rec.get("commission", 0.0) or 0.0)
        cash_before = cash_now
        pos_before = int(round(float(positions.get(symbol, 0.0))))
        fill_status = "SKIPPED"
        fill_reason = str(rec.get("reason", "") or "rejected_by_broker")
        net_cash_flow = 0.0
        pos_after = pos_before

        if status == ORDER_STATUS_FILLED:
            fill_status = "FILLED"
            fill_reason = str(rec.get("reason", "") or "filled")
            if side == "BUY":
                net_cash_flow = -(gross + commission)
            elif side == "SELL":
                net_cash_flow = gross - commission
            cash_now = float(rec.get("cash_after", cash_before + net_cash_flow))
            pos_after = int(round(float(rec.get("position_after", pos_before))))
            if pos_after <= 0 and symbol in positions.index:
                positions = positions.drop(symbol)
            elif symbol:
                positions.loc[symbol] = float(pos_after)

        rows.append(
            {
                "date": rec.get("date", ""),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "gross_amount": gross,
                "commission": commission,
                "net_cash_flow": net_cash_flow,
                "cash_before": cash_before,
                "cash_after": cash_now,
                "position_before": pos_before,
                "position_after": pos_after,
                "fill_status": fill_status,
                "fill_reason": fill_reason,
            }
        )
    return pd.DataFrame(rows, columns=PAPER_TRADE_COLUMNS)


def run_daily_paper_trade(
    settings: Settings,
    *,
    strategy: str,
    target_weights: Mapping[str, float] | pd.Series | pd.DataFrame,
    latest_prices: Mapping[str, float] | pd.Series | pd.DataFrame,
    trade_date: Any,
    trade_status: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
    risk_blacklist: pd.DataFrame | Mapping[str, Any] | None = None,
    default_cash: float | None = None,
    total_asset: float | None = None,
    persist_outputs: bool = True,
    execution_mode: str = EXECUTION_MODE_PAPER_TRADING,
) -> dict[str, Any]:
    """
    运行单日纸面交易流程。

    流程：
    1. 读取纸面账户状态
    2. 生成订单计划
    3. 做订单预检查
    4. 执行纸面交易；可直接走旧版纸面成交，也可走统一模拟券商适配器
    5. 生成账户快照并保存状态

    :return: dict，含 `orders/order_checks/paper_trades/account_snapshot/positions/paths`
    """
    mode = str(execution_mode).strip().lower()
    if mode not in {EXECUTION_MODE_PAPER_TRADING, EXECUTION_MODE_SIMULATED_BROKER}:
        raise ValueError(
            "execution_mode 仅支持 %s 或 %s"
            % (EXECUTION_MODE_PAPER_TRADING, EXECUTION_MODE_SIMULATED_BROKER)
        )

    start_cash, current_positions = load_account_state(
        settings,
        strategy=strategy,
        default_cash=(
            float(settings.paper_initial_cash)
            if default_cash is None
            else float(default_cash)
        ),
    )
    orders = build_order_plan(
        target_weights,
        current_positions,
        latest_prices,
        cash=start_cash,
        total_asset=total_asset,
        trade_date=trade_date,
        lot_size=int(settings.order_lot_size),
        min_order_amount=float(settings.min_order_amount),
        include_holds=False,
        board_rules=settings.order_board_rules,
    )
    checks = precheck_order_plan(
        orders,
        cash=start_cash,
        current_positions=current_positions,
        trade_status=trade_status,
        risk_blacklist=risk_blacklist,
        lot_size=int(settings.order_lot_size),
        min_order_amount=float(settings.min_order_amount),
        cash_buffer=float(settings.order_cash_buffer),
        board_rules=settings.order_board_rules,
        commission_rate=float(settings.commission_rate) if settings.order_board_rules else 0.0,
    )
    broker_orders = pd.DataFrame()
    if mode == EXECUTION_MODE_SIMULATED_BROKER:
        broker = SimulatedBroker(
            cash=start_cash,
            positions=current_positions,
            latest_prices=latest_prices,
            commission_rate=float(settings.commission_rate),
        )
        broker_orders = broker.submit_order_plan(orders, order_checks=checks)
        trades = _broker_orders_to_paper_trades(
            broker_orders,
            initial_cash=start_cash,
            current_positions=current_positions,
        )
        latest_cash = broker.get_cash()
        latest_positions = broker.get_positions()
        account = broker.get_account()
        snapshot = {
            "cash": float(account.cash),
            "market_value": float(account.market_value),
            "total_asset": float(account.total_asset),
            "n_positions": float(len(latest_positions)),
        }
    else:
        trades = run_paper_trading(
            initial_cash=start_cash,
            orders=orders,
            order_checks=checks,
            current_positions=current_positions,
            commission_rate=float(settings.commission_rate),
        )
        latest_cash = float(trades["cash_after"].iloc[-1]) if not trades.empty else float(start_cash)
        latest_positions = positions_from_trades(
            trades,
            current_positions,
            updated_at=trade_date,
        )
        snapshot = paper_account_snapshot(
            trades,
            latest_prices,
            current_positions=current_positions,
            cash=start_cash,
        )
        if trades.empty:
            snapshot["cash"] = latest_cash
            snapshot["total_asset"] = latest_cash + float(snapshot.get("market_value", 0.0))

    paths: dict[str, Path | dict[str, Path]] = {}
    if persist_outputs:
        paths["order_plans"] = save_order_plans(settings, {strategy: orders}).get(strategy)
        paths["order_checks"] = save_order_checks(settings, {strategy: checks}).get(strategy)
        paths["paper_trades"] = save_paper_trades(settings, {strategy: trades}).get(strategy)
        paths["account_state"] = save_account_state(
            settings,
            strategy=strategy,
            cash=latest_cash,
            positions=latest_positions,
            snapshot=snapshot,
            trade_date=trade_date,
        )

    return {
        "strategy": strategy,
        "execution_mode": mode,
        "trade_date": trade_date,
        "starting_cash": start_cash,
        "starting_positions": current_positions,
        "orders": orders,
        "order_checks": checks,
        "broker_orders": broker_orders,
        "paper_trades": trades,
        "cash": latest_cash,
        "positions": latest_positions,
        "account_snapshot": snapshot,
        "paths": paths,
    }
