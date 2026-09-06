"""
统一券商接口协议与模拟券商实现。

这一层只定义交易适配器的共同语言。策略、订单生成和风控只依赖
`BrokerAdapter` 的方法；真实券商接入时实现同一组方法即可。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

import pandas as pd


BROKER_MODE_SIMULATED = "simulated"
BROKER_MODE_REAL_READONLY = "real_readonly"
BROKER_MODE_REAL_TRADING = "real_trading"

ORDER_STATUS_NEW = "NEW"
ORDER_STATUS_PARTIALLY_FILLED = "PARTIALLY_FILLED"
ORDER_STATUS_FILLED = "FILLED"
ORDER_STATUS_REJECTED = "REJECTED"
ORDER_STATUS_CANCELLED = "CANCELLED"


class BrokerReadOnlyError(RuntimeError):
    """只读券商适配器收到交易请求时抛出。"""


@dataclass(frozen=True)
class BrokerAccount:
    """券商账户快照。"""

    cash: float
    market_value: float
    total_asset: float
    updated_at: str = ""


@dataclass(frozen=True)
class BrokerPosition:
    """券商持仓快照。"""

    symbol: str
    shares: int
    available_shares: int
    market_value: float = 0.0
    price: float = 0.0
    updated_at: str = ""


@dataclass(frozen=True)
class BrokerOrder:
    """统一订单 / 成交回报。"""

    order_id: str
    date: str
    symbol: str
    side: str
    qty: int
    price: float
    status: str
    reason: str = ""
    filled_qty: int = 0
    avg_price: float = 0.0
    gross_amount: float = 0.0
    commission: float = 0.0
    cash_after: float = 0.0
    position_after: int = 0
    submitted_at: str = ""


@dataclass(frozen=True)
class RealBrokerConfig:
    """真实券商适配器的非敏感配置。"""

    provider: str
    account_id: str = ""
    mode: str = BROKER_MODE_REAL_READONLY

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {BROKER_MODE_REAL_READONLY, BROKER_MODE_REAL_TRADING}:
            raise ValueError(
                "真实券商 mode 仅支持 %s 或 %s"
                % (BROKER_MODE_REAL_READONLY, BROKER_MODE_REAL_TRADING)
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "provider", str(self.provider).strip())
        object.__setattr__(self, "account_id", str(self.account_id).strip())
        if not str(self.provider).strip():
            raise ValueError("真实券商 provider 不能为空")


class BrokerAdapter(Protocol):
    """交易适配器协议；真实券商和模拟券商都应实现。"""

    def sync(self) -> None:
        """同步券商端账户、持仓、订单状态。"""

    def get_account(self) -> BrokerAccount:
        """返回账户快照。"""

    def get_cash(self) -> float:
        """返回可用现金。"""

    def get_positions(self) -> pd.DataFrame:
        """返回持仓表，至少包含 symbol/shares/available_shares。"""

    def get_orders(self) -> pd.DataFrame:
        """返回订单 / 成交回报表。"""

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        date: Any = None,
        reason: str = "",
    ) -> BrokerOrder:
        """提交单笔订单。"""

    def cancel_order(self, order_id: str, *, reason: str = "") -> BrokerOrder:
        """撤销订单。"""


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def _prices_to_series(prices: pd.DataFrame | Mapping[str, float] | pd.Series | None) -> pd.Series:
    if prices is None:
        return pd.Series(dtype=float)
    if isinstance(prices, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in prices.columns else "ts_code"
        price_col = "price"
        if price_col not in prices.columns:
            for candidate in ("close", "last_price", "latest_price"):
                if candidate in prices.columns:
                    price_col = candidate
                    break
        if symbol_col not in prices.columns or price_col not in prices.columns:
            raise ValueError("prices DataFrame 须包含 symbol/ts_code 与 price/close 列")
        out = pd.Series(
            prices[price_col].astype(float).to_numpy(),
            index=prices[symbol_col].astype(str),
            dtype=float,
        )
    elif isinstance(prices, pd.Series):
        out = prices.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in prices.items()}, dtype=float)
    return out.groupby(level=0).last().sort_index()


def _positions_to_frame(
    positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
    *,
    latest_prices: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    if isinstance(positions, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in positions.columns else "ts_code"
        if symbol_col in positions.columns and "shares" in positions.columns:
            out = positions.copy().rename(columns={symbol_col: "symbol"})
            out["symbol"] = out["symbol"].astype(str)
            out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0).round().astype(int)
            if "available_shares" not in out.columns:
                out["available_shares"] = out["shares"]
            if "price" not in out.columns:
                out["price"] = 0.0
            if "market_value" not in out.columns:
                out["market_value"] = out["shares"] * pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
            if "updated_at" not in out.columns:
                out["updated_at"] = _now_str()
            cols = ["symbol", "shares", "available_shares", "price", "market_value", "updated_at"]
            return out.loc[out["shares"] > 0, cols].reset_index(drop=True)
    prices = _prices_to_series(latest_prices)
    series = _positions_to_series(positions)
    rows: list[dict[str, Any]] = []
    for symbol, shares_f in series.items():
        shares = int(round(float(shares_f)))
        if shares <= 0:
            continue
        price = float(prices.get(symbol, 0.0))
        rows.append(
            {
                "symbol": str(symbol),
                "shares": shares,
                "available_shares": shares,
                "price": price,
                "market_value": shares * price if price > 0.0 else 0.0,
                "updated_at": _now_str(),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["symbol", "shares", "available_shares", "price", "market_value", "updated_at"],
    )


def _order_records_to_frame(records: list[BrokerOrder]) -> pd.DataFrame:
    cols = list(BrokerOrder.__dataclass_fields__.keys())
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([asdict(x) for x in records], columns=cols)


class RealBrokerReadOnlyAdapter:
    """
    真实券商只读适配器骨架。

    该类先固定安全边界：可以查账户、查持仓、查订单，但不能下单或撤单。
    具体券商接入时，可继承本类并实现 `sync` 中的真实 API 同步逻辑。
    """

    def __init__(
        self,
        config: RealBrokerConfig,
        *,
        account: BrokerAccount | Mapping[str, Any] | None = None,
        positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
        orders: pd.DataFrame | None = None,
        latest_prices: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    ) -> None:
        if config.mode != BROKER_MODE_REAL_READONLY:
            raise ValueError("RealBrokerReadOnlyAdapter 只允许 mode=%s" % BROKER_MODE_REAL_READONLY)
        self.config = config
        self._account = self._normalize_account(account)
        self._positions = _positions_to_frame(positions, latest_prices=latest_prices)
        self._orders = self._normalize_orders(orders)

    def _normalize_account(self, account: BrokerAccount | Mapping[str, Any] | None) -> BrokerAccount | None:
        if account is None:
            return None
        if isinstance(account, BrokerAccount):
            return account
        return BrokerAccount(
            cash=float(account.get("cash", 0.0)),
            market_value=float(account.get("market_value", 0.0)),
            total_asset=float(account.get("total_asset", 0.0)),
            updated_at=str(account.get("updated_at", "")),
        )

    def _normalize_orders(self, orders: pd.DataFrame | None) -> pd.DataFrame:
        cols = list(BrokerOrder.__dataclass_fields__.keys())
        if orders is None:
            return pd.DataFrame(columns=cols)
        out = orders.copy()
        for col in cols:
            if col not in out.columns:
                out[col] = "" if col in {"order_id", "date", "symbol", "side", "status", "reason", "submitted_at"} else 0
        return out.loc[:, cols].copy()

    def sync(self) -> None:
        """
        同步真实券商状态。

        基类不连接任何券商，只保留接口形态；具体券商 adapter 应覆盖本方法。
        """

    def update_snapshot(
        self,
        *,
        account: BrokerAccount | Mapping[str, Any] | None = None,
        positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
        orders: pd.DataFrame | None = None,
        latest_prices: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    ) -> None:
        """更新只读快照，便于本地测试或未来真实 API 同步后落入统一结构。"""
        if account is not None:
            self._account = self._normalize_account(account)
        if positions is not None:
            self._positions = _positions_to_frame(positions, latest_prices=latest_prices)
        if orders is not None:
            self._orders = self._normalize_orders(orders)

    def get_cash(self) -> float:
        return float(self.get_account().cash)

    def get_account(self) -> BrokerAccount:
        if self._account is not None:
            return self._account
        market_value = float(self._positions["market_value"].sum()) if not self._positions.empty else 0.0
        return BrokerAccount(
            cash=0.0,
            market_value=market_value,
            total_asset=market_value,
            updated_at=_now_str(),
        )

    def get_positions(self) -> pd.DataFrame:
        return self._positions.copy()

    def get_orders(self) -> pd.DataFrame:
        return self._orders.copy()

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        date: Any = None,
        reason: str = "",
    ) -> BrokerOrder:
        del symbol, side, qty, price, date, reason
        raise BrokerReadOnlyError("当前券商适配器处于 real_readonly 模式，禁止提交订单")

    def cancel_order(self, order_id: str, *, reason: str = "") -> BrokerOrder:
        del order_id, reason
        raise BrokerReadOnlyError("当前券商适配器处于 real_readonly 模式，禁止撤单")


class SimulatedBroker:
    """
    模拟券商适配器。

    默认以订单价格立即成交，用于验证统一交易协议和上层流程；它不是撮合引擎，
    也不模拟盘口、滑点或部分成交。
    """

    def __init__(
        self,
        *,
        cash: float = 1_000_000.0,
        positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
        latest_prices: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
        commission_rate: float = 0.0003,
    ) -> None:
        if cash < 0:
            raise ValueError("cash 不能为负")
        if commission_rate < 0:
            raise ValueError("commission_rate 不能为负")
        self._cash = float(cash)
        self._positions = _positions_to_series(positions)
        self._latest_prices = _prices_to_series(latest_prices)
        self._commission_rate = float(commission_rate)
        self._orders: list[BrokerOrder] = []

    def sync(self) -> None:
        """模拟券商无远端状态，保留该方法以匹配真实券商协议。"""

    def update_prices(self, latest_prices: pd.DataFrame | Mapping[str, float] | pd.Series) -> None:
        self._latest_prices = _prices_to_series(latest_prices)

    def get_cash(self) -> float:
        return float(self._cash)

    def get_account(self) -> BrokerAccount:
        positions = self.get_positions()
        market_value = float(positions["market_value"].sum()) if not positions.empty else 0.0
        return BrokerAccount(
            cash=float(self._cash),
            market_value=market_value,
            total_asset=float(self._cash) + market_value,
            updated_at=_now_str(),
        )

    def get_positions(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol, shares_f in self._positions.items():
            shares = int(round(float(shares_f)))
            if shares <= 0:
                continue
            price = float(self._latest_prices.get(symbol, 0.0))
            rows.append(
                {
                    "symbol": symbol,
                    "shares": shares,
                    "available_shares": shares,
                    "price": price,
                    "market_value": shares * price if price > 0.0 else 0.0,
                    "updated_at": _now_str(),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["symbol", "shares", "available_shares", "price", "market_value", "updated_at"],
        )

    def get_orders(self) -> pd.DataFrame:
        return _order_records_to_frame(self._orders)

    def _record_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        date: Any,
        status: str,
        reason: str,
        filled_qty: int = 0,
        avg_price: float = 0.0,
        gross_amount: float = 0.0,
        commission: float = 0.0,
    ) -> BrokerOrder:
        order = BrokerOrder(
            order_id=uuid4().hex,
            date=_date_to_str(date),
            symbol=str(symbol),
            side=str(side).upper(),
            qty=int(qty),
            price=float(price),
            status=status,
            reason=reason,
            filled_qty=int(filled_qty),
            avg_price=float(avg_price),
            gross_amount=float(gross_amount),
            commission=float(commission),
            cash_after=float(self._cash),
            position_after=int(round(float(self._positions.get(str(symbol), 0.0)))),
            submitted_at=_now_str(),
        )
        self._orders.append(order)
        return order

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        date: Any = None,
        reason: str = "",
    ) -> BrokerOrder:
        symbol_s = str(symbol)
        side_s = str(side).upper()
        qty_i = int(qty)
        price_f = float(price)
        if side_s not in {"BUY", "SELL"}:
            return self._record_order(
                symbol=symbol_s,
                side=side_s,
                qty=qty_i,
                price=price_f,
                date=date,
                status=ORDER_STATUS_REJECTED,
                reason="invalid_side",
            )
        if qty_i <= 0 or price_f <= 0.0:
            return self._record_order(
                symbol=symbol_s,
                side=side_s,
                qty=qty_i,
                price=price_f,
                date=date,
                status=ORDER_STATUS_REJECTED,
                reason="invalid_qty_or_price",
            )

        gross = qty_i * price_f
        commission = gross * self._commission_rate
        current_pos = int(round(float(self._positions.get(symbol_s, 0.0))))
        if side_s == "BUY":
            required_cash = gross + commission
            if required_cash > self._cash + 1e-8:
                return self._record_order(
                    symbol=symbol_s,
                    side=side_s,
                    qty=qty_i,
                    price=price_f,
                    date=date,
                    status=ORDER_STATUS_REJECTED,
                    reason="insufficient_cash",
                )
            self._cash -= required_cash
            self._positions.loc[symbol_s] = float(current_pos + qty_i)
        else:
            if qty_i > current_pos:
                return self._record_order(
                    symbol=symbol_s,
                    side=side_s,
                    qty=qty_i,
                    price=price_f,
                    date=date,
                    status=ORDER_STATUS_REJECTED,
                    reason="insufficient_position",
                )
            self._cash += gross - commission
            new_pos = current_pos - qty_i
            if new_pos <= 0 and symbol_s in self._positions.index:
                self._positions = self._positions.drop(symbol_s)
            else:
                self._positions.loc[symbol_s] = float(new_pos)

        self._latest_prices.loc[symbol_s] = price_f
        return self._record_order(
            symbol=symbol_s,
            side=side_s,
            qty=qty_i,
            price=price_f,
            date=date,
            status=ORDER_STATUS_FILLED,
            reason=reason or "filled",
            filled_qty=qty_i,
            avg_price=price_f,
            gross_amount=gross,
            commission=commission,
        )

    def cancel_order(self, order_id: str, *, reason: str = "") -> BrokerOrder:
        for order in self._orders:
            if order.order_id == order_id:
                if order.status != ORDER_STATUS_NEW:
                    raise ValueError("仅 NEW 状态订单可撤销；当前状态=%s" % order.status)
                cancelled = BrokerOrder(
                    **{
                        **asdict(order),
                        "status": ORDER_STATUS_CANCELLED,
                        "reason": reason or "cancelled",
                        "submitted_at": _now_str(),
                    }
                )
                self._orders.append(cancelled)
                return cancelled
        raise KeyError(order_id)

    def submit_order_plan(
        self,
        order_plan: pd.DataFrame,
        *,
        order_checks: pd.DataFrame | None = None,
        reject_blocked: bool = True,
    ) -> pd.DataFrame:
        """
        执行订单计划。

        若传入 `order_checks`，仅执行 `check_status=PASS` 的订单；被 BLOCK 的订单
        默认记录为 REJECTED，便于统一审计。
        """
        if order_plan.empty:
            return self.get_orders()
        checks: dict[tuple[str, str, int], str] = {}
        if order_checks is not None and not order_checks.empty:
            for rec in order_checks.to_dict("records"):
                key = (
                    str(rec.get("symbol", "")),
                    str(rec.get("side", "")).upper(),
                    int(round(float(rec.get("delta_shares", 0)))),
                )
                checks[key] = str(rec.get("check_status", "")).upper()

        for rec in order_plan.to_dict("records"):
            side = str(rec.get("side", "")).upper()
            if side not in {"BUY", "SELL"}:
                continue
            delta = int(round(float(rec.get("delta_shares", 0))))
            key = (str(rec.get("symbol", "")), side, delta)
            if checks and checks.get(key) != "PASS":
                if reject_blocked:
                    self._record_order(
                        symbol=rec.get("symbol", ""),
                        side=side,
                        qty=abs(delta),
                        price=float(rec.get("price", 0.0)),
                        date=rec.get("date", ""),
                        status=ORDER_STATUS_REJECTED,
                        reason="blocked_by_precheck",
                    )
                continue
            self.submit_order(
                symbol=rec.get("symbol", ""),
                side=side,
                qty=abs(delta),
                price=float(rec.get("price", 0.0)),
                date=rec.get("date", ""),
                reason=str(rec.get("trade_reason", "")),
            )
        return self.get_orders()
