"""
实盘 / 纸面交易表现归因。

本模块读取纸面账户快照、当前持仓、价格缓存和真实成交回填结果，把“账户涨跌”
拆成可复盘的几类来源：市场基准、当前持仓贡献、执行偏差和无法解释的残差。
它不修改账户、不生成订单，只负责日终解释。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


SUMMARY_COLUMNS = [
    "date",
    "previous_date",
    "strategy",
    "previous_total_asset",
    "total_asset",
    "cash",
    "market_value",
    "cash_weight",
    "market_weight",
    "n_positions",
    "account_return",
    "benchmark_return",
    "active_return",
    "stock_contribution_return",
    "execution_slippage_cost",
    "execution_slippage_return",
    "unexplained_return",
    "status",
    "detail",
]

STOCK_CONTRIBUTION_COLUMNS = [
    "date",
    "previous_date",
    "strategy",
    "symbol",
    "shares",
    "previous_price",
    "price",
    "price_return",
    "current_value",
    "current_weight",
    "contribution_amount",
    "contribution_return",
    "status",
]


def performance_attribution_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "performance_attribution" / safe


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def _normalize_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "total_asset"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError("账户快照缺少必要列: %s" % ", ".join(sorted(missing)))
    frame = snapshots.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    for col in ["cash", "market_value", "total_asset", "n_positions"]:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("账户快照为空或没有有效日期")
    return frame


def _normalize_prices(prices: pd.DataFrame | None) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})

    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code" if "ts_code" in frame.columns else None
    if symbol_col is not None:
        value_col = None
        for candidate in ["close", "adj_close", "price"]:
            if candidate in frame.columns:
                value_col = candidate
                break
        if value_col is None:
            raise ValueError("长表价格数据须包含 close/adj_close/price 之一")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        wide = frame.pivot_table(index="date", columns=symbol_col, values=value_col, aggfunc="last")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        wide = frame.set_index("date")
        wide = wide.apply(pd.to_numeric, errors="coerce")

    wide = wide[wide.index.notna()].sort_index()
    wide.columns = wide.columns.astype(str)
    return wide


def _price_row_on_or_before(wide_prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    if wide_prices.empty:
        return pd.Series(dtype=float)
    subset = wide_prices[wide_prices.index <= date]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[-1].astype(float)


def _current_and_previous_snapshot(
    snapshots: pd.DataFrame,
    trade_date: Any,
) -> tuple[pd.Series, pd.Series | None]:
    dt = pd.Timestamp(trade_date)
    frame = snapshots[snapshots["date"] <= dt].copy()
    if frame.empty:
        raise ValueError("账户快照中没有不晚于 %s 的记录" % dt.strftime("%Y-%m-%d"))
    current = frame.iloc[-1]
    previous_frame = frame[frame["date"] < current["date"]]
    previous = previous_frame.iloc[-1] if not previous_frame.empty else None
    return current, previous


def _position_frame(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame(columns=["symbol", "shares"])
    symbol_col = "symbol" if "symbol" in positions.columns else "ts_code" if "ts_code" in positions.columns else None
    if symbol_col is None or "shares" not in positions.columns:
        raise ValueError("持仓表须包含 symbol/ts_code 与 shares 列")
    frame = pd.DataFrame(
        {
            "symbol": positions[symbol_col].astype(str),
            "shares": pd.to_numeric(positions["shares"], errors="coerce").fillna(0.0),
        }
    )
    frame = frame[frame["shares"] > 0.0]
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "shares"])
    return frame.groupby("symbol", as_index=False)["shares"].sum().sort_values("symbol").reset_index(drop=True)


def _benchmark_return(
    wide_prices: pd.DataFrame,
    symbols: list[str],
    *,
    previous_date: pd.Timestamp | None,
    current_date: pd.Timestamp,
) -> tuple[float, str]:
    if previous_date is None:
        return float("nan"), "previous_snapshot_missing"
    if wide_prices.empty:
        return float("nan"), "price_cache_missing"
    prev = _price_row_on_or_before(wide_prices, previous_date)
    cur = _price_row_on_or_before(wide_prices, current_date)
    if prev.empty or cur.empty:
        return float("nan"), "price_row_missing"
    cols = symbols if symbols else sorted(set(prev.index.astype(str)).intersection(set(cur.index.astype(str))))
    rows = []
    for symbol in cols:
        p0 = _num(prev.get(symbol, float("nan")), float("nan"))
        p1 = _num(cur.get(symbol, float("nan")), float("nan"))
        if p0 > 0.0 and p1 > 0.0:
            rows.append(p1 / p0 - 1.0)
    if not rows:
        return float("nan"), "benchmark_symbols_missing"
    return float(pd.Series(rows).mean()), "ok"


def _stock_contribution(
    positions: pd.DataFrame,
    wide_prices: pd.DataFrame,
    *,
    strategy: str,
    current_date: pd.Timestamp,
    previous_date: pd.Timestamp | None,
    previous_total_asset: float,
    current_total_asset: float,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(columns=STOCK_CONTRIBUTION_COLUMNS)
    if previous_date is None:
        out = positions.copy()
        out["date"] = _date_to_str(current_date)
        out["previous_date"] = ""
        out["strategy"] = strategy
        out["previous_price"] = float("nan")
        out["price"] = float("nan")
        out["price_return"] = float("nan")
        out["current_value"] = float("nan")
        out["current_weight"] = float("nan")
        out["contribution_amount"] = float("nan")
        out["contribution_return"] = float("nan")
        out["status"] = "previous_snapshot_missing"
        return out[STOCK_CONTRIBUTION_COLUMNS]
    prev = _price_row_on_or_before(wide_prices, previous_date)
    cur = _price_row_on_or_before(wide_prices, current_date)
    rows: list[dict[str, Any]] = []
    for rec in positions.to_dict("records"):
        symbol = str(rec["symbol"])
        shares = _num(rec["shares"])
        p0 = _num(prev.get(symbol, float("nan")), float("nan"))
        p1 = _num(cur.get(symbol, float("nan")), float("nan"))
        status = "ok"
        if not (p0 > 0.0 and p1 > 0.0):
            status = "price_missing"
        price_return = p1 / p0 - 1.0 if status == "ok" else float("nan")
        current_value = shares * p1 if p1 > 0.0 else float("nan")
        contribution_amount = shares * (p1 - p0) if status == "ok" else float("nan")
        contribution_return = (
            contribution_amount / previous_total_asset
            if status == "ok" and previous_total_asset > 0.0
            else float("nan")
        )
        rows.append(
            {
                "date": _date_to_str(current_date),
                "previous_date": _date_to_str(previous_date),
                "strategy": strategy,
                "symbol": symbol,
                "shares": shares,
                "previous_price": p0,
                "price": p1,
                "price_return": price_return,
                "current_value": current_value,
                "current_weight": current_value / current_total_asset if current_total_asset > 0.0 and p1 > 0.0 else float("nan"),
                "contribution_amount": contribution_amount,
                "contribution_return": contribution_return,
                "status": status,
            }
        )
    out = pd.DataFrame(rows, columns=STOCK_CONTRIBUTION_COLUMNS)
    return out.sort_values("contribution_return", ascending=False, na_position="last").reset_index(drop=True)


def _execution_slippage_cost(execution_feedback: pd.DataFrame | None) -> float:
    if execution_feedback is None or execution_feedback.empty:
        return 0.0
    required = {"side", "suggested_price", "executed_price", "executed_qty"}
    if not required.issubset(execution_feedback.columns):
        return 0.0
    cost = 0.0
    for rec in execution_feedback.to_dict("records"):
        side = str(rec.get("side", "")).upper()
        qty = abs(_num(rec.get("executed_qty", 0.0)))
        suggested = _num(rec.get("suggested_price", 0.0))
        executed = _num(rec.get("executed_price", 0.0))
        if qty <= 0.0 or suggested <= 0.0 or executed <= 0.0:
            continue
        if side == "BUY":
            cost += (executed - suggested) * qty
        elif side == "SELL":
            cost += (suggested - executed) * qty
    return float(cost)


def build_performance_attribution(
    snapshots: pd.DataFrame,
    *,
    strategy: str,
    trade_date: Any,
    prices: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    execution_feedback: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成账户表现归因汇总表和逐股票贡献表。"""
    snapshots_n = _normalize_snapshots(snapshots)
    current, previous = _current_and_previous_snapshot(snapshots_n, trade_date)
    positions_n = _position_frame(positions)
    wide_prices = _normalize_prices(prices)

    current_date = pd.Timestamp(current["date"])
    previous_date = pd.Timestamp(previous["date"]) if previous is not None else None
    current_total_asset = _num(current.get("total_asset", 0.0))
    previous_total_asset = _num(previous.get("total_asset", 0.0)) if previous is not None else float("nan")
    account_return = (
        current_total_asset / previous_total_asset - 1.0
        if previous is not None and previous_total_asset > 0.0 and current_total_asset > 0.0
        else float("nan")
    )
    cash = _num(current.get("cash", 0.0))
    market_value = _num(current.get("market_value", 0.0))
    cash_weight = cash / current_total_asset if current_total_asset > 0.0 else float("nan")
    market_weight = market_value / current_total_asset if current_total_asset > 0.0 else float("nan")

    symbols = positions_n["symbol"].astype(str).tolist() if not positions_n.empty else []
    benchmark, benchmark_status = _benchmark_return(
        wide_prices,
        symbols,
        previous_date=previous_date,
        current_date=current_date,
    )
    stock = _stock_contribution(
        positions_n,
        wide_prices,
        strategy=strategy,
        current_date=current_date,
        previous_date=previous_date,
        previous_total_asset=previous_total_asset,
        current_total_asset=current_total_asset,
    )
    stock_contribution_return = float(stock["contribution_return"].sum(skipna=True)) if not stock.empty else 0.0
    slippage_cost = _execution_slippage_cost(execution_feedback)
    slippage_return = -slippage_cost / previous_total_asset if previous is not None and previous_total_asset > 0.0 else 0.0
    active_return = account_return - benchmark if pd.notna(account_return) and pd.notna(benchmark) else float("nan")
    unexplained = (
        account_return - stock_contribution_return - slippage_return
        if pd.notna(account_return)
        else float("nan")
    )

    status = "PASS"
    detail_parts = []
    if previous is None:
        status = "NA"
        detail_parts.append("previous_snapshot_missing")
    if benchmark_status != "ok":
        status = "NA" if status == "PASS" else status
        detail_parts.append("benchmark_%s" % benchmark_status)
    if not stock.empty and (stock["status"] != "ok").any():
        status = "NA" if status == "PASS" else status
        detail_parts.append("stock_price_missing=%d" % int((stock["status"] != "ok").sum()))
    if not detail_parts:
        detail_parts.append("ok")

    summary = pd.DataFrame(
        [
            {
                "date": _date_to_str(current_date),
                "previous_date": _date_to_str(previous_date) if previous_date is not None else "",
                "strategy": strategy,
                "previous_total_asset": previous_total_asset,
                "total_asset": current_total_asset,
                "cash": cash,
                "market_value": market_value,
                "cash_weight": cash_weight,
                "market_weight": market_weight,
                "n_positions": _num(current.get("n_positions", len(positions_n)), float(len(positions_n))),
                "account_return": account_return,
                "benchmark_return": benchmark,
                "active_return": active_return,
                "stock_contribution_return": stock_contribution_return,
                "execution_slippage_cost": slippage_cost,
                "execution_slippage_return": slippage_return,
                "unexplained_return": unexplained,
                "status": status,
                "detail": ";".join(detail_parts),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return summary, stock


def _fmt_pct(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(out):
        return "NA"
    return "%.2f%%" % (out * 100.0)


def _fmt_money(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(out):
        return "NA"
    return "%.2f" % out


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "无\n"
    rows = frame.head(max_rows)
    lines = [
        "| " + " | ".join(rows.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(rows.columns)) + " |",
    ]
    for rec in rows.to_dict("records"):
        vals: list[str] = []
        for col in rows.columns:
            value = rec.get(col, "")
            if isinstance(value, float):
                vals.append("%.4f" % value if pd.notna(value) else "NA")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_performance_attribution_report(summary: pd.DataFrame, stock: pd.DataFrame) -> str:
    rec = summary.iloc[0].to_dict() if not summary.empty else {}
    strategy = str(rec.get("strategy", ""))
    date_s = str(rec.get("date", ""))
    lines = [
        "# 实盘表现归因 - %s - %s" % (strategy, date_s),
        "",
        "这份报告用于解释当日纸面 / 小资金实盘账户的收益来源。它不会重新生成交易信号，也不会修改账户，只读取已经落盘的账户快照、持仓、价格缓存和真实成交回填结果。",
        "",
        "## 总览",
        "",
        "- 上一快照日期：%s" % str(rec.get("previous_date", "")),
        "- 上一总资产：%s" % _fmt_money(rec.get("previous_total_asset")),
        "- 当前总资产：%s" % _fmt_money(rec.get("total_asset")),
        "- 账户收益：%s" % _fmt_pct(rec.get("account_return")),
        "- 股票池等权基准收益：%s" % _fmt_pct(rec.get("benchmark_return")),
        "- 主动收益：%s" % _fmt_pct(rec.get("active_return")),
        "- 现金权重：%s" % _fmt_pct(rec.get("cash_weight")),
        "- 股票仓位：%s" % _fmt_pct(rec.get("market_weight")),
        "- 执行滑点成本：%s" % _fmt_money(rec.get("execution_slippage_cost")),
        "- 归因状态：%s，%s" % (str(rec.get("status", "")), str(rec.get("detail", ""))),
        "",
        "## 收益拆解",
        "",
        "| 项目 | 数值 | 解释 |",
        "| --- | ---: | --- |",
        "| 账户收益 | %s | 账户总资产相对上一快照的变化 |" % _fmt_pct(rec.get("account_return")),
        "| 基准收益 | %s | 当前持仓股票等权价格收益，用来近似市场 / 股票池环境 |" % _fmt_pct(rec.get("benchmark_return")),
        "| 主动收益 | %s | 账户收益减去基准收益，粗略观察策略是否跑赢持仓环境 |" % _fmt_pct(rec.get("active_return")),
        "| 持仓价格贡献 | %s | 当前持仓按价格变化估算出的贡献 |" % _fmt_pct(rec.get("stock_contribution_return")),
        "| 执行滑点贡献 | %s | 真实成交价格相对建议价格的影响，负数代表执行拖累 |" % _fmt_pct(rec.get("execution_slippage_return")),
        "| 未解释残差 | %s | 费用、盘中交易、持仓口径差异、现金流等未被上面几项解释的部分 |" % _fmt_pct(rec.get("unexplained_return")),
        "",
        "## 个股贡献",
        "",
    ]
    if stock.empty:
        lines.append("当前没有可计算的持仓贡献。")
    else:
        display = stock[
            [
                "symbol",
                "shares",
                "previous_price",
                "price",
                "price_return",
                "current_weight",
                "contribution_return",
                "status",
            ]
        ].copy()
        lines.append(_markdown_table(display, max_rows=30))
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "这是一张日终解释表，不是选股模型。它适合每天纸面交易或小资金人工实盘之后运行：如果收益好，要知道主要是谁贡献；如果收益差，要知道是市场拖累、个股拖累、执行滑点，还是数据口径还没解释清楚。",
            "",
            "当前个股贡献使用“当前持仓 × 前后两个快照价格变化”估算。如果当天发生了大量盘中买卖，它只能作为近似归因；后续接入真实逐笔成交和持仓流水后，可以升级成更精确的 Brinson / 交易级归因。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_performance_attribution(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    summary: pd.DataFrame,
    stock: pd.DataFrame,
) -> dict[str, Path]:
    base = performance_attribution_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    date_s = _date_to_str(trade_date)
    summary_path = base / ("%s_performance_attribution_summary.csv" % date_s)
    stock_path = base / ("%s_stock_contribution.csv" % date_s)
    report_path = base / ("%s_performance_attribution.md" % date_s)
    summary.to_csv(summary_path, index=False)
    stock.to_csv(stock_path, index=False)
    report_path.write_text(build_performance_attribution_report(summary, stock), encoding="utf-8")
    return {
        "summary": summary_path,
        "stock_contribution": stock_path,
        "markdown": report_path,
    }
