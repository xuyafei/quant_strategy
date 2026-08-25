"""
真实成交回填与执行偏差分析。

读取人工确认实盘单中回填的真实成交字段，比较系统建议订单和真实执行结果，
输出成交数量、成交价格、金额和状态差异。该模块不连接券商、不修改账户状态。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


FEEDBACK_COLUMNS = [
    "date",
    "strategy",
    "symbol",
    "side",
    "suggested_qty",
    "executed_qty",
    "qty_diff",
    "suggested_price",
    "executed_price",
    "price_diff",
    "price_slippage_pct",
    "suggested_amount",
    "executed_amount",
    "amount_diff",
    "check_status",
    "manual_action",
    "execution_status",
    "operator",
    "confirmed_at",
    "execution_note",
]


SUMMARY_COLUMNS = [
    "date",
    "strategy",
    "n_orders",
    "n_filled",
    "n_partial",
    "n_not_executed",
    "n_blocked",
    "suggested_buy_amount",
    "executed_buy_amount",
    "suggested_sell_amount",
    "executed_sell_amount",
    "net_executed_cash_flow",
    "avg_abs_price_slippage_pct",
    "max_abs_price_slippage_pct",
]

NEXT_DAY_REVIEW_COLUMNS = [
    "date",
    "review_date",
    "strategy",
    "symbol",
    "side",
    "execution_status",
    "executed_qty",
    "executed_price",
    "review_price",
    "executed_amount",
    "review_value",
    "next_day_return",
    "next_day_effect",
    "effect_type",
    "review_status",
]

NEXT_DAY_SUMMARY_COLUMNS = [
    "date",
    "review_date",
    "strategy",
    "n_orders",
    "n_reviewed",
    "n_missing_review_price",
    "buy_next_day_pnl",
    "sell_avoidance_pnl",
    "total_next_day_effect",
    "avg_buy_next_day_return",
    "avg_sell_avoidance_return",
]


def execution_feedback_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "execution_feedback" / safe


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


def _execution_status(row: pd.Series) -> str:
    check_status = str(row.get("check_status", "")).upper()
    action = str(row.get("manual_action", "")).upper()
    suggested_qty = abs(_num(row.get("suggested_qty", 0.0)))
    executed_qty = abs(_num(row.get("executed_qty", 0.0)))
    executed_price = _num(row.get("executed_price", 0.0))
    if check_status == "BLOCK" or action == "DO_NOT_EXECUTE":
        return "BLOCKED"
    if executed_qty <= 0 or executed_price <= 0:
        return "NOT_EXECUTED"
    if suggested_qty > 0 and executed_qty < suggested_qty:
        return "PARTIAL"
    if suggested_qty > 0 and executed_qty > suggested_qty:
        return "OVERFILLED"
    return "FILLED"


def build_execution_feedback(manual_confirmation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由人工确认单生成逐笔执行偏差和汇总表。"""
    if manual_confirmation.empty:
        empty_detail = pd.DataFrame(columns=FEEDBACK_COLUMNS)
        empty_summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return empty_detail, empty_summary

    required = {"date", "strategy", "symbol", "side", "delta_shares", "price", "estimated_amount"}
    missing = required - set(manual_confirmation.columns)
    if missing:
        raise ValueError("人工确认单缺少必要列: %s" % ", ".join(sorted(missing)))

    rows: list[dict[str, Any]] = []
    for rec in manual_confirmation.to_dict("records"):
        side = str(rec.get("side", "")).upper()
        suggested_qty = abs(int(round(_num(rec.get("delta_shares", 0.0)))))
        executed_qty = abs(int(round(_num(rec.get("executed_qty", 0.0)))))
        suggested_price = _num(rec.get("price", 0.0))
        executed_price = _num(rec.get("executed_price", 0.0))
        suggested_amount = abs(_num(rec.get("estimated_amount", suggested_qty * suggested_price)))
        executed_amount = executed_qty * executed_price if executed_qty > 0 and executed_price > 0 else 0.0
        price_diff = executed_price - suggested_price if executed_amount > 0 else 0.0
        price_slippage_pct = price_diff / suggested_price if suggested_price > 0 and executed_amount > 0 else 0.0
        row = {
            "date": _date_to_str(rec.get("date", "")),
            "strategy": str(rec.get("strategy", "")),
            "symbol": str(rec.get("symbol", "")),
            "side": side,
            "suggested_qty": suggested_qty,
            "executed_qty": executed_qty,
            "qty_diff": executed_qty - suggested_qty,
            "suggested_price": suggested_price,
            "executed_price": executed_price,
            "price_diff": price_diff,
            "price_slippage_pct": price_slippage_pct,
            "suggested_amount": suggested_amount,
            "executed_amount": executed_amount,
            "amount_diff": executed_amount - suggested_amount,
            "check_status": str(rec.get("check_status", "")),
            "manual_action": str(rec.get("manual_action", "")),
            "execution_status": "",
            "operator": str(rec.get("operator", "")),
            "confirmed_at": str(rec.get("confirmed_at", "")),
            "execution_note": str(rec.get("execution_note", "")),
        }
        row["execution_status"] = _execution_status(pd.Series(row))
        rows.append(row)

    detail = pd.DataFrame(rows, columns=FEEDBACK_COLUMNS)
    dates = detail["date"].dropna().astype(str)
    strategies = detail["strategy"].dropna().astype(str)
    buy = detail[detail["side"] == "BUY"]
    sell = detail[detail["side"] == "SELL"]
    executed_buy = float(buy["executed_amount"].sum()) if not buy.empty else 0.0
    executed_sell = float(sell["executed_amount"].sum()) if not sell.empty else 0.0
    filled_mask = detail["execution_status"].isin(["FILLED", "OVERFILLED"])
    slippage = detail.loc[detail["executed_amount"] > 0, "price_slippage_pct"].astype(float).abs()
    summary = pd.DataFrame(
        [
            {
                "date": dates.iloc[0] if len(dates) else "",
                "strategy": strategies.iloc[0] if len(strategies) else "",
                "n_orders": int(len(detail)),
                "n_filled": int(filled_mask.sum()),
                "n_partial": int((detail["execution_status"] == "PARTIAL").sum()),
                "n_not_executed": int((detail["execution_status"] == "NOT_EXECUTED").sum()),
                "n_blocked": int((detail["execution_status"] == "BLOCKED").sum()),
                "suggested_buy_amount": float(buy["suggested_amount"].sum()) if not buy.empty else 0.0,
                "executed_buy_amount": executed_buy,
                "suggested_sell_amount": float(sell["suggested_amount"].sum()) if not sell.empty else 0.0,
                "executed_sell_amount": executed_sell,
                "net_executed_cash_flow": executed_sell - executed_buy,
                "avg_abs_price_slippage_pct": float(slippage.mean()) if not slippage.empty else 0.0,
                "max_abs_price_slippage_pct": float(slippage.max()) if not slippage.empty else 0.0,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return detail, summary


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
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
                vals.append("%.4f" % value)
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_execution_feedback_report(detail: pd.DataFrame, summary: pd.DataFrame) -> str:
    """生成真实成交回填与执行偏差 Markdown 报告。"""
    rec = summary.iloc[0].to_dict() if not summary.empty else {}
    strategy = str(rec.get("strategy", ""))
    date_s = str(rec.get("date", ""))
    lines = [
        "# 真实成交回填与执行偏差分析 - %s - %s" % (strategy, date_s),
        "",
        "## 摘要",
        "",
        "- 订单数：%s" % rec.get("n_orders", 0),
        "- 完全成交：%s" % rec.get("n_filled", 0),
        "- 部分成交：%s" % rec.get("n_partial", 0),
        "- 未执行：%s" % rec.get("n_not_executed", 0),
        "- 阻断：%s" % rec.get("n_blocked", 0),
        "- 建议买入金额：%.2f" % float(rec.get("suggested_buy_amount", 0.0) or 0.0),
        "- 实际买入金额：%.2f" % float(rec.get("executed_buy_amount", 0.0) or 0.0),
        "- 建议卖出金额：%.2f" % float(rec.get("suggested_sell_amount", 0.0) or 0.0),
        "- 实际卖出金额：%.2f" % float(rec.get("executed_sell_amount", 0.0) or 0.0),
        "- 实际净现金流：%.2f" % float(rec.get("net_executed_cash_flow", 0.0) or 0.0),
        "- 平均绝对滑点：%.4f%%" % (float(rec.get("avg_abs_price_slippage_pct", 0.0) or 0.0) * 100.0),
        "- 最大绝对滑点：%.4f%%" % (float(rec.get("max_abs_price_slippage_pct", 0.0) or 0.0) * 100.0),
        "",
        "## 逐笔执行偏差",
        "",
        _markdown_table(detail),
        "",
        "## 说明",
        "",
        "`price_slippage_pct = (executed_price - suggested_price) / suggested_price`。",
        "",
        "买入时正滑点通常表示成交价高于建议价；卖出时正滑点通常表示成交价高于建议价。该表只记录执行偏差，不判断交易是否应该发生。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _price_lookup(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    frame = prices.copy()
    if {"trade_date", "ts_code", "close"}.issubset(frame.columns):
        out = frame.rename(columns={"trade_date": "date", "ts_code": "symbol", "close": "price"})
        out = out[["date", "symbol", "price"]].copy()
    elif {"date", "symbol", "price"}.issubset(frame.columns):
        out = frame[["date", "symbol", "price"]].copy()
    elif {"date", "symbol", "close"}.issubset(frame.columns):
        out = frame.rename(columns={"close": "price"})[["date", "symbol", "price"]].copy()
    else:
        date_col = "date" if "date" in frame.columns else frame.columns[0]
        wide = frame.rename(columns={date_col: "date"}).copy()
        out = wide.melt(id_vars=["date"], var_name="symbol", value_name="price")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["date", "symbol", "price"])
    out = out[out["price"] > 0.0].sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _resolve_review_date(price_frame: pd.DataFrame, trade_date: pd.Timestamp, review_date: Any | None) -> pd.Timestamp | None:
    if price_frame.empty:
        return None
    dates = pd.Index(sorted(price_frame["date"].dropna().unique()))
    if review_date is not None:
        target = pd.Timestamp(review_date)
        candidates = dates[dates >= target]
    else:
        candidates = dates[dates > trade_date]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[0])


def build_next_day_execution_review(
    execution_detail: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    review_date: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    用成交回填和次日价格做执行后复盘。

    BUY 的 `next_day_effect` 表示买入后次日浮盈浮亏；
    SELL 的 `next_day_effect` 表示卖出后次日规避损益，价格下跌为正。
    """
    if execution_detail.empty:
        return (
            pd.DataFrame(columns=NEXT_DAY_REVIEW_COLUMNS),
            pd.DataFrame(columns=NEXT_DAY_SUMMARY_COLUMNS),
        )
    required = {"date", "strategy", "symbol", "side", "executed_qty", "executed_price", "execution_status"}
    missing = required - set(execution_detail.columns)
    if missing:
        raise ValueError("执行偏差明细缺少必要列: %s" % ", ".join(sorted(missing)))

    detail = execution_detail.copy()
    detail["date"] = pd.to_datetime(detail["date"], errors="coerce")
    trade_date = pd.Timestamp(detail["date"].dropna().min())
    price_frame = _price_lookup(prices)
    resolved_review_date = _resolve_review_date(price_frame, trade_date, review_date)
    if resolved_review_date is None:
        resolved_review_date = pd.NaT
        review_prices = pd.Series(dtype=float)
    else:
        review_prices = (
            price_frame[price_frame["date"] == resolved_review_date]
            .drop_duplicates(subset=["symbol"], keep="last")
            .set_index("symbol")["price"]
        )

    rows: list[dict[str, Any]] = []
    for rec in detail.to_dict("records"):
        side = str(rec.get("side", "")).upper()
        symbol = str(rec.get("symbol", ""))
        status = str(rec.get("execution_status", "")).upper()
        qty = abs(int(round(_num(rec.get("executed_qty", 0.0)))))
        executed_price = _num(rec.get("executed_price", 0.0))
        executed_amount = qty * executed_price if qty > 0 and executed_price > 0 else 0.0
        review_price = _num(review_prices.get(symbol, 0.0), default=0.0)
        review_value = qty * review_price if qty > 0 and review_price > 0 else 0.0
        if status in {"FILLED", "PARTIAL", "OVERFILLED"} and qty > 0 and executed_price > 0 and review_price > 0:
            next_ret = review_price / executed_price - 1.0
            if side == "BUY":
                effect = (review_price - executed_price) * qty
                effect_type = "buy_mark_to_market_pnl"
            elif side == "SELL":
                effect = (executed_price - review_price) * qty
                effect_type = "sell_avoidance_pnl"
            else:
                effect = 0.0
                effect_type = "unknown"
            review_status = "REVIEWED"
        elif status not in {"FILLED", "PARTIAL", "OVERFILLED"}:
            next_ret = 0.0
            effect = 0.0
            effect_type = "not_executed"
            review_status = "NOT_EXECUTED"
        else:
            next_ret = 0.0
            effect = 0.0
            effect_type = "missing_review_price"
            review_status = "MISSING_REVIEW_PRICE"
        rows.append(
            {
                "date": _date_to_str(rec.get("date", "")),
                "review_date": "" if pd.isna(resolved_review_date) else resolved_review_date.strftime("%Y-%m-%d"),
                "strategy": str(rec.get("strategy", "")),
                "symbol": symbol,
                "side": side,
                "execution_status": status,
                "executed_qty": qty,
                "executed_price": executed_price,
                "review_price": review_price,
                "executed_amount": executed_amount,
                "review_value": review_value,
                "next_day_return": next_ret,
                "next_day_effect": effect,
                "effect_type": effect_type,
                "review_status": review_status,
            }
        )

    review = pd.DataFrame(rows, columns=NEXT_DAY_REVIEW_COLUMNS)
    buy_reviewed = review[(review["side"] == "BUY") & (review["review_status"] == "REVIEWED")]
    sell_reviewed = review[(review["side"] == "SELL") & (review["review_status"] == "REVIEWED")]
    summary = pd.DataFrame(
        [
            {
                "date": str(review["date"].iloc[0]) if not review.empty else "",
                "review_date": str(review["review_date"].iloc[0]) if not review.empty else "",
                "strategy": str(review["strategy"].iloc[0]) if not review.empty else "",
                "n_orders": int(len(review)),
                "n_reviewed": int((review["review_status"] == "REVIEWED").sum()) if not review.empty else 0,
                "n_missing_review_price": int((review["review_status"] == "MISSING_REVIEW_PRICE").sum()) if not review.empty else 0,
                "buy_next_day_pnl": float(buy_reviewed["next_day_effect"].sum()) if not buy_reviewed.empty else 0.0,
                "sell_avoidance_pnl": float(sell_reviewed["next_day_effect"].sum()) if not sell_reviewed.empty else 0.0,
                "total_next_day_effect": float(review["next_day_effect"].sum()) if not review.empty else 0.0,
                "avg_buy_next_day_return": float(buy_reviewed["next_day_return"].mean()) if not buy_reviewed.empty else 0.0,
                "avg_sell_avoidance_return": float((-sell_reviewed["next_day_return"]).mean()) if not sell_reviewed.empty else 0.0,
            }
        ],
        columns=NEXT_DAY_SUMMARY_COLUMNS,
    )
    return review, summary


def build_next_day_review_report(review: pd.DataFrame, summary: pd.DataFrame) -> str:
    """生成次日复盘 Markdown 报告。"""
    rec = summary.iloc[0].to_dict() if not summary.empty else {}
    strategy = str(rec.get("strategy", ""))
    date_s = str(rec.get("date", ""))
    review_date = str(rec.get("review_date", ""))
    lines = [
        "# 真实成交次日复盘 - %s - %s" % (strategy, date_s),
        "",
        "## 摘要",
        "",
        "- 交易日：%s" % date_s,
        "- 复盘价格日：%s" % review_date,
        "- 订单数：%s" % rec.get("n_orders", 0),
        "- 已复盘订单：%s" % rec.get("n_reviewed", 0),
        "- 缺少复盘价格：%s" % rec.get("n_missing_review_price", 0),
        "- 买入次日浮盈浮亏：%.2f" % float(rec.get("buy_next_day_pnl", 0.0) or 0.0),
        "- 卖出次日规避损益：%.2f" % float(rec.get("sell_avoidance_pnl", 0.0) or 0.0),
        "- 次日总影响：%.2f" % float(rec.get("total_next_day_effect", 0.0) or 0.0),
        "- 买入平均次日收益：%.4f%%" % (float(rec.get("avg_buy_next_day_return", 0.0) or 0.0) * 100.0),
        "- 卖出平均规避收益：%.4f%%" % (float(rec.get("avg_sell_avoidance_return", 0.0) or 0.0) * 100.0),
        "",
        "## 逐笔次日复盘",
        "",
        _markdown_table(review),
        "",
        "## 说明",
        "",
        "买入订单的 `next_day_effect` 是次日浮盈浮亏：`(review_price - executed_price) * executed_qty`。",
        "",
        "卖出订单的 `next_day_effect` 是次日规避损益：`(executed_price - review_price) * executed_qty`，卖出后股价下跌为正。",
        "",
        "这张表不判断策略长期好坏，只检查真实成交之后，次日价格变化对执行结果的短期影响。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_execution_feedback(
    settings: Settings,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    next_day_review: pd.DataFrame | None = None,
    next_day_summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """保存执行偏差 CSV 与 Markdown 报告。"""
    if summary.empty:
        strategy = "UNKNOWN"
        date_s = "unknown"
    else:
        rec = summary.iloc[0].to_dict()
        strategy = str(rec.get("strategy", "UNKNOWN") or "UNKNOWN")
        date_s = str(rec.get("date", "unknown") or "unknown")
    base = execution_feedback_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    detail_path = base / ("%s_execution_feedback.csv" % date_s)
    summary_path = base / ("%s_execution_summary.csv" % date_s)
    report_path = base / ("%s_execution_feedback.md" % date_s)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_execution_feedback_report(detail, summary), encoding="utf-8")
    paths = {
        "detail": detail_path,
        "summary": summary_path,
        "report": report_path,
    }
    if next_day_review is not None and next_day_summary is not None:
        review_path = base / ("%s_next_day_review.csv" % date_s)
        review_summary_path = base / ("%s_next_day_review_summary.csv" % date_s)
        review_report_path = base / ("%s_next_day_review.md" % date_s)
        next_day_review.to_csv(review_path, index=False)
        next_day_summary.to_csv(review_summary_path, index=False)
        review_report_path.write_text(build_next_day_review_report(next_day_review, next_day_summary), encoding="utf-8")
        paths.update(
            {
                "next_day_review": review_path,
                "next_day_summary": review_summary_path,
                "next_day_report": review_report_path,
            }
        )
    return paths
