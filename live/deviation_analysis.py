"""
实盘偏差分析。

本模块把目标持仓、纸面持仓、可选真实券商持仓和真实成交回填放到同一张
检查表里，观察策略从“目标”走到“账户”过程中是否出现偏离。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


POSITION_DEVIATION_COLUMNS = [
    "date",
    "strategy",
    "symbol",
    "target_weight",
    "paper_weight",
    "weight_diff",
    "abs_weight_diff",
    "paper_shares",
    "broker_shares",
    "share_diff",
    "price",
    "status",
    "detail",
]

SUMMARY_COLUMNS = [
    "date",
    "strategy",
    "module",
    "status",
    "metric",
    "value",
    "threshold",
    "detail",
]


def deviation_analysis_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "live_deviation" / safe


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


def _normalize_target_weights(target_weights: pd.DataFrame | pd.Series | dict[str, float] | None) -> pd.Series:
    if target_weights is None:
        return pd.Series(dtype=float)
    if isinstance(target_weights, pd.Series):
        out = target_weights.astype(float).copy()
        out.index = out.index.astype(str)
        return out.groupby(level=0).sum().sort_index()
    if isinstance(target_weights, dict):
        out = pd.Series({str(k): float(v) for k, v in target_weights.items()}, dtype=float)
        return out.groupby(level=0).sum().sort_index()
    if target_weights.empty:
        return pd.Series(dtype=float)
    frame = target_weights.copy()
    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code" if "ts_code" in frame.columns else None
    weight_col = "weight" if "weight" in frame.columns else "target_weight" if "target_weight" in frame.columns else None
    if symbol_col is None or weight_col is None:
        raise ValueError("目标权重表须包含 symbol/ts_code 与 weight/target_weight 列")
    if "selected" in frame.columns:
        selected = frame["selected"]
        if selected.dtype != bool:
            selected = selected.astype(str).str.lower().isin({"1", "true", "yes", "y"})
        frame = frame[selected]
    out = pd.Series(
        pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).to_numpy(),
        index=frame[symbol_col].astype(str),
        dtype=float,
    )
    return out.groupby(level=0).sum().sort_index()


def _normalize_positions(positions: pd.DataFrame | pd.Series | dict[str, float] | None, *, prefix: str) -> pd.DataFrame:
    cols = ["symbol", "%s_shares" % prefix]
    if positions is None:
        return pd.DataFrame(columns=cols)
    if isinstance(positions, pd.Series):
        out = pd.DataFrame({"symbol": positions.index.astype(str), "%s_shares" % prefix: positions.astype(float).to_numpy()})
    elif isinstance(positions, dict):
        out = pd.DataFrame({"symbol": [str(k) for k in positions], "%s_shares" % prefix: [float(v) for v in positions.values()]})
    else:
        if positions.empty:
            return pd.DataFrame(columns=cols)
        symbol_col = "symbol" if "symbol" in positions.columns else "ts_code" if "ts_code" in positions.columns else None
        if symbol_col is None or "shares" not in positions.columns:
            raise ValueError("持仓表须包含 symbol/ts_code 与 shares 列")
        out = pd.DataFrame(
            {
                "symbol": positions[symbol_col].astype(str),
                "%s_shares" % prefix: pd.to_numeric(positions["shares"], errors="coerce").fillna(0.0),
            }
        )
    out = out[out["%s_shares" % prefix] > 0.0]
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out.groupby("symbol", as_index=False)["%s_shares" % prefix].sum().sort_values("symbol").reset_index(drop=True)


def _normalize_prices(prices: pd.DataFrame | None) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code" if "ts_code" in frame.columns else None
    if symbol_col is not None:
        value_col = None
        for candidate in ["close", "price", "adj_close"]:
            if candidate in frame.columns:
                value_col = candidate
                break
        if value_col is None:
            raise ValueError("长表价格数据须包含 close/price/adj_close 之一")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        wide = frame.pivot_table(index="date", columns=symbol_col, values=value_col, aggfunc="last")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        wide = frame.set_index("date").apply(pd.to_numeric, errors="coerce")
    wide = wide[wide.index.notna()].sort_index()
    wide.columns = wide.columns.astype(str)
    return wide


def _price_row_on_or_before(prices: pd.DataFrame, trade_date: Any) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    dt = pd.Timestamp(trade_date)
    subset = prices[prices.index <= dt]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[-1].astype(float)


def _latest_total_asset(snapshots: pd.DataFrame | None, trade_date: Any) -> float:
    if snapshots is None or snapshots.empty or "total_asset" not in snapshots.columns:
        return 0.0
    frame = snapshots.copy()
    if "date" not in frame.columns:
        return 0.0
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["total_asset"] = pd.to_numeric(frame["total_asset"], errors="coerce")
    frame = frame[(frame["date"].notna()) & (frame["date"] <= pd.Timestamp(trade_date))]
    if frame.empty:
        return 0.0
    return _num(frame.sort_values("date").iloc[-1].get("total_asset", 0.0))


def build_position_deviation(
    *,
    strategy: str,
    trade_date: Any,
    target_weights: pd.DataFrame | pd.Series | dict[str, float] | None,
    paper_positions: pd.DataFrame | pd.Series | dict[str, float] | None,
    prices: pd.DataFrame | None,
    total_asset: float,
    broker_positions: pd.DataFrame | pd.Series | dict[str, float] | None = None,
    weight_watch_threshold: float = 0.02,
    weight_block_threshold: float = 0.05,
    share_tolerance: float = 0.0,
) -> pd.DataFrame:
    """生成目标权重、纸面持仓和可选真实券商持仓的逐股票偏差表。"""
    target = _normalize_target_weights(target_weights)
    paper = _normalize_positions(paper_positions, prefix="paper")
    broker = _normalize_positions(broker_positions, prefix="broker")
    price_row = _price_row_on_or_before(_normalize_prices(prices), trade_date)
    symbols = sorted(set(target.index.astype(str)).union(set(paper["symbol"].astype(str))).union(set(broker["symbol"].astype(str))))
    if not symbols:
        return pd.DataFrame(columns=POSITION_DEVIATION_COLUMNS)
    frame = pd.DataFrame({"symbol": symbols})
    frame["target_weight"] = frame["symbol"].map(target).fillna(0.0).astype(float)
    frame = frame.merge(paper, on="symbol", how="left").merge(broker, on="symbol", how="left")
    frame["paper_shares"] = pd.to_numeric(frame.get("paper_shares", 0.0), errors="coerce").fillna(0.0)
    frame["broker_shares"] = pd.to_numeric(frame.get("broker_shares", 0.0), errors="coerce").fillna(0.0)
    frame["price"] = frame["symbol"].map(price_row).astype(float)
    frame["paper_value"] = frame["paper_shares"] * frame["price"]
    frame["paper_weight"] = frame["paper_value"] / float(total_asset) if total_asset > 0.0 else 0.0
    frame["paper_weight"] = pd.to_numeric(frame["paper_weight"], errors="coerce").fillna(0.0)
    frame["weight_diff"] = frame["paper_weight"] - frame["target_weight"]
    frame["abs_weight_diff"] = frame["weight_diff"].abs()
    frame["share_diff"] = frame["broker_shares"] - frame["paper_shares"]

    def _status(row: pd.Series) -> tuple[str, str]:
        details: list[str] = []
        status = "PASS"
        if not (_num(row.get("price", float("nan")), float("nan")) > 0.0):
            return "NA", "price_missing"
        if float(row["abs_weight_diff"]) > float(weight_block_threshold):
            status = "BLOCK"
            details.append("target_tracking_block")
        elif float(row["abs_weight_diff"]) > float(weight_watch_threshold):
            status = "WATCH"
            details.append("target_tracking_watch")
        if broker_positions is not None and abs(float(row["share_diff"])) > float(share_tolerance):
            if status == "PASS":
                status = "WATCH"
            details.append("broker_share_mismatch")
        if not details:
            details.append("ok")
        return status, ";".join(details)

    status_detail = frame.apply(_status, axis=1)
    frame["status"] = [x[0] for x in status_detail]
    frame["detail"] = [x[1] for x in status_detail]
    out = frame.copy()
    out["date"] = _date_to_str(trade_date)
    out["strategy"] = strategy
    return out[POSITION_DEVIATION_COLUMNS].sort_values(["status", "abs_weight_diff", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def _status_from_position_deviation(position_deviation: pd.DataFrame) -> str:
    if position_deviation.empty:
        return "NA"
    statuses = set(position_deviation["status"].astype(str))
    if "BLOCK" in statuses:
        return "BLOCK"
    if "WATCH" in statuses:
        return "WATCH"
    if statuses == {"PASS"}:
        return "PASS"
    return "NA"


def _execution_summary(
    execution_feedback: pd.DataFrame | None,
    *,
    strategy: str,
    trade_date: Any,
    max_unfilled_ratio: float,
    max_slippage_pct: float,
) -> list[dict[str, Any]]:
    if execution_feedback is None or execution_feedback.empty:
        return [
            {
                "date": _date_to_str(trade_date),
                "strategy": strategy,
                "module": "execution_feedback",
                "status": "NA",
                "metric": "n_orders",
                "value": 0.0,
                "threshold": 0.0,
                "detail": "execution_feedback_missing",
            }
        ]
    frame = execution_feedback.copy()
    if "execution_status" in frame.columns:
        status_s = frame["execution_status"].astype(str).str.upper()
        unfilled = (~status_s.isin({"FILLED", "OVERFILLED"})).sum()
    else:
        unfilled = 0
    n_orders = len(frame)
    unfilled_ratio = float(unfilled) / float(n_orders) if n_orders else 0.0
    slip = pd.to_numeric(frame.get("price_slippage_pct", pd.Series(dtype=float)), errors="coerce").abs().dropna()
    max_slip = float(slip.max()) if not slip.empty else 0.0
    status = "PASS"
    detail: list[str] = []
    if unfilled_ratio > float(max_unfilled_ratio):
        status = "WATCH"
        detail.append("unfilled_ratio_high")
    if max_slip > float(max_slippage_pct):
        status = "WATCH"
        detail.append("slippage_high")
    if not detail:
        detail.append("ok")
    return [
        {
            "date": _date_to_str(trade_date),
            "strategy": strategy,
            "module": "execution_feedback",
            "status": status,
            "metric": "unfilled_ratio",
            "value": unfilled_ratio,
            "threshold": float(max_unfilled_ratio),
            "detail": ";".join(detail),
        },
        {
            "date": _date_to_str(trade_date),
            "strategy": strategy,
            "module": "execution_feedback",
            "status": status,
            "metric": "max_abs_price_slippage_pct",
            "value": max_slip,
            "threshold": float(max_slippage_pct),
            "detail": ";".join(detail),
        },
    ]


def build_live_deviation_analysis(
    *,
    strategy: str,
    trade_date: Any,
    snapshots: pd.DataFrame | None,
    target_weights: pd.DataFrame | pd.Series | dict[str, float] | None,
    paper_positions: pd.DataFrame | pd.Series | dict[str, float] | None,
    prices: pd.DataFrame | None,
    broker_positions: pd.DataFrame | pd.Series | dict[str, float] | None = None,
    execution_feedback: pd.DataFrame | None = None,
    weight_watch_threshold: float = 0.02,
    weight_block_threshold: float = 0.05,
    share_tolerance: float = 0.0,
    max_unfilled_ratio: float = 0.2,
    max_slippage_pct: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成实盘偏差汇总表和逐股票偏差表。"""
    total_asset = _latest_total_asset(snapshots, trade_date)
    position = build_position_deviation(
        strategy=strategy,
        trade_date=trade_date,
        target_weights=target_weights,
        paper_positions=paper_positions,
        broker_positions=broker_positions,
        prices=prices,
        total_asset=total_asset,
        weight_watch_threshold=weight_watch_threshold,
        weight_block_threshold=weight_block_threshold,
        share_tolerance=share_tolerance,
    )
    rows: list[dict[str, Any]] = []
    max_abs_weight_diff = float(position["abs_weight_diff"].max()) if not position.empty else 0.0
    rows.append(
        {
            "date": _date_to_str(trade_date),
            "strategy": strategy,
            "module": "target_tracking",
            "status": _status_from_position_deviation(position),
            "metric": "max_abs_weight_diff",
            "value": max_abs_weight_diff,
            "threshold": float(weight_block_threshold),
            "detail": "n_deviation=%d" % int((position["status"] != "PASS").sum()) if not position.empty else "position_empty",
        }
    )
    if broker_positions is None:
        rows.append(
            {
                "date": _date_to_str(trade_date),
                "strategy": strategy,
                "module": "broker_position_sync",
                "status": "NA",
                "metric": "max_abs_share_diff",
                "value": 0.0,
                "threshold": float(share_tolerance),
                "detail": "broker_positions_missing",
            }
        )
    else:
        max_abs_share_diff = float(position["share_diff"].abs().max()) if not position.empty else 0.0
        broker_status = "PASS" if max_abs_share_diff <= float(share_tolerance) else "WATCH"
        rows.append(
            {
                "date": _date_to_str(trade_date),
                "strategy": strategy,
                "module": "broker_position_sync",
                "status": broker_status,
                "metric": "max_abs_share_diff",
                "value": max_abs_share_diff,
                "threshold": float(share_tolerance),
                "detail": "ok" if broker_status == "PASS" else "broker_share_mismatch",
            }
        )
    rows.extend(
        _execution_summary(
            execution_feedback,
            strategy=strategy,
            trade_date=trade_date,
            max_unfilled_ratio=max_unfilled_ratio,
            max_slippage_pct=max_slippage_pct,
        )
    )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary, position


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
                vals.append("%.4f" % value if pd.notna(value) else "NA")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def summarize_deviation_status(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "NA", "summary_empty"
    statuses = set(summary["status"].astype(str))
    if "BLOCK" in statuses:
        final = "BLOCK"
    elif "WATCH" in statuses:
        final = "WATCH"
    elif "PASS" in statuses:
        final = "PASS"
    else:
        final = "NA"
    detail = ";".join(
        "%s.%s=%s" % (rec["module"], rec["metric"], rec["status"])
        for rec in summary.to_dict("records")
        if str(rec.get("status", "")) not in {"PASS", "NA"}
    )
    return final, detail or "ok"


def build_live_deviation_report(summary: pd.DataFrame, position: pd.DataFrame) -> str:
    final_status, final_detail = summarize_deviation_status(summary)
    rec = summary.iloc[0].to_dict() if not summary.empty else {}
    strategy = str(rec.get("strategy", ""))
    date_s = str(rec.get("date", ""))
    lines = [
        "# 实盘偏差分析 - %s - %s" % (strategy, date_s),
        "",
        "这份报告用于观察目标持仓、纸面账户、可选真实券商持仓和真实成交之间是否出现偏离。它不生成订单、不修改账户，只负责日终诊断。",
        "",
        "## 总体状态",
        "",
        "- 状态：`%s`" % final_status,
        "- 说明：%s" % final_detail,
        "",
        "## 偏差汇总",
        "",
        _markdown_table(summary),
        "",
        "## 逐股票偏差",
        "",
    ]
    if position.empty:
        lines.append("当前没有可展示的逐股票偏差。")
    else:
        display = position[
            [
                "symbol",
                "target_weight",
                "paper_weight",
                "weight_diff",
                "paper_shares",
                "broker_shares",
                "share_diff",
                "status",
                "detail",
            ]
        ].copy()
        lines.append(_markdown_table(display, max_rows=50))
    lines.extend(
        [
            "",
            "## 怎么使用",
            "",
            "偏差分析适合在每天纸面交易或小资金人工实盘之后运行。`target_tracking` 关注纸面持仓是否贴近目标权重；`broker_position_sync` 关注纸面持仓和真实券商持仓是否一致；`execution_feedback` 关注真实成交是否大量未成交或滑点过大。",
            "",
            "如果状态是 `PASS`，说明偏差在阈值内；如果是 `WATCH`，需要人工复核；如果是 `BLOCK`，说明目标跟踪偏离已经明显，不宜继续机械执行下一步。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_live_deviation_analysis(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    summary: pd.DataFrame,
    position: pd.DataFrame,
) -> dict[str, Path]:
    base = deviation_analysis_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    date_s = _date_to_str(trade_date)
    summary_path = base / ("%s_deviation_summary.csv" % date_s)
    position_path = base / ("%s_position_deviation.csv" % date_s)
    report_path = base / ("%s_deviation_report.md" % date_s)
    summary.to_csv(summary_path, index=False)
    position.to_csv(position_path, index=False)
    report_path.write_text(build_live_deviation_report(summary, position), encoding="utf-8")
    return {"summary": summary_path, "position_deviation": position_path, "markdown": report_path}
