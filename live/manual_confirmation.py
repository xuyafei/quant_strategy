"""
小资金人工确认实盘单。

本模块把日终纸面交易结果整理成“人工可执行确认单”：
系统只给出订单建议、预检查状态和风险提示，不连接券商、不自动下单。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


CONFIRM_COLUMNS = [
    "date",
    "strategy",
    "symbol",
    "side",
    "delta_shares",
    "price",
    "estimated_amount",
    "current_shares",
    "target_shares",
    "current_weight",
    "target_weight",
    "trade_reason",
    "check_status",
    "check_reason",
    "factor_health_status",
    "factor_health_reasons",
    "freeze_as_of_date",
    "freeze_strategy",
    "freeze_stock_pool_sha256",
    "freeze_git_commit",
    "freeze_git_dirty",
    "freeze_manifest_path",
    "manual_action",
    "operator",
    "confirmed_at",
    "executed_qty",
    "executed_price",
    "execution_note",
]

FACTOR_HEALTH_SEVERITY = {"FAILED": 4, "DEGRADED": 3, "WATCH": 2, "OK": 1}


def manual_confirmation_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "live_orders" / safe


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_money(value: Any) -> str:
    try:
        return "%.2f" % float(value)
    except (TypeError, ValueError):
        return ""


def _fmt_pct(value: Any) -> str:
    try:
        return "%.2f%%" % (float(value) * 100.0)
    except (TypeError, ValueError):
        return ""


def load_factor_decay_monitor(settings: Settings, path: Path | None = None) -> pd.DataFrame:
    """读取因子失效监控表；文件不存在时返回空表。"""
    monitor_path = path or settings.output_dir / "factor_validation" / "factor_decay_monitor.csv"
    if not monitor_path.exists():
        return pd.DataFrame()
    return pd.read_csv(monitor_path)


def summarize_factor_health(factor_monitor: pd.DataFrame | None) -> tuple[str, str]:
    """把因子失效监控表压缩成整体状态和原因说明。"""
    if factor_monitor is None or factor_monitor.empty or "status" not in factor_monitor.columns:
        return "UNKNOWN", "factor_monitor_missing"

    monitor = factor_monitor.copy()
    monitor["status"] = monitor["status"].astype(str).str.upper()
    monitor["_severity_rank"] = monitor["status"].map(FACTOR_HEALTH_SEVERITY).fillna(0).astype(int)
    worst_rank = int(monitor["_severity_rank"].max()) if not monitor.empty else 0
    status = next((k for k, v in FACTOR_HEALTH_SEVERITY.items() if v == worst_rank), "UNKNOWN")
    risky = monitor[monitor["_severity_rank"] >= 2].copy()
    if risky.empty:
        return status, "all_factors_ok"

    parts: list[str] = []
    for rec in risky.sort_values(["_severity_rank", "factor"], ascending=[False, True]).to_dict("records"):
        factor = str(rec.get("factor", ""))
        st = str(rec.get("status", ""))
        reasons = str(rec.get("reasons", "") or "-")
        parts.append("%s:%s:%s" % (factor, st, reasons))
    return status, ";".join(parts)


def _factor_health_summary(factor_monitor: pd.DataFrame | None) -> tuple[str, str]:
    return summarize_factor_health(factor_monitor)


def _freeze_summary(freeze_manifest: dict[str, Any] | None) -> dict[str, str]:
    if not freeze_manifest:
        return {
            "freeze_as_of_date": "",
            "freeze_strategy": "",
            "freeze_stock_pool_sha256": "",
            "freeze_git_commit": "",
            "freeze_git_dirty": "",
            "freeze_manifest_path": "",
        }
    policy = freeze_manifest.get("live_policy", {}) or {}
    stock_pool = freeze_manifest.get("stock_pool", {}) or {}
    git = freeze_manifest.get("git", {}) or {}
    return {
        "freeze_as_of_date": str(freeze_manifest.get("as_of_date", "") or ""),
        "freeze_strategy": str(policy.get("strategy", "") or ""),
        "freeze_stock_pool_sha256": str(stock_pool.get("sha256", "") or ""),
        "freeze_git_commit": str(git.get("commit", "") or ""),
        "freeze_git_dirty": str(bool(git.get("is_dirty", False))),
        "freeze_manifest_path": str(freeze_manifest.get("manifest_path", "") or ""),
    }


def _manual_action(check_status: str, factor_health_status: str) -> str:
    if str(check_status).upper() == "BLOCK":
        return "DO_NOT_EXECUTE"
    if str(factor_health_status).upper() in {"FAILED", "DEGRADED"}:
        return "REVIEW_FACTOR_HEALTH"
    if str(factor_health_status).upper() == "WATCH":
        return "CONFIRM_WITH_CAUTION"
    return "CONFIRM_MANUALLY"


def build_manual_confirmation_sheet(
    result: dict[str, Any],
    *,
    factor_monitor: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """由日终纸面结果生成待人工确认的订单表。"""
    orders = result.get("orders", pd.DataFrame())
    checks = result.get("order_checks", pd.DataFrame())
    strategy = str(result.get("strategy", ""))
    trade_date = _date_to_str(result.get("trade_date"))
    factor_status, factor_reasons = _factor_health_summary(factor_monitor)
    freeze = _freeze_summary(result.get("freeze_manifest"))

    if orders is None or orders.empty:
        return pd.DataFrame(columns=CONFIRM_COLUMNS)

    order_cols = [
        "date",
        "symbol",
        "side",
        "delta_shares",
        "price",
        "estimated_amount",
        "current_shares",
        "target_shares",
        "current_weight",
        "target_weight",
        "trade_reason",
    ]
    sheet = orders[[c for c in order_cols if c in orders.columns]].copy()
    if "date" not in sheet.columns:
        sheet["date"] = trade_date
    sheet["date"] = sheet["date"].replace("", trade_date)

    if checks is not None and not checks.empty:
        check_cols = ["symbol", "side", "check_status", "check_reason"]
        check_frame = checks[[c for c in check_cols if c in checks.columns]].copy()
        sheet = sheet.merge(check_frame, on=["symbol", "side"], how="left")
    else:
        sheet["check_status"] = "UNKNOWN"
        sheet["check_reason"] = "missing_order_precheck"

    sheet.insert(1, "strategy", strategy)
    sheet["factor_health_status"] = factor_status
    sheet["factor_health_reasons"] = factor_reasons
    for col, value in freeze.items():
        sheet[col] = value
    sheet["check_status"] = sheet["check_status"].fillna("UNKNOWN").astype(str)
    sheet["check_reason"] = sheet["check_reason"].fillna("missing_order_precheck").astype(str)
    sheet["manual_action"] = [
        _manual_action(cs, factor_status) for cs in sheet["check_status"].tolist()
    ]
    sheet["operator"] = ""
    sheet["confirmed_at"] = ""
    sheet["executed_qty"] = ""
    sheet["executed_price"] = ""
    sheet["execution_note"] = ""

    for col in CONFIRM_COLUMNS:
        if col not in sheet.columns:
            sheet[col] = ""
    return sheet[CONFIRM_COLUMNS].reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    cols = [c for c in columns if c in frame.columns]
    if frame.empty or not cols:
        return "无\n"
    lines = [
        "| " + " | ".join(headers[: len(cols)]) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in frame.loc[:, cols].to_dict("records"):
        vals: list[str] = []
        for col in cols:
            value = rec.get(col, "")
            if col in {"estimated_amount", "price", "executed_price"}:
                vals.append(_fmt_money(value))
            elif col.endswith("weight"):
                vals.append(_fmt_pct(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def build_manual_confirmation_report(
    result: dict[str, Any],
    sheet: pd.DataFrame,
    *,
    factor_monitor: pd.DataFrame | None = None,
) -> str:
    """生成人工确认单 Markdown。"""
    strategy = str(result.get("strategy", ""))
    trade_date = _date_to_str(result.get("trade_date"))
    target_date = _date_to_str(result.get("target_date", result.get("trade_date")))
    price_date = _date_to_str(result.get("price_date", result.get("trade_date")))
    factor_status, factor_reasons = _factor_health_summary(factor_monitor)
    freeze = _freeze_summary(result.get("freeze_manifest"))
    pass_orders = sheet[sheet["check_status"].astype(str).str.upper() == "PASS"] if not sheet.empty else sheet
    blocked = sheet[sheet["check_status"].astype(str).str.upper() == "BLOCK"] if not sheet.empty else sheet
    buys = pass_orders[pass_orders["side"].astype(str).str.upper() == "BUY"] if not pass_orders.empty else pass_orders
    sells = pass_orders[pass_orders["side"].astype(str).str.upper() == "SELL"] if not pass_orders.empty else pass_orders

    lines = [
        "# 小资金人工确认实盘单 - %s - %s" % (strategy, trade_date),
        "",
        "## 使用说明",
        "",
        "本文件只用于人工复核和手动执行，不会自动连接券商，也不会自动下单。",
        "",
        "人工执行前至少确认：账户资金、可用持仓、价格、涨跌停 / 停牌状态、因子健康状态和订单金额。",
        "",
        "## 摘要",
        "",
        "- 策略：`%s`" % strategy,
        "- 运行日期：%s" % trade_date,
        "- 目标权重日期：%s" % target_date,
        "- 价格日期：%s" % price_date,
        "- 待确认订单：%d" % int(len(sheet)),
        "- 预检查通过：%d" % int(len(pass_orders)),
        "- 预检查阻断：%d" % int(len(blocked)),
        "- 通过买入金额：%s" % _fmt_money(buys["estimated_amount"].astype(float).sum() if not buys.empty else 0.0),
        "- 通过卖出金额：%s" % _fmt_money(sells["estimated_amount"].astype(float).sum() if not sells.empty else 0.0),
        "- 因子健康状态：`%s`" % factor_status,
        "- 因子健康原因：%s" % factor_reasons,
        "",
        "## 版本冻结",
        "",
        "- 冻结日期：%s" % (freeze["freeze_as_of_date"] or "未提供"),
        "- 冻结策略：`%s`" % (freeze["freeze_strategy"] or "未提供"),
        "- 股票池 SHA256：`%s`" % (freeze["freeze_stock_pool_sha256"] or "未提供"),
        "- Git Commit：`%s`" % (freeze["freeze_git_commit"] or "未提供"),
        "- Git 工作区未提交改动：%s" % (freeze["freeze_git_dirty"] or "未提供"),
        "- 冻结清单：%s" % (freeze["freeze_manifest_path"] or "未提供"),
        "",
        "## 可人工确认订单",
        "",
        _markdown_table(
            pass_orders,
            ["symbol", "side", "delta_shares", "price", "estimated_amount", "target_weight", "manual_action"],
            ["标的", "方向", "股数变化", "价格", "预估金额", "目标权重", "人工动作"],
        ),
        "## 不应执行 / 需先处理订单",
        "",
        _markdown_table(
            blocked,
            ["symbol", "side", "delta_shares", "estimated_amount", "check_reason", "manual_action"],
            ["标的", "方向", "股数变化", "预估金额", "原因", "人工动作"],
        ),
        "## 人工回填字段",
        "",
        "CSV 中预留了 `operator`、`confirmed_at`、`executed_qty`、`executed_price`、`execution_note` 字段。",
        "",
        "真实在券商终端执行后，应把实际成交数量、成交价格和备注回填，后续用于真实成交回填和纸面 / 真实账户对账。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_manual_confirmation(
    settings: Settings,
    result: dict[str, Any],
    *,
    factor_monitor: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """保存 CSV 与 Markdown 人工确认单。"""
    strategy = str(result.get("strategy", ""))
    trade_date = _date_to_str(result.get("trade_date"))
    base = manual_confirmation_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    sheet = build_manual_confirmation_sheet(result, factor_monitor=factor_monitor)
    csv_path = base / ("%s_manual_confirm.csv" % trade_date)
    md_path = base / ("%s_manual_confirm.md" % trade_date)
    sheet.to_csv(csv_path, index=False)
    md_path.write_text(
        build_manual_confirmation_report(result, sheet, factor_monitor=factor_monitor),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path}
