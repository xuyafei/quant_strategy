"""Live run monitoring report.

The monitor checks whether the daily live/paper workflow produced the expected
artifacts and whether their dates line up with the run date.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


RUN_MONITOR_COLUMNS = [
    "trade_date",
    "category",
    "item",
    "status",
    "severity_rank",
    "detail",
    "action",
    "path",
]

_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "NA": 2, "PASS": 3}


@dataclass(frozen=True)
class RunMonitorItem:
    trade_date: str
    category: str
    item: str
    status: str
    severity_rank: int
    detail: str
    action: str
    path: str


def _date_to_str(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_strategy(strategy: str) -> str:
    return str(strategy).replace("/", "_")


def _normalize_status(status: Any) -> str:
    text = str(status or "NA").strip().upper()
    return text if text in _STATUS_RANK else "NA"


def _rank(status: Any) -> int:
    return int(_STATUS_RANK.get(_normalize_status(status), 2))


def _action(status: str) -> str:
    if status == "BLOCK":
        return "暂停实盘动作，先补齐或重跑该环节。"
    if status == "WATCH":
        return "允许继续观察，但需要人工复核该环节。"
    if status == "NA":
        return "当前环节没有纳入检查或缺少输入，不能当作通过。"
    return "无需额外动作，继续记录。"


def _item(
    *,
    trade_date: str,
    category: str,
    item: str,
    status: str,
    detail: str,
    path: Path | str | None = None,
) -> RunMonitorItem:
    st = _normalize_status(status)
    return RunMonitorItem(
        trade_date=trade_date,
        category=category,
        item=item,
        status=st,
        severity_rank=_rank(st),
        detail=detail,
        action=_action(st),
        path=str(path or ""),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _check_file(
    *,
    trade_date: str,
    category: str,
    item: str,
    path: Path,
    required: bool = True,
) -> RunMonitorItem:
    if path.exists():
        return _item(
            trade_date=trade_date,
            category=category,
            item=item,
            status="PASS",
            detail="file_exists",
            path=path,
        )
    return _item(
        trade_date=trade_date,
        category=category,
        item=item,
        status="BLOCK" if required else "NA",
        detail="file_missing",
        path=path,
    )


def _check_rebalance_log(path: Path, *, trade_date: str, max_age_days: int) -> RunMonitorItem:
    if not path.exists():
        return _item(
            trade_date=trade_date,
            category="data",
            item="target_weights",
            status="BLOCK",
            detail="rebalance_log_missing",
            path=path,
        )
    frame = _read_csv(path)
    if frame.empty or "date" not in frame.columns or "symbol" not in frame.columns or "weight" not in frame.columns:
        return _item(
            trade_date=trade_date,
            category="data",
            item="target_weights",
            status="BLOCK",
            detail="rebalance_log_invalid",
            path=path,
        )
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dt = pd.Timestamp(trade_date)
    usable = df[df["date"] <= dt].copy()
    if usable.empty:
        return _item(
            trade_date=trade_date,
            category="data",
            item="target_weights",
            status="BLOCK",
            detail="no_target_weight_before_trade_date",
            path=path,
        )
    latest_date = pd.Timestamp(usable["date"].max())
    latest = usable[usable["date"] == latest_date].copy()
    if "selected" in latest.columns:
        selected = latest["selected"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        latest = latest[selected].copy()
    latest["weight"] = pd.to_numeric(latest["weight"], errors="coerce").fillna(0.0)
    n_positive = int((latest["weight"] > 0.0).sum())
    age = int((dt.normalize() - latest_date.normalize()).days)
    if n_positive <= 0:
        status = "BLOCK"
        detail = "latest_target_empty date=%s" % latest_date.strftime("%Y-%m-%d")
    elif age > int(max_age_days):
        status = "WATCH"
        detail = "target_age_days=%d n_targets=%d latest_date=%s" % (age, n_positive, latest_date.strftime("%Y-%m-%d"))
    else:
        status = "PASS"
        detail = "target_age_days=%d n_targets=%d latest_date=%s" % (age, n_positive, latest_date.strftime("%Y-%m-%d"))
    return _item(
        trade_date=trade_date,
        category="data",
        item="target_weights",
        status=status,
        detail=detail,
        path=path,
    )


def _check_price_cache(path: Path, *, trade_date: str, max_age_days: int) -> RunMonitorItem:
    if not path.exists():
        return _item(
            trade_date=trade_date,
            category="data",
            item="price_cache",
            status="BLOCK",
            detail="price_cache_missing",
            path=path,
        )
    frame = _read_csv(path)
    if frame.empty:
        return _item(
            trade_date=trade_date,
            category="data",
            item="price_cache",
            status="BLOCK",
            detail="price_cache_empty",
            path=path,
        )
    date_col = "date" if "date" in frame.columns else frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if dates.empty:
        return _item(
            trade_date=trade_date,
            category="data",
            item="price_cache",
            status="BLOCK",
            detail="price_cache_missing_date",
            path=path,
        )
    latest_date = pd.Timestamp(dates.max())
    dt = pd.Timestamp(trade_date)
    if latest_date > dt:
        status = "BLOCK"
        detail = "future_price_date=%s" % latest_date.strftime("%Y-%m-%d")
    else:
        age = int((dt.normalize() - latest_date.normalize()).days)
        status = "WATCH" if age > int(max_age_days) else "PASS"
        detail = "price_age_days=%d latest_date=%s" % (age, latest_date.strftime("%Y-%m-%d"))
    return _item(
        trade_date=trade_date,
        category="data",
        item="price_cache",
        status=status,
        detail=detail,
        path=path,
    )


def _check_snapshot(path: Path, *, trade_date: str) -> RunMonitorItem:
    if not path.exists():
        return _item(
            trade_date=trade_date,
            category="account",
            item="paper_account_snapshot",
            status="BLOCK",
            detail="snapshot_file_missing",
            path=path,
        )
    frame = _read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return _item(
            trade_date=trade_date,
            category="account",
            item="paper_account_snapshot",
            status="BLOCK",
            detail="snapshot_file_invalid",
            path=path,
        )
    dates = frame["date"].astype(str)
    if trade_date not in set(dates):
        latest = str(dates.iloc[-1]) if len(dates) else ""
        return _item(
            trade_date=trade_date,
            category="account",
            item="paper_account_snapshot",
            status="WATCH",
            detail="snapshot_for_trade_date_missing latest=%s" % latest,
            path=path,
        )
    return _item(
        trade_date=trade_date,
        category="account",
        item="paper_account_snapshot",
        status="PASS",
        detail="snapshot_exists_for_trade_date",
        path=path,
    )


def _check_risk_control(path: Path, *, trade_date: str) -> RunMonitorItem:
    if not path.exists():
        return _item(
            trade_date=trade_date,
            category="risk",
            item="risk_control_report",
            status="BLOCK",
            detail="risk_control_report_missing",
            path=path,
        )
    frame = _read_csv(path)
    if frame.empty or "status" not in frame.columns:
        return _item(
            trade_date=trade_date,
            category="risk",
            item="risk_control_report",
            status="BLOCK",
            detail="risk_control_report_invalid",
            path=path,
        )
    statuses = frame["status"].astype(str).str.upper()
    n_block = int((statuses == "BLOCK").sum())
    n_watch = int((statuses == "WATCH").sum())
    n_na = int((statuses == "NA").sum())
    if n_block:
        status = "BLOCK"
    elif n_watch:
        status = "WATCH"
    elif n_na:
        status = "NA"
    else:
        status = "PASS"
    return _item(
        trade_date=trade_date,
        category="risk",
        item="risk_control_report",
        status=status,
        detail="BLOCK=%d WATCH=%d NA=%d rows=%d" % (n_block, n_watch, n_na, len(frame)),
        path=path,
    )


def build_live_run_monitor(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    freeze_manifest_path: Path | None = None,
    require_execution_feedback: bool = False,
    require_next_day_review: bool = False,
    max_price_age_days: int = 7,
    max_target_age_days: int = 45,
) -> pd.DataFrame:
    """Build a monitoring table for one live/paper run."""
    date_s = _date_to_str(trade_date)
    safe = _safe_strategy(strategy)
    tag = pd.Timestamp(date_s).strftime("%Y%m%d")
    rows: list[RunMonitorItem] = []

    rows.append(
        _check_file(
            trade_date=date_s,
            category="version",
            item="freeze_manifest",
            path=freeze_manifest_path or (settings.output_dir / "live_freeze" / date_s / "freeze_manifest.json"),
            required=True,
        )
    )
    rows.append(
        _check_rebalance_log(
            settings.output_dir / "rebalance_logs" / ("%s.csv" % safe),
            trade_date=date_s,
            max_age_days=max_target_age_days,
        )
    )
    rows.append(
        _check_price_cache(
            settings.output_dir / "cache" / "prices_wide_close.csv",
            trade_date=date_s,
            max_age_days=max_price_age_days,
        )
    )
    rows.append(
        _check_file(
            trade_date=date_s,
            category="orders",
            item="manual_confirmation",
            path=settings.output_dir / "live_orders" / safe / ("%s_manual_confirm.csv" % date_s),
            required=True,
        )
    )
    rows.append(
        _check_snapshot(
            settings.output_dir / "paper_account" / safe / "snapshots.csv",
            trade_date=date_s,
        )
    )
    rows.append(
        _check_risk_control(
            settings.output_dir / "risk_control_reports" / safe / ("daily_risk_control_report_%s.csv" % tag),
            trade_date=date_s,
        )
    )
    rows.append(
        _check_file(
            trade_date=date_s,
            category="reports",
            item="paper_report",
            path=settings.output_dir / "paper_reports" / safe / ("%s.md" % date_s),
            required=True,
        )
    )
    rows.append(
        _check_file(
            trade_date=date_s,
            category="execution",
            item="execution_feedback",
            path=settings.output_dir / "execution_feedback" / safe / ("%s_execution_feedback.csv" % date_s),
            required=require_execution_feedback,
        )
    )
    rows.append(
        _check_file(
            trade_date=date_s,
            category="execution",
            item="next_day_review",
            path=settings.output_dir / "execution_feedback" / safe / ("%s_next_day_review.csv" % date_s),
            required=require_next_day_review,
        )
    )

    out = pd.DataFrame([r.__dict__ for r in rows], columns=RUN_MONITOR_COLUMNS)
    return out.sort_values(["severity_rank", "category", "item"]).reset_index(drop=True)


def summarize_live_run_monitor(monitor: pd.DataFrame | None) -> tuple[str, str]:
    """Return overall status and a compact detail string."""
    if monitor is None or monitor.empty or "status" not in monitor.columns:
        return "NA", "未生成运行监控表"
    status = monitor["status"].astype(str).str.upper()
    n_block = int((status == "BLOCK").sum())
    n_watch = int((status == "WATCH").sum())
    n_na = int((status == "NA").sum())
    n_pass = int((status == "PASS").sum())
    overall = "BLOCK" if n_block else "WATCH" if n_watch else "NA" if n_na else "PASS"
    focus = monitor[monitor["status"].astype(str).str.upper().isin({"BLOCK", "WATCH", "NA"})]
    detail = "；".join(
        "%s.%s=%s" % (row["category"], row["item"], row["status"])
        for row in focus.head(8).to_dict("records")
    )
    if not detail:
        detail = "全部检查通过"
    return overall, "BLOCK=%d WATCH=%d NA=%d PASS=%d；%s" % (n_block, n_watch, n_na, n_pass, detail)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无\n"
    lines = [
        "| " + " | ".join(frame.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for rec in frame.to_dict("records"):
        lines.append("| " + " | ".join(str(rec.get(col, "")) for col in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def build_live_run_monitor_report(monitor: pd.DataFrame) -> str:
    """Render the monitor as Markdown."""
    status, detail = summarize_live_run_monitor(monitor)
    date_s = str(monitor["trade_date"].iloc[0]) if not monitor.empty else ""
    lines = [
        "# 实盘运行监控日报 - %s" % date_s,
        "",
        "## 摘要",
        "",
        "- 总状态：`%s`" % status,
        "- 明细：%s" % detail,
        "",
        "## 监控明细",
        "",
        _markdown_table(
            monitor[
                [
                    "category",
                    "item",
                    "status",
                    "detail",
                    "action",
                    "path",
                ]
            ]
        ),
        "",
        "## 状态说明",
        "",
        "- `PASS`：该环节已生成或日期正常。",
        "- `WATCH`：可以继续观察，但需要人工复核。",
        "- `NA`：该环节未纳入必查或缺少输入，不能当作通过。",
        "- `BLOCK`：关键环节缺失或异常，暂停实盘动作。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_live_run_monitor(
    settings: Settings,
    monitor: pd.DataFrame,
    *,
    strategy: str,
    trade_date: Any,
) -> dict[str, Path]:
    """Save monitor CSV and Markdown report."""
    safe = _safe_strategy(strategy)
    date_s = _date_to_str(trade_date)
    base = settings.output_dir / "live_run_monitor" / safe
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / ("%s_run_monitor.csv" % date_s)
    md_path = base / ("%s_run_monitor.md" % date_s)
    monitor.to_csv(csv_path, index=False)
    md_path.write_text(build_live_run_monitor_report(monitor), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}
