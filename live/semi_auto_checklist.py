"""
半自动实盘执行清单。

这个模块不自动下单，只把版本冻结、运行监控、风险总控、人工确认单、
成交回填、表现归因和偏差分析压缩成一张日常执行清单。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from live.deviation_analysis import summarize_deviation_status
from live.risk_control_report import summarize_risk_control_report
from live.run_monitor import summarize_live_run_monitor


CHECKLIST_COLUMNS = [
    "date",
    "strategy",
    "stage",
    "item",
    "status",
    "severity_rank",
    "required_before_order",
    "detail",
    "action",
    "path",
]

DECISION_COLUMNS = [
    "date",
    "strategy",
    "decision",
    "status",
    "blocking_count",
    "watch_count",
    "missing_required_count",
    "detail",
]

_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "NA": 2, "PASS": 3}


@dataclass(frozen=True)
class ChecklistItem:
    date: str
    strategy: str
    stage: str
    item: str
    status: str
    severity_rank: int
    required_before_order: bool
    detail: str
    action: str
    path: str


def semi_auto_checklist_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "semi_auto_checklist" / safe


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_status(status: Any) -> str:
    text = str(status or "NA").strip().upper()
    return text if text in _STATUS_RANK else "NA"


def _rank(status: Any) -> int:
    return int(_STATUS_RANK.get(_safe_status(status), 2))


def _action(status: str, *, required: bool) -> str:
    if status == "BLOCK":
        return "停止人工下单，先修复或重跑该环节。"
    if status == "WATCH":
        return "允许进入人工复核，不建议无脑执行。"
    if status == "NA":
        if required:
            return "必需环节缺失，先补齐再判断。"
        return "可选复盘环节缺失，记录但不阻断。"
    return "通过，继续下一步。"


def _item(
    *,
    date_s: str,
    strategy: str,
    stage: str,
    item: str,
    status: Any,
    required: bool,
    detail: str,
    path: Path | str | None = None,
) -> ChecklistItem:
    st = _safe_status(status)
    return ChecklistItem(
        date=date_s,
        strategy=strategy,
        stage=stage,
        item=item,
        status=st,
        severity_rank=_rank(st),
        required_before_order=bool(required),
        detail=str(detail),
        action=_action(st, required=required),
        path=str(path or ""),
    )


def _file_status(path: Path | str | None, *, required: bool) -> tuple[str, str]:
    if path is None:
        return ("NA" if not required else "BLOCK"), "path_not_provided"
    p = Path(path)
    if p.exists():
        return "PASS", "file_exists"
    return ("NA" if not required else "BLOCK"), "file_missing"


def _manual_confirmation_status(frame: pd.DataFrame | None) -> tuple[str, str]:
    if frame is None or frame.empty:
        return "BLOCK", "manual_confirmation_missing_or_empty"
    n_orders = int(len(frame))
    if "check_status" in frame.columns:
        blocked = int((frame["check_status"].astype(str).str.upper() == "BLOCK").sum())
        if blocked:
            return "WATCH", "n_orders=%d blocked_checks=%d" % (n_orders, blocked)
    if "manual_action" in frame.columns:
        action = frame["manual_action"].astype(str).str.upper()
        do_not = int(action.isin({"DO_NOT_EXECUTE", "SKIP", "CANCEL"}).sum())
        if do_not:
            return "WATCH", "n_orders=%d manual_skip=%d" % (n_orders, do_not)
    return "PASS", "n_orders=%d" % n_orders


def _execution_feedback_status(frame: pd.DataFrame | None) -> tuple[str, str]:
    if frame is None or frame.empty:
        return "NA", "execution_feedback_missing"
    if "execution_status" not in frame.columns:
        return "NA", "execution_status_missing"
    status = frame["execution_status"].astype(str).str.upper()
    n_orders = int(len(frame))
    bad = int(status.isin({"NOT_EXECUTED", "PARTIAL", "BLOCKED"}).sum())
    if bad:
        return "WATCH", "n_orders=%d incomplete_or_blocked=%d" % (n_orders, bad)
    return "PASS", "n_orders=%d all_executed" % n_orders


def _summary_status(frame: pd.DataFrame | None, *, status_col: str = "status") -> tuple[str, str]:
    if frame is None or frame.empty or status_col not in frame.columns:
        return "NA", "summary_missing"
    statuses = frame[status_col].astype(str).str.upper()
    if int((statuses == "BLOCK").sum()) > 0:
        return "BLOCK", "has_block=%d" % int((statuses == "BLOCK").sum())
    if int((statuses == "WATCH").sum()) > 0:
        return "WATCH", "has_watch=%d" % int((statuses == "WATCH").sum())
    if int((statuses == "PASS").sum()) > 0:
        return "PASS", "pass_rows=%d" % int((statuses == "PASS").sum())
    return "NA", "no_pass_watch_block"


def build_semi_auto_checklist(
    *,
    strategy: str,
    trade_date: Any,
    freeze_manifest_path: Path | str | None = None,
    paper_report_path: Path | str | None = None,
    run_monitor: pd.DataFrame | None = None,
    risk_control_report: pd.DataFrame | None = None,
    manual_confirmation: pd.DataFrame | None = None,
    execution_feedback: pd.DataFrame | None = None,
    performance_attribution: pd.DataFrame | None = None,
    deviation_summary: pd.DataFrame | None = None,
    require_execution_feedback: bool = False,
    require_performance_attribution: bool = False,
    require_deviation_analysis: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成半自动实盘执行清单和最终执行决策。"""
    date_s = _date_to_str(trade_date)
    items: list[ChecklistItem] = []

    st, detail = _file_status(freeze_manifest_path, required=True)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="pre_trade",
            item="version_freeze",
            status=st,
            required=True,
            detail=detail,
            path=freeze_manifest_path,
        )
    )

    status, detail = summarize_live_run_monitor(run_monitor)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="pre_trade",
            item="live_run_monitor",
            status=status,
            required=True,
            detail=detail,
        )
    )

    status, detail = summarize_risk_control_report(risk_control_report)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="pre_trade",
            item="risk_control_report",
            status=status,
            required=True,
            detail=detail,
        )
    )

    status, detail = _manual_confirmation_status(manual_confirmation)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="pre_order",
            item="manual_confirmation",
            status=status,
            required=True,
            detail=detail,
        )
    )

    st, detail = _file_status(paper_report_path, required=True)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="pre_order",
            item="paper_report",
            status=st,
            required=True,
            detail=detail,
            path=paper_report_path,
        )
    )

    status, detail = _execution_feedback_status(execution_feedback)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="post_order",
            item="execution_feedback",
            status=status,
            required=require_execution_feedback,
            detail=detail,
        )
    )

    status, detail = _summary_status(performance_attribution)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="post_order",
            item="performance_attribution",
            status=status,
            required=require_performance_attribution,
            detail=detail,
        )
    )

    if deviation_summary is None or deviation_summary.empty:
        status, detail = "NA", "deviation_summary_missing"
    else:
        status, detail = summarize_deviation_status(deviation_summary)
    items.append(
        _item(
            date_s=date_s,
            strategy=strategy,
            stage="post_order",
            item="deviation_analysis",
            status=status,
            required=require_deviation_analysis,
            detail=detail,
        )
    )

    checklist = pd.DataFrame([x.__dict__ for x in items], columns=CHECKLIST_COLUMNS)
    decision = build_semi_auto_decision(checklist)
    return checklist, decision


def build_semi_auto_decision(checklist: pd.DataFrame) -> pd.DataFrame:
    """由清单生成总决策。"""
    if checklist.empty:
        return pd.DataFrame(
            [
                {
                    "date": "",
                    "strategy": "",
                    "decision": "DO_NOT_TRADE",
                    "status": "BLOCK",
                    "blocking_count": 0,
                    "watch_count": 0,
                    "missing_required_count": 1,
                    "detail": "checklist_empty",
                }
            ],
            columns=DECISION_COLUMNS,
        )
    frame = checklist.copy()
    statuses = frame["status"].astype(str).str.upper()
    required = frame["required_before_order"].astype(bool)
    blocking_count = int((statuses == "BLOCK").sum())
    watch_count = int((statuses == "WATCH").sum())
    missing_required = int(((statuses == "NA") & required).sum())
    if blocking_count or missing_required:
        decision = "DO_NOT_TRADE"
        status = "BLOCK"
    elif watch_count:
        decision = "MANUAL_REVIEW"
        status = "WATCH"
    else:
        decision = "READY_FOR_MANUAL_ORDER"
        status = "PASS"
    focus = frame[(statuses.isin({"BLOCK", "WATCH"})) | ((statuses == "NA") & required)]
    detail = ";".join("%s=%s" % (str(r["item"]), str(r["status"])) for r in focus.to_dict("records")) or "ok"
    rec0 = frame.iloc[0]
    return pd.DataFrame(
        [
            {
                "date": str(rec0.get("date", "")),
                "strategy": str(rec0.get("strategy", "")),
                "decision": decision,
                "status": status,
                "blocking_count": blocking_count,
                "watch_count": watch_count,
                "missing_required_count": missing_required,
                "detail": detail,
            }
        ],
        columns=DECISION_COLUMNS,
    )


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 50) -> str:
    if frame.empty:
        return "无\n"
    rows = frame.head(max_rows)
    lines = [
        "| " + " | ".join(rows.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(rows.columns)) + " |",
    ]
    for rec in rows.to_dict("records"):
        values: list[str] = []
        for col in rows.columns:
            value = rec.get(col, "")
            if isinstance(value, float):
                values.append("%.4f" % value if pd.notna(value) else "NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_semi_auto_checklist_report(checklist: pd.DataFrame, decision: pd.DataFrame) -> str:
    rec = decision.iloc[0].to_dict() if not decision.empty else {}
    strategy = str(rec.get("strategy", ""))
    date_s = str(rec.get("date", ""))
    lines = [
        "# 半自动实盘执行清单 - %s - %s" % (strategy, date_s),
        "",
        "这份清单用于把实盘前后的关键产物压缩成一个执行决策。它不自动下单，只告诉人工操作者：今天是否可以进入人工下单、是否需要复核、是否必须停止。",
        "",
        "## 总决策",
        "",
        "- 决策：`%s`" % str(rec.get("decision", "")),
        "- 状态：`%s`" % str(rec.get("status", "")),
        "- 阻断项：%s" % str(rec.get("blocking_count", "")),
        "- 观察项：%s" % str(rec.get("watch_count", "")),
        "- 必需缺失项：%s" % str(rec.get("missing_required_count", "")),
        "- 说明：%s" % str(rec.get("detail", "")),
        "",
        "## 执行清单",
        "",
        _markdown_table(checklist),
        "",
        "## 使用口径",
        "",
        "`READY_FOR_MANUAL_ORDER` 表示系统产物齐全且关键检查通过，可以进入人工下单确认；`MANUAL_REVIEW` 表示没有硬阻断，但有观察项，需要人工判断是否继续；`DO_NOT_TRADE` 表示存在阻断或必需输入缺失，不应继续下单。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_semi_auto_checklist(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    checklist: pd.DataFrame,
    decision: pd.DataFrame,
) -> dict[str, Path]:
    base = semi_auto_checklist_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    date_s = _date_to_str(trade_date)
    checklist_path = base / ("%s_semi_auto_checklist.csv" % date_s)
    decision_path = base / ("%s_semi_auto_decision.csv" % date_s)
    report_path = base / ("%s_semi_auto_checklist.md" % date_s)
    checklist.to_csv(checklist_path, index=False)
    decision.to_csv(decision_path, index=False)
    report_path.write_text(build_semi_auto_checklist_report(checklist, decision), encoding="utf-8")
    return {
        "checklist": checklist_path,
        "decision": decision_path,
        "markdown": report_path,
    }
