#!/usr/bin/env python3
"""Generate semi-auto live execution checklist."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.daily_paper_cli import DEFAULT_STRATEGY
from live.semi_auto_checklist import build_semi_auto_checklist, save_semi_auto_checklist


def _optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成半自动实盘执行清单")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", required=True, help="执行日期")
    parser.add_argument("--freeze-manifest", type=Path, default=None, help="冻结清单 JSON")
    parser.add_argument("--paper-report", type=Path, default=None, help="纸面交易日报 Markdown")
    parser.add_argument("--run-monitor", type=Path, default=None, help="实盘运行监控 CSV")
    parser.add_argument("--risk-control-report", type=Path, default=None, help="风险总控日报 CSV")
    parser.add_argument("--manual-confirm", type=Path, default=None, help="人工确认单 CSV")
    parser.add_argument("--execution-feedback", type=Path, default=None, help="真实成交回填 CSV")
    parser.add_argument("--performance-attribution", type=Path, default=None, help="表现归因汇总 CSV")
    parser.add_argument("--deviation-summary", type=Path, default=None, help="偏差分析汇总 CSV")
    parser.add_argument("--require-execution-feedback", action="store_true", help="人工下单前要求已有成交回填")
    parser.add_argument("--require-performance-attribution", action="store_true", help="人工下单前要求已有表现归因")
    parser.add_argument("--no-require-deviation-analysis", action="store_true", help="不把偏差分析作为必需项")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    safe = str(args.strategy).replace("/", "_")
    date_s = pd.Timestamp(args.trade_date).strftime("%Y-%m-%d")
    tag = pd.Timestamp(args.trade_date).strftime("%Y%m%d")

    freeze_manifest = args.freeze_manifest or settings.output_dir / "live_freeze" / date_s / "freeze_manifest.json"
    paper_report = args.paper_report or settings.output_dir / "paper_reports" / safe / ("%s.md" % date_s)
    run_monitor_path = args.run_monitor or settings.output_dir / "live_run_monitor" / safe / ("%s_run_monitor.csv" % date_s)
    risk_control_path = (
        args.risk_control_report
        or settings.output_dir / "risk_control_reports" / safe / ("daily_risk_control_report_%s.csv" % tag)
    )
    manual_confirm_path = args.manual_confirm or settings.output_dir / "live_orders" / safe / ("%s_manual_confirm.csv" % date_s)
    execution_feedback_path = (
        args.execution_feedback
        or settings.output_dir / "execution_feedback" / safe / ("%s_execution_feedback.csv" % date_s)
    )
    attribution_path = (
        args.performance_attribution
        or settings.output_dir / "performance_attribution" / safe / ("%s_performance_attribution_summary.csv" % date_s)
    )
    deviation_path = args.deviation_summary or settings.output_dir / "live_deviation" / safe / ("%s_deviation_summary.csv" % date_s)

    checklist, decision = build_semi_auto_checklist(
        strategy=args.strategy,
        trade_date=date_s,
        freeze_manifest_path=freeze_manifest,
        paper_report_path=paper_report,
        run_monitor=_optional_csv(run_monitor_path),
        risk_control_report=_optional_csv(risk_control_path),
        manual_confirmation=_optional_csv(manual_confirm_path),
        execution_feedback=_optional_csv(execution_feedback_path),
        performance_attribution=_optional_csv(attribution_path),
        deviation_summary=_optional_csv(deviation_path),
        require_execution_feedback=args.require_execution_feedback,
        require_performance_attribution=args.require_performance_attribution,
        require_deviation_analysis=not args.no_require_deviation_analysis,
    )
    paths = save_semi_auto_checklist(
        settings,
        strategy=args.strategy,
        trade_date=date_s,
        checklist=checklist,
        decision=decision,
    )
    rec = decision.iloc[0].to_dict()
    print("semi_auto_checklist=%s" % paths["checklist"])
    print("semi_auto_decision=%s" % paths["decision"])
    print("semi_auto_report=%s" % paths["markdown"])
    print("decision=%s status=%s detail=%s" % (rec["decision"], rec["status"], rec["detail"]))
    return 0 if str(rec["decision"]) != "DO_NOT_TRADE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
