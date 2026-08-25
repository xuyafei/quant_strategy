#!/usr/bin/env python3
"""Generate a live run monitoring report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.daily_paper_cli import DEFAULT_STRATEGY
from live.run_monitor import build_live_run_monitor, save_live_run_monitor, summarize_live_run_monitor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成实盘运行监控日报")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", required=True, help="监控日期")
    parser.add_argument("--freeze-manifest", type=Path, default=None, help="实盘冻结清单 JSON")
    parser.add_argument("--require-execution-feedback", action="store_true", help="要求当天已生成真实成交回填")
    parser.add_argument("--require-next-day-review", action="store_true", help="要求当天已生成次日复盘")
    parser.add_argument("--max-price-age-days", type=int, default=7, help="价格缓存最大允许滞后天数")
    parser.add_argument("--max-target-age-days", type=int, default=45, help="目标权重最大允许滞后天数")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    monitor = build_live_run_monitor(
        settings,
        strategy=args.strategy,
        trade_date=args.trade_date,
        freeze_manifest_path=args.freeze_manifest,
        require_execution_feedback=args.require_execution_feedback,
        require_next_day_review=args.require_next_day_review,
        max_price_age_days=args.max_price_age_days,
        max_target_age_days=args.max_target_age_days,
    )
    paths = save_live_run_monitor(settings, monitor, strategy=args.strategy, trade_date=args.trade_date)
    status, detail = summarize_live_run_monitor(monitor)
    print("live_run_monitor=%s" % paths["csv"])
    print("live_run_monitor_report=%s" % paths["markdown"])
    print("status=%s detail=%s" % (status, detail))
    return 0 if status != "BLOCK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
