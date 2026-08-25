#!/usr/bin/env python3
"""Generate a live version freeze manifest."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.version_freeze import DEFAULT_LIVE_STRATEGY, build_freeze_manifest, save_freeze_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成实盘前版本冻结清单")
    parser.add_argument("--as-of-date", default=None, help="冻结日期，默认今天")
    parser.add_argument("--strategy", default=DEFAULT_LIVE_STRATEGY, help="冻结策略名")
    parser.add_argument("--run-time", default="09:35", help="固定日常运行时间，例如 09:35")
    parser.add_argument("--capital", type=float, default=None, help="小资金实盘观察资金；默认使用 Settings.paper_initial_cash")
    parser.add_argument("--stock-pool", type=Path, default=None, help="本次冻结使用的股票池文件；默认使用 Settings.stock_pool_path")
    parser.add_argument("--operator", default="", help="操作人或确认人")
    parser.add_argument("--notes", default="", help="备注")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录，默认 output/live_freeze/<as_of_date>")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    if args.stock_pool is not None:
        settings = replace(settings, stock_pool_path=args.stock_pool.expanduser())
    manifest = build_freeze_manifest(
        settings,
        strategy=args.strategy,
        as_of_date=args.as_of_date,
        run_time=args.run_time,
        capital=args.capital,
        operator=args.operator,
        notes=args.notes,
    )
    paths = save_freeze_outputs(settings, manifest, output_dir=args.output_dir)
    print("freeze_json=%s" % paths["json"])
    print("freeze_csv=%s" % paths["csv"])
    print("freeze_report=%s" % paths["markdown"])
    print("strategy=%s as_of_date=%s" % (manifest["live_policy"]["strategy"], manifest["as_of_date"]))
    print("stock_pool_active_count=%s" % manifest["stock_pool"].get("active_count", 0))
    print("git_dirty=%s" % manifest["git"].get("is_dirty", False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
