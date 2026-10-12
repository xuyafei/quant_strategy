#!/usr/bin/env python3
"""Prepare, execute or inspect the six-strategy forward paper workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.six_strategy_forward import (
    FORWARD_MANIFEST,
    execute_six_strategy_paper,
    mark_six_strategy_accounts,
    prepare_six_strategy_forward_paper,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="冻结六策略信号并初始化六个纸面账户")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--paper-protocol", type=Path, default=None)
    prepare.add_argument("--signal-date", required=True)
    prepare.add_argument("--execution-date", required=True)
    prepare.add_argument("--initial-cash", type=float, default=1_000_000.0)

    execute = sub.add_parser("execute", help="按冻结日期执行六个模拟券商账户")
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--trade-date", required=True)
    execute.add_argument("--trade-status", type=Path, default=None)
    execute.add_argument("--allow-rerun", action="store_true")

    mark = sub.add_parser("mark", help="仅按当日收盘价更新六账户净值，不生成订单")
    mark.add_argument("--data-dir", type=Path, required=True)
    mark.add_argument("--output", type=Path, required=True)
    mark.add_argument("--trade-date", required=True)
    mark.add_argument("--allow-rerun", action="store_true")

    status = sub.add_parser("status", help="查看六策略纸面交易清单状态")
    status.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command != "status":
        raise RuntimeError(
            "旧入口只保留历史回放状态查看。严格前向流程请使用 scripts/run_forward_ledger.py "
            "和 scripts/run_forward_daily.py；不可将旧账户重标为新前向账户。"
        )
    if args.command == "prepare":
        paths = prepare_six_strategy_forward_paper(
            args.data_dir,
            args.output,
            args.protocol,
            signal_date=args.signal_date,
            execution_date=args.execution_date,
            initial_cash=args.initial_cash,
            paper_protocol_path=args.paper_protocol,
        )
        for name, path in sorted(paths.items()):
            print("%s=%s" % (name, path))
        return 0
    if args.command == "execute":
        summary = execute_six_strategy_paper(
            args.data_dir,
            args.output,
            trade_date=args.trade_date,
            trade_status_path=args.trade_status,
            allow_rerun=args.allow_rerun,
        )
        print(summary.to_string(index=False))
        return 0
    if args.command == "mark":
        summary = mark_six_strategy_accounts(
            args.data_dir,
            args.output,
            trade_date=args.trade_date,
            allow_rerun=args.allow_rerun,
        )
        print(summary.to_string(index=False))
        return 0
    manifest_path = args.output / FORWARD_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
