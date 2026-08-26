#!/usr/bin/env python3
"""Generate a daily small-capital live trading SOP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.daily_paper_cli import DEFAULT_STRATEGY
from live.live_sop import build_live_daily_sop, save_live_daily_sop


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成小资金实盘每日 SOP")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", required=True, help="交易日期")
    parser.add_argument("--freeze-manifest", type=Path, default=None, help="版本冻结清单 JSON")
    parser.add_argument("--broker-positions", type=Path, default=None, help="券商只读持仓 CSV")
    parser.add_argument("--include-broker-reconcile", action="store_true", help="把真实账户只读对账加入 SOP")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    sop = build_live_daily_sop(
        settings,
        strategy=args.strategy,
        trade_date=args.trade_date,
        freeze_manifest_path=args.freeze_manifest,
        broker_positions_path=args.broker_positions,
        include_broker_reconcile=args.include_broker_reconcile,
    )
    paths = save_live_daily_sop(settings, strategy=args.strategy, trade_date=args.trade_date, sop=sop)
    print("live_sop_csv=%s" % paths["csv"])
    print("live_sop_markdown=%s" % paths["markdown"])
    print("steps=%d" % len(sop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
