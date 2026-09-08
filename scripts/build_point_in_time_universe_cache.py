#!/usr/bin/env python3
"""Filter a union price cache by point-in-time index constituent membership."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.stock_pool import load_stock_pool
from live.universe_history import (
    build_membership_intervals,
    filter_prices_by_membership,
    load_universe_changes,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True, help="所有历史成分股的并集行情 CSV")
    parser.add_argument("--current-pool", type=Path, required=True, help="截至 as-of 已生效的成分股快照")
    parser.add_argument("--changes", type=Path, required=True, help="成分调整事件 CSV")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--as-of", required=True, help="current-pool 对应的快照日期")
    parser.add_argument("--expected-size", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prices = pd.read_csv(args.prices)
    changes = load_universe_changes(args.changes)
    current_symbols = load_stock_pool(args.current_pool)
    membership = build_membership_intervals(
        current_symbols,
        changes,
        start=args.start,
        end=args.end,
        as_of=args.as_of,
        expected_size=args.expected_size,
    )
    filtered = filter_prices_by_membership(prices, membership)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.membership_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    membership.to_csv(args.membership_output, index=False, date_format="%Y-%m-%d")

    dates = pd.to_datetime(filtered["trade_date"])
    active_counts = filtered.groupby("trade_date")["ts_code"].nunique()
    print("rows=%d" % len(filtered))
    print("union_symbols=%d" % filtered["ts_code"].nunique())
    print("date_range=%s..%s" % (dates.min().date(), dates.max().date()))
    print("daily_active_min=%d daily_active_max=%d" % (active_counts.min(), active_counts.max()))
    print("membership_intervals=%d" % len(membership))
    print("output=%s" % args.output)
    print("membership_output=%s" % args.membership_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
