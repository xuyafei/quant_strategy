#!/usr/bin/env python3
"""Fetch Tushare daily prices with adj_factor and adj_close."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.data_feed import fetch_daily_panel
from live.stock_pool import load_stock_pool


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-pool", type=Path, default=settings.stock_pool_path)
    parser.add_argument("--code-col", default=settings.stock_pool_code_col)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjustment-mode", choices=["qfq", "hfq"], default=settings.adjustment_mode)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    symbols = load_stock_pool(args.stock_pool, code_col=args.code_col)
    prices = fetch_daily_panel(
        symbols,
        args.start,
        args.end,
        include_adj_factor=True,
        adjustment_mode=args.adjustment_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    coverage = float(prices["adj_close"].notna().mean()) if "adj_close" in prices.columns else 0.0
    print("symbols=%d" % len(symbols))
    print("rows=%d" % len(prices))
    print("start=%s" % prices["trade_date"].min().strftime("%Y-%m-%d"))
    print("end=%s" % prices["trade_date"].max().strftime("%Y-%m-%d"))
    print("adj_close_coverage=%.2f%%" % (coverage * 100.0))
    print("output=%s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
