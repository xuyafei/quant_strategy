#!/usr/bin/env python3
"""Build SQLite database quality report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.stock_pool import load_stock_pool
from storage.database import default_database_path
from storage.inspection import save_database_quality_report


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="生成 SQLite 数据库巡检日报。")
    parser.add_argument("--database", type=Path, default=default_database_path(settings), help="SQLite 数据库路径")
    parser.add_argument("--stock-pool", type=Path, default=settings.stock_pool_path, help="可选股票池，用于覆盖率检查")
    parser.add_argument("--code-col", default=settings.stock_pool_code_col, help="股票池代码列名")
    parser.add_argument("--as-of-date", default="", help="巡检日期；用于判断行情是否过旧")
    parser.add_argument("--cache-dir", type=Path, default=settings.output_dir / "cache", help="缓存目录")
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir / "database_quality", help="巡检报告输出目录")
    parser.add_argument("--max-price-stale-days", type=int, default=5, help="最新行情距离巡检日超过多少天进入 WATCH")
    parser.add_argument("--factor-min-coverage", type=float, default=0.5, help="因子有效覆盖率低于该值进入 WATCH")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    symbols: list[str] = []
    if args.stock_pool is not None and Path(args.stock_pool).is_file():
        symbols = load_stock_pool(args.stock_pool, code_col=args.code_col)

    paths = save_database_quality_report(
        args.database,
        args.output_dir,
        expected_symbols=symbols or None,
        as_of_date=args.as_of_date or None,
        cache_dir=args.cache_dir,
        max_price_stale_days=args.max_price_stale_days,
        factor_min_coverage=args.factor_min_coverage,
    )
    for name, path in paths.items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
