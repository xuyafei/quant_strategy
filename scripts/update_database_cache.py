#!/usr/bin/env python3
"""Import local caches into SQLite and export backtest-compatible cache CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.data_feed import load_fina_indicator_from_csv, load_prices_from_csv
from live.stock_pool import load_stock_pool
from storage.database import default_database_path, initialize_database
from storage.warehouse import (
    export_factor_panel_cache,
    export_price_cache,
    upsert_factor_panel_daily,
    upsert_fina_indicator,
    upsert_prices_daily,
)


def _existing_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_file() else None


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    default_factor_panel = settings.output_dir / "cache" / "factor_panel.csv"
    parser = argparse.ArgumentParser(
        description=(
            "把本地行情/财务/因子缓存增量写入 SQLite，再导出 main.py 和纸面交易兼容的 cache CSV。"
        )
    )
    parser.add_argument("--database", type=Path, default=default_database_path(settings), help="SQLite 数据库路径")
    parser.add_argument("--prices-csv", type=Path, default=_existing_path(settings.tushare_price_cache_path), help="行情长表 CSV；默认读取 QUANT_TUSHARE_PRICE_CACHE / data/prices_tushare_cache.csv")
    parser.add_argument("--fina-csv", type=Path, default=_existing_path(settings.fina_indicator_cache_path), help="财务指标 CSV；默认读取 QUANT_TUSHARE_FINA_CACHE")
    parser.add_argument("--factor-panel-csv", type=Path, default=_existing_path(default_factor_panel), help="因子面板 CSV；默认 output/cache/factor_panel.csv")
    parser.add_argument("--stock-pool", type=Path, default=_existing_path(settings.stock_pool_path), help="可选股票池，用于导出时过滤 symbol")
    parser.add_argument("--code-col", default=settings.stock_pool_code_col, help="股票池代码列名")
    parser.add_argument("--start", default="", help="导出开始日期；默认不限制")
    parser.add_argument("--end", default="", help="导出结束日期；默认不限制")
    parser.add_argument("--output-cache-dir", type=Path, default=settings.output_dir / "cache", help="导出 cache 目录")
    parser.add_argument("--factor-version", default="v1", help="因子版本")
    parser.add_argument("--no-export", action="store_true", help="只入库，不导出 cache CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    db_path = initialize_database(args.database)
    symbols: list[str] = []
    if args.stock_pool is not None and Path(args.stock_pool).is_file():
        symbols = load_stock_pool(args.stock_pool, code_col=args.code_col)

    print("database=%s" % db_path)

    if args.prices_csv is not None and Path(args.prices_csv).is_file():
        prices = load_prices_from_csv(args.prices_csv)
        n_prices = upsert_prices_daily(db_path, prices, source=str(args.prices_csv))
        print("prices_daily upsert rows=%d source=%s" % (n_prices, args.prices_csv))
    else:
        print("prices_daily skipped: no prices csv")

    if args.fina_csv is not None and Path(args.fina_csv).is_file():
        fina = load_fina_indicator_from_csv(args.fina_csv)
        n_fina = upsert_fina_indicator(db_path, fina, source=str(args.fina_csv))
        print("fina_indicator upsert rows=%d source=%s" % (n_fina, args.fina_csv))
    else:
        print("fina_indicator skipped: no fina csv")

    if args.factor_panel_csv is not None and Path(args.factor_panel_csv).is_file():
        panel = pd.read_csv(args.factor_panel_csv)
        n_factor = upsert_factor_panel_daily(
            db_path,
            panel,
            source=str(args.factor_panel_csv),
            factor_version=args.factor_version,
        )
        print("factor_panel_daily upsert rows=%d source=%s" % (n_factor, args.factor_panel_csv))
    else:
        print("factor_panel_daily skipped: no factor panel csv")

    if not args.no_export:
        start = args.start or None
        end = args.end or None
        price_paths = export_price_cache(
            db_path,
            args.output_cache_dir,
            start=start,
            end=end,
            symbols=symbols or None,
        )
        factor_path = export_factor_panel_cache(
            db_path,
            args.output_cache_dir,
            start=start,
            end=end,
            symbols=symbols or None,
            factor_version=args.factor_version,
        )
        print("export prices_long=%s" % price_paths["prices_long"])
        print("export prices_wide_close=%s" % price_paths["prices_wide_close"])
        print("export factor_panel=%s" % factor_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
