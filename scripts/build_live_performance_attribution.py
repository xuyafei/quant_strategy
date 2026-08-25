#!/usr/bin/env python3
"""Generate live / paper account performance attribution."""
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
from live.performance_attribution import (
    build_performance_attribution,
    save_performance_attribution,
)


def _optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成实盘 / 纸面交易表现归因报告")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", required=True, help="归因日期")
    parser.add_argument("--snapshots", type=Path, default=None, help="账户快照 CSV")
    parser.add_argument("--positions", type=Path, default=None, help="当前持仓 CSV")
    parser.add_argument("--prices", type=Path, default=None, help="价格缓存 CSV，支持宽表或长表")
    parser.add_argument("--execution-feedback", type=Path, default=None, help="真实成交回填逐笔 CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    safe = str(args.strategy).replace("/", "_")
    snapshots_path = args.snapshots or settings.output_dir / "paper_account" / safe / "snapshots.csv"
    positions_path = args.positions or settings.output_dir / "paper_account" / safe / "positions.csv"
    prices_path = args.prices or settings.output_dir / "cache" / "prices_wide_close.csv"
    feedback_path = (
        args.execution_feedback
        or settings.output_dir / "execution_feedback" / safe / ("%s_execution_feedback.csv" % args.trade_date)
    )

    if not snapshots_path.exists():
        raise FileNotFoundError("未找到账户快照: %s" % snapshots_path)
    snapshots = pd.read_csv(snapshots_path)
    positions = _optional_csv(positions_path)
    prices = _optional_csv(prices_path)
    execution_feedback = _optional_csv(feedback_path)

    summary, stock = build_performance_attribution(
        snapshots,
        strategy=args.strategy,
        trade_date=args.trade_date,
        prices=prices,
        positions=positions,
        execution_feedback=execution_feedback,
    )
    paths = save_performance_attribution(
        settings,
        strategy=args.strategy,
        trade_date=args.trade_date,
        summary=summary,
        stock=stock,
    )
    print("performance_attribution_summary=%s" % paths["summary"])
    print("stock_contribution=%s" % paths["stock_contribution"])
    print("performance_attribution_report=%s" % paths["markdown"])
    print("status=%s detail=%s" % (summary["status"].iloc[0], summary["detail"].iloc[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
