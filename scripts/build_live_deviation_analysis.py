#!/usr/bin/env python3
"""Generate live deviation analysis report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.daily_paper_cli import DEFAULT_STRATEGY, load_latest_target_weights
from live.deviation_analysis import build_live_deviation_analysis, save_live_deviation_analysis, summarize_deviation_status


def _optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成实盘偏差分析报告")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", required=True, help="分析日期")
    parser.add_argument("--target-weights", type=Path, default=None, help="目标权重 / 调仓日志 CSV")
    parser.add_argument("--snapshots", type=Path, default=None, help="纸面账户快照 CSV")
    parser.add_argument("--paper-positions", type=Path, default=None, help="纸面持仓 CSV")
    parser.add_argument("--broker-positions", type=Path, default=None, help="可选券商持仓 CSV")
    parser.add_argument("--prices", type=Path, default=None, help="价格缓存 CSV")
    parser.add_argument("--execution-feedback", type=Path, default=None, help="真实成交回填 CSV")
    parser.add_argument("--weight-watch-threshold", type=float, default=0.02, help="目标跟踪 WATCH 权重偏差阈值")
    parser.add_argument("--weight-block-threshold", type=float, default=0.05, help="目标跟踪 BLOCK 权重偏差阈值")
    parser.add_argument("--share-tolerance", type=float, default=0.0, help="纸面 / 券商股数差异容忍值")
    parser.add_argument("--max-unfilled-ratio", type=float, default=0.2, help="成交未完成比例 WATCH 阈值")
    parser.add_argument("--max-slippage-pct", type=float, default=0.01, help="最大绝对价格滑点 WATCH 阈值")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    safe = str(args.strategy).replace("/", "_")
    target_path = args.target_weights or settings.output_dir / "rebalance_logs" / ("%s.csv" % safe)
    snapshots_path = args.snapshots or settings.output_dir / "paper_account" / safe / "snapshots.csv"
    paper_positions_path = args.paper_positions or settings.output_dir / "paper_account" / safe / "positions.csv"
    prices_path = args.prices or settings.output_dir / "cache" / "prices_wide_close.csv"
    feedback_path = (
        args.execution_feedback
        or settings.output_dir / "execution_feedback" / safe / ("%s_execution_feedback.csv" % args.trade_date)
    )

    target_date, target_weights = load_latest_target_weights(target_path, trade_date=args.trade_date)
    snapshots = _optional_csv(snapshots_path)
    paper_positions = _optional_csv(paper_positions_path)
    broker_positions = _optional_csv(args.broker_positions)
    prices = _optional_csv(prices_path)
    execution_feedback = _optional_csv(feedback_path)

    summary, position = build_live_deviation_analysis(
        strategy=args.strategy,
        trade_date=args.trade_date,
        snapshots=snapshots,
        target_weights=target_weights,
        paper_positions=paper_positions,
        broker_positions=broker_positions,
        prices=prices,
        execution_feedback=execution_feedback,
        weight_watch_threshold=args.weight_watch_threshold,
        weight_block_threshold=args.weight_block_threshold,
        share_tolerance=args.share_tolerance,
        max_unfilled_ratio=args.max_unfilled_ratio,
        max_slippage_pct=args.max_slippage_pct,
    )
    paths = save_live_deviation_analysis(
        settings,
        strategy=args.strategy,
        trade_date=args.trade_date,
        summary=summary,
        position=position,
    )
    status, detail = summarize_deviation_status(summary)
    print("target_weight_date=%s" % target_date.strftime("%Y-%m-%d"))
    print("deviation_summary=%s" % paths["summary"])
    print("position_deviation=%s" % paths["position_deviation"])
    print("deviation_report=%s" % paths["markdown"])
    print("status=%s detail=%s" % (status, detail))
    return 0 if status != "BLOCK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
