#!/usr/bin/env python3
"""从一次完整回测产物生成适合文章引用的聚焦图，并校验策略净值可复现。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark import equal_weight_benchmark_nav
from backtest.backtest_single import run_single_backtest
from config import get_settings
from factors.preprocess import cross_sectional_zscore
from live.universe_history import load_membership_intervals, mask_wide_prices_by_membership


STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"


def _settings_from_run_config(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = {
        "backtest_start",
        "backtest_end",
        "commission_rate",
        "rebalance_freq",
        "force_final_rebalance",
        "top_k",
        "portfolio_weighting",
        "max_position_weight",
        "max_rebalance_turnover",
        "liquidity_lookback_days",
        "min_avg_volume",
        "min_avg_amount",
        "enable_trade_status_filter",
        "max_industry_weight",
        "industry_col",
        "target_volatility",
        "volatility_target_lookback_days",
        "volatility_target_min_obs",
        "min_positions",
        "min_positions_exposure",
        "optimizer_return_window",
        "optimizer_min_obs",
        "trading_days_per_year",
    }
    overrides = {key: raw[key] for key in keys if key in raw}
    return replace(get_settings(), **overrides), raw


def _drawdowns(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def build_charts(output_dir: Path, stock_pool: Path) -> dict[str, Path]:
    settings, run_config = _settings_from_run_config(output_dir / "cache" / "run_config.json")
    price_path = output_dir / "cache" / "prices_wide_adj_close.csv"
    long_path = output_dir / "cache" / "prices_long.csv"

    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    prices.index.name = "trade_date"
    long_prices = pd.read_csv(long_path, parse_dates=["trade_date"])

    composite = pd.read_csv(
        output_dir / "factor_diagnostics" / "factor_composite_scores.csv",
        parse_dates=["date"],
    ).set_index(["date", "symbol"])
    weight_log = pd.read_csv(
        output_dir / "factor_diagnostics" / "rolling_factor_weight_log.csv",
        parse_dates=["date"],
    )
    zscore = cross_sectional_zscore(composite)
    pieces: list[pd.Series] = []
    for dt, rows in weight_log.groupby("date"):
        weights = rows.set_index("factor")["final_weight"].astype(float)
        score = zscore.xs(pd.Timestamp(dt), level="date").mul(weights, axis=1).sum(axis=1)
        score.index = pd.MultiIndex.from_product(
            [[pd.Timestamp(dt)], score.index], names=["date", "symbol"]
        )
        pieces.append(score)
    fused = pd.concat(pieces).sort_index()

    strategy_nav, _ = run_single_backtest(
        STRATEGY,
        prices=prices,
        settings=settings,
        factor_values=fused,
        long_prices=long_prices,
    )
    membership_path = run_config.get("universe_membership_path")
    benchmark_prices = prices
    if membership_path:
        benchmark_prices = mask_wide_prices_by_membership(
            prices,
            load_membership_intervals(Path(membership_path)),
        )
    benchmark_nav = equal_weight_benchmark_nav(benchmark_prices, dates=strategy_nav.index)

    performance = pd.read_csv(output_dir / "performance_summary.csv").set_index("strategy")
    expected = float(performance.loc[STRATEGY, "final_nav"])
    actual = float(strategy_nav.iloc[-1])
    if abs(expected - actual) > 1e-9:
        raise RuntimeError("策略净值重建不一致: expected=%.12f actual=%.12f" % (expected, actual))

    article_dir = output_dir / "article_charts"
    article_dir.mkdir(parents=True, exist_ok=True)

    nav_path = article_dir / "strategy_vs_benchmark.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(strategy_nav.index, strategy_nav, label="Live-candidate strategy", linewidth=2.1, color="#d95f02")
    axes[0].plot(benchmark_nav.index, benchmark_nav, label="Point-in-time A50 equal weight", linewidth=2.1, color="#1b9e77")
    axes[0].axhline(1.0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("Normalized NAV")
    axes[0].set_title("Strategy vs point-in-time A50 benchmark")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    axes[1].fill_between(strategy_nav.index, _drawdowns(strategy_nav), 0.0, alpha=0.28, color="#d95f02", label="Strategy")
    axes[1].plot(benchmark_nav.index, _drawdowns(benchmark_nav), color="#1b9e77", linewidth=1.5, label="Benchmark")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(nav_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    rebalance = pd.read_csv(output_dir / "rebalance_logs" / (STRATEGY + ".csv"), parse_dates=["date"])
    latest_date = rebalance["date"].max()
    latest = rebalance[rebalance["date"] == latest_date].copy()
    latest["weight"] = pd.to_numeric(latest["weight"], errors="coerce").fillna(0.0)
    latest = latest[latest["weight"] > 1e-6].sort_values("weight", ascending=False)

    # 图中使用证券代码，避免不同运行环境缺少中文字体时生成方框字；文章正文再给出中文名。
    pd.read_csv(stock_pool)  # 同时验证文章使用的期末股票池文件可读。
    latest["label"] = latest["symbol"].astype(str)
    cash = max(1.0 - float(latest["weight"].sum()), 0.0)
    allocation = pd.concat(
        [latest[["label", "weight"]], pd.DataFrame([{"label": "Cash", "weight": cash}])],
        ignore_index=True,
    ).sort_values("weight", ascending=True)

    allocation_path = article_dir / "latest_target_allocation.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#bdbdbd" if label == "Cash" else "#4c78a8" for label in allocation["label"]]
    ax.barh(allocation["label"], allocation["weight"] * 100.0, color=colors)
    for i, value in enumerate(allocation["weight"] * 100.0):
        ax.text(value + 0.25, i, "%.1f%%" % value, va="center", fontsize=9)
    ax.set_title("Target allocation on %s" % latest_date.strftime("%Y-%m-%d"))
    ax.set_xlabel("Portfolio weight")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(allocation_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"nav": nav_path, "allocation": allocation_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成回测文章聚焦图")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stock-pool", type=Path, required=True)
    args = parser.parse_args()
    for name, path in build_charts(args.output_dir, args.stock_pool).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
