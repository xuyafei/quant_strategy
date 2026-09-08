#!/usr/bin/env python3
"""Generate the comparison charts used by the full-length article 111."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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


def _rebuild_nav(output_dir: Path) -> tuple[pd.Series, pd.Series]:
    settings, run_config = _settings_from_run_config(output_dir / "cache" / "run_config.json")
    prices = pd.read_csv(
        output_dir / "cache" / "prices_wide_adj_close.csv",
        index_col=0,
        parse_dates=True,
    )
    long_prices = pd.read_csv(output_dir / "cache" / "prices_long.csv", parse_dates=["trade_date"])
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
    if abs(float(strategy_nav.iloc[-1]) - expected) > 1e-9:
        raise RuntimeError(f"NAV reconstruction mismatch for {output_dir}")
    return strategy_nav, benchmark_nav


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def build_chart(output_dir: Path) -> Path:
    source = output_dir / "factor_validation" / "multi_horizon_summary.csv"
    frame = pd.read_csv(source)
    required = {"factor", "ic_1d", "ic_5d", "ic_20d", "ic_next_rebalance"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    frame = frame[frame["factor"] != "ANNOUNCEMENT_EVENT_SCORE"].copy()
    columns = ["ic_1d", "ic_5d", "ic_20d", "ic_next_rebalance"]
    frame["mean_ic"] = frame[columns].mean(axis=1, skipna=True)
    frame = frame.sort_values("mean_ic", ascending=True)
    values = frame[columns].to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), 0.01)

    target_dir = output_dir / "article_charts"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "multi_horizon_factor_ic.png"
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), ["1D", "5D", "20D", "Next rebalance"])
    ax.set_yticks(range(len(frame)), frame["factor"].tolist())
    ax.set_title("Out-of-sample IC across decision horizons")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "NA" if not np.isfinite(value) else f"{value:.3f}"
            color = "white" if np.isfinite(value) and abs(value) > limit * 0.55 else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=8.5, color=color)
    fig.colorbar(image, ax=ax, label="Spearman IC", fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def build_comparison_charts(before_dir: Path, after_dir: Path) -> dict[str, Path]:
    target_dir = after_dir / "article_charts"
    target_dir.mkdir(parents=True, exist_ok=True)
    targets: dict[str, Path] = {}

    before_nav, before_benchmark = _rebuild_nav(before_dir)
    after_nav, after_benchmark = _rebuild_nav(after_dir)
    if not before_benchmark.equals(after_benchmark):
        raise RuntimeError("Before/after benchmark series are not identical")
    target = target_dir / "before_after_nav_drawdown.png"
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].plot(before_nav.index, before_nav, label="Before repair", color="#7f8c8d", linewidth=2)
    axes[0].plot(after_nav.index, after_nav, label="After repair", color="#d95f02", linewidth=2.2)
    axes[0].plot(
        before_benchmark.index,
        before_benchmark,
        label="Point-in-time A50 benchmark",
        color="#1b9e77",
        linewidth=1.8,
    )
    axes[0].set_title("Rolling fusion strategy: before vs after validation repair")
    axes[0].set_ylabel("Normalized NAV")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    axes[1].plot(before_nav.index, _drawdown(before_nav), label="Before repair", color="#7f8c8d")
    axes[1].plot(after_nav.index, _drawdown(after_nav), label="After repair", color="#d95f02")
    axes[1].fill_between(after_nav.index, _drawdown(after_nav), 0, color="#d95f02", alpha=0.12)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["nav_drawdown"] = target

    before_perf = pd.read_csv(before_dir / "performance_summary.csv").set_index("strategy").loc[STRATEGY]
    after_perf = pd.read_csv(after_dir / "performance_summary.csv").set_index("strategy").loc[STRATEGY]
    metrics = [
        ("Total return", "total_return", 100.0, "%"),
        ("Annual return", "ann_return", 100.0, "%"),
        ("Annual volatility", "ann_vol", 100.0, "%"),
        ("Sharpe ratio", "sharpe", 1.0, ""),
        ("Max drawdown", "max_drawdown", 100.0, "%"),
        ("Annual excess", "excess_ann_return", 100.0, "%"),
        ("Information ratio", "information_ratio", 1.0, ""),
    ]
    target = target_dir / "performance_before_after.png"
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.1))
    for ax, (title, column, scale, suffix) in zip(axes.ravel(), metrics):
        values = [float(before_perf[column]) * scale, float(after_perf[column]) * scale]
        bars = ax.bar(["Before", "After"], values, color=["#9aa0a6", "#d95f02"], width=0.62)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.2)
        low, high = ax.get_ylim()
        span = high - low
        ax.set_ylim(low - span * 0.08, high + span * 0.12)
        pad = max(abs(v) for v in values) * 0.04 + 0.015
        for bar, value in zip(bars, values):
            y = value + pad if value >= 0 else value - pad
            va = "bottom" if value >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.2f}{suffix}", ha="center", va=va, fontsize=9)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Performance metrics before and after the five validation repairs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["performance"] = target

    before_selection = pd.read_csv(before_dir / "factor_diagnostics" / "factor_selection_summary.csv")
    after_selection = pd.read_csv(after_dir / "factor_diagnostics" / "factor_selection_summary.csv")
    order = after_selection["factor"].tolist()
    before_decisions = before_selection.set_index("factor")["decision"].reindex(order)
    after_decisions = after_selection.set_index("factor")["decision"].reindex(order)
    decision_value = {"REJECT": 0, "WATCH": 1, "PASS": 2}
    matrix = np.column_stack(
        [before_decisions.map(decision_value).to_numpy(), after_decisions.map(decision_value).to_numpy()]
    )
    target = target_dir / "factor_admission_before_after.png"
    fig, ax = plt.subplots(figsize=(9.5, 8.8))
    ax.imshow(matrix, cmap=ListedColormap(["#d73027", "#fdae61", "#1a9850"]), vmin=-0.5, vmax=2.5)
    ax.set_xticks([0, 1], ["Before", "After"])
    ax.set_yticks(range(len(order)), order)
    ax.set_title("Factor admission decisions")
    for row in range(len(order)):
        for col, decisions in enumerate([before_decisions, after_decisions]):
            ax.text(col, row, decisions.iloc[row], ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["admission"] = target

    before_coverage = pd.read_csv(before_dir / "data_quality" / "factor_coverage.csv").set_index("factor")
    after_coverage = pd.read_csv(after_dir / "data_quality" / "factor_coverage.csv").set_index("factor")
    order = after_coverage["coverage"].sort_values().index.tolist()
    old_values = before_coverage.reindex(order)["coverage"] * 100
    new_values = after_coverage.reindex(order)["coverage"] * 100
    target = target_dir / "coverage_denominator_before_after.png"
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    y = np.arange(len(order))
    for idx in range(len(order)):
        ax.plot([old_values.iloc[idx], new_values.iloc[idx]], [idx, idx], color="#c7c7c7", linewidth=2)
    ax.scatter(old_values, y, label="Union-universe denominator", color="#7f8c8d", s=42, zorder=3)
    ax.scatter(new_values, y, label="Point-in-time active denominator", color="#2ca25f", s=42, zorder=3)
    ax.set_yticks(y, order)
    ax.set_xlim(40, 102)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Factor coverage after correcting the denominator")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["coverage"] = target

    before_panel = pd.read_csv(before_dir / "cache" / "factor_panel.csv", parse_dates=["date"])
    after_panel = pd.read_csv(after_dir / "cache" / "factor_panel.csv", parse_dates=["date"])
    before_counts = before_panel.groupby("date")["ML_SCORE"].count()
    after_counts = after_panel.groupby("date")["ML_SCORE"].count()
    target = target_dir / "ml_universe_mask_before_after.png"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(before_counts.index, before_counts, label="Before repair", color="#7f8c8d", linewidth=1.8)
    ax.plot(after_counts.index, after_counts, label="After point-in-time mask", color="#5e3c99", linewidth=2)
    ax.axhline(50, color="#1b9e77", linestyle="--", linewidth=1.2, label="Active A50 size")
    ax.set_title("Number of stocks receiving an ML score each day")
    ax.set_ylabel("Valid ML scores")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["ml_mask"] = target

    rolling = pd.read_csv(after_dir / "factor_validation" / "rolling_out_of_sample_summary.csv")
    rolling = rolling[rolling["factor"] != "ANNOUNCEMENT_EVENT_SCORE"].copy()
    rolling = rolling.sort_values("supportive_window_rate", ascending=True)
    target = target_dir / "rolling_oos_evidence.png"
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    y = np.arange(len(rolling))
    ax.barh(y - 0.18, rolling["stable_window_rate"] * 100, height=0.34, color="#9aa0a6", label="Strict stable windows")
    ax.barh(y + 0.18, rolling["supportive_window_rate"] * 100, height=0.34, color="#4c78a8", label="Supportive windows")
    ax.axvline(50, color="#d95f02", linestyle="--", linewidth=1.2, label="50% support line")
    ax.set_yticks(y, rolling["factor"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of rolling validation windows (%)")
    ax.set_title("Rolling out-of-sample evidence used by the repaired gate")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["rolling_oos"] = target

    old_validation = before_selection.set_index("factor")["validation_excess_ann_return"]
    new_validation = after_selection.set_index("factor")["validation_excess_ann_return"]
    compare = pd.concat(
        [old_validation.rename("before"), new_validation.rename("after")],
        axis=1,
    ).drop(index="ANNOUNCEMENT_EVENT_SCORE", errors="ignore")
    compare = compare.dropna().sort_values("after")
    target = target_dir / "validation_excess_before_after.png"
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    y = np.arange(len(compare))
    for idx in range(len(compare)):
        ax.plot(
            [compare.iloc[idx]["before"] * 100, compare.iloc[idx]["after"] * 100],
            [idx, idx],
            color="#c7c7c7",
            linewidth=2,
        )
    ax.scatter(compare["before"] * 100, y, label="Before repair", color="#7f8c8d", s=42, zorder=3)
    ax.scatter(compare["after"] * 100, y, label="After repair", color="#756bb1", s=42, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, compare.index)
    ax.set_xlabel("Validation annualized excess return (%)")
    ax.set_title("Validation excess return after aligning the benchmark universe")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    targets["benchmark_alignment"] = target

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="生成第111篇因子验证文章图")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--before-output-dir", type=Path)
    args = parser.parse_args()
    print(build_chart(args.output_dir))
    if args.before_output_dir:
        for name, path in build_comparison_charts(args.before_output_dir, args.output_dir).items():
            print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
