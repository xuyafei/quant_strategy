#!/usr/bin/env python3
"""Run the article-112 point-in-time factor-gate walk-forward backtest."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import ConstantInputWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark import equal_weight_benchmark_nav, summarize_excess
from analysis.performance import summarize
from analysis.walk_forward_selection import (
    build_walk_forward_factor_fusion,
    validate_walk_forward_audit,
)
from backtest.backtest_multi import run_multi_backtest
from config import Settings, get_settings
from factors.panel_builder import DEFAULT_FACTOR_ORDER
from factors.preprocess import cross_sectional_zscore
from live.cache_io import save_rebalance_logs
from live.universe_history import (
    load_membership_intervals,
    mask_wide_prices_by_membership,
    membership_mask,
)


REFERENCE_STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"
WALK_FORWARD_STRATEGY = "FUSED_WALK_FORWARD_SCORE_WEIGHTED"
warnings.filterwarnings("ignore", category=ConstantInputWarning)
_PATH_FIELDS = {
    "project_root",
    "data_dir",
    "output_dir",
    "stock_pool_path",
    "universe_membership_path",
    "database_path",
    "tushare_price_cache_path",
    "fina_indicator_cache_path",
    "announcement_event_path",
}


def _settings_from_source(source_dir: Path, output_dir: Path) -> tuple[Settings, dict[str, object]]:
    raw = json.loads((source_dir / "cache" / "run_config.json").read_text(encoding="utf-8"))
    base = get_settings()
    allowed = {item.name for item in fields(Settings)}
    overrides: dict[str, object] = {}
    for key, value in raw.items():
        if key not in allowed or key == "output_dir":
            continue
        if key in _PATH_FIELDS:
            overrides[key] = Path(value) if value else None
        elif isinstance(getattr(base, key), tuple):
            overrides[key] = tuple(value)
        else:
            overrides[key] = value
    overrides["output_dir"] = output_dir
    return replace(base, **overrides), raw


def _load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["date"])
    panel["symbol"] = panel["symbol"].astype(str)
    return panel.set_index(["date", "symbol"]).sort_index()


def _rebuild_reference_nav(
    source_dir: Path,
    prices: pd.DataFrame,
    long_prices: pd.DataFrame,
    settings: Settings,
) -> pd.Series:
    composite = _load_panel(source_dir / "factor_diagnostics" / "factor_composite_scores.csv")
    weight_log = pd.read_csv(
        source_dir / "factor_diagnostics" / "rolling_factor_weight_log.csv",
        parse_dates=["date"],
    )
    zscore = cross_sectional_zscore(composite)
    parts: list[pd.Series] = []
    for dt, rows in weight_log.groupby("date"):
        weights = rows.set_index("factor")["final_weight"].astype(float)
        current = zscore.xs(pd.Timestamp(dt), level="date").mul(weights, axis=1).sum(axis=1)
        current.index = pd.MultiIndex.from_product(
            [[pd.Timestamp(dt)], current.index],
            names=["date", "symbol"],
        )
        parts.append(current)
    fused = pd.concat(parts).sort_index()
    nav, _ = run_multi_backtest(
        fused=fused,
        prices=prices,
        settings=settings,
        factor_name=REFERENCE_STRATEGY,
        long_prices=long_prices,
    )
    expected = pd.read_csv(source_dir / "performance_summary.csv").set_index("strategy")
    expected_final = float(expected.loc[REFERENCE_STRATEGY, "final_nav"])
    if abs(float(nav.iloc[-1]) - expected_final) > 1e-9:
        raise RuntimeError("reference NAV reconstruction mismatch")
    return nav.rename(REFERENCE_STRATEGY)


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def _stats_row(
    strategy: str,
    period: str,
    nav: pd.Series,
    benchmark: pd.Series,
    settings: Settings,
) -> dict[str, object]:
    row: dict[str, object] = {"strategy": strategy, "period": period}
    row.update(summarize(nav, periods=settings.trading_days_per_year))
    row.update(summarize_excess(nav, benchmark, periods=settings.trading_days_per_year))
    row["start"] = pd.Timestamp(nav.index.min()).strftime("%Y-%m-%d")
    row["end"] = pd.Timestamp(nav.index.max()).strftime("%Y-%m-%d")
    row["n_days"] = int(len(nav))
    return row


def _normalize_from(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    out = series.loc[series.index >= start].dropna().astype(float)
    return out / float(out.iloc[0]) if not out.empty else out


def _plot_outputs(
    output_dir: Path,
    navs: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    active_start: pd.Timestamp | None,
) -> dict[str, Path]:
    chart_dir = output_dir / "article_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    path = chart_dir / "walk_forward_nav_drawdown.png"
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {
        WALK_FORWARD_STRATEGY: "#d95f02",
        REFERENCE_STRATEGY: "#7f8c8d",
        "BENCH_EQUAL_WEIGHT": "#1b9e77",
    }
    labels = {
        WALK_FORWARD_STRATEGY: "Walk-forward gate",
        REFERENCE_STRATEGY: "Final-sample gate (reference)",
        "BENCH_EQUAL_WEIGHT": "Point-in-time A50 benchmark",
    }
    for column in [WALK_FORWARD_STRATEGY, REFERENCE_STRATEGY, "BENCH_EQUAL_WEIGHT"]:
        axes[0].plot(navs.index, navs[column], label=labels[column], color=colors[column], linewidth=2)
    if active_start is not None:
        axes[0].axvline(active_start, color="#4c78a8", linestyle="--", linewidth=1.2, label="First active gate")
    axes[0].set_title("Walk-forward factor admission: full-period NAV")
    axes[0].set_ylabel("Normalized NAV")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    axes[1].plot(navs.index, _drawdown(navs[WALK_FORWARD_STRATEGY]), label="Walk-forward gate", color=colors[WALK_FORWARD_STRATEGY])
    axes[1].plot(navs.index, _drawdown(navs[REFERENCE_STRATEGY]), label="Final-sample gate", color=colors[REFERENCE_STRATEGY])
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["nav_drawdown"] = path

    if active_start is not None:
        path = chart_dir / "walk_forward_active_period.png"
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for column in [WALK_FORWARD_STRATEGY, REFERENCE_STRATEGY, "BENCH_EQUAL_WEIGHT"]:
            series = _normalize_from(navs[column], active_start)
            ax.plot(series.index, series, label=labels[column], color=colors[column], linewidth=2)
        ax.set_title("Performance after the first eligible walk-forward rebalance")
        ax.set_ylabel("NAV rebased to 1.0")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["active_period"] = path

    if not decisions.empty:
        decision_order = {"WARMUP": 0, "ERROR": 0, "NO_PASS": 0, "REJECT": 1, "WATCH": 2, "PASS": 3}
        pivot = decisions.pivot_table(
            index="factor",
            columns="as_of_date",
            values="decision",
            aggfunc="last",
        )
        factor_order = [x for x in DEFAULT_FACTOR_ORDER + ["ML_SCORE"] if x in pivot.index]
        pivot = pivot.reindex(factor_order)
        matrix = pivot.apply(lambda col: col.map(decision_order)).fillna(0).to_numpy(dtype=float)
        path = chart_dir / "walk_forward_admission_timeline.png"
        fig, ax = plt.subplots(figsize=(13, 8.5))
        ax.imshow(matrix, cmap=ListedColormap(["#bdbdbd", "#d73027", "#fdae61", "#1a9850"]), vmin=-0.5, vmax=3.5, aspect="auto")
        dates = pd.to_datetime(pivot.columns)
        ax.set_xticks(range(len(dates)), [d.strftime("%Y-%m") for d in dates], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("Point-in-time factor admission at every rebalance")
        ax.set_xlabel("Rebalance date")
        ax.set_ylabel("Factor")
        ax.legend(
            handles=[
                Patch(color="#bdbdbd", label="Warm-up / unavailable"),
                Patch(color="#d73027", label="REJECT"),
                Patch(color="#fdae61", label="WATCH"),
                Patch(color="#1a9850", label="PASS"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["admission_timeline"] = path

    path = chart_dir / "walk_forward_gate_activity.png"
    fig, ax = plt.subplots(figsize=(11, 5.2))
    active = summary.copy()
    active["date"] = pd.to_datetime(active["date"])
    ax.step(active["date"], active["n_pass"], where="mid", color="#4c78a8", linewidth=2, label="PASS factors")
    ax.step(active["date"], active["n_styles"], where="mid", color="#d95f02", linewidth=1.8, label="Active style factors")
    warm = active[active["signal_status"] != "ACTIVE"]
    if not warm.empty:
        ax.scatter(warm["date"], np.zeros(len(warm)), color="#9e9e9e", s=25, label="Warm-up / no trade", zorder=3)
    ax.set_title("Walk-forward gate activity by rebalance")
    ax.set_ylabel("Count")
    ax.set_xlabel("Rebalance date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["gate_activity"] = path

    if not weights.empty:
        wide = weights.pivot(index="date", columns="style_factor", values="weight").fillna(0.0)
        path = chart_dir / "walk_forward_style_weights.png"
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax.stackplot(
            wide.index,
            *[wide[col] for col in wide.columns],
            labels=wide.columns,
            alpha=0.82,
            step="post",
        )
        ax.set_ylim(0, 1)
        ax.set_title("Point-in-time style weights after each gate decision")
        ax.set_ylabel("Portfolio factor weight")
        ax.set_xlabel("Rebalance date")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["style_weights_chart"] = path
    return paths


def run(
    source_dir: Path,
    output_dir: Path,
    *,
    rebalance_freq: str | None = None,
    history_lookback_days: int | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "thread",
) -> dict[str, Path]:
    source_settings, source_config = _settings_from_source(source_dir, output_dir)
    settings = (
        replace(source_settings, rebalance_freq=str(rebalance_freq))
        if rebalance_freq
        else source_settings
    )
    if history_lookback_days is not None:
        if int(history_lookback_days) < 0:
            raise ValueError("history_lookback_days 须 >= 0")
        settings = replace(
            settings,
            walk_forward_history_lookback_days=int(history_lookback_days),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel(source_dir / "cache" / "factor_panel_zscore.csv")
    prices = pd.read_csv(
        source_dir / "cache" / "prices_wide_adj_close.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source_dir / "cache" / "prices_long.csv", parse_dates=["trade_date"])
    factors = [x for x in DEFAULT_FACTOR_ORDER + ["ML_SCORE"] if x in panel.columns]

    eligible = None
    benchmark_prices = prices
    membership_path = settings.universe_membership_path
    if membership_path is not None:
        membership = load_membership_intervals(membership_path)
        eligible = membership_mask(panel.index, membership)
        benchmark_prices = mask_wide_prices_by_membership(prices, membership)

    fused, decisions, summary, weights = build_walk_forward_factor_fusion(
        panel,
        prices,
        settings,
        factors=factors,
        eligible_mask=eligible,
        benchmark_prices=benchmark_prices,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        gate_cache_dir=output_dir / "gate_cache",
    )
    audit_checks = validate_walk_forward_audit(decisions, summary, weights)
    nav, meta = run_multi_backtest(
        fused=fused,
        prices=prices,
        settings=settings,
        factor_name=WALK_FORWARD_STRATEGY,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )
    reference_nav = _rebuild_reference_nav(
        source_dir,
        prices,
        long_prices,
        source_settings,
    )
    benchmark_nav = equal_weight_benchmark_nav(benchmark_prices, dates=nav.index)
    navs = pd.concat([nav.rename(WALK_FORWARD_STRATEGY), reference_nav, benchmark_nav], axis=1)

    active_rows = summary[summary["signal_status"] == "ACTIVE"]
    active_start = pd.Timestamp(active_rows["date"].min()) if not active_rows.empty else None
    rows = [
        _stats_row(WALK_FORWARD_STRATEGY, "FULL_PERIOD", nav, benchmark_nav, settings),
        _stats_row(REFERENCE_STRATEGY, "FULL_PERIOD", reference_nav, benchmark_nav, settings),
    ]
    if active_start is not None:
        wf_active = _normalize_from(nav, active_start)
        ref_active = _normalize_from(reference_nav, active_start)
        bench_active = _normalize_from(benchmark_nav, active_start)
        rows.extend(
            [
                _stats_row(WALK_FORWARD_STRATEGY, "ACTIVE_PERIOD", wf_active, bench_active, settings),
                _stats_row(REFERENCE_STRATEGY, "ACTIVE_PERIOD", ref_active, bench_active, settings),
            ]
        )

    audit_dir = output_dir / "walk_forward"
    audit_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "factor_decisions": audit_dir / "factor_decisions.csv",
        "rebalance_summary": audit_dir / "rebalance_summary.csv",
        "style_weights": audit_dir / "style_weights.csv",
        "fused_scores": audit_dir / "fused_scores.csv",
        "nav_comparison": output_dir / "nav_comparison.csv",
        "performance": output_dir / "performance_summary.csv",
        "run_config": output_dir / "run_config.json",
        "audit_checks": audit_dir / "audit_checks.json",
    }
    decisions.to_csv(paths["factor_decisions"], index=False, date_format="%Y-%m-%d")
    summary.to_csv(paths["rebalance_summary"], index=False, date_format="%Y-%m-%d")
    weights.to_csv(paths["style_weights"], index=False, date_format="%Y-%m-%d")
    fused.rename("score").reset_index().to_csv(paths["fused_scores"], index=False, date_format="%Y-%m-%d")
    navs.to_csv(paths["nav_comparison"], date_format="%Y-%m-%d")
    pd.DataFrame(rows).to_csv(paths["performance"], index=False)
    config = dict(source_config)
    config.update(
        {
            "output_dir": str(output_dir),
            "source_output_dir": str(source_dir),
            "walk_forward_min_history_days": settings.walk_forward_min_history_days,
            "walk_forward_min_rolling_windows": settings.walk_forward_min_rolling_windows,
            "walk_forward_history_lookback_days": settings.walk_forward_history_lookback_days,
            "walk_forward_information_cutoff": "strictly_before_rebalance_date",
            "rebalance_freq": settings.rebalance_freq,
            "reference_rebalance_freq": source_settings.rebalance_freq,
            "walk_forward_n_jobs": max(1, int(n_jobs)),
            "walk_forward_parallel_backend": parallel_backend,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    paths["run_config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["audit_checks"].write_text(
        json.dumps(
            {
                **audit_checks,
                "decision_history_strictly_before_rebalance": True,
                "weight_history_strictly_before_rebalance": True,
                "style_weights_sum_to_one": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    save_rebalance_logs(settings, {WALK_FORWARD_STRATEGY: meta})
    paths.update(_plot_outputs(output_dir, navs, decisions, summary, weights, active_start))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="逐调仓日 Walk-forward 因子准入与回测")
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rebalance-freq",
        default=None,
        help="覆盖实验调仓频率，例如 W-FRI；参照策略仍按源运行频率重建",
    )
    parser.add_argument(
        "--history-lookback-days",
        type=int,
        default=None,
        help="覆盖因子门禁最多使用的历史交易日数；0 表示扩展历史",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="并行计算独立调仓日门禁的工作单元数")
    parser.add_argument(
        "--parallel-backend",
        choices=("thread", "process"),
        default="thread",
        help="门禁并行后端；大截面 CPU 密集任务建议 process",
    )
    args = parser.parse_args()
    for name, path in run(
        args.source_output_dir,
        args.output_dir,
        rebalance_freq=args.rebalance_freq,
        history_lookback_days=args.history_lookback_days,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
    ).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
