#!/usr/bin/env python3
"""Build and audit the frozen CSI300 multi-memory consensus V2 candidate."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark import equal_weight_benchmark_nav, summarize_excess
from analysis.performance import summarize
from backtest.backtest_multi import run_multi_backtest
from live.universe_history import load_membership_intervals, mask_wide_prices_by_membership
from scripts.build_walk_forward_factor_backtest import _settings_from_source


SOURCE_ORDER = ["ROLLING_260D", "ROLLING_504D", "EXPANDING"]
PRIMARY = "CONSENSUS_3WAY"
FULL_RISK_DIAGNOSTIC = "CONSENSUS_3WAY_FULL_RISK_DIAGNOSTIC"
VARIANT_SOURCES = {
    PRIMARY: SOURCE_ORDER,
    "LEAVE_OUT_260D": ["ROLLING_504D", "EXPANDING"],
    "LEAVE_OUT_504D": ["ROLLING_260D", "EXPANDING"],
    "LEAVE_OUT_EXPANDING": ["ROLLING_260D", "ROLLING_504D"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_score(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.set_index(["date", "symbol"])["score"].sort_index()


def cross_sectional_source_ranks(scores: dict[str, pd.Series]) -> pd.DataFrame:
    """Convert every source score to a same-date percentile rank without filling gaps."""
    if set(scores) != set(SOURCE_ORDER):
        raise ValueError("scores must contain exactly the three frozen source models")
    frame = pd.concat(
        [scores[source].rename(source) for source in SOURCE_ORDER], axis=1
    ).sort_index()
    frame.index = frame.index.set_names(["date", "symbol"])
    return frame.groupby(level="date", sort=False).rank(method="average", pct=True)


def build_consensus_score(
    ranks: pd.DataFrame,
    sources: list[str],
    *,
    minimum_valid_sources: int,
) -> pd.Series:
    """Take the row median of frozen source ranks and require explicit source coverage."""
    selected = ranks.loc[:, sources]
    valid = selected.notna().sum(axis=1)
    score = selected.median(axis=1, skipna=True).where(valid >= minimum_valid_sources)
    score.name = "score"
    return score


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _top_set(series: pd.Series, top_k: int) -> set[str]:
    clean = series.dropna().astype(float)
    return set(clean.nlargest(top_k).index.astype(str))


def build_disagreement_monitor(
    ranks: pd.DataFrame,
    *,
    threshold: float = 0.40,
    top_k: int = 10,
) -> pd.DataFrame:
    """Map exact-date source Top-K agreement to the frozen 100/75/50 risk schedule."""
    rows: list[dict[str, Any]] = []
    for date, cross in ranks.groupby(level="date", sort=True):
        cross = cross.droplevel("date")
        sets = {source: _top_set(cross[source], top_k) for source in SOURCE_ORDER}
        pair_values: dict[str, float] = {}
        for left, right in itertools.combinations(SOURCE_ORDER, 2):
            pair_values[f"jaccard_{left.lower()}_{right.lower()}"] = jaccard(
                sets[left], sets[right]
            )
        values = list(pair_values.values())
        agreement_pairs = int(sum(value >= threshold for value in values))
        if agreement_pairs == 3:
            state, multiplier = "GREEN", 1.0
        elif agreement_pairs >= 1:
            state, multiplier = "YELLOW", 0.75
        else:
            state, multiplier = "RED", 0.50
        rows.append(
            {
                "date": pd.Timestamp(date),
                **pair_values,
                "minimum_pairwise_jaccard": float(min(values)),
                "median_pairwise_jaccard": float(np.median(values)),
                "maximum_pairwise_jaccard": float(max(values)),
                "agreement_pairs_at_or_above_threshold": agreement_pairs,
                "risk_state": state,
                "gross_exposure_multiplier": multiplier,
                **{f"{source.lower()}_top10": ",".join(sorted(sets[source])) for source in SOURCE_ORDER},
                **{f"{source.lower()}_valid_signal": bool(sets[source]) for source in SOURCE_ORDER},
            }
        )
    return pd.DataFrame(rows)


def _normalize(nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    selected = nav.loc[(nav.index >= start) & (nav.index <= end)].dropna().astype(float)
    return selected / float(selected.iloc[0]) if not selected.empty else selected


def _stats(
    variant: str,
    period: str,
    nav: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": variant, "period": period}
    row.update(summarize(nav))
    row.update(summarize_excess(nav, benchmark))
    row["benchmark_total_return"] = (
        float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0) if len(benchmark) else np.nan
    )
    row["return_gap_vs_benchmark"] = (
        float(row["total_return"] - row["benchmark_total_return"])
        if np.isfinite(row["benchmark_total_return"])
        else np.nan
    )
    row["start"] = nav.index.min().strftime("%Y-%m-%d") if not nav.empty else ""
    row["end"] = nav.index.max().strftime("%Y-%m-%d") if not nav.empty else ""
    row["n_days"] = len(nav)
    return row


def _actual_target_turnover(log: list[dict[str, Any]], start: pd.Timestamp) -> float:
    previous: dict[str, float] = {}
    total = 0.0
    for rec in sorted(log, key=lambda value: pd.Timestamp(value["date"])):
        dt = pd.Timestamp(rec["date"])
        current = {
            str(symbol): float(weight)
            for symbol, weight in zip(rec.get("picks", []), rec.get("weights", []))
        }
        if dt < start:
            previous = current
            continue
        symbols = set(previous) | set(current)
        total += sum(
            abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
            for symbol in symbols
        )
        previous = current
    return float(total)


def _execution_summary(
    variant: str,
    meta: dict[str, Any],
    start: pd.Timestamp,
) -> dict[str, Any]:
    logs = [
        rec
        for rec in meta.get("rebalance_log", [])
        if pd.Timestamp(rec["date"]) >= start
    ]
    rows: list[dict[str, float]] = []
    for rec in logs:
        weights = [float(value) for value in rec.get("weights", []) if float(value) > 1e-12]
        gross = float(sum(weights))
        normalized = [value / gross for value in weights] if gross > 1e-12 else []
        hhi = sum(value * value for value in normalized)
        rows.append(
            {
                "holding_count": float(len(weights)),
                "holding_count_above_50bp": float(sum(value >= 0.005 for value in weights)),
                "effective_n": 1.0 / hhi if hhi > 1e-12 else 0.0,
                "gross": gross,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "variant": variant,
        "n_rebalances": len(logs),
        "actual_target_turnover": _actual_target_turnover(logs, start),
        "average_holding_count": float(frame["holding_count"].mean()),
        "max_holding_count": int(frame["holding_count"].max()),
        "average_holding_count_above_50bp": float(
            frame["holding_count_above_50bp"].mean()
        ),
        "average_effective_n": float(frame["effective_n"].mean()),
        "average_gross_target_weight": float(frame["gross"].mean()),
    }


def _source_cutoff_audit(variant_dirs: dict[str, Path]) -> dict[str, Any]:
    violations = 0
    checked = 0
    for source in SOURCE_ORDER:
        frame = pd.read_csv(
            variant_dirs[source] / "walk_forward" / "rebalance_summary.csv",
            parse_dates=["date", "history_end"],
        )
        active = frame[frame["signal_status"] == "ACTIVE"].dropna(subset=["history_end"])
        checked += len(active)
        violations += int((active["history_end"] >= active["date"]).sum())
    return {
        "source_cutoff_rows_checked": checked,
        "source_cutoff_violations": violations,
        "source_information_cutoff_pass": violations == 0,
    }


def _consensus_support_diagnostic(
    ranks: pd.DataFrame,
    primary_meta: dict[str, Any],
    common_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Describe source Top10 support for the selected consensus names post hoc."""
    rows: list[dict[str, Any]] = []
    for rec in primary_meta.get("rebalance_log", []):
        date = pd.Timestamp(rec["date"])
        if date < common_start or date not in ranks.index.get_level_values("date"):
            continue
        cross = ranks.xs(date, level="date")
        source_top10 = {
            source: _top_set(cross[source], 10) for source in SOURCE_ORDER
        }
        picks = [str(symbol) for symbol in rec.get("selected_picks", rec.get("picks", []))]
        support = {
            symbol: sum(symbol in source_top10[source] for source in SOURCE_ORDER)
            for symbol in picks
        }
        counts = {votes: sum(value == votes for value in support.values()) for votes in range(4)}
        rows.append(
            {
                "date": date,
                "n_selected": len(picks),
                **{f"selected_with_{votes}_source_votes": counts[votes] for votes in range(4)},
                "selected_with_at_least_2_source_votes": sum(
                    value >= 2 for value in support.values()
                ),
                "share_with_at_least_2_source_votes": (
                    sum(value >= 2 for value in support.values()) / len(picks)
                    if picks
                    else np.nan
                ),
                "mean_source_votes": (
                    float(np.mean(list(support.values()))) if support else np.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    period_masks = {
        "FULL_ACTIVE": frame["date"] >= common_start,
        "HISTORICAL_ROBUSTNESS": frame["date"] <= pd.Timestamp("2025-09-11"),
        "KNOWN_DEVELOPMENT": frame["date"] >= pd.Timestamp("2025-09-12"),
    }
    for period, mask in period_masks.items():
        selected = frame.loc[mask]
        summary[period] = {
            "n_rebalances": int(len(selected)),
            "average_selected_with_0_source_votes": float(
                selected["selected_with_0_source_votes"].mean()
            ),
            "average_selected_with_1_source_vote": float(
                selected["selected_with_1_source_votes"].mean()
            ),
            "average_selected_with_2_source_votes": float(
                selected["selected_with_2_source_votes"].mean()
            ),
            "average_selected_with_3_source_votes": float(
                selected["selected_with_3_source_votes"].mean()
            ),
            "average_share_with_at_least_2_source_votes": float(
                selected["share_with_at_least_2_source_votes"].mean()
            ),
            "median_share_with_at_least_2_source_votes": float(
                selected["share_with_at_least_2_source_votes"].median()
            ),
            "average_mean_source_votes": float(selected["mean_source_votes"].mean()),
        }
    return frame, summary


def _state_forward_return_diagnostic(
    full_risk_nav: pd.Series,
    benchmark: pd.Series,
    monitor: pd.DataFrame,
    primary_meta: dict[str, Any],
    common_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure next-rebalance returns by agreement state post hoc."""
    dates = sorted(
        pd.Timestamp(rec["date"])
        for rec in primary_meta.get("rebalance_log", [])
        if pd.Timestamp(rec["date"]) >= common_start
    )
    states = monitor.set_index("date")["risk_state"]
    rows: list[dict[str, Any]] = []
    for current, following in zip(dates, dates[1:]):
        if current not in states.index:
            continue
        strategy_return = float(full_risk_nav.loc[following] / full_risk_nav.loc[current] - 1.0)
        benchmark_return = float(benchmark.loc[following] / benchmark.loc[current] - 1.0)
        rows.append(
            {
                "date": current,
                "next_rebalance_date": following,
                "risk_state": states.loc[current],
                "full_risk_consensus_next_return": strategy_return,
                "benchmark_next_return": benchmark_return,
                "next_excess_return": strategy_return - benchmark_return,
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for state, selected in frame.groupby("risk_state", sort=True):
        summary[str(state)] = {
            "n_intervals": int(len(selected)),
            "mean_full_risk_consensus_next_return": float(
                selected["full_risk_consensus_next_return"].mean()
            ),
            "mean_benchmark_next_return": float(selected["benchmark_next_return"].mean()),
            "mean_next_excess_return": float(selected["next_excess_return"].mean()),
            "positive_full_risk_consensus_return_share": float(
                (selected["full_risk_consensus_next_return"] > 0.0).mean()
            ),
        }
    return frame, summary


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    monitor: pd.DataFrame,
    performance: pd.DataFrame,
    execution: pd.DataFrame,
    common_start: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "article_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        PRIMARY: "#7c3aed",
        "LEAVE_OUT_260D": "#4c78a8",
        "LEAVE_OUT_504D": "#d95f02",
        "LEAVE_OUT_EXPANDING": "#54a24b",
        "CSI300_EQUAL_WEIGHT": "#0f766e",
    }
    paths: list[Path] = []

    active = navs.loc[navs.index >= common_start]
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name in list(VARIANT_SOURCES) + ["CSI300_EQUAL_WEIGHT"]:
        series = active[name] / float(active[name].iloc[0])
        axes[0].plot(series.index, series, label=name, color=colors[name], linewidth=2)
        if name != "CSI300_EQUAL_WEIGHT":
            axes[1].plot(
                series.index,
                series / series.cummax() - 1.0,
                label=name,
                color=colors[name],
            )
    axes[0].set_title("Frozen multi-memory consensus and leave-one-out stress tests")
    axes[0].set_ylabel("NAV")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "consensus_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    m = monitor[monitor["date"] >= common_start].set_index("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    pair_cols = [column for column in m.columns if column.startswith("jaccard_")]
    for column in pair_cols:
        axes[0].plot(m.index, m[column], label=column.replace("jaccard_", ""), linewidth=1.4)
    axes[0].axhline(0.40, color="#d62728", ls="--", label="frozen threshold")
    axes[0].set_ylabel("Top10 Jaccard")
    axes[0].set_title("Source-model agreement and exact-date risk budget")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].step(
        m.index,
        m["gross_exposure_multiplier"],
        where="post",
        color="#7c3aed",
        label="gross exposure multiplier",
    )
    axes[1].set_yticks([0.5, 0.75, 1.0], ["RED 50%", "YELLOW 75%", "GREEN 100%"])
    axes[1].set_ylim(0.45, 1.05)
    axes[1].set_ylabel("State")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "agreement_risk_state_timeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    period_order = ["HISTORICAL_ROBUSTNESS", "KNOWN_DEVELOPMENT", "FULL_ACTIVE"]
    pivot = performance.pivot(index="variant", columns="period", values="total_return")
    pivot = pivot.reindex(index=list(VARIANT_SOURCES), columns=period_order)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.24
    period_colors = ["#4c78a8", "#f58518", "#7c3aed"]
    for i, period in enumerate(period_order):
        ax.bar(
            x + (i - 1) * width,
            pivot[period] * 100,
            width=width,
            label=period,
            color=period_colors[i],
        )
    ax.set_xticks(x, pivot.index, rotation=20)
    ax.set_ylabel("Total return (%)")
    ax.set_title("Subperiod returns: no leave-one-out winner may be selected")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "consensus_subperiod_returns.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    full = performance[performance["period"] == "FULL_ACTIVE"].set_index("variant")
    loo = full.reindex([name for name in VARIANT_SOURCES if name != PRIMARY])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    axes[0].bar(loo.index, loo["total_return"] * 100, color="#4c78a8")
    axes[1].bar(loo.index, loo["excess_ann_return"] * 100, color="#d95f02")
    axes[2].bar(loo.index, loo["max_drawdown"] * 100, color="#7c3aed")
    for ax, title in zip(axes, ["Total return", "Annualized excess", "Maximum drawdown"]):
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Leave-one-window-out stability audit")
    fig.tight_layout()
    path = chart_dir / "leave_one_out_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    e = execution.set_index("variant").reindex(list(VARIANT_SOURCES))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].bar(e.index, e["average_gross_target_weight"] * 100, color="#0f766e")
    axes[0].set_ylabel("Average gross target (%)")
    axes[0].set_title("Realized risk budget")
    axes[1].bar(e.index, e["actual_target_turnover"], color="#7c3aed")
    axes[1].set_ylabel("Cumulative L1 turnover")
    axes[1].set_title("Execution footprint")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "consensus_execution_footprint.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_risk_gate_ablation(
    output: Path,
    gated_nav: pd.Series,
    full_risk_nav: pd.Series,
    benchmark: pd.Series,
    start: pd.Timestamp,
) -> Path:
    chart_dir = output / "article_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    series_map = {
        "CONSENSUS_WITH_DISAGREEMENT_GATE": _normalize(gated_nav, start, gated_nav.index.max()),
        "CONSENSUS_FULL_RISK_DIAGNOSTIC": _normalize(full_risk_nav, start, full_risk_nav.index.max()),
        "CSI300_EQUAL_WEIGHT": _normalize(benchmark, start, benchmark.index.max()),
    }
    colors = {
        "CONSENSUS_WITH_DISAGREEMENT_GATE": "#7c3aed",
        "CONSENSUS_FULL_RISK_DIAGNOSTIC": "#d95f02",
        "CSI300_EQUAL_WEIGHT": "#0f766e",
    }
    for name, series in series_map.items():
        axes[0].plot(series.index, series, label=name, color=colors[name], linewidth=2)
        if name != "CSI300_EQUAL_WEIGHT":
            axes[1].plot(
                series.index,
                series / series.cummax() - 1.0,
                label=name,
                color=colors[name],
            )
    axes[0].set_title("Post-hoc ablation only: rank consensus with and without disagreement gate")
    axes[0].set_ylabel("NAV")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "risk_gate_posthoc_ablation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run(
    source: Path,
    variant_dirs: dict[str, Path],
    output: Path,
    protocol_path: Path,
) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch; frozen V2 protocol was modified")
    if set(variant_dirs) != set(SOURCE_ORDER):
        raise ValueError("variant directories must contain exactly the frozen three sources")

    settings, _ = _settings_from_source(source, output)
    settings = replace(settings, rebalance_freq="W-FRI", max_tech_growth_weight=0.0)
    prices = pd.read_csv(
        source / "cache" / "prices_wide_adj_close.csv", index_col=0, parse_dates=True
    ).sort_index()
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source / "cache" / "prices_long.csv", parse_dates=["trade_date"])
    membership = load_membership_intervals(settings.universe_membership_path)
    benchmark = equal_weight_benchmark_nav(
        mask_wide_prices_by_membership(prices, membership),
        dates=prices.index,
        name="CSI300_EQUAL_WEIGHT",
    )

    source_scores = {
        name: _load_score(path / "walk_forward" / "fused_scores.csv")
        for name, path in variant_dirs.items()
    }
    ranks = cross_sectional_source_ranks(source_scores)
    monitor = build_disagreement_monitor(
        ranks,
        threshold=float(protocol["model_disagreement_gate"]["agreement_threshold"]),
        top_k=int(protocol["model_disagreement_gate"]["source_topk"]),
    )
    consensus_scores = {
        variant: build_consensus_score(
            ranks,
            sources,
            minimum_valid_sources=2,
        )
        for variant, sources in VARIANT_SOURCES.items()
    }

    nav_by_variant: dict[str, pd.Series] = {}
    meta_by_variant: dict[str, dict[str, Any]] = {}
    for variant, score in consensus_scores.items():
        nav, meta = run_multi_backtest(
            fused=score,
            prices=prices,
            settings=settings,
            factor_name=variant,
            top_k=10,
            long_prices=long_prices,
            empty_signal_policy="cash",
            rebalance_overrides=monitor,
        )
        nav_by_variant[variant] = nav.rename(variant)
        meta_by_variant[variant] = meta
    full_risk_nav, full_risk_meta = run_multi_backtest(
        fused=consensus_scores[PRIMARY],
        prices=prices,
        settings=settings,
        factor_name=FULL_RISK_DIAGNOSTIC,
        top_k=10,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )

    active_starts = []
    for score in consensus_scores.values():
        active_dates = score.dropna().index.get_level_values("date")
        active_starts.append(pd.Timestamp(active_dates.min()))
    common_start = max(active_starts)
    common_end = pd.Timestamp("2026-09-04")
    periods = {
        "FULL_ACTIVE": (common_start, common_end),
        "HISTORICAL_ROBUSTNESS": (common_start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), common_end),
    }
    performance_rows = []
    for period, (start, end) in periods.items():
        bench_period = _normalize(benchmark, start, end)
        for variant in VARIANT_SOURCES:
            performance_rows.append(
                _stats(
                    variant,
                    period,
                    _normalize(nav_by_variant[variant], start, end),
                    bench_period,
                )
            )
    performance = pd.DataFrame(performance_rows)
    diagnostic_performance = pd.DataFrame(
        [
            _stats(
                FULL_RISK_DIAGNOSTIC,
                period,
                _normalize(full_risk_nav, start, end),
                _normalize(benchmark, start, end),
            )
            for period, (start, end) in periods.items()
        ]
    )
    execution = pd.DataFrame(
        [
            _execution_summary(variant, meta_by_variant[variant], common_start)
            for variant in VARIANT_SOURCES
        ]
    )
    support_diagnostic, support_summary = _consensus_support_diagnostic(
        ranks, meta_by_variant[PRIMARY], common_start
    )
    state_forward_diagnostic, state_forward_summary = _state_forward_return_diagnostic(
        full_risk_nav,
        benchmark,
        monitor,
        meta_by_variant[PRIMARY],
        common_start,
    )

    full = performance[performance["period"] == "FULL_ACTIVE"].set_index("variant")
    historical = performance[
        performance["period"] == "HISTORICAL_ROBUSTNESS"
    ].set_index("variant")
    known = performance[performance["period"] == "KNOWN_DEVELOPMENT"].set_index("variant")
    leave_out = [variant for variant in VARIANT_SOURCES if variant != PRIMARY]
    cutoff_audit = _source_cutoff_audit(variant_dirs)
    primary_turnover = float(
        execution.set_index("variant").loc[PRIMARY, "actual_target_turnover"]
    )
    checks: dict[str, Any] = {
        "protocol_sha256": actual,
        "common_active_start": common_start.strftime("%Y-%m-%d"),
        "common_active_end": common_end.strftime("%Y-%m-%d"),
        **cutoff_audit,
        "primary_full_active_annualized_excess": float(
            full.loc[PRIMARY, "excess_ann_return"]
        ),
        "primary_full_active_annualized_excess_pass": bool(
            full.loc[PRIMARY, "excess_ann_return"] > 0.0
        ),
        "primary_full_active_information_ratio": float(
            full.loc[PRIMARY, "information_ratio"]
        ),
        "primary_full_active_information_ratio_pass": bool(
            full.loc[PRIMARY, "information_ratio"] >= 0.25
        ),
        "primary_full_active_max_drawdown": float(full.loc[PRIMARY, "max_drawdown"]),
        "primary_full_active_max_drawdown_pass": bool(
            abs(full.loc[PRIMARY, "max_drawdown"]) <= 0.145
        ),
        "primary_full_active_total_return": float(full.loc[PRIMARY, "total_return"]),
        "primary_full_active_total_return_pass": bool(
            full.loc[PRIMARY, "total_return"] >= 0.4394
        ),
        "primary_historical_return_gap_vs_benchmark": float(
            historical.loc[PRIMARY, "return_gap_vs_benchmark"]
        ),
        "primary_historical_return_gap_vs_benchmark_pass": bool(
            historical.loc[PRIMARY, "return_gap_vs_benchmark"] >= -0.05
        ),
        "primary_positive_total_return_in_both_subperiods_pass": bool(
            historical.loc[PRIMARY, "total_return"] > 0.0
            and known.loc[PRIMARY, "total_return"] > 0.0
        ),
        "leave_one_out_full_active_total_return_range": float(
            full.loc[leave_out, "total_return"].max()
            - full.loc[leave_out, "total_return"].min()
        ),
        "leave_one_out_full_active_total_return_range_pass": bool(
            full.loc[leave_out, "total_return"].max()
            - full.loc[leave_out, "total_return"].min()
            <= 0.15
        ),
        "leave_one_out_full_active_max_drawdown_range": float(
            full.loc[leave_out, "max_drawdown"].max()
            - full.loc[leave_out, "max_drawdown"].min()
        ),
        "leave_one_out_full_active_max_drawdown_range_pass": bool(
            full.loc[leave_out, "max_drawdown"].max()
            - full.loc[leave_out, "max_drawdown"].min()
            <= 0.05
        ),
        "leave_one_out_full_active_positive_excess_all_pass": bool(
            (full.loc[leave_out, "excess_ann_return"] > 0.0).all()
        ),
        "primary_actual_target_turnover": primary_turnover,
        "primary_actual_target_turnover_pass": bool(primary_turnover <= 133.57),
        "all_signal_and_gate_dates_auditable_pass": bool(
            cutoff_audit["source_information_cutoff_pass"]
            and monitor["date"].is_unique
            and monitor["gross_exposure_multiplier"].isin([0.5, 0.75, 1.0]).all()
        ),
        "posthoc_full_risk_diagnostic_excluded_from_acceptance": True,
    }
    required = [key for key in checks if key.endswith("_pass")]
    checks["historical_acceptance_pass"] = bool(all(checks[key] for key in required))
    checks["verdict"] = (
        "HISTORICAL_STABILITY_CANDIDATE_ONLY"
        if checks["historical_acceptance_pass"]
        else "CONSENSUS_V2_NO_GO"
    )
    checks["live_approval"] = False

    output.mkdir(parents=True, exist_ok=True)
    ranks.rename_axis(index=["date", "symbol"]).reset_index().to_csv(
        output / "source_rank_scores.csv", index=False, date_format="%Y-%m-%d"
    )
    pd.concat(consensus_scores, axis=1).rename_axis(index=["date", "symbol"]).reset_index().to_csv(
        output / "consensus_scores.csv", index=False, date_format="%Y-%m-%d"
    )
    monitor.to_csv(output / "disagreement_monitor.csv", index=False, date_format="%Y-%m-%d")
    navs = pd.concat(
        [*(nav_by_variant[name] for name in VARIANT_SOURCES), benchmark], axis=1
    ).dropna()
    navs.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    diagnostic_performance.to_csv(
        output / "posthoc_full_risk_diagnostic_performance.csv", index=False
    )
    support_diagnostic.to_csv(
        output / "posthoc_consensus_top10_support.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    state_forward_diagnostic.to_csv(
        output / "posthoc_state_forward_returns.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    (output / "posthoc_diagnostic_summary.json").write_text(
        json.dumps(
            {
                "excluded_from_frozen_acceptance": True,
                "consensus_top10_support": support_summary,
                "state_forward_returns": state_forward_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    execution.to_csv(output / "execution_summary.csv", index=False)
    pd.DataFrame(
        [_execution_summary(FULL_RISK_DIAGNOSTIC, full_risk_meta, common_start)]
    ).to_csv(output / "posthoc_full_risk_diagnostic_execution.csv", index=False)
    pd.concat(
        [
            nav_by_variant[PRIMARY],
            full_risk_nav.rename(FULL_RISK_DIAGNOSTIC),
            benchmark,
        ],
        axis=1,
    ).to_csv(output / "posthoc_full_risk_diagnostic_nav.csv", date_format="%Y-%m-%d")
    (output / "acceptance_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for variant, meta in meta_by_variant.items():
        pd.DataFrame(meta.get("rebalance_log", [])).to_json(
            output / f"{variant.lower()}_rebalance_log.json",
            orient="records",
            date_format="iso",
            indent=2,
        )
    charts = _plot_outputs(
        output, navs, monitor, performance, execution, common_start
    )
    risk_gate_ablation_chart = _plot_risk_gate_ablation(
        output,
        nav_by_variant[PRIMARY],
        full_risk_nav,
        benchmark,
        common_start,
    )

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
    except Exception:
        git_head = "UNKNOWN"
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": actual,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "historical_data_cutoff": "2026-09-04",
        "source_artifact_sha256": {
            source_name: {
                "run_config": _sha256(path / "run_config.json"),
                "fused_scores": _sha256(path / "walk_forward" / "fused_scores.csv"),
                "rebalance_summary": _sha256(
                    path / "walk_forward" / "rebalance_summary.csv"
                ),
            }
            for source_name, path in variant_dirs.items()
        },
        "code_sha256": {
            "consensus_script": _sha256(Path(__file__)),
            "backtest_engine": _sha256(ROOT / "backtest" / "backtest_single.py"),
        },
        "git_head_at_evaluation": git_head,
        "historical_acceptance_pass": checks["historical_acceptance_pass"],
        "verdict": checks["verdict"],
        "live_approval": False,
        "true_forward_status": "NOT_STARTED_FOR_V2",
        "posthoc_full_risk_diagnostic_excluded_from_acceptance": True,
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "performance": output / "performance_by_period.csv",
        "monitor": output / "disagreement_monitor.csv",
        "execution": output / "execution_summary.csv",
        "acceptance": output / "acceptance_checks.json",
        "manifest": output / "audit_manifest.json",
        "posthoc_diagnostic": output / "posthoc_full_risk_diagnostic_performance.csv",
        "posthoc_support": output / "posthoc_consensus_top10_support.csv",
        "posthoc_state_forward": output / "posthoc_state_forward_returns.csv",
        "posthoc_summary": output / "posthoc_diagnostic_summary.json",
        "posthoc_chart": risk_gate_ablation_chart,
        **{f"chart_{index + 1}": path for index, path in enumerate(charts)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--rolling-260-dir", type=Path, required=True)
    parser.add_argument("--rolling-504-dir", type=Path, required=True)
    parser.add_argument("--expanding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    variants = {
        "ROLLING_260D": args.rolling_260_dir,
        "ROLLING_504D": args.rolling_504_dir,
        "EXPANDING": args.expanding_dir,
    }
    for name, path in run(
        args.source_output_dir, variants, args.output_dir, args.protocol
    ).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
