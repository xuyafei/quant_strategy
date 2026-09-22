#!/usr/bin/env python3
"""Audit whether the frozen CSI300 Top10 process is stable across history windows."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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


VARIANT_ORDER = ["ROLLING_260D", "ROLLING_504D", "EXPANDING"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _load_score(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.set_index(["date", "symbol"])["score"].sort_index()


def _normalize(nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    selected = nav.loc[(nav.index >= start) & (nav.index <= end)].dropna().astype(float)
    return selected / float(selected.iloc[0]) if not selected.empty else selected


def _stats(variant: str, period: str, nav: pd.Series, benchmark: pd.Series) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": variant, "period": period}
    row.update(summarize(nav))
    row.update(summarize_excess(nav, benchmark))
    row["start"] = nav.index.min().strftime("%Y-%m-%d")
    row["end"] = nav.index.max().strftime("%Y-%m-%d")
    row["n_days"] = len(nav)
    return row


def _sets_by_date(frame: pd.DataFrame, date_col: str, value_col: str) -> dict[pd.Timestamp, set[str]]:
    if frame.empty:
        return {}
    return {
        pd.Timestamp(date): set(group[value_col].dropna().astype(str))
        for date, group in frame.groupby(date_col)
    }


def _signal_sets(meta: dict[str, Any]) -> dict[pd.Timestamp, set[str]]:
    return {
        pd.Timestamp(rec["date"]): set(str(x) for x in rec.get("selected_picks", []))
        for rec in meta.get("rebalance_log", [])
    }


def _style_vectors(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    return {
        pd.Timestamp(date): group.set_index("style_factor")["weight"].astype(float)
        for date, group in frame.groupby("date")
    }


def _style_l1(left: pd.Series | None, right: pd.Series | None) -> float:
    if left is None or right is None:
        return float("nan")
    idx = left.index.union(right.index)
    return float((left.reindex(idx, fill_value=0.0) - right.reindex(idx, fill_value=0.0)).abs().sum())


def _actual_turnover(log: list[dict[str, Any]]) -> float:
    previous: dict[str, float] = {}
    total = 0.0
    for rec in sorted(log, key=lambda value: pd.Timestamp(value["date"])):
        current = {
            str(symbol): float(weight)
            for symbol, weight in zip(rec.get("picks", []), rec.get("weights", []))
        }
        symbols = set(previous) | set(current)
        total += sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)
        previous = current
    return float(total)


def _execution_summary(variant: str, meta: dict[str, Any], common_start: pd.Timestamp) -> dict[str, Any]:
    logs = [rec for rec in meta.get("rebalance_log", []) if pd.Timestamp(rec["date"]) >= common_start]
    rows = []
    for rec in logs:
        weights = [float(x) for x in rec.get("weights", []) if float(x) > 1e-12]
        gross = float(sum(weights))
        normalized = [value / gross for value in weights] if gross > 1e-12 else []
        hhi = sum(value * value for value in normalized)
        rows.append({
            "holding_count": len(weights),
            "holding_count_above_10bp": sum(value >= 0.001 for value in weights),
            "holding_count_above_50bp": sum(value >= 0.005 for value in weights),
            "effective_n": 1.0 / hhi if hhi > 1e-12 else 0.0,
            "gross": gross,
        })
    frame = pd.DataFrame(rows)
    return {
        "variant": variant,
        "n_rebalances": len(logs),
        "actual_target_turnover": _actual_turnover(logs),
        "average_holding_count": float(frame["holding_count"].mean()),
        "max_holding_count": int(frame["holding_count"].max()),
        "average_holding_count_above_10bp": float(frame["holding_count_above_10bp"].mean()),
        "average_holding_count_above_50bp": float(frame["holding_count_above_50bp"].mean()),
        "average_effective_n": float(frame["effective_n"].mean()),
        "average_gross_target_weight": float(frame["gross"].mean()),
    }


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    common_start: pd.Timestamp,
    pairwise: pd.DataFrame,
    factor_rates: pd.DataFrame,
    performance: pd.DataFrame,
    execution: pd.DataFrame,
) -> list[Path]:
    chart_dir = output / "article_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    colors = {"ROLLING_260D": "#d95f02", "ROLLING_504D": "#4c78a8", "EXPANDING": "#7c3aed", "CSI300_EQUAL_WEIGHT": "#0f766e"}
    paths: list[Path] = []

    active = navs.loc[navs.index >= common_start]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name in VARIANT_ORDER + ["CSI300_EQUAL_WEIGHT"]:
        series = active[name] / float(active[name].iloc[0])
        axes[0].plot(series.index, series, label=name, color=colors[name], linewidth=2)
        if name != "CSI300_EQUAL_WEIGHT":
            axes[1].plot(series.index, series / series.cummax() - 1.0, label=name, color=colors[name])
    axes[0].set_title("CSI300 Top10 across frozen history windows")
    axes[0].set_ylabel("NAV")
    axes[0].legend(ncol=2)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(ncol=3)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    path = chart_dir / "history_window_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    for pair, group in pairwise.groupby("pair"):
        axes[0].plot(group["date"], group["factor_pass_jaccard"], label=pair, linewidth=1.5)
        axes[1].plot(group["date"], group["signal_top10_jaccard"], label=pair, linewidth=1.5)
        axes[2].plot(group["date"], group["style_weight_l1"], label=pair, linewidth=1.5)
    axes[0].axhline(.5, color="#d62728", ls="--")
    axes[1].axhline(.4, color="#d62728", ls="--")
    axes[2].axhline(.5, color="#d62728", ls="--")
    axes[0].set_ylabel("PASS Jaccard")
    axes[1].set_ylabel("Top10 Jaccard")
    axes[2].set_ylabel("Style-weight L1")
    axes[0].set_title("Decision stability by rebalance date")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = chart_dir / "decision_stability_timeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    pivot = factor_rates.pivot(index="factor", columns="variant", values="pass_rate").reindex(columns=VARIANT_ORDER)
    pivot = pivot.sort_values("EXPANDING", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value > .55 else "black", fontsize=8)
    ax.set_title("Factor PASS rate under each history window")
    fig.colorbar(image, ax=ax, label="PASS rate")
    fig.tight_layout()
    path = chart_dir / "factor_pass_rate_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    full = performance[performance["period"] == "FULL_COMMON"].set_index("variant").reindex(VARIANT_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].bar(VARIANT_ORDER, full["total_return"] * 100, color=[colors[x] for x in VARIANT_ORDER])
    axes[1].bar(VARIANT_ORDER, full["excess_ann_return"] * 100, color=[colors[x] for x in VARIANT_ORDER])
    axes[2].bar(VARIANT_ORDER, full["max_drawdown"] * 100, color=[colors[x] for x in VARIANT_ORDER])
    axes[0].set_title("Total return")
    axes[1].set_title("Annualized excess")
    axes[2].set_title("Maximum drawdown")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Performance dispersion across history windows")
    fig.tight_layout()
    path = chart_dir / "history_window_performance_bars.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(VARIANT_ORDER))
    e = execution.set_index("variant").reindex(VARIANT_ORDER)
    axes[0].bar(x - .18, e["average_holding_count"], width=.36, label="Any weight")
    axes[0].bar(x + .18, e["average_holding_count_above_50bp"], width=.36, label=">=0.5%")
    axes[0].set_xticks(x, VARIANT_ORDER, rotation=25)
    axes[0].set_ylabel("Average holdings")
    axes[0].legend()
    axes[1].bar(VARIANT_ORDER, e["actual_target_turnover"], color=[colors[v] for v in VARIANT_ORDER])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Cumulative L1 target turnover")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Execution footprint under the same Top10 rule")
    fig.tight_layout()
    path = chart_dir / "execution_footprint.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(source: Path, variant_dirs: dict[str, Path], output: Path, protocol_path: Path) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch")
    if set(variant_dirs) != set(VARIANT_ORDER):
        raise ValueError("variant directories must contain exactly the frozen three variants")

    settings, _ = _settings_from_source(source, output)
    settings = replace(settings, rebalance_freq="W-FRI", max_tech_growth_weight=0.0)
    prices = pd.read_csv(source / "cache" / "prices_wide_adj_close.csv", index_col=0, parse_dates=True).sort_index()
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source / "cache" / "prices_long.csv", parse_dates=["trade_date"])
    membership = load_membership_intervals(settings.universe_membership_path)
    benchmark = equal_weight_benchmark_nav(mask_wide_prices_by_membership(prices, membership), dates=prices.index, name="CSI300_EQUAL_WEIGHT")

    nav_by_variant: dict[str, pd.Series] = {}
    meta_by_variant: dict[str, dict[str, Any]] = {}
    decisions: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    styles: dict[str, pd.DataFrame] = {}
    active_starts: list[pd.Timestamp] = []
    active_ends: list[pd.Timestamp] = []
    for variant in VARIANT_ORDER:
        base = variant_dirs[variant]
        fused = _load_score(base / "walk_forward" / "fused_scores.csv")
        summary = pd.read_csv(base / "walk_forward" / "rebalance_summary.csv", parse_dates=["date"])
        decision = pd.read_csv(base / "walk_forward" / "factor_decisions.csv", parse_dates=["as_of_date"])
        style = pd.read_csv(base / "walk_forward" / "style_weights.csv", parse_dates=["date"])
        active = summary[summary["signal_status"] == "ACTIVE"]
        active_starts.append(pd.Timestamp(active["date"].min()))
        active_ends.append(pd.Timestamp(active["date"].max()))
        nav, meta = run_multi_backtest(
            fused=fused,
            prices=prices,
            settings=settings,
            factor_name=f"CSI300_TOP10_{variant}",
            top_k=10,
            long_prices=long_prices,
            empty_signal_policy="cash",
        )
        nav_by_variant[variant] = nav.rename(variant)
        meta_by_variant[variant] = meta
        decisions[variant] = decision
        summaries[variant] = summary
        styles[variant] = style

    common_start = max(active_starts)
    common_end = min(active_ends)
    navs = pd.concat([*(nav_by_variant[v] for v in VARIANT_ORDER), benchmark], axis=1).dropna()
    periods = {
        "FULL_COMMON": (common_start, common_end),
        "HISTORICAL_ROBUSTNESS": (common_start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), pd.Timestamp("2026-09-04")),
    }
    performance_rows = []
    for period, (start, end) in periods.items():
        bench_period = _normalize(benchmark, start, end)
        for variant in VARIANT_ORDER:
            performance_rows.append(_stats(variant, period, _normalize(nav_by_variant[variant], start, end), bench_period))
    performance = pd.DataFrame(performance_rows)

    factor_sets = {
        variant: _sets_by_date(frame[frame["decision"] == "PASS"], "as_of_date", "factor")
        for variant, frame in decisions.items()
    }
    signal_sets = {variant: _signal_sets(meta) for variant, meta in meta_by_variant.items()}
    style_vectors = {variant: _style_vectors(frame) for variant, frame in styles.items()}
    rebalance_dates = sorted(set.intersection(*[
        set(summary.loc[summary["signal_status"] == "ACTIVE", "date"].map(pd.Timestamp))
        for summary in summaries.values()
    ]))
    pair_rows = []
    for left, right in itertools.combinations(VARIANT_ORDER, 2):
        for date in rebalance_dates:
            pair_rows.append({
                "date": date,
                "left": left,
                "right": right,
                "pair": f"{left} vs {right}",
                "factor_pass_jaccard": jaccard(factor_sets[left].get(date, set()), factor_sets[right].get(date, set())),
                "signal_top10_jaccard": jaccard(signal_sets[left].get(date, set()), signal_sets[right].get(date, set())),
                "style_weight_l1": _style_l1(style_vectors[left].get(date), style_vectors[right].get(date)),
            })
    pairwise = pd.DataFrame(pair_rows)
    aggregate = pairwise.groupby(["left", "right", "pair"], as_index=False).agg(
        n_dates=("date", "size"),
        factor_pass_jaccard_median=("factor_pass_jaccard", "median"),
        factor_pass_jaccard_mean=("factor_pass_jaccard", "mean"),
        signal_top10_jaccard_median=("signal_top10_jaccard", "median"),
        signal_top10_jaccard_mean=("signal_top10_jaccard", "mean"),
        style_weight_l1_median=("style_weight_l1", "median"),
        style_weight_l1_mean=("style_weight_l1", "mean"),
    )
    aggregate_by_period_parts = []
    for period, (start, end) in periods.items():
        selected = pairwise[(pairwise["date"] >= start) & (pairwise["date"] <= end)]
        grouped = selected.groupby(["left", "right", "pair"], as_index=False).agg(
            n_dates=("date", "size"),
            factor_pass_jaccard_median=("factor_pass_jaccard", "median"),
            factor_pass_jaccard_mean=("factor_pass_jaccard", "mean"),
            signal_top10_jaccard_median=("signal_top10_jaccard", "median"),
            signal_top10_jaccard_mean=("signal_top10_jaccard", "mean"),
            style_weight_l1_median=("style_weight_l1", "median"),
            style_weight_l1_mean=("style_weight_l1", "mean"),
        )
        grouped.insert(0, "period", period)
        aggregate_by_period_parts.append(grouped)
    aggregate_by_period = pd.concat(aggregate_by_period_parts, ignore_index=True)

    factor_rows = []
    for variant, frame in decisions.items():
        active_dates = summaries[variant].loc[summaries[variant]["signal_status"] == "ACTIVE", "date"].nunique()
        counts = frame[frame["decision"] == "PASS"].groupby("factor").size()
        for factor in sorted(frame["factor"].dropna().astype(str).unique()):
            factor_rows.append({"variant": variant, "factor": factor, "pass_count": int(counts.get(factor, 0)), "active_dates": int(active_dates), "pass_rate": float(counts.get(factor, 0) / active_dates)})
    factor_rates = pd.DataFrame(factor_rows)
    execution = pd.DataFrame([
        _execution_summary(variant, meta_by_variant[variant], common_start)
        for variant in VARIANT_ORDER
    ])

    full = performance[performance["period"] == "FULL_COMMON"].set_index("variant")
    known = performance[performance["period"] == "KNOWN_DEVELOPMENT"].set_index("variant")
    checks = {
        "protocol_sha256": actual,
        "common_active_start": common_start.strftime("%Y-%m-%d"),
        "common_active_end": common_end.strftime("%Y-%m-%d"),
        "common_active_rebalances": len(rebalance_dates),
        "factor_pass_set_median_jaccard_each_pair_pass": bool((aggregate["factor_pass_jaccard_median"] >= .50).all()),
        "signal_top10_median_jaccard_each_pair_pass": bool((aggregate["signal_top10_jaccard_median"] >= .40).all()),
        "style_weight_median_l1_each_pair_pass": bool((aggregate["style_weight_l1_median"] <= .50).all()),
        "full_active_total_return_range": float(full["total_return"].max() - full["total_return"].min()),
        "full_active_total_return_range_pass": bool(full["total_return"].max() - full["total_return"].min() <= .15),
        "full_active_max_drawdown_range": float(full["max_drawdown"].max() - full["max_drawdown"].min()),
        "full_active_max_drawdown_range_pass": bool(full["max_drawdown"].max() - full["max_drawdown"].min() <= .05),
        "full_active_positive_excess_all_pass": bool((full["excess_ann_return"] > 0).all()),
        "known_development_total_return_range": float(known["total_return"].max() - known["total_return"].min()),
        "known_development_total_return_range_pass": bool(known["total_return"].max() - known["total_return"].min() <= .20),
    }
    required = [key for key in checks if key.endswith("_pass")]
    checks["window_stability_acceptance_pass"] = bool(all(checks[key] for key in required))
    checks["verdict"] = "WINDOW_STABLE_CANDIDATE_ONLY" if checks["window_stability_acceptance_pass"] else "WINDOW_SENSITIVE_RESEARCH_REQUIRED"
    checks["live_approval"] = False

    output.mkdir(parents=True, exist_ok=True)
    navs.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    pairwise.to_csv(output / "pairwise_stability_by_date.csv", index=False, date_format="%Y-%m-%d")
    aggregate.to_csv(output / "pairwise_stability_summary.csv", index=False)
    aggregate_by_period.to_csv(output / "pairwise_stability_by_period.csv", index=False)
    factor_rates.to_csv(output / "factor_pass_rates.csv", index=False)
    execution.to_csv(output / "execution_summary.csv", index=False)
    (output / "acceptance_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "protocol_sha256": actual,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _sha256(source / "cache" / "run_config.json"),
        "variant_artifact_sha256": {
            variant: {
                "run_config": _sha256(path / "run_config.json"),
                "fused_scores": _sha256(path / "walk_forward" / "fused_scores.csv"),
                "factor_decisions": _sha256(path / "walk_forward" / "factor_decisions.csv"),
            }
            for variant, path in variant_dirs.items()
        },
        "result": checks["verdict"],
        "live_approval": False,
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    charts = _plot_outputs(output, navs, common_start, pairwise, factor_rates, performance, execution)
    return {
        "performance": output / "performance_by_period.csv",
        "pairwise_summary": output / "pairwise_stability_summary.csv",
        "pairwise_by_period": output / "pairwise_stability_by_period.csv",
        "factor_rates": output / "factor_pass_rates.csv",
        "execution": output / "execution_summary.csv",
        "acceptance": output / "acceptance_checks.json",
        **{f"chart_{i + 1}": path for i, path in enumerate(charts)},
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
    variants = {"ROLLING_260D": args.rolling_260_dir, "ROLLING_504D": args.rolling_504_dir, "EXPANDING": args.expanding_dir}
    for key, path in run(args.source_output_dir, variants, args.output_dir, args.protocol).items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
