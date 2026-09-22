#!/usr/bin/env python3
"""Run the frozen simple factor-admission and fixed-family-fusion comparison."""
from __future__ import annotations

import argparse
import hashlib
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
from analysis.factor_redundancy import prune_redundant_factors
from analysis.performance import summarize
from analysis.walk_forward_selection import evaluate_factor_gate_asof
from backtest.backtest_multi import run_multi_backtest
from factors.preprocess import cross_sectional_zscore
from live.universe_history import (
    load_membership_intervals,
    mask_wide_prices_by_membership,
    membership_mask,
)
from scripts.build_walk_forward_factor_backtest import _settings_from_source


PRIMARY = "SIMPLE_ADMISSION_FIXED_FUSION_TOP50"
TOP10 = "SIMPLE_ADMISSION_FIXED_FUSION_TOP10"
WEEKLY_BENCHMARK = "CSI300_PIT_WEEKLY_EQUAL_WEIGHT_COSTED"
LEGACY_BENCHMARK = "CSI300_PIT_DAILY_EQUAL_WEIGHT_COSTLESS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.set_index(["date", "symbol"]).sort_index()


def _normalize(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    selected = series.loc[(series.index >= start) & (series.index <= end)].dropna().astype(float)
    return selected / float(selected.iloc[0]) if not selected.empty else selected


def _stats(
    strategy: str,
    period: str,
    nav: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    row: dict[str, Any] = {"strategy": strategy, "period": period}
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
    row["n_days"] = int(len(nav))
    return row


def select_factors_within_families(
    selection: pd.DataFrame,
    redundancy: pd.DataFrame,
    families: dict[str, list[str]],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Keep PASS only and apply the existing redundancy report within each family."""
    passed = set(
        selection.loc[selection["decision"].astype(str) == "PASS", "factor"].astype(str)
    )
    selected: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    for family, candidates in families.items():
        family_passed = [str(factor) for factor in candidates if str(factor) in passed]
        retained = prune_redundant_factors(family_passed, redundancy)
        if retained:
            selected[str(family)] = retained
        for factor in candidates:
            decision_rows = selection.loc[selection["factor"].astype(str) == str(factor)]
            decision = (
                str(decision_rows.iloc[0]["decision"]) if not decision_rows.empty else "UNAVAILABLE"
            )
            rows.append(
                {
                    "family": str(family),
                    "factor": str(factor),
                    "admission_decision": decision,
                    "passed_gate": str(factor) in family_passed,
                    "retained_after_within_family_redundancy": str(factor) in retained,
                }
            )
    return selected, pd.DataFrame(rows)


def build_fixed_family_fusion(
    panel: pd.DataFrame,
    selected: dict[str, list[str]],
    eligible: pd.Series,
    admission_as_of: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame]:
    """Average factors within family, z-score families, then equally average families."""
    family_scores: dict[str, pd.Series] = {}
    for family, factors in selected.items():
        family_scores[family] = panel[factors].mean(axis=1, skipna=True)
    if not family_scores:
        raise RuntimeError("no PASS factor remained after within-family redundancy")
    family_panel = pd.DataFrame(family_scores, index=panel.index)
    standardized = cross_sectional_zscore(family_panel)
    required = len(standardized.columns)
    fused = standardized.mean(axis=1, skipna=True).where(
        standardized.notna().sum(axis=1) == required
    )
    fused = fused.where(eligible.reindex(fused.index).fillna(False).astype(bool))
    fused = fused.where(fused.index.get_level_values("date") >= admission_as_of)
    fused.name = "score"
    return fused, standardized


def build_weekly_costed_pit_benchmark(
    pit_prices: pd.DataFrame,
    prices: pd.DataFrame,
    long_prices: pd.DataFrame,
    settings: Any,
) -> tuple[pd.Series, dict[str, Any]]:
    """Run all point-in-time members through the same weekly equal-weight engine."""
    valid = pit_prices.notna()
    score = valid.astype(float).where(valid).stack().rename("score")
    score.index = score.index.set_names(["date", "symbol"])
    benchmark_settings = replace(
        settings,
        portfolio_weighting="equal",
        max_position_weight=0.0,
        max_industry_weight=0.0,
        max_tech_growth_weight=0.0,
        target_volatility=0.0,
        min_positions=0,
        max_rebalance_turnover=0.0,
    )
    return run_multi_backtest(
        fused=score,
        prices=prices,
        settings=benchmark_settings,
        factor_name=WEEKLY_BENCHMARK,
        top_k=len(prices.columns),
        long_prices=long_prices,
        empty_signal_policy="cash",
    )


def _execution_summary(name: str, meta: dict[str, Any], start: pd.Timestamp) -> dict[str, Any]:
    logs = [
        rec
        for rec in meta.get("rebalance_log", [])
        if pd.Timestamp(rec["date"]) >= start
    ]
    prior: dict[str, float] = {}
    total_turnover = 0.0
    holding_counts: list[int] = []
    gross_weights: list[float] = []
    for rec in logs:
        current = {
            str(symbol): float(weight)
            for symbol, weight in zip(rec.get("picks", []), rec.get("weights", []))
        }
        symbols = set(prior) | set(current)
        total_turnover += sum(
            abs(current.get(symbol, 0.0) - prior.get(symbol, 0.0)) for symbol in symbols
        )
        prior = current
        holding_counts.append(sum(weight > 1e-12 for weight in current.values()))
        gross_weights.append(sum(current.values()))
    return {
        "strategy": name,
        "n_rebalances": len(logs),
        "actual_target_turnover": float(total_turnover),
        "average_holding_count": float(np.mean(holding_counts)) if holding_counts else np.nan,
        "max_holding_count": int(max(holding_counts)) if holding_counts else 0,
        "average_gross_target_weight": float(np.mean(gross_weights)) if gross_weights else np.nan,
    }


def _load_prior_nav_comparison(root: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "112",
            "A50",
            "STRICT_WALK_FORWARD",
            root / "output/a50_walk_forward_20260904/nav_comparison.csv",
            "FUSED_WALK_FORWARD_SCORE_WEIGHTED",
        ),
        (
            "113",
            "CSI300",
            "STRICT_WALK_FORWARD_TOP10",
            root / "output/csi300_walk_forward_weekly_20260904/nav_comparison.csv",
            "FUSED_WALK_FORWARD_SCORE_WEIGHTED",
        ),
        (
            "114",
            "CSI300",
            "EXPANDING_TOP10_BASELINE",
            root / "output/csi300_preregistered_state_gate_20260909/nav_comparison.csv",
            "CSI300_WF_TOP10_BASELINE",
        ),
        (
            "114",
            "CSI300",
            "STATE_GATE_V1",
            root / "output/csi300_preregistered_state_gate_20260909/nav_comparison.csv",
            "CSI300_WF_TOP10_STATE_GATE_V1",
        ),
        (
            "115",
            "CSI300",
            "ROLLING_260D",
            root / "output/csi300_history_window_stability_20260910/nav_comparison.csv",
            "ROLLING_260D",
        ),
        (
            "115",
            "CSI300",
            "ROLLING_504D",
            root / "output/csi300_history_window_stability_20260910/nav_comparison.csv",
            "ROLLING_504D",
        ),
        (
            "115",
            "CSI300",
            "EXPANDING",
            root / "output/csi300_history_window_stability_20260910/nav_comparison.csv",
            "EXPANDING",
        ),
        (
            "116",
            "CSI300",
            "CONSENSUS_3WAY",
            root / "output/csi300_multi_memory_consensus_v2_20260912/nav_comparison.csv",
            "CONSENSUS_3WAY",
        ),
    ]
    rows: list[dict[str, Any]] = []
    periods = {
        "COMMON_2025_EVAL": (pd.Timestamp("2025-01-03"), pd.Timestamp("2026-09-04")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), pd.Timestamp("2026-09-04")),
    }
    cache: dict[Path, pd.DataFrame] = {}
    for article, universe, label, path, column in specs:
        if path not in cache:
            cache[path] = pd.read_csv(path, index_col=0, parse_dates=True)
        nav = cache[path][column].dropna().astype(float)
        for period, (start, end) in periods.items():
            selected = _normalize(nav, start, end)
            stats = summarize(selected)
            rows.append(
                {
                    "source_article": article,
                    "universe": universe,
                    "strategy": label,
                    "period": period,
                    **stats,
                    "start": selected.index.min().strftime("%Y-%m-%d"),
                    "end": selected.index.max().strftime("%Y-%m-%d"),
                    "comparison_note": "normalized_saved_nav; methodology differs",
                }
            )
    return rows


def _plot_results(
    output: Path,
    navs: pd.DataFrame,
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    component_performance: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    active = navs.loc[(navs.index >= start) & (navs.index <= end)].copy()
    active = active.apply(lambda series: series / float(series.dropna().iloc[0]))
    colors = {
        PRIMARY: "#7c3aed",
        TOP10: "#d95f02",
        WEEKLY_BENCHMARK: "#0f766e",
        LEGACY_BENCHMARK: "#7f8c8d",
    }
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for column in active.columns:
        axes[0].plot(active.index, active[column], label=column, color=colors[column], linewidth=2)
        if column in {PRIMARY, TOP10}:
            axes[1].plot(
                active.index,
                active[column] / active[column].cummax() - 1.0,
                label=column,
                color=colors[column],
            )
    axes[0].set_title("Simple fixed-family fusion versus like-for-like and legacy benchmarks")
    axes[0].set_ylabel("NAV")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "simple_fusion_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    counts = selection["decision"].value_counts().reindex(["PASS", "WATCH", "REJECT"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(counts.index, counts.values, color=["#1a9850", "#fdae61", "#d73027"])
    ax.set_ylabel("Factor count")
    ax.set_title("One-time admission decision as of 2025-01-03")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "admission_decision_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    known = comparison[comparison["period"] == "KNOWN_DEVELOPMENT"].copy()
    known = known.sort_values("total_return")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    labels = known["source_article"].astype(str) + " / " + known["strategy"].astype(str)
    ax.barh(labels, known["total_return"] * 100, color="#4c78a8")
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("Total return (%)")
    ax.set_title("Retrospective common-period context; rows are not a model-selection ranking")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "comparison_with_articles_112_116.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    components = component_performance[
        component_performance["period"] == "FULL_EVALUATION"
    ].sort_values("return_gap_vs_benchmark")
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.barh(
        components["component"],
        components["return_gap_vs_benchmark"] * 100,
        color=np.where(
            components["return_gap_vs_benchmark"] >= 0.0, "#1a9850", "#d73027"
        ),
    )
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("Return gap versus weekly costed PIT equal weight (percentage points)")
    ax.set_title("Post-hoc component attribution; excluded from frozen acceptance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "posthoc_component_return_gaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(source: Path, output: Path, protocol_path: Path) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch; frozen simple-fusion protocol changed")

    source_settings, _ = _settings_from_source(source, output)
    settings = replace(
        source_settings,
        rebalance_freq="W-FRI",
        top_k=50,
        portfolio_weighting="equal",
        max_position_weight=0.0,
        max_industry_weight=0.0,
        max_tech_growth_weight=0.0,
        target_volatility=0.0,
        min_positions=0,
        max_rebalance_turnover=0.0,
    )
    panel = _load_panel(source / "cache/factor_panel_zscore.csv")
    prices = pd.read_csv(
        source / "cache/prices_wide_adj_close.csv", index_col=0, parse_dates=True
    ).sort_index()
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source / "cache/prices_long.csv", parse_dates=["trade_date"])
    membership = load_membership_intervals(settings.universe_membership_path)
    eligible = membership_mask(panel.index, membership)
    pit_prices = mask_wide_prices_by_membership(prices, membership)

    families = {
        str(family): [str(factor) for factor in candidates]
        for family, candidates in protocol["candidate_families"].items()
    }
    candidates = [factor for values in families.values() for factor in values]
    missing = [factor for factor in candidates if factor not in panel.columns]
    if missing:
        raise RuntimeError(f"protocol candidate factors missing from panel: {missing}")

    admission_as_of = pd.Timestamp(protocol["data"]["admission_as_of"])
    history_dates = prices.index[prices.index < admission_as_of]
    panel_history = panel.loc[
        panel.index.get_level_values("date").isin(history_dates), candidates
    ]
    eligible_history = eligible.reindex(panel_history.index).fillna(False).astype(bool)
    panel_history = panel_history.where(eligible_history)
    prices_history = prices.loc[history_dates]
    benchmark_history = pit_prices.loc[history_dates]
    selection, rolling, multi, redundancy = evaluate_factor_gate_asof(
        panel_history,
        prices_history,
        settings,
        factors=candidates,
        eligible_mask_history=eligible_history,
        benchmark_prices_history=benchmark_history,
    )
    selected, family_audit = select_factors_within_families(
        selection, redundancy, families
    )
    fused, family_scores = build_fixed_family_fusion(
        panel[candidates], selected, eligible, admission_as_of
    )

    nav50, meta50 = run_multi_backtest(
        fused=fused,
        prices=prices,
        settings=settings,
        factor_name=PRIMARY,
        top_k=50,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )
    nav10, meta10 = run_multi_backtest(
        fused=fused,
        prices=prices,
        settings=settings,
        factor_name=TOP10,
        top_k=10,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )
    weekly_benchmark, weekly_meta = build_weekly_costed_pit_benchmark(
        pit_prices, prices, long_prices, settings
    )
    weekly_benchmark = weekly_benchmark.rename(WEEKLY_BENCHMARK)
    legacy_benchmark = equal_weight_benchmark_nav(
        pit_prices, dates=prices.index, name=LEGACY_BENCHMARK
    )

    periods = {
        "FULL_EVALUATION": (admission_as_of, pd.Timestamp("2026-09-04")),
        "EARLY_COMPARISON": (admission_as_of, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), pd.Timestamp("2026-09-04")),
    }
    performance_rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        bench = _normalize(weekly_benchmark, start, end)
        for name, nav in [(PRIMARY, nav50), (TOP10, nav10)]:
            performance_rows.append(
                _stats(name, period, _normalize(nav, start, end), bench)
            )
    performance = pd.DataFrame(performance_rows)

    component_navs: dict[str, pd.Series] = {}
    component_metas: dict[str, dict[str, Any]] = {}
    retained_factors = [factor for values in selected.values() for factor in values]
    diagnostic_scores: dict[str, pd.Series] = {
        f"FACTOR_{factor}": panel[factor]
        .where(eligible.reindex(panel.index).fillna(False).astype(bool))
        .where(panel.index.get_level_values("date") >= admission_as_of)
        for factor in retained_factors
    }
    diagnostic_scores.update(
        {
            f"FAMILY_{family}": family_scores[family]
            .where(eligible.reindex(family_scores.index).fillna(False).astype(bool))
            .where(family_scores.index.get_level_values("date") >= admission_as_of)
            for family in selected
        }
    )
    for name, score in diagnostic_scores.items():
        component_nav, component_meta = run_multi_backtest(
            fused=score.rename("score"),
            prices=prices,
            settings=settings,
            factor_name=name,
            top_k=50,
            long_prices=long_prices,
            empty_signal_policy="cash",
        )
        component_navs[name] = component_nav.rename(name)
        component_metas[name] = component_meta
    component_rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        bench = _normalize(weekly_benchmark, start, end)
        for name, component_nav in component_navs.items():
            row = _stats(name, period, _normalize(component_nav, start, end), bench)
            row["component"] = name
            component_rows.append(row)
    component_performance = pd.DataFrame(component_rows)

    comparison_rows = _load_prior_nav_comparison(ROOT)
    current_navs = {
        "CURRENT_TOP50": nav50,
        "CURRENT_TOP10": nav10,
        "CURRENT_WEEKLY_PIT_BENCHMARK": weekly_benchmark,
    }
    comparison_periods = {
        "COMMON_2025_EVAL": (admission_as_of, pd.Timestamp("2026-09-04")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), pd.Timestamp("2026-09-04")),
    }
    for label, nav in current_navs.items():
        for period, (start, end) in comparison_periods.items():
            selected_nav = _normalize(nav, start, end)
            comparison_rows.append(
                {
                    "source_article": "RESET_TEST",
                    "universe": "CSI300",
                    "strategy": label,
                    "period": period,
                    **summarize(selected_nav),
                    "start": selected_nav.index.min().strftime("%Y-%m-%d"),
                    "end": selected_nav.index.max().strftime("%Y-%m-%d"),
                    "comparison_note": "current frozen retrospective test",
                }
            )
    comparison = pd.DataFrame(comparison_rows)

    full = performance[performance["period"] == "FULL_EVALUATION"].set_index("strategy")
    early = performance[performance["period"] == "EARLY_COMPARISON"].set_index("strategy")
    known = performance[performance["period"] == "KNOWN_DEVELOPMENT"].set_index("strategy")
    checks = {
        "protocol_sha256": actual,
        "admission_as_of": admission_as_of.strftime("%Y-%m-%d"),
        "admission_history_end": pd.Timestamp(history_dates[-1]).strftime("%Y-%m-%d"),
        "admission_history_strictly_before_signal": bool(history_dates[-1] < admission_as_of),
        "selected_families": selected,
        "selected_factor_count": int(sum(len(values) for values in selected.values())),
        "primary_full_total_return": float(full.loc[PRIMARY, "total_return"]),
        "primary_full_benchmark_total_return": float(
            full.loc[PRIMARY, "benchmark_total_return"]
        ),
        "primary_full_return_greater_than_benchmark_pass": bool(
            full.loc[PRIMARY, "return_gap_vs_benchmark"] > 0.0
        ),
        "primary_full_annualized_excess": float(
            full.loc[PRIMARY, "excess_ann_return"]
        ),
        "primary_full_annualized_excess_pass": bool(
            full.loc[PRIMARY, "excess_ann_return"] > 0.0
        ),
        "primary_full_information_ratio": float(
            full.loc[PRIMARY, "information_ratio"]
        ),
        "primary_full_information_ratio_pass": bool(
            full.loc[PRIMARY, "information_ratio"] > 0.0
        ),
        "primary_early_return_gap": float(
            early.loc[PRIMARY, "return_gap_vs_benchmark"]
        ),
        "primary_known_return_gap": float(
            known.loc[PRIMARY, "return_gap_vs_benchmark"]
        ),
        "primary_positive_return_gap_in_both_subperiods_pass": bool(
            early.loc[PRIMARY, "return_gap_vs_benchmark"] > 0.0
            and known.loc[PRIMARY, "return_gap_vs_benchmark"] > 0.0
        ),
        "top10_is_sensitivity_only": True,
        "posthoc_component_diagnostics_excluded_from_acceptance": True,
        "historical_data_already_seen": True,
        "live_approval": False,
    }
    required = [key for key in checks if key.endswith("_pass")]
    checks["historical_acceptance_pass"] = bool(all(checks[key] for key in required))
    checks["verdict"] = (
        "RETROSPECTIVE_SIMPLE_FUSION_CANDIDATE_ONLY"
        if checks["historical_acceptance_pass"]
        else "SIMPLE_FUSION_NO_GO"
    )

    output.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output / "factor_admission.csv", index=False)
    family_audit.to_csv(output / "family_selection_audit.csv", index=False)
    rolling.to_csv(output / "rolling_oos_summary_at_admission.csv", index=False)
    multi.to_csv(output / "multi_horizon_summary_at_admission.csv", index=False)
    redundancy.to_csv(output / "redundancy_at_admission.csv", index=False)
    family_scores.reset_index().to_csv(
        output / "family_scores.csv", index=False, date_format="%Y-%m-%d"
    )
    fused.reset_index().to_csv(
        output / "fused_scores.csv", index=False, date_format="%Y-%m-%d"
    )
    navs = pd.concat(
        [
            nav50.rename(PRIMARY),
            nav10.rename(TOP10),
            weekly_benchmark,
            legacy_benchmark,
        ],
        axis=1,
    )
    navs.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    component_performance.to_csv(
        output / "posthoc_component_performance_by_period.csv", index=False
    )
    comparison.to_csv(output / "comparison_with_articles_112_116.csv", index=False)
    pd.DataFrame(
        [
            _execution_summary(PRIMARY, meta50, admission_as_of),
            _execution_summary(TOP10, meta10, admission_as_of),
            _execution_summary(WEEKLY_BENCHMARK, weekly_meta, admission_as_of),
            *[
                _execution_summary(name, component_metas[name], admission_as_of)
                for name in component_metas
            ],
        ]
    ).to_csv(output / "execution_summary.csv", index=False)
    article111 = pd.read_csv(
        ROOT
        / "output/a50_pit_integrated_backtest_20260904_multihorizon/performance_summary.csv"
    )
    article111.loc[
        article111["strategy"].isin(["FUSED_ROLLING_SCORE_WEIGHTED", "BENCH_EQUAL_WEIGHT"])
    ].to_csv(output / "article_111_original_reported_context.csv", index=False)
    (output / "acceptance_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chart_paths = _plot_results(
        output,
        navs,
        comparison,
        selection,
        component_performance,
        admission_as_of,
        pd.Timestamp("2026-09-04"),
    )
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_head = "UNKNOWN"
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": actual,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_factor_panel_sha256": _sha256(source / "cache/factor_panel_zscore.csv"),
        "source_prices_sha256": _sha256(source / "cache/prices_wide_adj_close.csv"),
        "code_sha256": _sha256(Path(__file__)),
        "git_head_at_evaluation": git_head,
        "verdict": checks["verdict"],
        "live_approval": False,
        "true_forward_status": "NOT_STARTED",
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "admission": output / "factor_admission.csv",
        "family_selection": output / "family_selection_audit.csv",
        "performance": output / "performance_by_period.csv",
        "comparison": output / "comparison_with_articles_112_116.csv",
        "acceptance": output / "acceptance_checks.json",
        "manifest": output / "audit_manifest.json",
        **{f"chart_{index + 1}": path for index, path in enumerate(chart_paths)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(args.source_output_dir, args.output_dir, args.protocol).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
