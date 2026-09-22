#!/usr/bin/env python3
"""Backtest soft factor governance against prior no-admission and strict controls."""
from __future__ import annotations

import argparse
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

from backtest.backtest_multi import run_multi_backtest
from factors.preprocess import cross_sectional_zscore
from live.universe_history import load_membership_intervals, membership_mask
from scripts.build_no_admission_fusion_backtest import _period_rows
from scripts.build_no_admission_rolling_ic_weight_backtest import PRIMARY as ROLLING_IC
from scripts.build_simple_admission_fusion_backtest import (
    WEEKLY_BENCHMARK,
    _execution_summary,
    _load_panel,
    _sha256,
)
from scripts.build_walk_forward_factor_backtest import _settings_from_source


SOFT_MAIN = "SOFT_GOVERNANCE_MAIN_TOP50"
SOFT_HALF = "SOFT_WATCH_HALF_SHADOW_TOP50"
FIXED_NO_ADMISSION = "NO_ADMISSION_ALL_FACTOR_FIXED_FUSION_TOP50"
STRICT_ADMISSION = "SIMPLE_ADMISSION_FIXED_FUSION_TOP50"


def classify_governance_state(
    row: pd.Series,
    *,
    minimum_coverage: float,
    minimum_valid_symbols: int,
) -> tuple[str, str]:
    """Map the old performance decision into the new governance state machine."""
    as_of = pd.Timestamp(row["as_of_date"])
    history_end = pd.Timestamp(row["history_end"])
    if pd.isna(history_end) or history_end >= as_of:
        return "QUARANTINE", "non_point_in_time_or_missing_history_cutoff"
    coverage = pd.to_numeric(pd.Series([row.get("coverage")]), errors="coerce").iloc[0]
    valid_symbols = pd.to_numeric(pd.Series([row.get("valid_symbols")]), errors="coerce").iloc[0]
    if not np.isfinite(coverage) or float(coverage) < float(minimum_coverage):
        return "QUARANTINE", "historical_coverage_below_threshold"
    if not np.isfinite(valid_symbols) or int(valid_symbols) < int(minimum_valid_symbols):
        return "QUARANTINE", "valid_symbols_below_threshold"
    old = str(row.get("decision", "")).upper()
    if old == "PASS":
        return "ACTIVE", "old_pass_maps_to_active"
    if old in {"WATCH", "REJECT"}:
        return "WATCH", f"old_{old.lower()}_maps_to_watch"
    return "QUARANTINE", "unknown_or_missing_old_decision"


def build_governance_log(
    decisions: pd.DataFrame,
    families: dict[str, list[str]],
    variants: dict[str, dict[str, Any]],
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    minimum_coverage: float,
    minimum_valid_symbols: int,
) -> pd.DataFrame:
    """Create auditable state multipliers and normalized factor weights."""
    family_by_factor = {
        factor: family for family, factors in families.items() for factor in factors
    }
    factors = list(family_by_factor)
    work = decisions[
        decisions["factor"].astype(str).isin(factors)
        & decisions["as_of_date"].between(evaluation_start, evaluation_end)
    ].copy()
    if work.duplicated(["as_of_date", "factor"]).any():
        raise RuntimeError("duplicate point-in-time decision rows")
    counts = work.groupby("as_of_date")["factor"].nunique()
    if counts.empty or not bool((counts == len(factors)).all()):
        raise RuntimeError("incomplete factor health audit on one or more rebalance dates")
    states = work.apply(
        lambda row: classify_governance_state(
            row,
            minimum_coverage=minimum_coverage,
            minimum_valid_symbols=minimum_valid_symbols,
        ),
        axis=1,
        result_type="expand",
    )
    work[["governance_state", "governance_reason"]] = states
    work["family"] = work["factor"].map(family_by_factor)
    work["retired"] = False
    rows: list[dict[str, Any]] = []
    for dt, date_rows in work.groupby("as_of_date", sort=True):
        for variant, config in variants.items():
            state_multiplier = {
                "ACTIVE": float(config["active_multiplier"]),
                "WATCH": float(config["watch_multiplier"]),
                "QUARANTINE": 0.0,
                "RETIRED": 0.0,
            }
            for family, factors_in_family in families.items():
                current = date_rows[date_rows["family"] == family].set_index("factor")
                raw = current["governance_state"].map(state_multiplier).astype(float)
                total = float(raw.sum())
                if total <= 0.0:
                    normalized = pd.Series(0.0, index=factors_in_family)
                else:
                    normalized = raw / total
                for factor in factors_in_family:
                    rec = current.loc[factor]
                    rows.append(
                        {
                            "date": pd.Timestamp(dt),
                            "variant": variant,
                            "family": family,
                            "factor": factor,
                            "old_decision": rec["decision"],
                            "old_reasons": rec.get("reasons", ""),
                            "governance_state": rec["governance_state"],
                            "governance_reason": rec["governance_reason"],
                            "coverage": float(rec["coverage"]),
                            "valid_symbols": int(rec["valid_symbols"]),
                            "history_start": pd.Timestamp(rec["history_start"]),
                            "history_end": pd.Timestamp(rec["history_end"]),
                            "state_multiplier": float(raw[factor]),
                            "within_family_weight": float(normalized[factor]),
                            "family_weight": 1.0 / len(families),
                            "total_factor_weight": float(normalized[factor] / len(families)),
                            "retired": False,
                        }
                    )
    return pd.DataFrame(rows)


def build_state_weighted_fusion(
    panel: pd.DataFrame,
    governance_log: pd.DataFrame,
    families: dict[str, list[str]],
    eligible: pd.Series,
    variant: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fuse all non-quarantined factors using the registered state multipliers."""
    score_parts: list[pd.Series] = []
    family_parts: list[pd.DataFrame] = []
    selected_log = governance_log[governance_log["variant"] == variant]
    for dt, rows in selected_log.groupby("date", sort=True):
        date = pd.Timestamp(dt)
        try:
            current = panel.xs(date, level="date")
        except KeyError:
            continue
        indexed = rows.set_index("factor")
        family_scores: dict[str, pd.Series] = {}
        for family, factors in families.items():
            weights = indexed.loc[factors, "within_family_weight"].astype(float)
            if float(weights.sum()) <= 0.0:
                family_scores[family] = pd.Series(np.nan, index=current.index)
                continue
            values = current[factors].astype(float)
            numerator = values.mul(weights, axis=1).sum(axis=1, min_count=1)
            denominator = values.notna().mul(weights, axis=1).sum(axis=1)
            family_scores[family] = numerator / denominator.replace(0.0, np.nan)
        family_panel = pd.DataFrame(family_scores, index=current.index)
        family_panel.index = pd.MultiIndex.from_product(
            [[date], family_panel.index.astype(str)], names=["date", "symbol"]
        )
        standardized = cross_sectional_zscore(family_panel)
        fused = standardized.mean(axis=1, skipna=True).where(
            standardized.notna().sum(axis=1) == len(families)
        )
        fused = fused.where(eligible.reindex(fused.index).fillna(False).astype(bool))
        score_parts.append(fused)
        family_parts.append(standardized)
    if not score_parts:
        raise RuntimeError(f"no scores produced for {variant}")
    return (
        pd.concat(score_parts).sort_index().rename("score"),
        pd.concat(family_parts).sort_index(),
    )


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    performance: pd.DataFrame,
    governance: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    order = [SOFT_MAIN, SOFT_HALF, FIXED_NO_ADMISSION, ROLLING_IC, STRICT_ADMISSION, WEEKLY_BENCHMARK]
    labels = {
        SOFT_MAIN: "Soft governance main",
        SOFT_HALF: "WATCH half-weight shadow",
        FIXED_NO_ADMISSION: "No-admission fixed",
        ROLLING_IC: "Rolling IC/ICIR",
        STRICT_ADMISSION: "Strict admission",
        WEEKLY_BENCHMARK: "Weekly PIT benchmark",
    }
    colors = {
        SOFT_MAIN: "#d95f02",
        SOFT_HALF: "#e6ab02",
        FIXED_NO_ADMISSION: "#7c3aed",
        ROLLING_IC: "#1f78b4",
        STRICT_ADMISSION: "#7570b3",
        WEEKLY_BENCHMARK: "#0f766e",
    }
    active = navs.loc[(navs.index >= start) & (navs.index <= end), order].copy()
    active = active.apply(lambda series: series / float(series.dropna().iloc[0]))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for strategy in order:
        width = 2.4 if strategy in {SOFT_MAIN, SOFT_HALF} else 1.6
        axes[0].plot(active.index, active[strategy], label=labels[strategy], color=colors[strategy], linewidth=width)
        if strategy != WEEKLY_BENCHMARK:
            axes[1].plot(active.index, active[strategy] / active[strategy].cummax() - 1.0, label=labels[strategy], color=colors[strategy], linewidth=width)
    axes[0].set_title("Soft factor governance versus prior fusion methods")
    axes[0].set_ylabel("NAV")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "soft_governance_method_comparison_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    periods = ["EARLY_COMPARISON", "KNOWN_DEVELOPMENT", "FULL_EVALUATION"]
    strategies = [SOFT_MAIN, SOFT_HALF, FIXED_NO_ADMISSION, ROLLING_IC, STRICT_ADMISSION]
    gap = performance.pivot(index="strategy", columns="period", values="return_gap_vs_benchmark").reindex(index=strategies, columns=periods)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(periods))
    width = 0.16
    offsets = np.arange(len(strategies)) - (len(strategies) - 1) / 2
    for index, strategy in enumerate(strategies):
        ax.bar(x + offsets[index] * width, gap.loc[strategy].to_numpy(float) * 100, width, label=labels[strategy], color=colors[strategy])
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x, periods)
    ax.set_ylabel("Return gap versus benchmark (percentage points)")
    ax.set_title("Five factor-governance methods by evaluation period")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "soft_governance_period_gaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    main = governance[governance["variant"] == "SOFT_GOVERNANCE_MAIN"]
    counts = main.groupby(["factor", "governance_state"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["ACTIVE", "WATCH", "QUARANTINE"], fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).sort_values("ACTIVE")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    left = np.zeros(len(shares))
    state_colors = {"ACTIVE": "#1b9e77", "WATCH": "#e6ab02", "QUARANTINE": "#d73027"}
    for state in shares.columns:
        values = shares[state].to_numpy(float) * 100
        ax.barh(shares.index, values, left=left, label=state, color=state_colors[state])
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of 87 weekly rebalances (%)")
    ax.set_title("Point-in-time governance state frequency by factor")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "soft_governance_state_frequency.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(
    source: Path,
    decisions_path: Path,
    fixed_output: Path,
    rolling_output: Path,
    strict_output: Path,
    output: Path,
    protocol_path: Path,
) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch; frozen soft-governance protocol changed")

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
    prices = pd.read_csv(source / "cache/prices_wide_adj_close.csv", index_col=0, parse_dates=True).sort_index()
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source / "cache/prices_long.csv", parse_dates=["trade_date"])
    membership = load_membership_intervals(settings.universe_membership_path)
    eligible = membership_mask(panel.index, membership)
    decisions = pd.read_csv(
        decisions_path,
        parse_dates=["as_of_date", "history_start", "history_end"],
    )
    families = {str(family): [str(factor) for factor in factors] for family, factors in protocol["candidate_families"].items()}
    factors = [factor for factors_in_family in families.values() for factor in factors_in_family]
    missing = [factor for factor in factors if factor not in panel.columns]
    if missing:
        raise RuntimeError(f"protocol factor missing from panel: {missing}")
    start = pd.Timestamp(protocol["data"]["evaluation_start"])
    end = pd.Timestamp(protocol["data"]["evaluation_end"])
    governance = build_governance_log(
        decisions,
        families,
        protocol["variants"],
        evaluation_start=start,
        evaluation_end=end,
        minimum_coverage=float(protocol["data_logic_gate"]["minimum_historical_coverage"]),
        minimum_valid_symbols=int(protocol["data_logic_gate"]["minimum_valid_symbols"]),
    )
    main_fused, main_families = build_state_weighted_fusion(panel[factors], governance, families, eligible, "SOFT_GOVERNANCE_MAIN")
    half_fused, half_families = build_state_weighted_fusion(panel[factors], governance, families, eligible, "SOFT_WATCH_HALF_SHADOW")
    main_nav, main_meta = run_multi_backtest(
        fused=main_fused,
        prices=prices,
        settings=settings,
        factor_name=SOFT_MAIN,
        top_k=50,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )
    half_nav, half_meta = run_multi_backtest(
        fused=half_fused,
        prices=prices,
        settings=settings,
        factor_name=SOFT_HALF,
        top_k=50,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )

    fixed_navs = pd.read_csv(fixed_output / "nav_comparison.csv", index_col=0, parse_dates=True)
    rolling_navs = pd.read_csv(rolling_output / "nav_comparison.csv", index_col=0, parse_dates=True)
    fixed = fixed_navs[FIXED_NO_ADMISSION].astype(float)
    strict = fixed_navs[STRICT_ADMISSION].astype(float)
    benchmark = fixed_navs[WEEKLY_BENCHMARK].astype(float)
    rolling = rolling_navs[ROLLING_IC].astype(float)
    aligned_benchmark = pd.concat([benchmark, rolling_navs[WEEKLY_BENCHMARK]], axis=1).dropna()
    if not np.allclose(aligned_benchmark.iloc[:, 0], aligned_benchmark.iloc[:, 1], atol=1e-12):
        raise RuntimeError("benchmark changed between prior experiments")
    periods = {
        "FULL_EVALUATION": (start, end),
        "EARLY_COMPARISON": (start, pd.Timestamp(protocol["data"]["early_comparison_period"][1])),
        "KNOWN_DEVELOPMENT": (pd.Timestamp(protocol["data"]["known_development_period"][0]), end),
    }
    strategies = {
        SOFT_MAIN: main_nav,
        SOFT_HALF: half_nav,
        FIXED_NO_ADMISSION: fixed,
        ROLLING_IC: rolling,
        STRICT_ADMISSION: strict,
    }
    performance = _period_rows(strategies, benchmark, periods)
    full = performance[performance["period"] == "FULL_EVALUATION"].set_index("strategy")
    main_aligned = pd.concat([main_nav.rename("main"), fixed.rename("fixed")], axis=1).dropna()
    state_counts = governance[governance["variant"] == "SOFT_GOVERNANCE_MAIN"]["governance_state"].value_counts()
    checks: dict[str, Any] = {
        "protocol_sha256": actual,
        "rebalance_dates": int(governance["date"].nunique()),
        "candidate_factors": factors,
        "main_active_watch_weights_strictly_positive_pass": bool(
            (governance.loc[
                (governance["variant"] == "SOFT_GOVERNANCE_MAIN")
                & governance["governance_state"].isin(["ACTIVE", "WATCH"]),
                "total_factor_weight",
            ] > 0.0).all()
        ),
        "history_cutoff_strict_pass": bool((governance["history_end"] < governance["date"]).all()),
        "no_factor_retired_from_seen_history_pass": bool(~governance["retired"].any()),
        "factor_weights_sum_to_one_pass": bool(
            np.allclose(
                governance.groupby(["variant", "date"])["total_factor_weight"].sum().to_numpy(float),
                1.0,
                atol=1e-10,
            )
        ),
        "families_equal_twenty_percent_pass": bool(
            np.allclose(
                governance.groupby(["variant", "date", "family"])["total_factor_weight"].sum().to_numpy(float),
                0.2,
                atol=1e-10,
            )
        ),
        "main_equals_no_admission_fixed_nav_pass": bool(
            np.allclose(main_aligned["main"], main_aligned["fixed"], atol=1e-12)
        ),
        "active_state_rows": int(state_counts.get("ACTIVE", 0)),
        "watch_state_rows": int(state_counts.get("WATCH", 0)),
        "quarantine_state_rows": int(state_counts.get("QUARANTINE", 0)),
        "retired_state_rows": 0,
        "main_full_total_return": float(full.loc[SOFT_MAIN, "total_return"]),
        "half_watch_full_total_return": float(full.loc[SOFT_HALF, "total_return"]),
        "fixed_no_admission_full_total_return": float(full.loc[FIXED_NO_ADMISSION, "total_return"]),
        "rolling_ic_full_total_return": float(full.loc[ROLLING_IC, "total_return"]),
        "strict_admission_full_total_return": float(full.loc[STRICT_ADMISSION, "total_return"]),
        "half_watch_minus_main_full_return": float(full.loc[SOFT_HALF, "total_return"] - full.loc[SOFT_MAIN, "total_return"]),
        "half_watch_improves_main": bool(full.loc[SOFT_HALF, "total_return"] > full.loc[SOFT_MAIN, "total_return"]),
        "historical_data_already_seen": True,
        "live_approval": False,
        "verdict": "RETROSPECTIVE_SOFT_GOVERNANCE_COMPARISON_ONLY",
    }

    output.mkdir(parents=True, exist_ok=True)
    governance.to_csv(output / "factor_governance_log.csv", index=False, date_format="%Y-%m-%d")
    main_families.reset_index().assign(variant="SOFT_GOVERNANCE_MAIN").to_csv(output / "main_family_scores.csv", index=False, date_format="%Y-%m-%d")
    half_families.reset_index().assign(variant="SOFT_WATCH_HALF_SHADOW").to_csv(output / "half_watch_family_scores.csv", index=False, date_format="%Y-%m-%d")
    pd.concat([main_fused.rename(SOFT_MAIN), half_fused.rename(SOFT_HALF)], axis=1).reset_index().to_csv(output / "fused_scores.csv", index=False, date_format="%Y-%m-%d")
    nav_frame = pd.concat(
        [
            main_nav.rename(SOFT_MAIN),
            half_nav.rename(SOFT_HALF),
            fixed.rename(FIXED_NO_ADMISSION),
            rolling.rename(ROLLING_IC),
            strict.rename(STRICT_ADMISSION),
            benchmark.rename(WEEKLY_BENCHMARK),
        ],
        axis=1,
    )
    nav_frame.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    execution = pd.DataFrame([
        _execution_summary(SOFT_MAIN, main_meta, start),
        _execution_summary(SOFT_HALF, half_meta, start),
    ])
    prior_execution = []
    for path in [
        fixed_output / "execution_summary.csv",
        rolling_output / "execution_summary.csv",
        strict_output / "execution_summary.csv",
    ]:
        prior_execution.append(pd.read_csv(path))
    prior = pd.concat(prior_execution, ignore_index=True).drop_duplicates("strategy", keep="first")
    prior = prior[prior["strategy"].isin([FIXED_NO_ADMISSION, ROLLING_IC, STRICT_ADMISSION, WEEKLY_BENCHMARK])]
    pd.concat([execution, prior], ignore_index=True).to_csv(output / "execution_summary.csv", index=False)
    (output / "comparison_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    charts = _plot_outputs(output, nav_frame, performance, governance, start, end)

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        git_head = "UNKNOWN"
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": actual,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_factor_panel_sha256": _sha256(source / "cache/factor_panel_zscore.csv"),
        "point_in_time_decisions_sha256": _sha256(decisions_path),
        "code_sha256": _sha256(Path(__file__)),
        "git_head_at_evaluation": git_head,
        "verdict": checks["verdict"],
        "live_approval": False,
        "true_forward_status": "NOT_STARTED",
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "performance": output / "performance_by_period.csv",
        "governance_log": output / "factor_governance_log.csv",
        "checks": output / "comparison_checks.json",
        "manifest": output / "audit_manifest.json",
        **{f"chart_{index + 1}": path for index, path in enumerate(charts)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--fixed-output-dir", type=Path, required=True)
    parser.add_argument("--rolling-output-dir", type=Path, required=True)
    parser.add_argument("--strict-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(
        args.source_output_dir,
        args.decisions,
        args.fixed_output_dir,
        args.rolling_output_dir,
        args.strict_output_dir,
        args.output_dir,
        args.protocol,
    ).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
