#!/usr/bin/env python3
"""Compare no-admission rolling IC/ICIR weights with fixed family fusion."""
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

from analysis.ic import daily_ic_spearman
from backtest.backtest_multi import run_multi_backtest
from factors.preprocess import cross_sectional_zscore
from live.universe_history import (
    load_membership_intervals,
    mask_wide_prices_by_membership,
    membership_mask,
)
from scripts.build_no_admission_fusion_backtest import _period_rows
from scripts.build_simple_admission_fusion_backtest import (
    WEEKLY_BENCHMARK,
    _execution_summary,
    _load_panel,
    _sha256,
    build_weekly_costed_pit_benchmark,
)
from scripts.build_walk_forward_factor_backtest import _settings_from_source


PRIMARY = "NO_ADMISSION_ROLLING_IC_ICIR_FUSION_TOP50"
FIXED_CONTROL = "NO_ADMISSION_ALL_FACTOR_FIXED_FUSION_TOP50"


def _known_ic_window(
    trading_dates: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    *,
    forward_days: int,
    lookback_days: int,
) -> tuple[pd.DatetimeIndex, pd.Timestamp | None, pd.Timestamp | None]:
    """Return factor dates whose complete forward return is known before rebalance."""
    history = pd.DatetimeIndex(trading_dates[trading_dates < pd.Timestamp(rebalance_date)])
    if len(history) <= int(forward_days):
        return pd.DatetimeIndex([]), None, None
    usable = history[: -int(forward_days)]
    if int(lookback_days) > 0:
        usable = usable[-int(lookback_days) :]
    if len(usable) == 0:
        return usable, None, None
    last_signal = pd.Timestamp(usable[-1])
    signal_location = int(trading_dates.get_indexer([last_signal])[0])
    last_outcome = pd.Timestamp(trading_dates[signal_location + int(forward_days)])
    return usable, last_signal, last_outcome


def _within_family_target_weights(
    metrics: pd.DataFrame,
    factors: list[str],
    *,
    minimum_weight: float,
) -> tuple[pd.Series, str]:
    """Turn positive mean IC and ICIR evidence into non-zero family weights."""
    names = pd.Index([str(factor) for factor in factors])
    if len(names) == 0:
        return pd.Series(dtype=float), "empty_family"
    equal = pd.Series(1.0 / len(names), index=names, dtype=float)
    aligned = metrics.set_index("factor").reindex(names) if not metrics.empty else pd.DataFrame(index=names)
    components: list[pd.Series] = []
    for column in ("mean_ic", "ic_ir"):
        values = (
            pd.to_numeric(aligned[column], errors="coerce")
            if column in aligned
            else pd.Series(np.nan, index=names)
        )
        positive = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        total = float(positive.sum())
        if total > 1e-15:
            components.append(positive / total)
    if not components:
        return equal, "equal_fallback_no_positive_ic_evidence"
    evidence = pd.concat(components, axis=1).mean(axis=1)
    evidence = evidence / float(evidence.sum())
    floor = float(minimum_weight)
    if not np.isfinite(floor) or floor < 0.0 or floor * len(names) >= 1.0:
        raise ValueError("within-family minimum weight is infeasible")
    target = floor + (1.0 - floor * len(names)) * evidence
    target = target / float(target.sum())
    return target.astype(float), "positive_mean_ic_icir_blend"


def build_point_in_time_weight_log(
    ic_by_factor: dict[str, pd.Series],
    trading_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    families: dict[str, list[str]],
    *,
    forward_days: int,
    lookback_days: int,
    minimum_valid_days: int,
    minimum_within_family_weight: float,
    smoothing: float,
) -> pd.DataFrame:
    """Build weekly factor weights without admission or future IC observations."""
    alpha = float(smoothing)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("smoothing must be between zero and one")
    previous = {
        family: pd.Series(1.0 / len(factors), index=factors, dtype=float)
        for family, factors in families.items()
    }
    rows: list[dict[str, Any]] = []
    for raw_date in rebalance_dates:
        dt = pd.Timestamp(raw_date)
        window, last_signal, last_outcome = _known_ic_window(
            trading_dates,
            dt,
            forward_days=forward_days,
            lookback_days=lookback_days,
        )
        metric_rows: list[dict[str, Any]] = []
        for family, factors in families.items():
            for factor in factors:
                values = ic_by_factor[factor].reindex(window).dropna().astype(float)
                n_days = int(len(values))
                mean_ic = float(values.mean()) if n_days >= minimum_valid_days else np.nan
                std_ic = (
                    float(values.std(ddof=1))
                    if n_days >= minimum_valid_days and n_days > 1
                    else np.nan
                )
                ic_ir = (
                    float(mean_ic / std_ic)
                    if np.isfinite(mean_ic) and np.isfinite(std_ic) and std_ic > 1e-15
                    else np.nan
                )
                metric_rows.append(
                    {
                        "family": family,
                        "factor": factor,
                        "valid_ic_days": n_days,
                        "mean_ic": mean_ic,
                        "ic_std": std_ic,
                        "ic_ir": ic_ir,
                    }
                )
        metrics = pd.DataFrame(metric_rows)
        for family, factors in families.items():
            family_metrics = metrics[metrics["family"] == family]
            target, reason = _within_family_target_weights(
                family_metrics,
                factors,
                minimum_weight=minimum_within_family_weight,
            )
            prior = previous[family].reindex(factors)
            final = alpha * target + (1.0 - alpha) * prior
            final = final / float(final.sum())
            previous[family] = final
            for factor in factors:
                metric = family_metrics[family_metrics["factor"] == factor].iloc[0]
                rows.append(
                    {
                        "date": dt,
                        "family": family,
                        "factor": factor,
                        "mean_ic": metric["mean_ic"],
                        "ic_std": metric["ic_std"],
                        "ic_ir": metric["ic_ir"],
                        "valid_ic_days": int(metric["valid_ic_days"]),
                        "target_within_family_weight": float(target[factor]),
                        "within_family_weight": float(final[factor]),
                        "family_weight": 1.0 / len(families),
                        "total_factor_weight": float(final[factor] / len(families)),
                        "ic_window_start": pd.Timestamp(window[0]) if len(window) else pd.NaT,
                        "ic_signal_end": last_signal,
                        "ic_outcome_end": last_outcome,
                        "information_end": (
                            pd.Timestamp(trading_dates[trading_dates < dt][-1])
                            if bool((trading_dates < dt).any())
                            else pd.NaT
                        ),
                        "weight_reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def build_dynamic_family_fusion(
    panel: pd.DataFrame,
    weight_log: pd.DataFrame,
    families: dict[str, list[str]],
    eligible: pd.Series,
    evaluation_start: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply weekly factor weights, then preserve the fixed five-family fusion."""
    score_parts: list[pd.Series] = []
    family_parts: list[pd.DataFrame] = []
    for dt, rows in weight_log.groupby("date", sort=True):
        date = pd.Timestamp(dt)
        if date < evaluation_start:
            continue
        try:
            current = panel.xs(date, level="date")
        except KeyError:
            continue
        family_scores: dict[str, pd.Series] = {}
        indexed = rows.set_index("factor")
        for family, factors in families.items():
            weights = indexed.loc[factors, "within_family_weight"].astype(float)
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
        raise RuntimeError("no dynamic fusion scores were produced")
    scores = pd.concat(score_parts).sort_index().rename("score")
    family_scores = pd.concat(family_parts).sort_index()
    return scores, family_scores


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    performance: pd.DataFrame,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    active = navs.loc[(navs.index >= start) & (navs.index <= end)].copy()
    active = active.apply(lambda series: series / float(series.dropna().iloc[0]))
    colors = {PRIMARY: "#d95f02", FIXED_CONTROL: "#7c3aed", WEEKLY_BENCHMARK: "#0f766e"}
    labels = {PRIMARY: "Rolling IC/ICIR", FIXED_CONTROL: "Fixed family equal", WEEKLY_BENCHMARK: "Weekly PIT benchmark"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for column in [PRIMARY, FIXED_CONTROL, WEEKLY_BENCHMARK]:
        axes[0].plot(active.index, active[column], label=labels[column], color=colors[column], linewidth=2)
        if column != WEEKLY_BENCHMARK:
            axes[1].plot(active.index, active[column] / active[column].cummax() - 1.0, label=labels[column], color=colors[column])
    axes[0].set_title("No-admission fusion: rolling IC/ICIR weights versus fixed weights")
    axes[0].set_ylabel("NAV")
    axes[0].legend()
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "rolling_ic_vs_fixed_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    periods = ["EARLY_COMPARISON", "KNOWN_DEVELOPMENT", "FULL_EVALUATION"]
    gap = performance.pivot(index="strategy", columns="period", values="return_gap_vs_benchmark").reindex(index=[PRIMARY, FIXED_CONTROL], columns=periods)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(periods))
    width = 0.35
    ax.bar(x - width / 2, gap.loc[PRIMARY].to_numpy(float) * 100, width, label="Rolling IC/ICIR", color=colors[PRIMARY])
    ax.bar(x + width / 2, gap.loc[FIXED_CONTROL].to_numpy(float) * 100, width, label="Fixed family equal", color=colors[FIXED_CONTROL])
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x, periods)
    ax.set_ylabel("Return gap versus benchmark (percentage points)")
    ax.set_title("Weighting-method comparison by evaluation period")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "rolling_ic_vs_fixed_period_gaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    evaluation_weights = weights[weights["date"] >= start]
    wide = evaluation_weights.pivot(index="date", columns="factor", values="total_factor_weight")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.stackplot(wide.index, *[wide[column] for column in wide.columns], labels=wide.columns, alpha=0.86, step="post")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Point-in-time rolling factor weights (all factors remain included)")
    ax.set_ylabel("Total factor weight")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "rolling_factor_weight_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(source: Path, fixed_output: Path, output: Path, protocol_path: Path) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch; frozen rolling-weight protocol changed")

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
    pit_prices = mask_wide_prices_by_membership(prices, membership)

    families = {str(family): [str(factor) for factor in factors] for family, factors in protocol["candidate_families"].items()}
    candidates = [factor for factors in families.values() for factor in factors]
    missing = [factor for factor in candidates if factor not in panel.columns]
    if missing:
        raise RuntimeError(f"protocol factor missing from existing panel: {missing}")

    factor_panel = panel[candidates].where(eligible.reindex(panel.index).fillna(False), np.nan)
    ic_by_factor = {
        factor: daily_ic_spearman(
            factor_panel[factor],
            prices,
            forward_days=int(protocol["rolling_weight"]["forward_return_days"]),
            min_names=20,
        )
        for factor in candidates
    }
    rebalance_dates = pd.DatetimeIndex(
        [group.index[-1] for _, group in prices.dropna(how="all").groupby(pd.Grouper(freq="W-FRI")) if not group.empty]
    )
    weight_log = build_point_in_time_weight_log(
        ic_by_factor,
        pd.DatetimeIndex(prices.index),
        rebalance_dates,
        families,
        forward_days=int(protocol["rolling_weight"]["forward_return_days"]),
        lookback_days=int(protocol["rolling_weight"]["history_window_trading_days"]),
        minimum_valid_days=int(protocol["rolling_weight"]["minimum_valid_ic_days"]),
        minimum_within_family_weight=float(protocol["rolling_weight"]["within_family_minimum_weight"]),
        smoothing=float(protocol["rolling_weight"]["weekly_weight_smoothing"]),
    )
    evaluation_start = pd.Timestamp(protocol["data"]["evaluation_start"])
    evaluation_end = pd.Timestamp(protocol["data"]["evaluation_end"])
    fused, family_scores = build_dynamic_family_fusion(panel[candidates], weight_log, families, eligible, evaluation_start)
    nav, meta = run_multi_backtest(
        fused=fused,
        prices=prices,
        settings=settings,
        factor_name=PRIMARY,
        top_k=50,
        long_prices=long_prices,
        empty_signal_policy="cash",
    )
    benchmark, benchmark_meta = build_weekly_costed_pit_benchmark(pit_prices, prices, long_prices, settings)
    benchmark = benchmark.rename(WEEKLY_BENCHMARK)

    fixed_navs = pd.read_csv(fixed_output / "nav_comparison.csv", index_col=0, parse_dates=True)
    fixed = fixed_navs[FIXED_CONTROL].astype(float)
    prior_benchmark = fixed_navs[WEEKLY_BENCHMARK].astype(float)
    aligned_benchmark = pd.concat([benchmark, prior_benchmark], axis=1).dropna()
    if not np.allclose(aligned_benchmark.iloc[:, 0], aligned_benchmark.iloc[:, 1], atol=1e-12):
        raise RuntimeError("weekly PIT benchmark changed between weighting runs")

    periods = {
        "FULL_EVALUATION": (evaluation_start, evaluation_end),
        "EARLY_COMPARISON": (evaluation_start, pd.Timestamp(protocol["data"]["early_comparison_period"][1])),
        "KNOWN_DEVELOPMENT": (pd.Timestamp(protocol["data"]["known_development_period"][0]), evaluation_end),
    }
    performance = _period_rows({PRIMARY: nav, FIXED_CONTROL: fixed}, benchmark, periods)
    full = performance[performance["period"] == "FULL_EVALUATION"].set_index("strategy")
    early = performance[performance["period"] == "EARLY_COMPARISON"].set_index("strategy")
    known = performance[performance["period"] == "KNOWN_DEVELOPMENT"].set_index("strategy")

    evaluation_weights = weight_log[weight_log["date"].between(evaluation_start, evaluation_end)]
    totals = evaluation_weights.groupby("date")["total_factor_weight"].sum()
    family_totals = evaluation_weights.groupby(["date", "family"])["total_factor_weight"].sum()
    pit_rows = evaluation_weights.dropna(subset=["ic_outcome_end", "information_end"])
    checks: dict[str, Any] = {
        "protocol_sha256": actual,
        "factor_admission_enabled": False,
        "redundancy_pruning_enabled": False,
        "all_candidate_factors_included": candidates,
        "all_factor_weights_strictly_positive_pass": bool((evaluation_weights["total_factor_weight"] > 0.0).all()),
        "factor_weights_sum_to_one_pass": bool(np.allclose(totals.to_numpy(float), 1.0, atol=1e-10)),
        "families_remain_equal_weight_pass": bool(np.allclose(family_totals.to_numpy(float), 0.2, atol=1e-10)),
        "ic_outcomes_known_before_rebalance_pass": bool((pd.to_datetime(pit_rows["ic_outcome_end"]) < pd.to_datetime(pit_rows["date"])).all()),
        "ic_information_cutoff_strict_pass": bool((pd.to_datetime(pit_rows["information_end"]) < pd.to_datetime(pit_rows["date"])).all()),
        "dynamic_full_total_return": float(full.loc[PRIMARY, "total_return"]),
        "fixed_full_total_return": float(full.loc[FIXED_CONTROL, "total_return"]),
        "dynamic_minus_fixed_full_return": float(full.loc[PRIMARY, "total_return"] - full.loc[FIXED_CONTROL, "total_return"]),
        "dynamic_full_return_gap_vs_benchmark": float(full.loc[PRIMARY, "return_gap_vs_benchmark"]),
        "fixed_full_return_gap_vs_benchmark": float(full.loc[FIXED_CONTROL, "return_gap_vs_benchmark"]),
        "dynamic_early_return_gap": float(early.loc[PRIMARY, "return_gap_vs_benchmark"]),
        "fixed_early_return_gap": float(early.loc[FIXED_CONTROL, "return_gap_vs_benchmark"]),
        "dynamic_known_return_gap": float(known.loc[PRIMARY, "return_gap_vs_benchmark"]),
        "fixed_known_return_gap": float(known.loc[FIXED_CONTROL, "return_gap_vs_benchmark"]),
        "dynamic_improves_full_return_over_fixed": bool(full.loc[PRIMARY, "total_return"] > full.loc[FIXED_CONTROL, "total_return"]),
        "dynamic_improves_both_subperiod_gaps_over_fixed": bool(
            early.loc[PRIMARY, "return_gap_vs_benchmark"] > early.loc[FIXED_CONTROL, "return_gap_vs_benchmark"]
            and known.loc[PRIMARY, "return_gap_vs_benchmark"] > known.loc[FIXED_CONTROL, "return_gap_vs_benchmark"]
        ),
        "historical_data_already_seen": True,
        "live_approval": False,
        "verdict": "RETROSPECTIVE_WEIGHTING_COMPARISON_ONLY",
    }

    output.mkdir(parents=True, exist_ok=True)
    weight_log.to_csv(output / "rolling_factor_weight_log.csv", index=False, date_format="%Y-%m-%d")
    family_scores.reset_index().to_csv(output / "family_scores.csv", index=False, date_format="%Y-%m-%d")
    fused.reset_index().to_csv(output / "fused_scores.csv", index=False, date_format="%Y-%m-%d")
    nav_frame = pd.concat([nav.rename(PRIMARY), fixed.rename(FIXED_CONTROL), benchmark], axis=1)
    nav_frame.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    execution = pd.DataFrame([
        _execution_summary(PRIMARY, meta, evaluation_start),
        _execution_summary(WEEKLY_BENCHMARK, benchmark_meta, evaluation_start),
    ])
    fixed_execution = pd.read_csv(fixed_output / "execution_summary.csv")
    fixed_row = fixed_execution[fixed_execution["strategy"] == FIXED_CONTROL]
    execution = pd.concat([execution, fixed_row], ignore_index=True)
    execution.to_csv(output / "execution_summary.csv", index=False)
    (output / "comparison_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    charts = _plot_outputs(output, nav_frame, performance, weight_log, evaluation_start, evaluation_end)

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        git_head = "UNKNOWN"
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": actual,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_factor_panel_sha256": _sha256(source / "cache/factor_panel_zscore.csv"),
        "source_prices_sha256": _sha256(source / "cache/prices_wide_adj_close.csv"),
        "fixed_control_checks_sha256": _sha256(fixed_output / "acceptance_checks.json"),
        "code_sha256": _sha256(Path(__file__)),
        "git_head_at_evaluation": git_head,
        "verdict": checks["verdict"],
        "live_approval": False,
        "true_forward_status": "NOT_STARTED",
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "performance": output / "performance_by_period.csv",
        "weights": output / "rolling_factor_weight_log.csv",
        "checks": output / "comparison_checks.json",
        "manifest": output / "audit_manifest.json",
        **{f"chart_{index + 1}": path for index, path in enumerate(charts)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--fixed-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(args.source_output_dir, args.fixed_output_dir, args.output_dir, args.protocol).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
