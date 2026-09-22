#!/usr/bin/env python3
"""Run the frozen all-traditional-factor fusion without an admission gate."""
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

from analysis.benchmark import summarize_excess
from analysis.performance import summarize
from backtest.backtest_multi import run_multi_backtest
from live.universe_history import (
    load_membership_intervals,
    mask_wide_prices_by_membership,
    membership_mask,
)
from scripts.build_simple_admission_fusion_backtest import (
    WEEKLY_BENCHMARK,
    _execution_summary,
    _load_panel,
    _normalize,
    _sha256,
    build_fixed_family_fusion,
    build_weekly_costed_pit_benchmark,
)
from scripts.build_walk_forward_factor_backtest import _settings_from_source


PRIMARY = "NO_ADMISSION_ALL_FACTOR_FIXED_FUSION_TOP50"
TOP10 = "NO_ADMISSION_ALL_FACTOR_FIXED_FUSION_TOP10"
ADMITTED_TOP50 = "SIMPLE_ADMISSION_FIXED_FUSION_TOP50"
ADMITTED_TOP10 = "SIMPLE_ADMISSION_FIXED_FUSION_TOP10"


def _stats(
    strategy: str,
    period: str,
    nav: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    row: dict[str, Any] = {"strategy": strategy, "period": period}
    row.update(summarize(nav))
    row.update(summarize_excess(nav, benchmark))
    benchmark_total = (
        float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        if not benchmark.empty
        else np.nan
    )
    row["benchmark_total_return"] = benchmark_total
    row["return_gap_vs_benchmark"] = (
        float(row["total_return"] - benchmark_total)
        if np.isfinite(benchmark_total)
        else np.nan
    )
    row["start"] = nav.index.min().strftime("%Y-%m-%d") if not nav.empty else ""
    row["end"] = nav.index.max().strftime("%Y-%m-%d") if not nav.empty else ""
    row["n_days"] = int(len(nav))
    return row


def _period_rows(
    navs: dict[str, pd.Series],
    benchmark: pd.Series,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        bench = _normalize(benchmark, start, end)
        for strategy, nav in navs.items():
            rows.append(
                _stats(strategy, period, _normalize(nav, start, end), bench)
            )
    return pd.DataFrame(rows)


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    performance: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    active = navs.loc[(navs.index >= start) & (navs.index <= end)].copy()
    active = active.apply(lambda series: series / float(series.dropna().iloc[0]))
    colors = {
        PRIMARY: "#d95f02",
        ADMITTED_TOP50: "#7c3aed",
        WEEKLY_BENCHMARK: "#0f766e",
    }
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for column in [PRIMARY, ADMITTED_TOP50, WEEKLY_BENCHMARK]:
        axes[0].plot(
            active.index,
            active[column],
            label=column,
            color=colors[column],
            linewidth=2,
        )
        if column != WEEKLY_BENCHMARK:
            axes[1].plot(
                active.index,
                active[column] / active[column].cummax() - 1.0,
                label=column,
                color=colors[column],
            )
    axes[0].set_title("Admission ablation: all-factor fusion versus admitted-factor fusion")
    axes[0].set_ylabel("NAV")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "no_admission_vs_admission_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    order = ["EARLY_COMPARISON", "KNOWN_DEVELOPMENT", "FULL_EVALUATION"]
    subset = performance[
        performance["strategy"].isin([PRIMARY, ADMITTED_TOP50])
    ].pivot(index="strategy", columns="period", values="return_gap_vs_benchmark")
    subset = subset.reindex(index=[PRIMARY, ADMITTED_TOP50], columns=order)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(order))
    width = 0.35
    ax.bar(
        x - width / 2,
        subset.loc[PRIMARY].to_numpy(dtype=float) * 100,
        width,
        label="No admission",
        color="#d95f02",
    )
    ax.bar(
        x + width / 2,
        subset.loc[ADMITTED_TOP50].to_numpy(dtype=float) * 100,
        width,
        label="Admission enabled",
        color="#7c3aed",
    )
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("Return gap versus weekly PIT benchmark (percentage points)")
    ax.set_title("Removing admission improves both periods in this retrospective ablation")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "admission_ablation_period_gaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(
    source: Path,
    admitted_output: Path,
    output: Path,
    protocol_path: Path,
) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch; frozen no-admission protocol changed")

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
        str(family): [str(factor) for factor in factors]
        for family, factors in protocol["candidate_families"].items()
    }
    candidates = [factor for factors in families.values() for factor in factors]
    missing = [factor for factor in candidates if factor not in panel.columns]
    if missing:
        raise RuntimeError(f"protocol factor missing from existing panel: {missing}")
    evaluation_start = pd.Timestamp(protocol["data"]["evaluation_start"])
    fused, family_scores = build_fixed_family_fusion(
        panel[candidates], families, eligible, evaluation_start
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

    admitted_navs = pd.read_csv(
        admitted_output / "nav_comparison.csv", index_col=0, parse_dates=True
    )
    admitted50 = admitted_navs[ADMITTED_TOP50].astype(float)
    admitted10 = admitted_navs[ADMITTED_TOP10].astype(float)
    prior_benchmark = admitted_navs[WEEKLY_BENCHMARK].astype(float)
    aligned = pd.concat([weekly_benchmark, prior_benchmark], axis=1).dropna()
    if not np.allclose(
        aligned.iloc[:, 0].to_numpy(dtype=float),
        aligned.iloc[:, 1].to_numpy(dtype=float),
        atol=1e-12,
    ):
        raise RuntimeError("weekly PIT benchmark changed between ablation runs")

    periods = {
        "FULL_EVALUATION": (evaluation_start, pd.Timestamp("2026-09-04")),
        "EARLY_COMPARISON": (evaluation_start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), pd.Timestamp("2026-09-04")),
    }
    performance = _period_rows(
        {
            PRIMARY: nav50,
            TOP10: nav10,
            ADMITTED_TOP50: admitted50,
            ADMITTED_TOP10: admitted10,
        },
        weekly_benchmark,
        periods,
    )

    full = performance[performance["period"] == "FULL_EVALUATION"].set_index("strategy")
    early = performance[performance["period"] == "EARLY_COMPARISON"].set_index("strategy")
    known = performance[performance["period"] == "KNOWN_DEVELOPMENT"].set_index("strategy")
    checks: dict[str, Any] = {
        "protocol_sha256": actual,
        "factor_admission_enabled": False,
        "redundancy_pruning_enabled": False,
        "all_candidate_factors_included": candidates,
        "all_five_families_included": list(families),
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
        "full_return_difference_vs_admission_top50": float(
            full.loc[PRIMARY, "total_return"] - full.loc[ADMITTED_TOP50, "total_return"]
        ),
        "early_return_gap_difference_vs_admission_top50": float(
            early.loc[PRIMARY, "return_gap_vs_benchmark"]
            - early.loc[ADMITTED_TOP50, "return_gap_vs_benchmark"]
        ),
        "known_return_gap_difference_vs_admission_top50": float(
            known.loc[PRIMARY, "return_gap_vs_benchmark"]
            - known.loc[ADMITTED_TOP50, "return_gap_vs_benchmark"]
        ),
        "historical_data_already_seen": True,
        "live_approval": False,
    }
    required = [key for key in checks if key.endswith("_pass")]
    checks["historical_acceptance_pass"] = bool(all(checks[key] for key in required))
    checks["verdict"] = (
        "RETROSPECTIVE_NO_ADMISSION_CANDIDATE_ONLY"
        if checks["historical_acceptance_pass"]
        else "NO_ADMISSION_FUSION_NO_GO"
    )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "family": family,
                "factors": ",".join(factors),
                "family_weight": 1.0 / len(families),
                "factor_weight_within_family": 1.0 / len(factors),
                "implied_each_factor_total_weight": 1.0 / len(families) / len(factors),
            }
            for family, factors in families.items()
        ]
    ).to_csv(output / "all_factor_fusion_definition.csv", index=False)
    family_scores.reset_index().to_csv(
        output / "family_scores.csv", index=False, date_format="%Y-%m-%d"
    )
    fused.reset_index().to_csv(
        output / "fused_scores.csv", index=False, date_format="%Y-%m-%d"
    )
    nav_frame = pd.concat(
        [
            nav50.rename(PRIMARY),
            nav10.rename(TOP10),
            admitted50.rename(ADMITTED_TOP50),
            admitted10.rename(ADMITTED_TOP10),
            weekly_benchmark,
        ],
        axis=1,
    )
    nav_frame.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    pd.DataFrame(
        [
            _execution_summary(PRIMARY, meta50, evaluation_start),
            _execution_summary(TOP10, meta10, evaluation_start),
            _execution_summary(WEEKLY_BENCHMARK, weekly_meta, evaluation_start),
        ]
    ).to_csv(output / "execution_summary.csv", index=False)
    (output / "acceptance_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    charts = _plot_outputs(
        output,
        nav_frame,
        performance,
        evaluation_start,
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
        "admission_control_acceptance_sha256": _sha256(
            admitted_output / "acceptance_checks.json"
        ),
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
        "definition": output / "all_factor_fusion_definition.csv",
        "performance": output / "performance_by_period.csv",
        "acceptance": output / "acceptance_checks.json",
        "manifest": output / "audit_manifest.json",
        **{f"chart_{index + 1}": path for index, path in enumerate(charts)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--admission-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(
        args.source_output_dir,
        args.admission_output_dir,
        args.output_dir,
        args.protocol,
    ).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
