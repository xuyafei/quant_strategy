#!/usr/bin/env python3
"""Compare Top-K breadth and a cross-industry technology-growth exposure cap."""
from __future__ import annotations

import argparse
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
from live.cache_io import save_rebalance_logs
from live.universe_history import load_membership_intervals, mask_wide_prices_by_membership
from scripts.build_walk_forward_factor_backtest import (
    WALK_FORWARD_STRATEGY,
    _settings_from_source,
)


def _load_scores(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    out = frame.set_index(["date", "symbol"])["score"].sort_index()
    out.index = out.index.set_names(["date", "symbol"])
    return out


def _normalize_from(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    out = series.loc[series.index >= start].dropna().astype(float)
    return out / float(out.iloc[0])


def _target_map(rec: dict[str, Any]) -> dict[str, float]:
    picks = [str(value) for value in rec.get("picks", [])]
    weights = [float(value) for value in rec.get("weights", [])]
    return dict(zip(picks, weights)) if len(picks) == len(weights) else {}


def _turnover_summary(logs: list[dict[str, Any]], commission_rate: float) -> dict[str, float | int]:
    previous: dict[str, float] = {}
    values: list[float] = []
    for rec in sorted(logs, key=lambda item: pd.Timestamp(item["date"])):
        current = _target_map(rec)
        symbols = set(previous) | set(current)
        values.append(float(sum(abs(current.get(sym, 0.0) - previous.get(sym, 0.0)) for sym in symbols)))
        previous = current
    arr = np.asarray(values, dtype=float)
    return {
        "n_rebalances": int(len(arr)),
        "avg_turnover": float(arr.mean()) if len(arr) else float("nan"),
        "max_turnover": float(arr.max()) if len(arr) else float("nan"),
        "total_turnover": float(arr.sum()) if len(arr) else float("nan"),
        "estimated_commission": float(arr.sum() * commission_rate) if len(arr) else float("nan"),
    }


def _industry_by_symbol(long_prices: pd.DataFrame, industry_col: str) -> dict[str, str]:
    if industry_col not in long_prices.columns:
        return {}
    work = long_prices[["trade_date", "ts_code", industry_col]].dropna(subset=[industry_col]).copy()
    work[industry_col] = work[industry_col].astype(str).str.strip()
    work = work[work[industry_col] != ""].sort_values("trade_date")
    return work.drop_duplicates("ts_code", keep="last").set_index("ts_code")[industry_col].to_dict()


def _exposure_summary(
    logs: list[dict[str, Any]],
    industries: dict[str, str],
    tech_industries: set[str],
) -> dict[str, float | int]:
    exposures: list[float] = []
    applied = 0
    for rec in logs:
        target = _target_map(rec)
        exposures.append(
            float(sum(weight for sym, weight in target.items() if industries.get(sym, "") in tech_industries))
        )
        applied += int(bool(rec.get("tech_growth_cap_applied", False)))
    arr = np.asarray(exposures, dtype=float)
    return {
        "avg_tech_growth_exposure": float(arr.mean()) if len(arr) else float("nan"),
        "max_tech_growth_exposure": float(arr.max()) if len(arr) else float("nan"),
        "tech_growth_cap_applied_rebalances": int(applied),
    }


def _performance_row(
    name: str,
    nav: pd.Series,
    benchmark: pd.Series,
    active_start: pd.Timestamp,
    periods: int,
    *,
    top_k: int,
    tech_cap: float,
) -> dict[str, Any]:
    active = _normalize_from(nav, active_start)
    bench = _normalize_from(benchmark, active_start)
    row: dict[str, Any] = {
        "strategy": name,
        "top_k": top_k,
        "max_tech_growth_weight": tech_cap,
        "start": active.index.min().strftime("%Y-%m-%d"),
        "end": active.index.max().strftime("%Y-%m-%d"),
        "n_days": int(len(active)),
    }
    row.update(summarize(active, periods=periods))
    row.update(summarize_excess(active, bench, periods=periods))
    return row


def _plot(
    navs: pd.DataFrame,
    benchmark: pd.Series,
    active_start: pd.Timestamp,
    path: Path,
    universe_label: str,
) -> None:
    normalized = navs.loc[navs.index >= active_start].copy()
    normalized = normalized.div(normalized.iloc[0])
    bench = _normalize_from(benchmark, active_start)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for name in [f"{universe_label}_TOP05", f"{universe_label}_TOP10", f"{universe_label}_TOP15"]:
        axes[0].plot(normalized.index, normalized[name], linewidth=2, label=name)
    axes[0].plot(bench.index, bench, color="black", linewidth=1.5, linestyle="--", label=f"PIT {universe_label} benchmark")
    axes[0].set_title("Position breadth sensitivity — no aggregate technology cap")
    axes[0].set_ylabel("NAV (rebased to 1.0)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    for name in [
        f"{universe_label}_TOP10",
        f"{universe_label}_TOP10_TECH25",
        f"{universe_label}_TOP15",
        f"{universe_label}_TOP15_TECH25",
    ]:
        axes[1].plot(normalized.index, normalized[name], linewidth=1.8, label=name)
    axes[1].plot(bench.index, bench, color="black", linewidth=1.5, linestyle="--", label=f"PIT {universe_label} benchmark")
    axes[1].set_title("Aggregate technology-growth cap sensitivity")
    axes[1].set_ylabel("NAV (rebased to 1.0)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best", ncol=2)
    axes[1].grid(alpha=0.25)
    fig.text(
        0.01,
        0.01,
        f"Source: local point-in-time {universe_label} walk-forward scores · {active_start:%Y-%m-%d} to {navs.index.max():%Y-%m-%d} · commission included",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    source_dir: Path,
    walk_forward_dir: Path,
    output_dir: Path,
    tech_cap: float = 0.25,
    universe_label: str = "A50",
) -> dict[str, Path]:
    source_settings, source_config = _settings_from_source(source_dir, output_dir)
    walk_config = json.loads((walk_forward_dir / "run_config.json").read_text(encoding="utf-8"))
    settings = replace(
        source_settings,
        output_dir=output_dir,
        rebalance_freq=str(walk_config["rebalance_freq"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = _load_scores(walk_forward_dir / "walk_forward" / "fused_scores.csv")
    prices = pd.read_csv(source_dir / "cache" / "prices_wide_adj_close.csv", index_col=0, parse_dates=True)
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source_dir / "cache" / "prices_long.csv", parse_dates=["trade_date"])
    membership = load_membership_intervals(Path(settings.universe_membership_path))
    benchmark_prices = mask_wide_prices_by_membership(prices, membership)
    benchmark = equal_weight_benchmark_nav(benchmark_prices, dates=prices.index)
    summary = pd.read_csv(walk_forward_dir / "walk_forward" / "rebalance_summary.csv", parse_dates=["date"])
    active_start = pd.Timestamp(summary.loc[summary["signal_status"] == "ACTIVE", "date"].min())

    label = str(universe_label).strip().upper().replace(" ", "_")
    variants = [
        (f"{label}_TOP05", 5, 0.0),
        (f"{label}_TOP10", 10, 0.0),
        (f"{label}_TOP15", 15, 0.0),
        (f"{label}_TOP05_TECH25", 5, tech_cap),
        (f"{label}_TOP10_TECH25", 10, tech_cap),
        (f"{label}_TOP15_TECH25", 15, tech_cap),
    ]
    nav_by_name: dict[str, pd.Series] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    performance_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    industries = _industry_by_symbol(long_prices, settings.industry_col)
    tech_industries = {str(value) for value in settings.tech_growth_industries}

    for name, top_k, cap in variants:
        variant_settings = replace(settings, top_k=top_k, max_tech_growth_weight=cap)
        nav, meta = run_multi_backtest(
            fused=scores,
            prices=prices,
            settings=variant_settings,
            factor_name=name,
            top_k=top_k,
            long_prices=long_prices,
            empty_signal_policy="cash",
        )
        nav_by_name[name] = nav.rename(name)
        meta_by_name[name] = meta
        performance_rows.append(
            _performance_row(
                name,
                nav,
                benchmark,
                active_start,
                settings.trading_days_per_year,
                top_k=top_k,
                tech_cap=cap,
            )
        )
        risk_rows.append(
            {
                "strategy": name,
                "top_k": top_k,
                "max_tech_growth_weight": cap,
                **_turnover_summary(meta.get("rebalance_log", []), settings.commission_rate),
                **_exposure_summary(meta.get("rebalance_log", []), industries, tech_industries),
            }
        )

    navs = pd.concat(nav_by_name.values(), axis=1)
    existing = pd.read_csv(walk_forward_dir / "nav_comparison.csv", index_col="date", parse_dates=True)[
        WALK_FORWARD_STRATEGY
    ]
    baseline_diff = float((navs[f"{label}_TOP05"] - existing.reindex(navs.index)).abs().max())
    risk = pd.DataFrame(risk_rows)
    capped = risk[risk["max_tech_growth_weight"] > 0]
    cap_ok = bool((capped["max_tech_growth_exposure"] <= tech_cap + 1e-9).all())

    paths = {
        "nav_comparison": output_dir / "nav_comparison.csv",
        "performance": output_dir / "performance_summary.csv",
        "risk_summary": output_dir / "turnover_exposure_summary.csv",
        "audit": output_dir / "audit_checks.json",
        "config": output_dir / "run_config.json",
        "chart": output_dir / "breadth_tech_cap_comparison.png",
    }
    navs.assign(BENCH_EQUAL_WEIGHT=benchmark.reindex(navs.index)).to_csv(paths["nav_comparison"], date_format="%Y-%m-%d")
    pd.DataFrame(performance_rows).to_csv(paths["performance"], index=False)
    risk.to_csv(paths["risk_summary"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "baseline_matches_existing_weekly_nav": baseline_diff <= 1e-9,
                "baseline_max_absolute_difference": baseline_diff,
                "capped_variants_respect_tech_growth_limit": cap_ok,
                "active_start": active_start.strftime("%Y-%m-%d"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["config"].write_text(
        json.dumps(
            {
                **source_config,
                "source_output_dir": str(source_dir),
                "walk_forward_output_dir": str(walk_forward_dir),
                "output_dir": str(output_dir),
                "rebalance_freq": settings.rebalance_freq,
                "top_k_values": [5, 10, 15],
                "tech_growth_cap_values": [0.0, tech_cap],
                "tech_growth_industries": list(settings.tech_growth_industries),
                "universe_label": label,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    save_rebalance_logs(settings, meta_by_name)
    _plot(navs, benchmark, active_start, paths["chart"], label)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="持仓数量与科技成长综合暴露上限敏感性测试")
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--walk-forward-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tech-cap", type=float, default=0.25)
    parser.add_argument("--universe-label", default="A50")
    args = parser.parse_args()
    for name, path in run(
        args.source_output_dir,
        args.walk_forward_output_dir,
        args.output_dir,
        tech_cap=args.tech_cap,
        universe_label=args.universe_label,
    ).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
