#!/usr/bin/env python3
"""Build and backtest reusable strategy universes derived from base universe V2."""
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

from backtest.backtest_multi import run_multi_backtest
from config import get_settings
from live.strategy_universe import save_runtime_exports
from scripts.build_dynamic_universe_comparison import (
    _benchmark_score,
    _execution_summary,
    _performance_row,
    _shift_score_to_next_session,
    _weekly_decision_dates,
    build_weekly_raw_factor_panel,
)
from strategies.style import build_fixed_family_score, build_focused_style_score
from universe.dynamic import build_dynamic_universe_report, eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import (
    StrategyUniverseProfile,
    build_strategy_universe_report,
    daily_basic_for_index,
    eligibility_for_profile,
    point_in_time_industry,
    profiles_from_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected(
    score: pd.Series,
    strategy: str,
    universe: str,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, values in score.groupby(level="date", sort=True):
        ranked = values.droplevel("date").sort_values(ascending=False).head(int(top_k))
        for rank, (symbol, value) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "signal_date": pd.Timestamp(date),
                    "strategy": strategy,
                    "universe": universe,
                    "rank": rank,
                    "symbol": str(symbol),
                    "score": float(value),
                }
            )
    return pd.DataFrame(rows)


def _profile_lookup(profiles: list[StrategyUniverseProfile]) -> dict[str, StrategyUniverseProfile]:
    return {profile.name: profile for profile in profiles}


def _universe_summary(
    report: pd.DataFrame,
    base_eligible: pd.Series,
    profiles: list[StrategyUniverseProfile],
) -> pd.DataFrame:
    rows = []
    profile_masks = {
        profile.name: eligibility_for_profile(report, profile) for profile in profiles
    }
    masks = {"BASE_V2": base_eligible, **profile_masks}
    indexed = report.set_index(["date", "symbol"]).sort_index()
    for universe, mask in masks.items():
        aligned = mask.reindex(indexed.index).fillna(False).astype(bool)
        for date, date_mask in aligned.groupby(level="date", sort=True):
            symbols = date_mask.index[date_mask.to_numpy()]
            if len(symbols) == 0:
                continue
            group = indexed.reindex(symbols)
            industry = group["industry_l1"].fillna("").astype(str).str.strip()
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "universe": universe,
                    "eligible_count": int(len(group)),
                    "median_circ_mv_cny": float(
                        pd.to_numeric(group["circ_mv_yuan"], errors="coerce").median()
                    ),
                    "median_adv20_cny": float(
                        pd.to_numeric(group["adv20_yuan"], errors="coerce").median()
                    ),
                    "industry_count": int(industry[industry.ne("")].nunique()),
                    "daily_basic_coverage": float(group["daily_basic_fresh"].mean()),
                    "industry_coverage": float(industry.ne("").mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    performance: pd.DataFrame,
    summary: pd.DataFrame,
    strategies: list[str],
    labels: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    active = navs.loc[(navs.index >= start) & (navs.index <= end), strategies].copy()
    active = active.apply(lambda value: value / float(value.dropna().iloc[0]))
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for strategy in strategies:
        axes[0].plot(active.index, active[strategy], label=labels[strategy], linewidth=1.8)
        drawdown = active[strategy] / active[strategy].cummax() - 1.0
        axes[1].plot(drawdown.index, drawdown * 100.0, label=labels[strategy], linewidth=1.5)
    axes[0].set_title("Point-in-time strategy universes: next-session-close NAV")
    axes[0].set_ylabel("NAV")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = chart_dir / "strategy_universe_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    ordered = performance.set_index("strategy").reindex(strategies)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.bar(x - 0.19, ordered["total_return"] * 100.0, 0.38, label="Strategy")
    ax.bar(x + 0.19, ordered["benchmark_total_return"] * 100.0, 0.38, label="Own universe")
    ax.set_xticks(x, [labels[value] for value in strategies], rotation=15, ha="right")
    ax.set_ylabel("Total return (%)")
    ax.set_title("Each strategy versus its own point-in-time equal-weight universe")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = chart_dir / "strategy_vs_own_universe.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for universe, group in summary.groupby("universe", sort=False):
        ax.plot(group["date"], group["eligible_count"], label=universe, linewidth=1.9)
    ax.set_title("Point-in-time breadth of base and derived strategy universes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Eligible stocks")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = chart_dir / "strategy_universe_breadth.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(data_dir: Path, output: Path, protocol_path: Path) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = _sha256(protocol_path)
    if expected != actual:
        raise RuntimeError("protocol hash mismatch")
    start = pd.Timestamp(protocol["data"]["evaluation_start"])
    end = pd.Timestamp(protocol["data"]["evaluation_end"])
    output.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(data_dir / "prices_all_a_qfq.csv.gz", parse_dates=["trade_date"])
    finance = pd.read_csv(data_dir / "fina_indicator_all_a.csv.gz")
    balance = pd.read_csv(data_dir / "balance_sheet_all_a.csv.gz")
    audit = pd.read_csv(data_dir / "fina_audit_all_a.csv.gz")
    basic = pd.read_csv(data_dir / "stock_basic_all_a.csv", dtype={"ts_code": str})
    names = pd.read_csv(data_dir / "namechange_all_a.csv", dtype={"ts_code": str})
    calendar = pd.read_csv(data_dir / "trade_calendar.csv", dtype={"cal_date": str})
    daily_basic = pd.read_csv(data_dir / "daily_basic_weekly.csv.gz", dtype={"ts_code": str})
    industry_membership = pd.read_csv(
        data_dir / "sw2021_l1_membership.csv.gz", dtype={"ts_code": str}
    )
    decision_dates = _weekly_decision_dates(calendar, start, end)

    base_cfg = protocol["base_universe"]
    tradable_report = build_dynamic_universe_report(
        prices,
        basic,
        names,
        calendar,
        decision_dates,
        min_listing_sessions=int(base_cfg["minimum_listing_open_sessions"]),
        liquidity_window=int(base_cfg["liquidity_lookback_open_sessions"]),
        min_valid_days=int(base_cfg["minimum_valid_trading_days"]),
        min_adv_yuan=float(base_cfg["minimum_adv20_cny"]),
        amount_col="amount_yuan",
    )
    risk_report = build_hard_risk_universe_report(
        tradable_report, balance, audit, names
    )
    base_eligible = eligibility_from_report(risk_report)
    raw_panel = build_weekly_raw_factor_panel(prices, finance, decision_dates)
    profiles = profiles_from_config(protocol["strategy_universe_profiles"])
    profile_by_name = _profile_lookup(profiles)
    report = build_strategy_universe_report(
        risk_report,
        daily_basic,
        raw_panel,
        industry_membership,
        profiles,
        max_daily_basic_staleness_days=int(
            protocol["data"]["maximum_daily_basic_staleness_calendar_days"]
        ),
    )
    profile_eligibility = {
        profile.name: eligibility_for_profile(report, profile) for profile in profiles
    }
    industry = point_in_time_industry(raw_panel.index, industry_membership)
    market = daily_basic_for_index(
        raw_panel.index,
        daily_basic,
        max_staleness_days=int(protocol["data"]["maximum_daily_basic_staleness_calendar_days"]),
    )
    pe_ttm = pd.to_numeric(market["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(market["pb"], errors="coerce")
    raw_panel = raw_panel.copy()
    raw_panel["EARNINGS_YIELD"] = (1.0 / pe_ttm).where(pe_ttm.gt(0))
    raw_panel["BOOK_TO_PRICE"] = (1.0 / pb).where(pb.gt(0))

    families = {
        str(name): [str(factor) for factor in factors]
        for name, factors in protocol["candidate_families"].items()
    }
    strategy_specs = protocol["strategies"]
    raw_scores: dict[str, pd.Series] = {}
    universe_by_strategy: dict[str, str] = {}
    top_k_by_strategy: dict[str, int] = {}
    score_components: dict[str, pd.DataFrame] = {}
    for spec in strategy_specs:
        name = str(spec["name"])
        universe = str(spec["universe"])
        eligible = base_eligible if universe == "BASE_V2" else profile_eligibility[universe]
        if spec["score"] == "fixed_equal_five_family_multi_factor":
            score, components = build_fixed_family_score(
                raw_panel, eligible, industry, families
            )
        else:
            score, components = build_focused_style_score(
                raw_panel,
                eligible,
                industry,
                [str(factor) for factor in spec["factors"]],
                minimum_components=int(spec["minimum_components"]),
            )
        raw_scores[name] = score
        score_components[name] = components
        universe_by_strategy[name] = universe
        top_k_by_strategy[name] = int(spec["top_k"])

    price_wide = prices.pivot(index="trade_date", columns="ts_code", values="adj_close").sort_index()
    price_wide = price_wide.loc[price_wide.index <= end]
    shifted_scores = {
        name: _shift_score_to_next_session(score, price_wide.index)
        for name, score in raw_scores.items()
    }
    universe_masks = {"BASE_V2": base_eligible, **profile_eligibility}
    benchmark_names = {universe: "BENCHMARK_" + universe for universe in universe_masks}
    benchmark_scores = {
        benchmark_names[universe]: _shift_score_to_next_session(
            _benchmark_score(mask, prices), price_wide.index
        )
        for universe, mask in universe_masks.items()
    }

    settings = replace(
        get_settings(),
        backtest_start=protocol["data"]["raw_start"],
        backtest_end=protocol["data"]["evaluation_end"],
        rebalance_freq="D",
        force_final_rebalance=False,
        top_k=max(top_k_by_strategy.values()),
        portfolio_weighting="equal",
        commission_rate=float(protocol["factor_and_portfolio"]["commission_rate"]),
        max_position_weight=0.0,
        max_rebalance_turnover=0.0,
        min_avg_volume=0.0,
        min_avg_amount=0.0,
        enable_trade_status_filter=False,
        max_industry_weight=0.0,
        max_tech_growth_weight=0.0,
        target_volatility=0.0,
        min_positions=0,
    )
    navs: dict[str, pd.Series] = {}
    metas: dict[str, dict[str, Any]] = {}
    for name, score in shifted_scores.items():
        navs[name], metas[name] = run_multi_backtest(
            fused=score,
            prices=price_wide,
            settings=settings,
            factor_name=name,
            top_k=top_k_by_strategy[name],
            long_prices=prices,
            empty_signal_policy="hold",
        )
    for name, score in benchmark_scores.items():
        navs[name], metas[name] = run_multi_backtest(
            fused=score,
            prices=price_wide,
            settings=settings,
            factor_name=name,
            top_k=len(price_wide.columns),
            long_prices=prices,
            empty_signal_policy="hold",
        )

    nav_frame = pd.concat(navs, axis=1).sort_index()
    nav_frame.to_csv(output / "nav_comparison.csv", index_label="date")
    strategy_names = [str(spec["name"]) for spec in strategy_specs]
    performance_rows = []
    for strategy in strategy_names:
        universe = universe_by_strategy[strategy]
        benchmark = benchmark_names[universe]
        row = _performance_row(
            strategy, universe, navs[strategy], navs[benchmark], start, end
        )
        row["benchmark"] = benchmark
        row["top_k"] = top_k_by_strategy[strategy]
        row["execution_timing"] = "NEXT_SESSION_CLOSE"
        performance_rows.append(row)
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(output / "performance_comparison.csv", index=False)

    periods = {
        "FULL_EVALUATION": (start, end),
        "EARLY_COMPARISON_PREEXISTING_SPLIT": (start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT_PREEXISTING_SPLIT": (pd.Timestamp("2025-09-12"), end),
    }
    period_rows = []
    for period, (period_start, period_end) in periods.items():
        for strategy in strategy_names:
            universe = universe_by_strategy[strategy]
            benchmark = benchmark_names[universe]
            row = _performance_row(
                strategy,
                universe,
                navs[strategy],
                navs[benchmark],
                period_start,
                period_end,
            )
            row["period"] = period
            row["execution_timing"] = "NEXT_SESSION_CLOSE"
            period_rows.append(row)
    pd.DataFrame(period_rows).to_csv(output / "performance_by_period.csv", index=False)

    selected = pd.concat(
        [
            _selected(
                raw_scores[strategy],
                strategy,
                universe_by_strategy[strategy],
                top_k_by_strategy[strategy],
            )
            for strategy in strategy_names
        ],
        ignore_index=True,
    )
    selected.to_csv(output / "selected_by_signal_date.csv", index=False)
    summary = _universe_summary(report, base_eligible, profiles)
    summary.to_csv(output / "strategy_universe_summary_by_date.csv", index=False)
    report.to_csv(
        output / "strategy_universe_audit.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(
        [_execution_summary(name, meta, start) for name, meta in metas.items()]
    ).to_csv(output / "execution_summary.csv", index=False)
    runtime_paths = save_runtime_exports(
        output / "runtime_exports",
        report=report,
        shifted_scores=shifted_scores,
        universe_by_strategy=universe_by_strategy,
        top_k_by_strategy=top_k_by_strategy,
        protocol_sha256=actual,
    )

    base_rows = report[report["eligible"].astype(bool)]
    labels = {
        "BASE_V2_MULTI_FACTOR_TOP50": "Base V2 multi-factor",
        "LARGE_CAP_MULTI_FACTOR_TOP50": "Large-cap multi-factor",
        "MID_SMALL_CAP_MULTI_FACTOR_TOP50": "Mid-small multi-factor",
        "VALUE_FOCUSED_TOP50": "Value focused",
        "GROWTH_FOCUSED_TOP50": "Growth focused",
        "TECH_GROWTH_THEME_MULTI_FACTOR_TOP50": "Tech-growth theme",
    }
    charts = _plot_outputs(
        output, nav_frame, performance, summary, strategy_names, labels, start, end
    )
    checks = {
        "protocol_sha256": actual,
        "decision_dates": int(len(decision_dates)),
        "daily_basic_point_in_time_pass": bool(
            (
                report["daily_basic_date"].isna()
                | report["daily_basic_date"].le(report["date"])
            ).all()
        ),
        "base_daily_basic_coverage": float(base_rows["daily_basic_fresh"].mean()),
        "base_industry_coverage": float(
            base_rows["industry_l1"].fillna("").astype(str).str.strip().ne("").mean()
        ),
        "average_universe_sizes": {
            key: float(value)
            for key, value in summary.groupby("universe")["eligible_count"].mean().items()
        },
        "strategies": strategy_names,
        "live_approval": False,
        "verdict": "RETROSPECTIVE_STRATEGY_UNIVERSE_ARCHITECTURE_AND_BACKTEST_ONLY",
    }
    (output / "comparison_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_head = ""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "protocol_path": str(protocol_path),
        "protocol_sha256": actual,
        "data_dir": str(data_dir),
        "outputs": sorted(path.name for path in output.iterdir()),
        "charts": [str(path) for path in charts],
        "runtime_exports": {key: str(path) for key, path in runtime_paths.items()},
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "performance": output / "performance_comparison.csv",
        "checks": output / "comparison_checks.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(args.data_dir, args.output, args.protocol).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
