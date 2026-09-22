#!/usr/bin/env python3
"""Compare a frozen all-A dynamic universe with the prior CSI300 PIT universe."""
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

from analysis.benchmark import summarize_excess
from analysis.performance import summarize
from backtest.backtest_multi import run_multi_backtest
from config import get_settings
from factors.preprocess import cross_sectional_zscore, preprocess_factor_panel
from live.universe_history import load_membership_intervals, membership_mask
from scripts.build_simple_admission_fusion_backtest import _execution_summary
from universe.dynamic import build_dynamic_universe_report, eligibility_from_report


PRIMARY = "ALL_A_DYNAMIC_ADV50M_FIXED_FUSION_TOP50"
ADV30 = "ALL_A_DYNAMIC_ADV30M_FIXED_FUSION_TOP50"
ADV100 = "ALL_A_DYNAMIC_ADV100M_FIXED_FUSION_TOP50"
CSI300 = "CSI300_PIT_FIXED_FUSION_TOP50"
PRIOR = "NO_ADMISSION_ALL_FACTOR_FIXED_FUSION_TOP50"
BENCH50 = "ALL_A_DYNAMIC_ADV50M_WEEKLY_EQUAL_WEIGHT_COSTED"
BENCH30 = "ALL_A_DYNAMIC_ADV30M_WEEKLY_EQUAL_WEIGHT_COSTED"
BENCH100 = "ALL_A_DYNAMIC_ADV100M_WEEKLY_EQUAL_WEIGHT_COSTED"
CSI_BENCH = "CSI300_PIT_WEEKLY_EQUAL_WEIGHT_COSTED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_tushare_dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, format="mixed", errors="coerce")
    return compact.fillna(fallback).astype("datetime64[ns]")


def _weekly_decision_dates(calendar: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    dates = pd.to_datetime(
        calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"],
        format="%Y%m%d",
        errors="coerce",
    ).dropna()
    dates = pd.DatetimeIndex(dates[(dates >= start) & (dates <= end)]).normalize()
    if dates.empty:
        return dates
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    return pd.DatetimeIndex(frame.groupby("week")["date"].max().values).sort_values()


def _stack(matrix: pd.DataFrame, dates: pd.DatetimeIndex, name: str) -> pd.Series:
    selected = matrix.reindex(index=dates)
    value = selected.stack().dropna()
    value.index = value.index.set_names(["date", "symbol"])
    return value.rename(name)


def build_weekly_raw_factor_panel(
    prices: pd.DataFrame,
    finance: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the frozen 13-factor raw panel only on weekly decision dates."""
    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], errors="raise").dt.normalize()
    px["ts_code"] = px["ts_code"].astype(str)
    px = px.sort_values(["trade_date", "ts_code"])
    adjusted = px.pivot(index="trade_date", columns="ts_code", values="adj_close").sort_index()
    returns = adjusted.pct_change(fill_method=None)
    factors = pd.concat(
        [
            _stack(adjusted / adjusted.shift(20) - 1.0, decision_dates, "MOMENTUM"),
            _stack(adjusted / adjusted.shift(60) - 1.0, decision_dates, "MOMENTUM_60D"),
            _stack(-(adjusted / adjusted.shift(5) - 1.0), decision_dates, "REVERSAL_5D"),
            _stack(
                -returns.rolling(20, min_periods=10).std() * np.sqrt(252.0),
                decision_dates,
                "VOLATILITY",
            ),
        ],
        axis=1,
    )

    weekly = px[px["trade_date"].isin(decision_dates)][
        ["trade_date", "ts_code", "close"]
    ].copy()
    weekly["trade_date"] = weekly["trade_date"].astype("datetime64[ns]")
    fin = finance.copy()
    fin["ann_date"] = _parse_tushare_dates(fin["ann_date"])
    fin["end_date"] = _parse_tushare_dates(fin["end_date"])
    fin = fin.dropna(subset=["ann_date"]).sort_values(
        ["ts_code", "ann_date", "end_date"]
    )
    fin = fin.drop_duplicates(["ts_code", "ann_date"], keep="last")
    fin_columns = [
        "ts_code",
        "ann_date",
        "eps",
        "roe",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "or_yoy",
        "netprofit_yoy",
        "ocfps",
        "ocf_to_profit",
    ]
    available = [column for column in fin_columns if column in fin.columns]
    merged = pd.merge_asof(
        weekly.sort_values(["trade_date", "ts_code"]),
        fin[available].sort_values(["ann_date", "ts_code"]),
        left_on="trade_date",
        right_on="ann_date",
        by="ts_code",
        direction="backward",
        allow_exact_matches=True,
    )
    if bool((merged["ann_date"].notna() & merged["ann_date"].gt(merged["trade_date"])).any()):
        raise RuntimeError("finance point-in-time violation: ann_date after decision date")
    merged["PE"] = -pd.to_numeric(merged["close"], errors="coerce") / pd.to_numeric(
        merged.get("eps"), errors="coerce"
    )
    merged["PE"] = merged["PE"].where(pd.to_numeric(merged.get("eps"), errors="coerce") > 0)
    merged["FREE_CASH_FLOW_YIELD"] = pd.to_numeric(
        merged.get("ocfps"), errors="coerce"
    ) / pd.to_numeric(merged["close"], errors="coerce")
    mapping = {
        "ROE": "roe",
        "GROSS_MARGIN": "grossprofit_margin",
        "NET_MARGIN": "netprofit_margin",
        "LOW_DEBT_TO_ASSETS": "debt_to_assets",
        "REVENUE_GROWTH": "or_yoy",
        "PROFIT_GROWTH": "netprofit_yoy",
        "CASH_PROFIT_QUALITY": "ocf_to_profit",
    }
    for target, source in mapping.items():
        merged[target] = pd.to_numeric(merged.get(source), errors="coerce")
    merged["LOW_DEBT_TO_ASSETS"] = -merged["LOW_DEBT_TO_ASSETS"]
    merged_index = pd.MultiIndex.from_arrays(
        [merged["trade_date"].values, merged["ts_code"].astype(str).values],
        names=["date", "symbol"],
    )
    finance_panel = merged[
        [
            "PE",
            "FREE_CASH_FLOW_YIELD",
            "ROE",
            "GROSS_MARGIN",
            "NET_MARGIN",
            "LOW_DEBT_TO_ASSETS",
            "CASH_PROFIT_QUALITY",
            "REVENUE_GROWTH",
            "PROFIT_GROWTH",
        ]
    ].copy()
    finance_panel.index = merged_index
    panel = pd.concat([factors, finance_panel], axis=1).replace([np.inf, -np.inf], np.nan)
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    return panel


def _eligibility_for_threshold(base: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    report = base.copy()
    report["liquidity_pass"] = pd.to_numeric(report["adv20_yuan"], errors="coerce").ge(threshold)
    report["eligible"] = (
        report["listed"].astype(bool)
        & report["not_delisted"].astype(bool)
        & ~report["is_st"].astype(bool)
        & report["price_available"].astype(bool)
        & report["listing_age_pass"].astype(bool)
        & report["trading_days_pass"].astype(bool)
        & report["liquidity_pass"].astype(bool)
    )
    low = ~report["liquidity_pass"]
    report.loc[low & report["exclude_reason"].eq(""), "exclude_reason"] = "adv20_below_min"
    report.loc[low & report["exclude_reason"].ne("") & ~report["exclude_reason"].str.contains("adv20_below_min"), "exclude_reason"] += ";adv20_below_min"
    return report, eligibility_from_report(report)


def _industry_for_panel(panel: pd.DataFrame, basic: pd.DataFrame) -> pd.Series:
    lookup = (
        basic.drop_duplicates("ts_code", keep="first")
        .set_index("ts_code")["industry"]
        .fillna("")
        .astype(str)
    )
    symbols = panel.index.get_level_values("symbol")
    return pd.Series(lookup.reindex(symbols).values, index=panel.index, name="industry")


def build_fixed_family_score(
    raw_panel: pd.DataFrame,
    eligible: pd.Series,
    industry: pd.Series,
    families: dict[str, list[str]],
) -> tuple[pd.Series, pd.DataFrame]:
    mask = eligible.reindex(raw_panel.index).fillna(False).astype(bool)
    masked = raw_panel.where(mask, axis=0)
    standardized = preprocess_factor_panel(
        masked,
        industry=industry,
        by_industry=True,
        min_industry_count=3,
    )
    family_raw = pd.DataFrame(
        {
            family: standardized[factors].mean(axis=1, skipna=True)
            for family, factors in families.items()
        },
        index=standardized.index,
    )
    family_scores = cross_sectional_zscore(family_raw.where(mask, axis=0))
    score = family_scores.mean(axis=1, skipna=True).where(
        family_scores.notna().sum(axis=1).eq(len(families))
    )
    return score.where(mask).dropna().rename("score"), family_scores


def _benchmark_score(eligible: pd.Series, prices: pd.DataFrame) -> pd.Series:
    available = prices.set_index(["trade_date", "ts_code"])["adj_close"].notna()
    available.index = available.index.set_names(["date", "symbol"])
    mask = eligible.reindex(available.index).fillna(False).astype(bool) & available
    return pd.Series(1.0, index=mask.index[mask], name="score").sort_index()


def _shift_score_to_next_session(score: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Move a close-known signal to the following session for timing sensitivity."""
    frame = score.rename("score").reset_index()
    session_values = pd.DatetimeIndex(sessions).sort_values()
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for raw_date in pd.to_datetime(frame["date"]).unique():
        date = pd.Timestamp(raw_date)
        position = int(session_values.searchsorted(date, side="right"))
        if position < len(session_values):
            mapping[date] = pd.Timestamp(session_values[position])
    frame["date"] = pd.to_datetime(frame["date"]).map(mapping)
    frame = frame.dropna(subset=["date"])
    shifted = frame.set_index(["date", "symbol"])["score"].sort_index()
    shifted.index = shifted.index.set_names(["date", "symbol"])
    return shifted


def _normalize(nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    out = nav.loc[(nav.index >= start) & (nav.index <= end)].dropna().astype(float)
    return out / float(out.iloc[0]) if not out.empty else out


def _performance_row(
    strategy: str,
    universe: str,
    nav: pd.Series,
    benchmark: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    selected = _normalize(nav, start, end)
    bench = _normalize(benchmark, start, end)
    row: dict[str, Any] = {"strategy": strategy, "universe": universe}
    row.update(summarize(selected))
    row.update(summarize_excess(selected, bench))
    benchmark_total = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    row["benchmark_total_return"] = benchmark_total
    row["return_gap_vs_benchmark"] = float(row["total_return"] - benchmark_total)
    row["start"] = selected.index.min().strftime("%Y-%m-%d")
    row["end"] = selected.index.max().strftime("%Y-%m-%d")
    return row


def _universe_summary(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, report in reports.items():
        for date, group in report.groupby("date", sort=True):
            eligible = group.loc[group["eligible"], "symbol"].astype(str)
            rows.append(
                {
                    "universe": name,
                    "date": pd.Timestamp(date),
                    "eligible_count": int(len(eligible)),
                    "median_adv20_yuan": float(group.loc[group["eligible"], "adv20_yuan"].median()),
                    "st_excluded": int(group["is_st"].sum()),
                    "age_excluded": int((~group["listing_age_pass"].astype(bool)).sum()),
                    "liquidity_excluded": int((~group["liquidity_pass"].astype(bool)).sum()),
                }
            )
    return pd.DataFrame(rows)


def _plot_outputs(
    output: Path,
    navs: pd.DataFrame,
    performance: pd.DataFrame,
    universe_summary: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    active = navs.loc[(navs.index >= start) & (navs.index <= end)].copy()
    active = active.apply(lambda x: x / float(x.dropna().iloc[0]))
    labels = {
        PRIMARY: "All-A dynamic ADV50m Top50",
        ADV30: "All-A dynamic ADV30m Top50",
        ADV100: "All-A dynamic ADV100m Top50",
        CSI300: "CSI300 PIT Top50 (same rebuild)",
        PRIOR: "Prior saved CSI300 Top50",
        BENCH50: "All-A ADV50m equal weight",
        CSI_BENCH: "CSI300 PIT equal weight",
    }
    colors = {
        PRIMARY: "#d95f02",
        ADV30: "#e6ab02",
        ADV100: "#a6761d",
        CSI300: "#7c3aed",
        PRIOR: "#b39ddb",
        BENCH50: "#1b9e77",
        CSI_BENCH: "#0f766e",
    }
    order = [PRIMARY, ADV30, ADV100, CSI300, PRIOR, BENCH50, CSI_BENCH]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name in order:
        if name not in active:
            continue
        width = 2.4 if name in {PRIMARY, CSI300} else 1.5
        axes[0].plot(active.index, active[name], label=labels[name], color=colors[name], linewidth=width)
        if name in {PRIMARY, ADV30, ADV100, CSI300, PRIOR}:
            axes[1].plot(active.index, active[name] / active[name].cummax() - 1.0, label=labels[name], color=colors[name], linewidth=width)
    axes[0].set_title("Custom dynamic universe versus CSI300 point-in-time control")
    axes[0].set_ylabel("NAV")
    axes[1].set_ylabel("Drawdown")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = chart_dir / "dynamic_universe_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    plot_perf = performance.set_index("strategy").reindex([PRIMARY, ADV30, ADV100, CSI300])
    x = np.arange(len(plot_perf))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - 0.18, plot_perf["total_return"] * 100, 0.36, label="Strategy total return", color="#4c78a8")
    ax.bar(x + 0.18, plot_perf["benchmark_total_return"] * 100, 0.36, label="Own-universe benchmark", color="#9ecae1")
    ax.set_xticks(x, ["ADV50m", "ADV30m", "ADV100m", "CSI300"])
    ax.set_ylabel("Total return (%)")
    ax.set_title("Each Top50 strategy against its own investable-universe benchmark")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = chart_dir / "dynamic_universe_strategy_vs_own_benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for name, group in universe_summary.groupby("universe"):
        ax.plot(group["date"], group["eligible_count"], label=name, linewidth=2)
    ax.set_ylabel("Eligible stocks")
    ax.set_title("Point-in-time custom-universe breadth by liquidity threshold")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = chart_dir / "dynamic_universe_breadth.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(data_dir: Path, membership_path: Path, prior_output: Path, output: Path, protocol_path: Path) -> dict[str, Path]:
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
    basic = pd.read_csv(data_dir / "stock_basic_all_a.csv", dtype={"ts_code": str})
    names = pd.read_csv(data_dir / "namechange_all_a.csv", dtype={"ts_code": str})
    calendar = pd.read_csv(data_dir / "trade_calendar.csv", dtype={"cal_date": str})
    decision_dates = _weekly_decision_dates(calendar, start, end)
    base_report = build_dynamic_universe_report(
        prices,
        basic,
        names,
        calendar,
        decision_dates,
        min_listing_sessions=int(protocol["universe"]["minimum_listing_open_sessions"]),
        liquidity_window=int(protocol["universe"]["liquidity_lookback_open_sessions"]),
        min_valid_days=int(protocol["universe"]["minimum_valid_trading_days"]),
        min_adv_yuan=0.0,
        amount_col="amount_yuan",
    )
    reports: dict[str, pd.DataFrame] = {}
    eligibilities: dict[str, pd.Series] = {}
    for label, threshold in (("ADV30M", 30_000_000.0), ("ADV50M", 50_000_000.0), ("ADV100M", 100_000_000.0)):
        reports[label], eligibilities[label] = _eligibility_for_threshold(base_report, threshold)

    raw_panel = build_weekly_raw_factor_panel(prices, finance, decision_dates)
    membership = load_membership_intervals(membership_path)
    csi_eligible = membership_mask(raw_panel.index, membership)
    industry = _industry_for_panel(raw_panel, basic)
    families = {str(k): [str(v) for v in values] for k, values in protocol["candidate_families"].items()}
    scores: dict[str, pd.Series] = {}
    family_outputs: dict[str, pd.DataFrame] = {}
    for strategy, key in ((ADV30, "ADV30M"), (PRIMARY, "ADV50M"), (ADV100, "ADV100M")):
        scores[strategy], family_outputs[strategy] = build_fixed_family_score(
            raw_panel, eligibilities[key], industry, families
        )
    scores[CSI300], family_outputs[CSI300] = build_fixed_family_score(
        raw_panel, csi_eligible, industry, families
    )

    price_wide = prices.pivot(index="trade_date", columns="ts_code", values="adj_close").sort_index()
    price_wide = price_wide.loc[price_wide.index <= end]
    settings = replace(
        get_settings(),
        backtest_start=protocol["data"]["raw_start"],
        backtest_end=protocol["data"]["evaluation_end"],
        rebalance_freq="W-FRI",
        force_final_rebalance=False,
        top_k=50,
        portfolio_weighting="equal",
        commission_rate=float(protocol["portfolio"]["commission_rate"]),
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
    for strategy, score in scores.items():
        navs[strategy], metas[strategy] = run_multi_backtest(
            fused=score,
            prices=price_wide,
            settings=settings,
            factor_name=strategy,
            top_k=50,
            long_prices=prices,
            empty_signal_policy="cash",
        )

    benchmark_specs = {
        BENCH30: eligibilities["ADV30M"],
        BENCH50: eligibilities["ADV50M"],
        BENCH100: eligibilities["ADV100M"],
        CSI_BENCH: csi_eligible,
    }
    for benchmark, eligible in benchmark_specs.items():
        score = _benchmark_score(eligible, prices)
        navs[benchmark], metas[benchmark] = run_multi_backtest(
            fused=score,
            prices=price_wide,
            settings=settings,
            factor_name=benchmark,
            top_k=len(price_wide.columns),
            long_prices=prices,
            empty_signal_policy="cash",
        )

    prior_frame = pd.read_csv(prior_output / "nav_comparison.csv", index_col=0, parse_dates=True)
    navs[PRIOR] = prior_frame[PRIOR].astype(float)
    nav_frame = pd.concat(navs, axis=1).sort_index()
    nav_frame.to_csv(output / "nav_comparison.csv", index_label="date")

    benchmark_by_strategy = {
        PRIMARY: BENCH50,
        ADV30: BENCH30,
        ADV100: BENCH100,
        CSI300: CSI_BENCH,
        PRIOR: CSI_BENCH,
    }
    universe_by_strategy = {
        PRIMARY: "ALL_A_DYNAMIC_ADV50M",
        ADV30: "ALL_A_DYNAMIC_ADV30M",
        ADV100: "ALL_A_DYNAMIC_ADV100M",
        CSI300: "CSI300_POINT_IN_TIME",
        PRIOR: "CSI300_POINT_IN_TIME_PRIOR_SAVED",
    }
    performance = pd.DataFrame(
        [
            _performance_row(
                strategy,
                universe_by_strategy[strategy],
                navs[strategy],
                navs[benchmark_by_strategy[strategy]],
                start,
                end,
            )
            for strategy in [PRIMARY, ADV30, ADV100, CSI300, PRIOR]
        ]
    )
    performance.to_csv(output / "performance_comparison.csv", index=False)
    universe_summary = _universe_summary(reports)
    universe_summary.to_csv(output / "universe_summary_by_date.csv", index=False)
    reports["ADV50M"].to_csv(output / "dynamic_universe_adv50m_audit.csv.gz", index=False, compression="gzip")
    raw_panel.reset_index().to_csv(output / "weekly_raw_factor_panel.csv.gz", index=False, compression="gzip")
    pd.DataFrame({name: value for name, value in scores.items()}).reset_index().to_csv(
        output / "weekly_fused_scores.csv.gz", index=False, compression="gzip"
    )
    execution = pd.DataFrame(
        [_execution_summary(name, meta, start) for name, meta in metas.items()]
    )
    execution.to_csv(output / "execution_summary.csv", index=False)

    holdings = []
    for strategy, score in scores.items():
        for date, values in score.groupby(level="date", sort=True):
            ranked = values.droplevel("date").sort_values(ascending=False).head(50)
            for rank, (symbol, value) in enumerate(ranked.items(), start=1):
                holdings.append(
                    {
                        "date": pd.Timestamp(date),
                        "strategy": strategy,
                        "rank": rank,
                        "symbol": str(symbol),
                        "score": float(value),
                    }
                )
    pd.DataFrame(holdings).to_csv(output / "selected_top50_by_date.csv", index=False)

    periods = {
        "FULL_EVALUATION": (start, end),
        "EARLY_COMPARISON_PREEXISTING_SPLIT": (start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT_PREEXISTING_SPLIT": (pd.Timestamp("2025-09-12"), end),
    }
    period_rows = []
    for period, (period_start, period_end) in periods.items():
        for strategy in [PRIMARY, ADV30, ADV100, CSI300, PRIOR]:
            row = _performance_row(
                strategy,
                universe_by_strategy[strategy],
                navs[strategy],
                navs[benchmark_by_strategy[strategy]],
                period_start,
                period_end,
            )
            row["period"] = period
            row["period_status"] = (
                "FROZEN_PRIMARY" if period == "FULL_EVALUATION" else "POSTHOC_REUSE_OF_PRIOR_FROZEN_SPLIT"
            )
            period_rows.append(row)
    pd.DataFrame(period_rows).to_csv(output / "performance_by_period.csv", index=False)

    # Post-hoc robustness only: signals known after the decision-day close are
    # moved to the next session and traded at that session's close.  This is
    # deliberately more conservative than a next-open implementation.
    lag_settings = replace(settings, rebalance_freq="D")
    lag_rows = []
    for strategy, benchmark, universe_name in (
        (PRIMARY, BENCH50, "ALL_A_DYNAMIC_ADV50M"),
        (CSI300, CSI_BENCH, "CSI300_POINT_IN_TIME"),
    ):
        lag_score = _shift_score_to_next_session(scores[strategy], price_wide.index)
        original_benchmark_score = _benchmark_score(benchmark_specs[benchmark], prices)
        lag_benchmark_score = _shift_score_to_next_session(original_benchmark_score, price_wide.index)
        lag_nav, _ = run_multi_backtest(
            fused=lag_score,
            prices=price_wide,
            settings=lag_settings,
            factor_name=strategy + "_TPLUS1_CLOSE",
            top_k=50,
            long_prices=prices,
            empty_signal_policy="hold",
        )
        lag_benchmark_nav, _ = run_multi_backtest(
            fused=lag_benchmark_score,
            prices=price_wide,
            settings=lag_settings,
            factor_name=benchmark + "_TPLUS1_CLOSE",
            top_k=len(price_wide.columns),
            long_prices=prices,
            empty_signal_policy="hold",
        )
        same_day = _performance_row(strategy, universe_name, navs[strategy], navs[benchmark], start, end)
        same_day["timing"] = "DECISION_DAY_CLOSE"
        same_day["status"] = "FROZEN_ORIGINAL_ENGINE"
        lagged = _performance_row(strategy, universe_name, lag_nav, lag_benchmark_nav, start, end)
        lagged["timing"] = "NEXT_SESSION_CLOSE"
        lagged["status"] = "POSTHOC_CONSERVATIVE_TIMING_SENSITIVITY"
        lag_rows.extend([same_day, lagged])
    timing = pd.DataFrame(lag_rows)
    timing.to_csv(output / "execution_timing_sensitivity.csv", index=False)

    full = performance.set_index("strategy")
    csi_rebuild_prior = pd.concat(
        [_normalize(navs[CSI300], start, end), _normalize(navs[PRIOR], start, end)], axis=1
    ).dropna()
    checks = {
        "protocol_sha256": actual,
        "decision_dates": int(len(decision_dates)),
        "raw_factor_rows": int(len(raw_panel)),
        "all_a_price_symbols": int(prices["ts_code"].nunique()),
        "primary_universe_average_size": float(universe_summary.loc[universe_summary["universe"].eq("ADV50M"), "eligible_count"].mean()),
        "primary_universe_min_size": int(universe_summary.loc[universe_summary["universe"].eq("ADV50M"), "eligible_count"].min()),
        "primary_universe_max_size": int(universe_summary.loc[universe_summary["universe"].eq("ADV50M"), "eligible_count"].max()),
        "primary_total_return": float(full.loc[PRIMARY, "total_return"]),
        "primary_own_benchmark_total_return": float(full.loc[PRIMARY, "benchmark_total_return"]),
        "primary_return_gap": float(full.loc[PRIMARY, "return_gap_vs_benchmark"]),
        "csi300_rebuild_total_return": float(full.loc[CSI300, "total_return"]),
        "prior_saved_csi300_total_return": float(full.loc[PRIOR, "total_return"]),
        "primary_minus_csi300_rebuild_return": float(full.loc[PRIMARY, "total_return"] - full.loc[CSI300, "total_return"]),
        "adv30_minus_primary_return": float(full.loc[ADV30, "total_return"] - full.loc[PRIMARY, "total_return"]),
        "adv100_minus_primary_return": float(full.loc[ADV100, "total_return"] - full.loc[PRIMARY, "total_return"]),
        "finance_point_in_time_pass": True,
        "market_cap_filter_enabled": False,
        "profitability_filter_enabled": False,
        "historical_data_already_seen": True,
        "live_approval": False,
        "csi300_rebuild_matches_prior_saved": bool(
            len(csi_rebuild_prior) > 0
            and np.allclose(csi_rebuild_prior.iloc[:, 0], csi_rebuild_prior.iloc[:, 1], atol=1e-10)
        ),
        "primary_next_session_close_total_return": float(
            timing.loc[
                timing["strategy"].eq(PRIMARY) & timing["timing"].eq("NEXT_SESSION_CLOSE"),
                "total_return",
            ].iloc[0]
        ),
        "primary_next_session_close_return_gap": float(
            timing.loc[
                timing["strategy"].eq(PRIMARY) & timing["timing"].eq("NEXT_SESSION_CLOSE"),
                "return_gap_vs_benchmark",
            ].iloc[0]
        ),
        "verdict": "RETROSPECTIVE_DYNAMIC_UNIVERSE_COMPARISON_ONLY",
    }
    (output / "comparison_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    charts = _plot_outputs(output, nav_frame, performance, universe_summary, start, end)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
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
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"performance": output / "performance_comparison.csv", "checks": output / "comparison_checks.json"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(args.data_dir, args.membership, args.prior_output, args.output, args.protocol).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
