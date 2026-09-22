#!/usr/bin/env python3
"""Evaluate the frozen CSI300 weekly Top10 state gate without tuning its rules."""
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
from analysis.performance import summarize
from backtest.backtest_multi import run_multi_backtest
from live.cache_io import save_rebalance_logs
from live.universe_history import load_membership_intervals, mask_wide_prices_by_membership
from scripts.build_walk_forward_factor_backtest import _settings_from_source


BASELINE = "CSI300_WF_TOP10_BASELINE"
STATE = "CSI300_WF_TOP10_STATE_GATE_V1"
CONTROL = "CSI300_WF_TOP5_CONTROL"


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


def _members_on(membership: pd.DataFrame, dt: pd.Timestamp) -> set[str]:
    active = membership[
        (membership["effective_from"] <= dt) & (membership["effective_to"] >= dt)
    ]
    return set(active["ts_code"].astype(str))


def _compound(series: pd.Series, n: int) -> float:
    tail = series.dropna().iloc[-n:]
    if len(tail) < n:
        return float("nan")
    return float((1.0 + tail).prod() - 1.0)


def build_state_monitor(
    prices: pd.DataFrame,
    long_prices: pd.DataFrame,
    membership: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    tech_industries: set[str],
) -> pd.DataFrame:
    """Compute every input through t-1 and apply the frozen hysteresis state machine."""
    pit_prices = mask_wide_prices_by_membership(prices, membership)
    returns = pit_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    industry_by_symbol = (
        long_prices.sort_values("trade_date").dropna(subset=["industry"])
        .drop_duplicates("ts_code", keep="last").set_index("ts_code")["industry"].astype(str).to_dict()
    )
    tech_symbols = {
        str(symbol) for symbol, industry in industry_by_symbol.items() if industry in tech_industries
    }
    tech_cols = [column for column in pit_prices.columns if str(column) in tech_symbols]
    tech_returns = returns[tech_cols].mean(axis=1, skipna=True)
    benchmark_returns = returns.mean(axis=1, skipna=True)

    rows: list[dict[str, Any]] = []
    for dt in pd.DatetimeIndex(rebalance_dates).sort_values():
        history = prices.index[prices.index < dt]
        if len(history) == 0:
            continue
        history_end = pd.Timestamp(history[-1])
        tech20 = _compound(tech_returns.loc[:history_end], 20)
        tech60 = _compound(tech_returns.loc[:history_end], 60)
        bench20 = _compound(benchmark_returns.loc[:history_end], 20)
        bench60 = _compound(benchmark_returns.loc[:history_end], 60)
        relative20 = tech20 - bench20
        relative60 = tech60 - bench60

        current_tech = sorted(_members_on(membership, history_end) & tech_symbols)
        current_px = pit_prices.loc[history_end, current_tech] if current_tech else pd.Series(dtype=float)
        ma60 = pit_prices[current_tech].loc[:history_end].tail(60).mean() if current_tech else pd.Series(dtype=float)
        comparable = pd.concat([current_px.rename("price"), ma60.rename("ma")], axis=1).dropna()
        breadth = float((comparable["price"] > comparable["ma"]).mean()) if not comparable.empty else float("nan")

        ret20 = tech_returns.loc[:history_end].dropna().iloc[-20:]
        ret120 = tech_returns.loc[:history_end].dropna().iloc[-120:]
        vol20 = float(ret20.std(ddof=0)) if len(ret20) >= 20 else float("nan")
        vol120 = float(ret120.std(ddof=0)) if len(ret120) >= 120 else float("nan")
        vol_ratio = vol20 / vol120 if np.isfinite(vol120) and vol120 > 1e-12 else float("nan")

        member_px = pit_prices[sorted(_members_on(membership, history_end))].loc[:history_end]
        member_20d = member_px.iloc[-1] / member_px.iloc[-21] - 1.0 if len(member_px) >= 21 else pd.Series(dtype=float)
        dispersion = float(member_20d.replace([np.inf, -np.inf], np.nan).std(ddof=0)) if not member_20d.empty else float("nan")

        weak_trend = bool(
            np.isfinite(relative20) and np.isfinite(relative60) and relative20 < 0.0 and relative60 < 0.0
        )
        weak_breadth = bool(np.isfinite(breadth) and breadth < 0.40)
        vol_shock = bool(np.isfinite(vol_ratio) and vol_ratio > 1.25)
        bad_count = int(weak_trend) + int(weak_breadth) + int(vol_shock)
        raw_state = "RED" if bad_count == 3 else ("YELLOW" if bad_count == 2 else "GREEN")
        rows.append(
            {
                "date": dt,
                "history_end": history_end,
                "tech_relative_return_20d": relative20,
                "tech_relative_return_60d": relative60,
                "tech_breadth_above_60d_ma": breadth,
                "tech_volatility_ratio_20d_120d": vol_ratio,
                "cross_section_return_dispersion_20d": dispersion,
                "weak_relative_trend": weak_trend,
                "weak_breadth": weak_breadth,
                "volatility_shock": vol_shock,
                "bad_input_count": bad_count,
                "raw_state": raw_state,
                "n_tech_members": len(current_tech),
            }
        )

    monitor = pd.DataFrame(rows)
    state = "GREEN"
    consecutive_raw_yellow = 0
    consecutive_raw_green = 0
    consecutive_non_red = 0
    effective: list[str] = []
    for raw in monitor["raw_state"]:
        consecutive_raw_yellow = consecutive_raw_yellow + 1 if raw == "YELLOW" else 0
        consecutive_raw_green = consecutive_raw_green + 1 if raw == "GREEN" else 0
        consecutive_non_red = consecutive_non_red + 1 if raw != "RED" else 0
        if raw == "RED":
            state = "RED"
        elif state == "GREEN" and consecutive_raw_yellow >= 2:
            state = "YELLOW"
        elif state == "YELLOW" and consecutive_raw_green >= 2:
            state = "GREEN"
        elif state == "RED" and consecutive_non_red >= 2:
            state = "YELLOW"
            consecutive_raw_green = 0
        effective.append(state)
    monitor["risk_state"] = effective
    monitor["gross_exposure_multiplier"] = monitor["risk_state"].map(
        {"GREEN": 1.0, "YELLOW": 0.75, "RED": 0.50}
    )
    monitor["max_tech_growth_weight"] = monitor["risk_state"].map(
        {"GREEN": 0.0, "YELLOW": 0.40, "RED": 0.25}
    )
    return monitor


def _normalize_period(nav: pd.Series, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    selected = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))].dropna()
    return selected / float(selected.iloc[0]) if not selected.empty else selected


def _stats(strategy: str, period: str, nav: pd.Series, benchmark: pd.Series) -> dict[str, Any]:
    row: dict[str, Any] = {"strategy": strategy, "period": period}
    row.update(summarize(nav))
    row.update(summarize_excess(nav, benchmark))
    row["start"] = nav.index.min().strftime("%Y-%m-%d") if not nav.empty else ""
    row["end"] = nav.index.max().strftime("%Y-%m-%d") if not nav.empty else ""
    row["n_days"] = len(nav)
    return row


def _turnover(meta: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> float:
    total = 0.0
    previous: dict[str, float] = {}
    for rec in sorted(meta.get("rebalance_log", []), key=lambda value: pd.Timestamp(value["date"])):
        dt = pd.Timestamp(rec["date"])
        current = {
            str(symbol): float(weight)
            for symbol, weight in zip(rec.get("picks", []), rec.get("weights", []))
        }
        if dt < start:
            previous = current
            continue
        if dt > end:
            break
        symbols = set(previous) | set(current)
        total += sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)
        previous = current
    return float(total)


def _portfolio_monitor(meta: dict[str, Any], industry_by_symbol: dict[str, str], tech: set[str]) -> pd.DataFrame:
    rows = []
    for rec in meta.get("rebalance_log", []):
        weights = dict(zip(rec.get("picks", []), rec.get("weights", [])))
        if not weights:
            continue
        industries: dict[str, float] = {}
        for symbol, weight in weights.items():
            industry = industry_by_symbol.get(str(symbol), "UNKNOWN")
            industries[industry] = industries.get(industry, 0.0) + float(weight)
        gross = float(sum(float(x) for x in weights.values()))
        normalized_weights = [float(x) / gross for x in weights.values()] if gross > 1e-12 else []
        normalized_industries = [float(x) / gross for x in industries.values()] if gross > 1e-12 else []
        pos_hhi = float(sum(x * x for x in normalized_weights))
        rows.append({
            "date": pd.Timestamp(rec["date"]),
            "gross_target_weight": gross,
            "holding_count": len(weights),
            "holding_count_above_10bp": int(sum(float(x) >= 0.001 for x in weights.values())),
            "holding_count_above_50bp": int(sum(float(x) >= 0.005 for x in weights.values())),
            "tech_growth_exposure": float(sum(w for s, w in weights.items() if industry_by_symbol.get(str(s), "") in tech)),
            "largest_industry_exposure": max(industries.values()) if industries else 0.0,
            "industry_hhi": float(sum(x * x for x in normalized_industries)),
            "position_hhi": pos_hhi,
            "effective_position_count": 1.0 / pos_hhi if pos_hhi > 1e-12 else 0.0,
            "cash_target_weight": max(0.0, 1.0 - float(sum(weights.values()))),
        })
    return pd.DataFrame(rows)


def _factor_crowding(
    panel: pd.DataFrame,
    style_weights: pd.DataFrame,
    fused: pd.Series,
) -> pd.DataFrame:
    rows = []
    for dt, group in style_weights.groupby("date"):
        try:
            cross = panel.xs(pd.Timestamp(dt), level="date")
            eligible = fused.xs(pd.Timestamp(dt), level="date").dropna().index
        except KeyError:
            continue
        components = sorted({item for value in group["components"].dropna().astype(str) for item in value.split(",") if item in cross.columns})
        data = cross.reindex(eligible)[components].dropna(axis=1, how="all")
        if data.shape[1] < 2:
            rows.append({"date": dt, "active_component_count": data.shape[1], "mean_abs_factor_correlation": np.nan, "mean_top20_jaccard": np.nan})
            continue
        corr = data.corr(method="spearman").abs().to_numpy()
        tri = corr[np.triu_indices_from(corr, k=1)]
        tops = {col: set(data[col].dropna().nlargest(20).index) for col in data.columns}
        jac = []
        cols = list(tops)
        for i, left in enumerate(cols):
            for right in cols[i + 1:]:
                union = tops[left] | tops[right]
                if union:
                    jac.append(len(tops[left] & tops[right]) / len(union))
        rows.append({
            "date": pd.Timestamp(dt),
            "active_component_count": data.shape[1],
            "mean_abs_factor_correlation": float(np.nanmean(tri)) if len(tri) else np.nan,
            "mean_top20_jaccard": float(np.mean(jac)) if jac else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_all(output: Path, navs: pd.DataFrame, state_monitor: pd.DataFrame, exposure: pd.DataFrame, crowding: pd.DataFrame) -> list[Path]:
    chart_dir = output / "article_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {BASELINE: "#6b7280", STATE: "#d95f02", CONTROL: "#8b5cf6", "CSI300_EQUAL_WEIGHT": "#0f766e"}
    for name in [BASELINE, STATE, CONTROL, "CSI300_EQUAL_WEIGHT"]:
        series = navs[name] / float(navs[name].iloc[0])
        axes[0].plot(series.index, series, label=name.replace("CSI300_WF_", "").replace("CSI300_", ""), color=colors[name], linewidth=2 if name in {BASELINE, STATE} else 1.4)
    for name in [BASELINE, STATE]:
        series = navs[name] / float(navs[name].iloc[0])
        axes[1].plot(series.index, series / series.cummax() - 1.0, label=name.replace("CSI300_WF_", ""), color=colors[name])
    axes[0].set_title("Frozen weekly Top10: baseline versus state gate")
    axes[0].set_ylabel("NAV")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=.2)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    path = chart_dir / "baseline_state_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    m = state_monitor.set_index("date")
    axes[0].plot(m.index, m["tech_relative_return_20d"], label="20d", color="#4c78a8")
    axes[0].plot(m.index, m["tech_relative_return_60d"], label="60d", color="#f58518")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set_ylabel("Relative\nreturn")
    axes[0].legend(ncol=2)
    axes[1].plot(m.index, m["tech_breadth_above_60d_ma"], color="#54a24b")
    axes[1].axhline(.4, color="#d62728", ls="--")
    axes[1].set_ylabel("Breadth")
    axes[2].plot(m.index, m["tech_volatility_ratio_20d_120d"], color="#e45756")
    axes[2].axhline(1.25, color="#d62728", ls="--")
    axes[2].set_ylabel("Vol ratio")
    state_num = m["risk_state"].map({"GREEN": 0, "YELLOW": 1, "RED": 2})
    axes[3].step(m.index, state_num, where="post", color="#7c3aed")
    axes[3].set_yticks([0, 1, 2], ["GREEN", "YELLOW", "RED"])
    axes[3].set_ylabel("State")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set_title("Lagged state inputs and hysteresis (all inputs end at t-1)")
    fig.tight_layout()
    path = chart_dir / "state_inputs_timeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    monitor = exposure.merge(crowding, on="date", how="left").merge(
        state_monitor[["date", "cross_section_return_dispersion_20d"]], on="date", how="left"
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(monitor["date"], monitor["tech_growth_exposure"], label="Tech-growth", color="#d95f02")
    axes[0].plot(monitor["date"], monitor["largest_industry_exposure"], label="Largest industry", color="#4c78a8")
    axes[0].set_ylabel("Weight")
    axes[0].legend(ncol=2)
    axes[1].plot(monitor["date"], monitor["effective_position_count"], color="#54a24b")
    axes[1].set_ylabel("Effective N")
    axes[2].plot(monitor["date"], monitor["mean_abs_factor_correlation"], label="Mean |rank corr|", color="#b279a2")
    axes[2].plot(monitor["date"], monitor["mean_top20_jaccard"], label="Top20 Jaccard", color="#ff9da6")
    axes[2].set_ylabel("Factor crowding")
    axes[2].legend(ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set_title("Portfolio-level common exposure monitor")
    fig.tight_layout()
    path = chart_dir / "portfolio_common_exposure.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    july = navs.loc["2026-07-01":"2026-07-31", [BASELINE, STATE, "CSI300_EQUAL_WEIGHT"]]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for name, color in [(BASELINE, "#6b7280"), (STATE, "#d95f02"), ("CSI300_EQUAL_WEIGHT", "#0f766e")]:
        series = july[name] / float(july[name].iloc[0])
        ax.plot(series.index, series, label=name.replace("CSI300_WF_", "").replace("CSI300_", ""), color=color, linewidth=2)
    ax.set_title("July 2026 diagnostic only — not a tuning target")
    ax.set_ylabel("NAV rebased to 1.0")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = chart_dir / "july_2026_diagnostic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(source: Path, walk_forward: Path, output: Path, protocol_path: Path) -> dict[str, Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_hash = _sha256(protocol_path)
    expected_hash_path = protocol_path.with_suffix(".sha256")
    expected_hash = expected_hash_path.read_text(encoding="utf-8").strip().split()[0]
    if frozen_hash != expected_hash:
        raise RuntimeError("protocol hash mismatch; frozen preregistration was modified")

    settings, _ = _settings_from_source(source, output)
    settings = replace(settings, rebalance_freq="W-FRI", max_tech_growth_weight=0.0)
    prices = pd.read_csv(source / "cache" / "prices_wide_adj_close.csv", index_col=0, parse_dates=True).sort_index()
    prices.columns = prices.columns.astype(str)
    long_prices = pd.read_csv(source / "cache" / "prices_long.csv", parse_dates=["trade_date"])
    long_prices["ts_code"] = long_prices["ts_code"].astype(str)
    membership = load_membership_intervals(settings.universe_membership_path)
    fused = _load_score(walk_forward / "walk_forward" / "fused_scores.csv")
    summary = pd.read_csv(walk_forward / "walk_forward" / "rebalance_summary.csv", parse_dates=["date"])
    style_weights = pd.read_csv(walk_forward / "walk_forward" / "style_weights.csv", parse_dates=["date"])
    panel = pd.read_csv(source / "cache" / "factor_panel_zscore.csv", parse_dates=["date"])
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel.set_index(["date", "symbol"]).sort_index()

    monitor = build_state_monitor(
        prices,
        long_prices,
        membership,
        pd.DatetimeIndex(summary["date"]),
        set(protocol["technology_growth_industries"]),
    )
    baseline_nav, baseline_meta = run_multi_backtest(
        fused=fused, prices=prices, settings=settings, factor_name=BASELINE,
        top_k=10, long_prices=long_prices, empty_signal_policy="cash",
    )
    state_nav, state_meta = run_multi_backtest(
        fused=fused, prices=prices, settings=settings, factor_name=STATE,
        top_k=10, long_prices=long_prices, empty_signal_policy="cash",
        rebalance_overrides=monitor,
    )
    control_nav, control_meta = run_multi_backtest(
        fused=fused, prices=prices, settings=settings, factor_name=CONTROL,
        top_k=5, long_prices=long_prices, empty_signal_policy="cash",
    )
    benchmark = equal_weight_benchmark_nav(mask_wide_prices_by_membership(prices, membership), dates=prices.index, name="CSI300_EQUAL_WEIGHT")
    navs = pd.concat([baseline_nav.rename(BASELINE), state_nav.rename(STATE), control_nav.rename(CONTROL), benchmark], axis=1).dropna()

    active = summary[summary["signal_status"] == "ACTIVE"]
    active_start = pd.Timestamp(active["date"].min())
    periods = {
        "FULL_ACTIVE": (active_start, pd.Timestamp("2026-09-04")),
        "HISTORICAL_ROBUSTNESS": (active_start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT": (pd.Timestamp("2025-09-12"), pd.Timestamp("2026-09-04")),
        "JULY_2026_DIAGNOSTIC": (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31")),
    }
    stats_rows = []
    for period, (start, end) in periods.items():
        bench_period = _normalize_period(benchmark, start, end)
        for name, nav in [(BASELINE, baseline_nav), (STATE, state_nav), (CONTROL, control_nav)]:
            stats_rows.append(_stats(name, period, _normalize_period(nav, start, end), bench_period))
    performance = pd.DataFrame(stats_rows)

    robust = performance[performance["period"] == "HISTORICAL_ROBUSTNESS"].set_index("strategy")
    full = performance[performance["period"] == "FULL_ACTIVE"].set_index("strategy")
    july = performance[performance["period"] == "JULY_2026_DIAGNOSTIC"].set_index("strategy")
    robust_start, robust_end = periods["HISTORICAL_ROBUSTNESS"]
    base_turnover = _turnover(baseline_meta, robust_start, robust_end)
    state_turnover = _turnover(state_meta, robust_start, robust_end)
    base_mdd = abs(float(robust.loc[BASELINE, "max_drawdown"]))
    state_mdd = abs(float(robust.loc[STATE, "max_drawdown"]))
    mdd_improvement = (base_mdd - state_mdd) / base_mdd if base_mdd > 1e-12 else np.nan
    july_base = float(july.loc[BASELINE, "total_return"])
    july_state = float(july.loc[STATE, "total_return"])
    july_reduction = (abs(july_base) - abs(min(july_state, 0.0))) / abs(july_base) if july_base < 0 else None

    active_navs = navs.loc[navs.index >= active_start, [BASELINE, STATE]]
    monthly = active_navs.pct_change(fill_method=None).add(1).groupby(active_navs.index.to_period("M")).prod().sub(1)
    adverse = monthly[monthly[BASELINE] < 0].copy()
    adverse["improvement_pp"] = (adverse[STATE] - adverse[BASELINE]) * 100.0
    # Ignore sub-basis-point floating/cost noise when counting distinct episodes.
    adverse["materially_improved"] = adverse["improvement_pp"] >= 0.10
    adverse.index = adverse.index.astype(str)
    adverse.index.name = "month"
    criteria = {
        "protocol_sha256": frozen_hash,
        "historical_robustness_max_drawdown_improvement": float(mdd_improvement),
        "historical_robustness_max_drawdown_pass": bool(mdd_improvement >= 0.20),
        "historical_robustness_return_difference_pp": float((robust.loc[STATE, "total_return"] - robust.loc[BASELINE, "total_return"]) * 100),
        "historical_robustness_return_tolerance_pass": bool(robust.loc[STATE, "total_return"] >= robust.loc[BASELINE, "total_return"] - 0.05),
        "turnover_ratio": float(state_turnover / base_turnover) if base_turnover > 1e-12 else np.nan,
        "turnover_limit_pass": bool(base_turnover <= 1e-12 or state_turnover <= 1.20 * base_turnover),
        "full_active_excess_return_pass": bool(full.loc[STATE, "excess_ann_return"] > 0),
        "full_active_information_ratio_pass": bool(full.loc[STATE, "information_ratio"] > 0),
        "july_2026_loss_reduction": float(july_reduction) if july_reduction is not None else None,
        "july_2026_diagnostic_status": "EVALUATED" if july_reduction is not None else "NOT_APPLICABLE_BASELINE_POSITIVE",
        "july_2026_diagnostic_pass": bool(july_reduction >= 0.30) if july_reduction is not None else None,
        "adverse_months_materially_improved": int(adverse["materially_improved"].sum()),
        "adverse_month_materiality_threshold_pp": 0.10,
        "anti_single_episode_pass": bool(adverse["materially_improved"].sum() >= 2),
    }
    required = [key for key in criteria if key.endswith("_pass") and key != "july_2026_diagnostic_pass"]
    criteria["historical_acceptance_pass"] = bool(all(criteria[key] for key in required))
    criteria["forward_holdout_pass"] = None
    criteria["overall_live_approval"] = False

    industry_map = long_prices.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")["industry"].astype(str).to_dict()
    exposure = _portfolio_monitor(baseline_meta, industry_map, set(protocol["technology_growth_industries"]))
    crowding = _factor_crowding(panel, style_weights, fused)
    output.mkdir(parents=True, exist_ok=True)
    monitor.to_csv(output / "state_monitor.csv", index=False, date_format="%Y-%m-%d")
    exposure.merge(crowding, on="date", how="left").to_csv(output / "portfolio_common_exposure.csv", index=False, date_format="%Y-%m-%d")
    navs.to_csv(output / "nav_comparison.csv", date_format="%Y-%m-%d")
    performance.to_csv(output / "performance_by_period.csv", index=False)
    adverse.to_csv(output / "adverse_months.csv")
    (output / "acceptance_checks.json").write_text(json.dumps(criteria, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(baseline_meta["rebalance_log"]).to_json(output / "baseline_rebalance_log.json", orient="records", date_format="iso", indent=2)
    pd.DataFrame(state_meta["rebalance_log"]).to_json(output / "state_rebalance_log.json", orient="records", date_format="iso", indent=2)
    save_rebalance_logs(
        replace(settings, output_dir=output),
        {BASELINE: baseline_meta, STATE: state_meta, CONTROL: control_meta},
    )
    charts = _plot_all(output, navs, monitor, exposure, crowding)

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        git_head = "UNKNOWN"
    try:
        git_status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip().splitlines()
    except Exception:
        git_status = ["UNKNOWN"]
    code_paths = [
        ROOT / "config.py",
        ROOT / "analysis" / "ic.py",
        ROOT / "analysis" / "factor_redundancy.py",
        ROOT / "analysis" / "walk_forward_selection.py",
        ROOT / "backtest" / "backtest_single.py",
        ROOT / "scripts" / "build_walk_forward_factor_backtest.py",
        ROOT / "scripts" / "build_preregistered_state_gate_backtest.py",
    ]
    freeze = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": frozen_hash,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "historical_data_cutoff": "2026-09-04",
        "true_forward_start": protocol["data_plan"]["true_forward_holdout_start"],
        "engineering_minimum_trading_days": 20,
        "research_minimum_trading_days": 60,
        "parameter_changes_allowed": False,
        "real_orders_authorized": False,
        "git_head_at_evaluation": git_head,
        "git_worktree_clean": not bool(git_status),
        "git_status_short": git_status,
        "code_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in code_paths
        },
        "input_artifact_sha256": {
            "source_run_config": _sha256(source / "cache" / "run_config.json"),
            "walk_forward_run_config": _sha256(walk_forward / "run_config.json"),
            "walk_forward_fused_scores": _sha256(
                walk_forward / "walk_forward" / "fused_scores.csv"
            ),
        },
        "historical_acceptance_pass": criteria["historical_acceptance_pass"],
        "forward_status": "RESERVED_NOT_YET_OBSERVED",
    }
    (output / "forward_freeze_manifest.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    run_config = {
        "source_output_dir": str(source),
        "walk_forward_output_dir": str(walk_forward),
        "protocol_path": str(protocol_path),
        "protocol_sha256": frozen_hash,
        "information_cutoff": "strictly_before_rebalance_date",
        "primary_candidate": protocol["primary_candidate"],
        "state_inputs": protocol["state_inputs"],
        "hysteresis": protocol["hysteresis"],
        "state_actions": protocol["state_actions"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "performance": output / "performance_by_period.csv",
        "acceptance": output / "acceptance_checks.json",
        "state_monitor": output / "state_monitor.csv",
        "forward_manifest": output / "forward_freeze_manifest.json",
        **{f"chart_{i+1}": path for i, path in enumerate(charts)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--walk-forward-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for key, path in run(args.source_output_dir, args.walk_forward_output_dir, args.output_dir, args.protocol).items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
