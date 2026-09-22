#!/usr/bin/env python3
"""Compare base, hard-risk and CSI300 universes using next-session execution."""
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
from live.universe_history import load_membership_intervals, membership_mask
from scripts.build_dynamic_universe_comparison import (
    _benchmark_score,
    _execution_summary,
    _industry_for_panel,
    _performance_row,
    _shift_score_to_next_session,
    _weekly_decision_dates,
    build_fixed_family_score,
    build_weekly_raw_factor_panel,
)
from universe.dynamic import build_dynamic_universe_report, eligibility_from_report
from universe.risk import build_hard_risk_universe_report


BASE = "ALL_A_BASE_ADV50M_TOP50_TPLUS1_CLOSE"
RISK = "ALL_A_HARD_RISK_ADV50M_TOP50_TPLUS1_CLOSE"
CSI = "CSI300_PIT_TOP50_TPLUS1_CLOSE"
BASE_BENCH = "ALL_A_BASE_ADV50M_EQUAL_WEIGHT_TPLUS1_CLOSE"
RISK_BENCH = "ALL_A_HARD_RISK_ADV50M_EQUAL_WEIGHT_TPLUS1_CLOSE"
CSI_BENCH = "CSI300_PIT_EQUAL_WEIGHT_TPLUS1_CLOSE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected(score: pd.Series, strategy: str, top_k: int = 50) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, values in score.groupby(level="date", sort=True):
        ranked = values.droplevel("date").sort_values(ascending=False).head(top_k)
        for rank, (symbol, value) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "signal_date": pd.Timestamp(date),
                    "strategy": strategy,
                    "rank": rank,
                    "symbol": str(symbol),
                    "score": float(value),
                }
            )
    return pd.DataFrame(rows)


def _risk_summary(report: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    reasons = []
    for date, group in report.groupby("date", sort=True):
        base = group["base_eligible"].astype(bool)
        eligible = group["eligible"].astype(bool)
        row = {
            "date": pd.Timestamp(date),
            "base_eligible": int(base.sum()),
            "risk_eligible": int(eligible.sum()),
            "incremental_removed": int((base & ~eligible).sum()),
            "negative_net_assets_removed": int((base & group["negative_net_assets"].astype(bool)).sum()),
            "severe_audit_removed": int((base & group["severe_audit_opinion"].astype(bool)).sum()),
            "delisting_arrangement_removed": int((base & group["delisting_arrangement"].astype(bool)).sum()),
            "qualified_audit_warning": int((base & group["qualified_audit_warning"].astype(bool)).sum()),
            "balance_coverage": float(group.loc[base, "balance_available"].mean()) if base.any() else np.nan,
            "audit_coverage": float(group.loc[base, "audit_available"].mean()) if base.any() else np.nan,
        }
        rows.append(row)
        for reason, column in (
            ("negative_net_assets", "negative_net_assets"),
            ("severe_audit_opinion", "severe_audit_opinion"),
            ("delisting_arrangement", "delisting_arrangement"),
        ):
            affected = group[base & group[column].astype(bool)]
            for rec in affected.to_dict("records"):
                reasons.append(
                    {
                        "date": pd.Timestamp(date),
                        "symbol": rec["symbol"],
                        "name": rec.get("name", ""),
                        "reason": reason,
                        "net_assets": rec.get("net_assets"),
                        "balance_effective_date": rec.get("balance_effective_date"),
                        "audit_result": rec.get("audit_result", ""),
                        "audit_effective_date": rec.get("audit_effective_date"),
                        "delisting_name": rec.get("delisting_name", ""),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(reasons)


def _plot(
    output: Path,
    navs: pd.DataFrame,
    performance: pd.DataFrame,
    risk_summary: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    chart_dir = output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    selected = navs.loc[(navs.index >= start) & (navs.index <= end), [BASE, RISK, CSI]].copy()
    selected = selected / selected.iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    labels = {BASE: "Base dynamic", RISK: "Hard-risk dynamic", CSI: "CSI300"}
    for column in selected:
        axes[0].plot(selected.index, selected[column], label=labels[column], linewidth=2)
        drawdown = selected[column] / selected[column].cummax() - 1.0
        axes[1].plot(drawdown.index, drawdown * 100.0, label=labels[column], linewidth=1.7)
    axes[0].set_title("Next-session-close strategies: base vs hard-risk vs CSI300")
    axes[0].set_ylabel("NAV")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "hard_risk_nav_drawdown.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    ordered = performance.set_index("strategy").reindex([BASE, RISK, CSI])
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - 0.18, ordered["total_return"] * 100.0, 0.36, label="Strategy")
    ax.bar(x + 0.18, ordered["benchmark_total_return"] * 100.0, 0.36, label="Own benchmark")
    ax.set_xticks(x, ["Base dynamic", "Hard-risk dynamic", "CSI300"])
    ax.set_ylabel("Total return (%)")
    ax.set_title("Each strategy versus its own point-in-time universe")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = chart_dir / "hard_risk_strategy_vs_benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(risk_summary["date"], risk_summary["incremental_removed"], label="Total removed", linewidth=2.4)
    ax.plot(risk_summary["date"], risk_summary["negative_net_assets_removed"], label="Negative equity")
    ax.plot(risk_summary["date"], risk_summary["severe_audit_removed"], label="Severe audit")
    ax.plot(risk_summary["date"], risk_summary["delisting_arrangement_removed"], label="Delisting arrangement")
    ax.set_ylabel("Stocks")
    ax.set_title("Incremental hard-risk exclusions among otherwise tradable stocks")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = chart_dir / "hard_risk_exclusions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(
    data_dir: Path,
    membership_path: Path,
    output: Path,
    protocol_path: Path,
) -> dict[str, Path]:
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
    decision_dates = _weekly_decision_dates(calendar, start, end)

    base_report = build_dynamic_universe_report(
        prices,
        basic,
        names,
        calendar,
        decision_dates,
        min_listing_sessions=int(protocol["base_universe"]["minimum_listing_open_sessions"]),
        liquidity_window=int(protocol["base_universe"]["liquidity_lookback_open_sessions"]),
        min_valid_days=int(protocol["base_universe"]["minimum_valid_trading_days"]),
        min_adv_yuan=float(protocol["base_universe"]["minimum_adv20_cny"]),
        amount_col="amount_yuan",
    )
    risk_report = build_hard_risk_universe_report(base_report, balance, audit, names)
    base_eligible = eligibility_from_report(base_report)
    risk_eligible = eligibility_from_report(risk_report)
    risk_summary, risk_exclusions = _risk_summary(risk_report)

    raw_panel = build_weekly_raw_factor_panel(prices, finance, decision_dates)
    membership = load_membership_intervals(membership_path)
    csi_eligible = membership_mask(raw_panel.index, membership)
    industry = _industry_for_panel(raw_panel, basic)
    families = {
        str(key): [str(value) for value in values]
        for key, values in protocol["candidate_families"].items()
    }
    base_score, _ = build_fixed_family_score(raw_panel, base_eligible, industry, families)
    risk_score, _ = build_fixed_family_score(raw_panel, risk_eligible, industry, families)
    csi_score, _ = build_fixed_family_score(raw_panel, csi_eligible, industry, families)

    price_wide = prices.pivot(index="trade_date", columns="ts_code", values="adj_close").sort_index()
    price_wide = price_wide.loc[price_wide.index <= end]
    scores = {
        BASE: _shift_score_to_next_session(base_score, price_wide.index),
        RISK: _shift_score_to_next_session(risk_score, price_wide.index),
        CSI: _shift_score_to_next_session(csi_score, price_wide.index),
    }
    benchmark_eligibility = {
        BASE_BENCH: base_eligible,
        RISK_BENCH: risk_eligible,
        CSI_BENCH: csi_eligible,
    }
    benchmark_scores = {
        name: _shift_score_to_next_session(_benchmark_score(eligible, prices), price_wide.index)
        for name, eligible in benchmark_eligibility.items()
    }
    settings = replace(
        get_settings(),
        backtest_start=protocol["data"]["raw_start"],
        backtest_end=protocol["data"]["evaluation_end"],
        rebalance_freq="D",
        force_final_rebalance=False,
        top_k=int(protocol["factor_and_portfolio"]["top_k"]),
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
    for strategy, score in scores.items():
        navs[strategy], metas[strategy] = run_multi_backtest(
            fused=score,
            prices=price_wide,
            settings=settings,
            factor_name=strategy,
            top_k=int(protocol["factor_and_portfolio"]["top_k"]),
            long_prices=prices,
            empty_signal_policy="hold",
        )
    for benchmark, score in benchmark_scores.items():
        navs[benchmark], metas[benchmark] = run_multi_backtest(
            fused=score,
            prices=price_wide,
            settings=settings,
            factor_name=benchmark,
            top_k=len(price_wide.columns),
            long_prices=prices,
            empty_signal_policy="hold",
        )

    nav_frame = pd.concat(navs, axis=1).sort_index()
    nav_frame.to_csv(output / "nav_comparison.csv", index_label="date")
    benchmark_by_strategy = {BASE: BASE_BENCH, RISK: RISK_BENCH, CSI: CSI_BENCH}
    universe_by_strategy = {
        BASE: "ALL_A_BASE_ADV50M",
        RISK: "ALL_A_HARD_RISK_ADV50M",
        CSI: "CSI300_POINT_IN_TIME",
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
            for strategy in (BASE, RISK, CSI)
        ]
    )
    performance["execution_timing"] = "NEXT_SESSION_CLOSE"
    performance.to_csv(output / "performance_comparison.csv", index=False)

    periods = {
        "FULL_EVALUATION": (start, end),
        "EARLY_COMPARISON_PREEXISTING_SPLIT": (start, pd.Timestamp("2025-09-11")),
        "KNOWN_DEVELOPMENT_PREEXISTING_SPLIT": (pd.Timestamp("2025-09-12"), end),
    }
    period_rows = []
    for period, (period_start, period_end) in periods.items():
        for strategy in (BASE, RISK, CSI):
            row = _performance_row(
                strategy,
                universe_by_strategy[strategy],
                navs[strategy],
                navs[benchmark_by_strategy[strategy]],
                period_start,
                period_end,
            )
            row["period"] = period
            row["execution_timing"] = "NEXT_SESSION_CLOSE"
            period_rows.append(row)
    pd.DataFrame(period_rows).to_csv(output / "performance_by_period.csv", index=False)

    base_selected = _selected(base_score, BASE)
    risk_selected = _selected(risk_score, RISK)
    csi_selected = _selected(csi_score, CSI)
    selected = pd.concat([base_selected, risk_selected, csi_selected], ignore_index=True)
    selected.to_csv(output / "selected_top50_by_signal_date.csv", index=False)
    overlap_rows = []
    for date in decision_dates:
        base_set = set(base_selected.loc[base_selected["signal_date"].eq(date), "symbol"])
        risk_set = set(risk_selected.loc[risk_selected["signal_date"].eq(date), "symbol"])
        directly_blocked = set(
            risk_report.loc[
                risk_report["date"].eq(date) & risk_report["hard_risk_block"].astype(bool),
                "symbol",
            ].astype(str)
        )
        overlap_rows.append(
            {
                "signal_date": date,
                "overlap_count": len(base_set & risk_set),
                "base_vs_risk_top50_difference": len(base_set - risk_set),
                "directly_blocked_base_top50": len(base_set & directly_blocked),
                "replacement_count": len(risk_set - base_set),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(output / "top50_overlap.csv", index=False)
    risk_summary.to_csv(output / "risk_universe_summary_by_date.csv", index=False)
    risk_exclusions.to_csv(output / "risk_exclusions_by_date.csv", index=False)
    risk_report.to_csv(output / "hard_risk_universe_audit.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        [_execution_summary(name, meta, start) for name, meta in metas.items()]
    ).to_csv(output / "execution_summary.csv", index=False)

    full = performance.set_index("strategy")
    overlap = pd.read_csv(output / "top50_overlap.csv")
    checks = {
        "protocol_sha256": actual,
        "decision_dates": int(len(decision_dates)),
        "finance_point_in_time_pass": bool(
            (~risk_report["balance_effective_date"].notna() | risk_report["balance_effective_date"].le(risk_report["date"])).all()
            and (~risk_report["audit_effective_date"].notna() | risk_report["audit_effective_date"].le(risk_report["date"])).all()
        ),
        "average_base_universe_size": float(risk_summary["base_eligible"].mean()),
        "average_risk_universe_size": float(risk_summary["risk_eligible"].mean()),
        "average_incremental_removed": float(risk_summary["incremental_removed"].mean()),
        "max_incremental_removed": int(risk_summary["incremental_removed"].max()),
        "average_balance_coverage": float(risk_summary["balance_coverage"].mean()),
        "average_audit_coverage": float(risk_summary["audit_coverage"].mean()),
        "average_base_risk_top50_overlap": float(overlap["overlap_count"].mean()),
        "average_base_risk_top50_difference": float(overlap["base_vs_risk_top50_difference"].mean()),
        "total_directly_blocked_base_top50_events": int(overlap["directly_blocked_base_top50"].sum()),
        "total_base_return": float(full.loc[BASE, "total_return"]),
        "total_risk_return": float(full.loc[RISK, "total_return"]),
        "risk_minus_base_return": float(full.loc[RISK, "total_return"] - full.loc[BASE, "total_return"]),
        "base_max_drawdown": float(full.loc[BASE, "max_drawdown"]),
        "risk_max_drawdown": float(full.loc[RISK, "max_drawdown"]),
        "risk_minus_base_own_benchmark_gap": float(
            full.loc[RISK, "return_gap_vs_benchmark"] - full.loc[BASE, "return_gap_vs_benchmark"]
        ),
        "live_approval": False,
        "verdict": "RETROSPECTIVE_HARD_RISK_UNIVERSE_ABLATION_ONLY",
    }
    (output / "comparison_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    charts = _plot(output, nav_frame, performance, risk_summary, start, end)
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
    return {
        "performance": output / "performance_comparison.csv",
        "checks": output / "comparison_checks.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(args.data_dir, args.membership, args.output, args.protocol).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
