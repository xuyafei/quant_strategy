#!/usr/bin/env python3
"""Read-only first-day analysis. Write reports outside the sealed paper ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from live.forward_ledger import LABELS, verify, sha
from live.order_builder import build_order_plan


def cash_components(orders: pd.DataFrame, checks: pd.DataFrame, commission: float, cash: float) -> dict:
    """Partition initial target budgets; do not attribute idle cash to alpha."""
    zero = orders.target_shares.eq(0)
    unavailable = float(orders.loc[zero, "target_value"].sum())
    rounding = float((orders.loc[~zero, "target_value"] - orders.loc[~zero, "estimated_amount"]).sum())
    blocked = float(checks.loc[checks.check_status.eq("BLOCK"), "estimated_amount"].sum())
    difference = unavailable + rounding + blocked - commission - cash
    if abs(difference) > 1e-6:
        raise ValueError("cash decomposition does not reconcile")
    return {"zero_quantity_targets": int(zero.sum()), "minimum_quantity_budget": unavailable,
            "rounding_residual": rounding, "blocked_order_notional": blocked,
            "commission": commission, "actual_cash": cash, "reconciliation_error": difference}


def build(root: Path, date: str, out: Path) -> dict:
    day = root / "days" / date.replace("-", "")
    plan = root / "plans" / date.replace("-", "")
    dm, pm = verify(day), verify(plan)
    verify(root / "initial")
    if dm["previous_manifest_sha256"] != sha(root / "initial" / "manifest.json"):
        raise ValueError("this report is for initial cash-only construction, not later rebalances")
    cfg = json.loads((root / "initial/session.json").read_text())
    initial = float(cfg["initial_cash"])
    targets = pd.read_csv(plan / "targets_combined.csv")
    summary = pd.read_csv(day / "summary.csv")
    reference = pd.read_csv(plan / "summary.csv")
    market = pd.read_csv(day / "market.csv").set_index("symbol")
    rows, details, holdings = [], [], {}
    for row in summary.to_dict("records"):
        strategy = row["strategy"]
        target = targets[targets.strategy.eq(strategy)]
        orders = build_order_plan(target.set_index("symbol").weight, {}, market.close,
                                  cash=initial, trade_date=date, include_holds=True, board_rules=True)
        checks = pd.read_csv(day / "order_checks" / (strategy + ".csv"))
        fills = pd.read_csv(day / "paper_trades" / (strategy + ".csv"))
        positions = pd.read_csv(day / "paper_account" / strategy / "positions.csv")
        holdings[strategy] = set(positions.symbol)
        ref = reference.set_index("strategy").loc[strategy]
        comp = cash_components(orders, checks, row["commission"], row["cash"])
        rows.append({**row, **comp, "label": LABELS[strategy], "target_count": len(target),
                     "reference_positions": int(ref.estimated_positions), "reference_cash": float(ref.estimated_cash),
                     "cash_change_from_reference": row["cash"] - ref.estimated_cash,
                     "cash_ratio_pct": row["cash"] / row["total_asset"] * 100,
                     "day_one_return_pct": (row["net_value"] - 1) * 100})
        detail = target.merge(orders[["symbol", "target_shares", "trade_reason"]], on="symbol", validate="one_to_one")
        detail = detail.merge(checks[["symbol", "check_status", "check_reason"]], on="symbol", how="left", validate="one_to_one")
        detail = detail.merge(fills[["symbol", "qty", "price", "fill_status", "commission"]], on="symbol", how="left", validate="one_to_one")
        detail["outcome"] = np.where(detail.target_shares.eq(0), "NO_ORDER_BELOW_MINIMUM_QUANTITY", detail.fill_status)
        details.append(detail)
    comparison = pd.DataFrame(rows)
    detailed = pd.concat(details, ignore_index=True)
    filled_count = int(summary.filled_orders.sum())
    facts = {"date": date, "signal_date": pm["signal_date"], "plan_created_at_utc": pm["created_at_utc"],
             "day_completed_at_utc": dm["completed_at_utc"], "initial_total_cash": initial * len(rows),
             "target_positions": len(targets), "unique_target_stocks": targets.symbol.nunique(),
             "filled_orders": filled_count, "unique_held_stocks": len(set.union(*holdings.values())),
             "no_order_positions": int(comparison.zero_quantity_targets.sum()),
             "blocked_orders": int(detailed.check_status.eq("BLOCK").sum()),
             "cash": float(summary.cash.sum()), "stock_value": float(summary.market_value.sum()),
             "total_asset": float(summary.total_asset.sum()), "commission": float(summary.commission.sum()),
             "simulation_only": True, "fee_model": cfg["fees"],
             "plan_manifest_sha256": sha(plan / "manifest.json"), "day_manifest_sha256": sha(day / "manifest.json")}
    if abs(facts["initial_total_cash"] - facts["commission"] - facts["total_asset"]) > 1e-6:
        raise ValueError("first-day loss is not fully explained by commission")
    overlap = pd.DataFrame({a: {b: len(holdings[a] & holdings[b]) for b in holdings} for a in holdings})
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "first_day_comparison.csv", index=False)
    detailed.to_csv(out / "all_300_target_outcomes.csv", index=False)
    overlap.to_csv(out / "holding_overlap_counts.csv")
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False})
    x = np.arange(len(comparison))
    labels = comparison.label
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].bar(x - .19, comparison.reference_positions, .38, color="#a7bcc9", label="9月24日参考价预估")
    bars = axes[0].bar(x + .19, comparison.n_positions, .38, color="#34678c", label="9月28日模拟成交后")
    axes[0].axhline(50, ls="--", color="#888888", label="每策略50个目标")
    axes[0].set_ylabel("持仓数量（只）")
    axes[0].set_ylim(0, 58)
    axes[0].legend(fontsize=9, loc="lower left")
    for bar, value in zip(bars, comparison.n_positions):
        axes[0].text(bar.get_x() + bar.get_width()/2, value + .7, str(int(value)), ha="center", fontsize=10)
    cash = comparison.actual_cash / 10000
    bars = axes[1].bar(x, cash, color="#ac8054")
    axes[1].set_ylabel("剩余虚拟现金（万元，已扣模拟佣金）")
    for bar, value in zip(bars, cash):
        axes[1].text(bar.get_x()+bar.get_width()/2, value + .5, f"{value:.2f}", ha="center", fontsize=10)
    axes[1].set_ylim(0, cash.max()+5)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_xlabel("六个独立虚拟账户")
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("首日纸面建仓：持仓数量与剩余现金")
    fig.text(.5, .01, "来源：已封存计划和2026-09-28账本｜每账户100万元｜收盘价模拟，非真实券商成交", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(out / "first_day_positions_cash.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(comparison))
    components = [("minimum_quantity_budget", "最低数量买不起的目标预算", "#34678c"),
                  ("rounding_residual", "股数取整剩余预算", "#a7bcc9"),
                  ("blocked_order_notional", "交易检查阻断的订单金额", "#ac8054")]
    for col, label, color in components:
        values = comparison[col].to_numpy()/10000
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        for i, value in enumerate(values):
            if value > 1:
                ax.text(i, bottom[i]+value/2, f"{value:.2f}", ha="center", va="center", color="white" if col=="minimum_quantity_budget" else "#202020", fontsize=11)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_xlabel("六个独立虚拟账户")
    ax.set_ylabel("未投入股票的预算（万元，扣佣金前）")
    ax.set_ylim(0, max(bottom)+5)
    ax.set_title("现金从哪里来：先拆预算，再减去佣金")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, fontsize=10)
    ax.grid(axis="y", alpha=.18)
    fig.text(.5, .01, "来源：2026-09-28目标、订单检查及成交日志｜柱子合计 − 当日佣金 = 实际剩余虚拟现金", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(out / "cash_decomposition.png", dpi=170)
    plt.close(fig)
    return facts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.session, a.date, a.output), ensure_ascii=False, indent=2))
