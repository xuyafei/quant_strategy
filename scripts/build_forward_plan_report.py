#!/usr/bin/env python3
"""Render frozen-plan reference estimates, never future fills or future returns."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from live.forward_ledger import LABELS, verify

ENGLISH = {"BASE_V2_MULTI_FACTOR_TOP50": "Base", "LARGE_CAP_MULTI_FACTOR_TOP50": "Large-cap",
           "MID_SMALL_CAP_MULTI_FACTOR_TOP50": "Mid-small", "VALUE_FOCUSED_TOP50": "Value",
           "GROWTH_FOCUSED_TOP50": "Growth", "TECH_GROWTH_THEME_MULTI_FACTOR_TOP50": "Tech theme"}


def table(frame: pd.DataFrame) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def build(plan: Path, output: Path) -> None:
    meta = verify(plan)
    targets = pd.read_csv(plan / "targets_combined.csv")
    summary = pd.read_csv(plan / "summary.csv")
    output.mkdir(parents=True, exist_ok=True)
    labels = summary.strategy.map(ENGLISH)
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    axes[0].bar(x, summary.estimated_positions, color="#34678c", label="Reference-price affordable positions")
    axes[0].axhline(50, color="#888888", ls="--", label="50 target stocks")
    axes[0].set_ylabel("Number of stocks")
    axes[0].set_ylim(0, 57)
    axes[0].legend(loc="lower left", fontsize=8)
    for i, n in enumerate(summary.estimated_positions):
        axes[0].text(i, n + .6, str(n), ha="center", fontsize=10)
    ratios = summary.estimated_cash / (summary.estimated_cash + summary.estimated_stock_value) * 100
    axes[1].bar(x, ratios, color="#ac8054")
    axes[1].set_ylabel("Reference cash / estimated assets (%)")
    for i, n in enumerate(ratios):
        axes[1].text(i, n + .4, f"{n:.1f}%", ha="center", fontsize=10)
    axes[1].set_ylim(0, max(10, ratios.max() + 5))
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_xlabel("Strategy (independent virtual account)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen targets: quantity and cash estimates, NOT executed positions")
    fig.text(.5, .01, f"Source: sealed plan | Reference close {meta['signal_date']} | Intended paper execution {meta['execution_date']} | Commission-only model", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / "reference_positions_cash.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    main = summary.targets - summary.star_targets - summary.chinext_targets
    ax.bar(x, main, label="Main board", color="#34678c")
    ax.bar(x, summary.chinext_targets, bottom=main, label="ChiNext", color="#91aebe")
    ax.bar(x, summary.star_targets, bottom=main + summary.chinext_targets, label="STAR", color="#ac8054")
    for i, rec in summary.iterrows():
        ax.text(i, 51, f"STAR {rec.star_targets} / ChiNext {rec.chinext_targets}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Strategy (independent virtual account)")
    ax.set_ylabel("Number of target stocks (not actual fills)")
    ax.set_ylim(0, 60)
    ax.set_title("Research targets by board: not the user's brokerage permissions")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, -.13))
    fig.text(.5, .01, f"Source: sealed plan | Signal {meta['signal_date']} | 50 target stocks per strategy", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "target_board_coverage.png", dpi=170)
    plt.close(fig)

    lines = ["# 六策略冻结目标明细：不是实盘荐股，也不是已成交记录", "",
             f"信号数据截止：{meta['signal_date']}；计划执行：{meta['execution_date']}。", "",
             f"实际冻结时间（UTC）：{meta['created_at_utc']}。", "",
             "每个账户100万元虚拟资金，Top50等权。参考价格是信号日未复权收盘价；股数为参考估算，执行日会按冻结的目标权重和规则重新计算。未买入的额度保留现金，不临时补选。", "",
             "参考数量检查不包含未来停牌、涨跌停或新闻风险；每个策略清单全部50只并非50条真实买入建议。", ""]
    for strategy, frame in targets.groupby("strategy", sort=True):
        lines += ["## " + LABELS[strategy], ""]
        display = frame[["rank", "symbol", "name", "industry", "board", "weight", "reference_price", "estimated_target_shares", "reference_quantity_check"]].copy()
        display["weight"] = display.weight.map(lambda v: f"{v:.0%}")
        display["reference_price"] = display.reference_price.map(lambda v: f"{v:.2f}")
        display.columns = ["排名", "代码", "名称", "行业", "板块", "目标权重", "参考价/元", "目标股数估算", "参考数量检查"]
        lines += [table(display), ""]
    (output / "六策略冻结目标明细.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.plan, a.output)
