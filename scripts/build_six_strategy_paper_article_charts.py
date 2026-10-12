#!/usr/bin/env python3
"""Build article charts for the first six-strategy forward paper session."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "BASE_V2_MULTI_FACTOR_TOP50": "Base multi-factor",
    "GROWTH_FOCUSED_TOP50": "Growth",
    "LARGE_CAP_MULTI_FACTOR_TOP50": "Large-cap",
    "MID_SMALL_CAP_MULTI_FACTOR_TOP50": "Mid-small",
    "TECH_GROWTH_THEME_MULTI_FACTOR_TOP50": "Tech theme",
    "VALUE_FOCUSED_TOP50": "Value",
}


def build(summary_path: Path, output_dir: Path, *, initial_cash: float) -> list[Path]:
    summary = pd.read_csv(summary_path)
    summary["label"] = summary["strategy"].map(LABELS).fillna(summary["strategy"])
    summary["cash_ratio"] = summary["cash"] / summary["total_asset"] * 100.0
    summary["market_ratio"] = summary["market_value"] / float(initial_cash) * 100.0
    summary["cash_initial_ratio"] = summary["cash"] / float(initial_cash) * 100.0
    summary["commission"] = float(initial_cash) - summary["total_asset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    x = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(12.5, 6.5))
    bars = axis.bar(x, summary["positions"], color="#4C78A8", width=0.62)
    axis.axhline(50, color="#4C78A8", linestyle="--", linewidth=1.4, alpha=0.7)
    axis.set_ylabel("Executed positions")
    axis.set_ylim(0, 56)
    axis.set_xticks(x, summary["label"], rotation=15, ha="right")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, summary["positions"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.8,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    right = axis.twinx()
    right.plot(
        x,
        summary["cash_ratio"],
        color="#F58518",
        marker="o",
        linewidth=2.2,
        label="Cash ratio",
    )
    right.set_ylabel("Cash ratio (%)")
    right.set_ylim(0, max(35.0, float(summary["cash_ratio"].max()) + 5.0))
    for xpos, value in zip(x, summary["cash_ratio"]):
        right.annotate(
            "%.1f%%" % value,
            (xpos, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#B75D00",
            fontsize=9,
        )
    axis.set_title("First forward paper execution: positions and remaining cash")
    fig.tight_layout()
    path = output_dir / "initial_positions_and_cash.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(12.5, 6.5))
    market = axis.bar(
        x,
        summary["market_ratio"],
        color="#54A24B",
        width=0.65,
        label="Market value",
    )
    axis.bar(
        x,
        summary["cash_initial_ratio"],
        bottom=summary["market_ratio"],
        color="#E7BA52",
        width=0.65,
        label="Cash",
    )
    axis.set_ylim(0, 106)
    axis.set_ylabel("Share of initial capital (%)")
    axis.set_xticks(x, summary["label"], rotation=15, ha="right")
    axis.set_title("One-million-CNY accounts after the initial paper fills")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower right")
    for xpos, rec in summary.reset_index(drop=True).iterrows():
        axis.text(
            xpos,
            101.0,
            "fee %.0f" % float(rec["commission"]),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#6B6B6B",
        )
    for bar, value in zip(market, summary["market_ratio"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(2.0, float(value) / 2.0),
            "%.1f%%" % value,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    path = output_dir / "initial_account_composition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    args = parser.parse_args()
    for path in build(args.summary, args.output, initial_cash=args.initial_cash):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
