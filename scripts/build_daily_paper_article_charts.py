#!/usr/bin/env python3
"""Build focused charts for a daily paper-trading article from audited outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")


def build_charts(output_dir: Path, strategy: str, trade_date: str) -> dict[str, Path]:
    order_path = output_dir / "order_plans" / f"{strategy}.csv"
    trade_path = output_dir / "paper_trades" / f"{strategy}.csv"
    snapshot_path = output_dir / "paper_account" / strategy / "snapshots.csv"
    risk_path = (
        output_dir
        / "risk_control_reports"
        / strategy
        / f"daily_risk_control_report_{trade_date.replace('-', '')}.csv"
    )

    orders = pd.read_csv(order_path)
    trades = pd.read_csv(trade_path)
    snapshots = pd.read_csv(snapshot_path)
    # “NA” is a business status here, not a missing-value token.
    risk = pd.read_csv(risk_path, keep_default_na=False)
    _require_columns(orders, {"date", "symbol", "target_weight"}, order_path)
    _require_columns(
        trades,
        {"date", "symbol", "gross_amount", "commission", "fill_status"},
        trade_path,
    )
    _require_columns(snapshots, {"date", "cash", "market_value", "total_asset"}, snapshot_path)
    _require_columns(risk, {"trade_date", "module", "status", "summary"}, risk_path)

    orders = orders[orders["date"].astype(str) == trade_date].copy()
    trades = trades[
        (trades["date"].astype(str) == trade_date) & (trades["fill_status"] == "FILLED")
    ].copy()
    snapshot = snapshots[snapshots["date"].astype(str) == trade_date].tail(1)
    risk = risk[risk["trade_date"].astype(str) == trade_date].copy()
    if orders.empty or trades.empty or snapshot.empty or risk.empty:
        raise ValueError(f"incomplete daily-paper outputs for {strategy} on {trade_date}")

    total_asset = float(snapshot.iloc[0]["total_asset"])
    cash = float(snapshot.iloc[0]["cash"])
    target = orders.groupby("symbol", as_index=True)["target_weight"].sum().sort_index()
    actual = trades.groupby("symbol", as_index=True)["gross_amount"].sum().div(total_asset)
    symbols = target.index.union(actual.index).tolist()
    target_values = target.reindex(symbols, fill_value=0.0).to_numpy() * 100.0
    actual_values = actual.reindex(symbols, fill_value=0.0).to_numpy() * 100.0
    target_cash = max(1.0 - float(target.sum()), 0.0) * 100.0
    actual_cash = cash / total_asset * 100.0

    chart_dir = output_dir / "paper_article_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    allocation_path = chart_dir / "target_vs_paper_fills.png"
    labels = symbols + ["Cash"]
    target_plot = np.append(target_values, target_cash)
    actual_plot = np.append(actual_values, actual_cash)
    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.bar(
        positions - width / 2,
        target_plot,
        width,
        label="Target",
        color="#4c78a8",
    )
    ax.bar(
        positions + width / 2,
        actual_plot,
        width,
        label="Paper fill",
        color="#f58518",
    )
    for x, value in zip(positions - width / 2, target_plot):
        ax.text(x, value + 0.55, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    for x, value in zip(positions + width / 2, actual_plot):
        ax.text(x, value + 0.55, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, labels, rotation=18, ha="right")
    ax.set_ylabel("Portfolio weight")
    ax.set_title(f"Target vs simulated fills — {trade_date}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(actual_cash, target_cash) + 8.0)
    fig.tight_layout()
    fig.savefig(allocation_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    preferred_modules = [
        "订单预检查",
        "容量与冲击成本",
        "回撤止损与降仓",
        "组合风险限额",
        "统一风险门禁",
        "组合压力测试",
    ]
    labels_en = {
        "订单预检查": "Order pre-check",
        "容量与冲击成本": "Capacity & impact",
        "回撤止损与降仓": "Drawdown control",
        "组合风险限额": "Portfolio limits",
        "统一风险门禁": "Unified risk gate",
        "组合压力测试": "Stress test",
    }
    status_colors = {
        "PASS": "#54a24b",
        "WATCH": "#eeca3b",
        "NA": "#9d9da1",
        "BLOCK": "#e45756",
        "FAILED": "#e45756",
    }
    rows = risk.set_index("module").reindex(preferred_modules).dropna(subset=["status"])
    risk_path_out = chart_dir / "daily_risk_dashboard.png"
    fig, (account_ax, risk_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 6.4),
        gridspec_kw={"width_ratios": [0.9, 1.5]},
    )
    stock_value = float(snapshot.iloc[0]["market_value"])
    account_ax.pie(
        [stock_value, cash],
        labels=["Stocks", "Cash"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4c78a8", "#bab0ac"],
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    fees = float(trades["commission"].sum())
    account_ax.set_title(
        "Paper account\n"
        f"Asset CNY {total_asset:,.2f}\n"
        f"Fees CNY {fees:,.2f}",
        pad=16,
    )

    y = np.arange(len(rows))
    colors = [status_colors.get(str(status), "#9d9da1") for status in rows["status"]]
    risk_ax.barh(y, np.ones(len(rows)), color=colors, height=0.62)
    risk_ax.set_yticks(y, [labels_en.get(name, name) for name in rows.index])
    risk_ax.invert_yaxis()
    risk_ax.set_xlim(0.0, 1.0)
    risk_ax.set_xticks([])
    for pos, (_, row) in enumerate(rows.iterrows()):
        risk_ax.text(0.5, pos, str(row["status"]), ha="center", va="center", weight="bold")
    risk_ax.set_title("Daily control status")
    for spine in risk_ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(f"Paper-trading control panel — {trade_date}", fontsize=14)
    fig.tight_layout()
    fig.savefig(risk_path_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {"allocation": allocation_path, "risk_dashboard": risk_path_out}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成每日纸面交易文章图")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", default="FUSED_ROLLING_SCORE_WEIGHTED")
    parser.add_argument("--trade-date", required=True)
    args = parser.parse_args()
    for name, path in build_charts(args.output_dir, args.strategy, args.trade_date).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
