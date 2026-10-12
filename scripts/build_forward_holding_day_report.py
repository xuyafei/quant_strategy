#!/usr/bin/env python3
"""Read-only holding-day article facts/figures; never write sealed account state."""
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

from live.forward_ledger import LABELS, sha, verify


def check_close(actual, expected, label):
    if not np.isfinite([actual, expected]).all() or abs(actual - expected) > 1e-6:
        raise ValueError("holding-day reconciliation failed: " + label)


def build(root: Path, date: str, out: Path):
    root, out = root.resolve(), out.resolve()
    # A reading report may live under reports, never in source ledger directories.
    if out == root or (out.is_relative_to(root) and not out.is_relative_to(root / "reports")):
        raise ValueError("reports must not overwrite sealed source directories")
    if out.exists():
        raise FileExistsError("choose a new report directory; do not overwrite")
    day = root / "days" / date.replace("-", "")
    dm = verify(day)
    initial = verify(root / "initial")
    earlier = sorted(p for p in (root / "days").glob("????????") if p.is_dir() and p.name < day.name)
    if not earlier:
        raise ValueError("use the first-day report for cash-only construction")
    prior = earlier[-1]
    pm = verify(prior)
    if dm["date"] != date or dm["previous_manifest_sha256"] != sha(prior / "manifest.json"):
        raise ValueError("broken date/account ancestry")
    config = json.loads((root / "initial/session.json").read_text())
    before = pd.read_csv(prior / "summary.csv").set_index("strategy")
    after = pd.read_csv(day / "summary.csv").set_index("strategy")
    strategies = config["strategies"]
    if (before.index.has_duplicates or after.index.has_duplicates
            or set(before.index) != set(strategies) or set(after.index) != set(strategies)):
        raise ValueError("strategy coverage mismatch")
    if not after.action.eq("HOLD_MARK").all() or not after.filled_orders.eq(0).all():
        raise ValueError("holding-day report does not support rebalances")
    if not after.date.eq(date).all() or not before.date.eq(pm["date"]).all():
        raise ValueError("summary dates mismatch")
    market = pd.read_csv(day / "market.csv").set_index("symbol")
    old_market = pd.read_csv(prior / "market.csv").set_index("symbol")
    source = verify(day / "market_source")
    dividends = pd.read_csv(day / "dividends.csv")
    if dividends.duplicated(["strategy", "event_id"]).any():
        raise ValueError("duplicate dividend history")
    if dm.get("corporate_action_manifest_sha256"):
        verify(day / "corporate_action_source")
        if sha(day / "corporate_action_source/manifest.json") != dm["corporate_action_manifest_sha256"]:
            raise ValueError("corporate action source mismatch")
    paid_today = dividends[dividends.pay_date.eq(date)]
    rows, detail = [], []
    for strategy in strategies:
        previous, current = before.loc[strategy], after.loc[strategy]
        old = pd.read_csv(prior / "paper_account" / strategy / "positions.csv").set_index("symbol")
        new = pd.read_csv(day / "paper_account" / strategy / "positions.csv").set_index("symbol")
        pd.testing.assert_series_equal(old.shares.sort_index(), new.shares.sort_index())
        p0, p1 = old_market.close.reindex(old.index), market.close.reindex(old.index)
        if p0.isna().any() or p1.isna().any():
            raise ValueError("missing held-stock valuation")
        payments = paid_today[paid_today.strategy.eq(strategy)]
        gross = float(payments.cash_gross.sum())
        price_pnl = float((old.shares * (p1 - p0)).sum())
        daily_pnl = float(current.total_asset - previous.total_asset)
        check_close(current.commission, 0, "unexpected holding-day fee")
        check_close(current.dividend_cash_gross, gross, "dividend history")
        check_close(current.cash, previous.cash + gross, "cash")
        check_close(current.market_value, float((new.shares * p1).sum()), "market value")
        check_close(current.total_asset, current.cash + current.market_value, "assets")
        check_close(daily_pnl, price_pnl + gross, "price change plus dividend")
        check_close(current.net_value, current.total_asset / config["initial_cash"], "net value")
        rows.append({"strategy": strategy, "label": LABELS.get(strategy, strategy),
                     "daily_pnl": daily_pnl, "price_pnl": price_pnl, "dividend_cash_gross": gross,
                     "daily_return_pct": daily_pnl / previous.total_asset * 100,
                     "cumulative_return_pct": (current.net_value - 1) * 100,
                     "total_asset": current.total_asset, "cash": current.cash,
                     "positions": int(current.n_positions),
                     "start_stock_exposure_pct": previous.market_value / previous.total_asset * 100})
        for rec in payments.to_dict("records"):
            symbol = rec["symbol"]
            quantity = float(old.loc[symbol, "shares"])
            check_close(quantity, rec["eligible_shares"], "dividend entitlement")
            change = quantity * (p1.loc[symbol] - p0.loc[symbol])
            detail.append({**rec, "label": LABELS.get(strategy,strategy), "previous_close": p0.loc[symbol],
                           "close": p1.loc[symbol], "price_pnl": change,
                           "holding_total_pnl": change + rec["cash_gross"]})
    comparison = pd.DataFrame(rows)
    details = pd.DataFrame(detail)
    facts = {"date": date, "previous_date": pm["date"], "initial_date": initial["date"],
             "completed_sessions": len(earlier)+1, "market_retrieved_at_utc": source["retrieved_at_utc"],
             "day_completed_at_utc": dm["completed_at_utc"], "required_symbols": source["required_symbols"],
             "sum_daily_pnl": float(comparison.daily_pnl.sum()),
             "sum_dividends_gross": float(comparison.dividend_cash_gross.sum()),
             "sum_assets": float(comparison.total_asset.sum()),
             "sum_initial_cash": float(config["initial_cash"] * len(strategies)),
             "sum_cumulative_pnl": float(comparison.total_asset.sum() - config["initial_cash"] * len(strategies)),
             "filled_orders": 0, "daily_commission": 0, "simulation_only": True,
             "dividend_tax_policy": dm.get("dividend_tax_policy"),
             "day_manifest_sha256": sha(day/"manifest.json"), "prior_manifest_sha256": sha(prior/"manifest.json"),
             "revision_sha256": dm.get("revision_sha256"), "report_code_sha256": sha(Path(__file__))}
    out.mkdir(parents=True)
    comparison.to_csv(out/"holding_day_comparison.csv", index=False)
    details.to_csv(out/"dividend_attribution.csv", index=False)
    (out/"facts.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding="utf-8")
    plt.rcParams.update({"font.family":"Arial Unicode MS", "axes.unicode_minus":False})
    display_date = f"{pd.Timestamp(date).month}月{pd.Timestamp(date).day}日"
    fig, axes = plt.subplots(1,2,figsize=(13,6))
    for ax, col, title, color in [(axes[0],"daily_return_pct",f"{display_date}：相对上一日资产的收益","#356b89"),
                                  (axes[1],"cumulative_return_pct","建仓以来：相对初始100万元的收益","#8e7855")]:
        bars = ax.barh(comparison.label,comparison[col],color=color)
        ax.invert_yaxis()
        ax.set_xlabel("账户收益率（%）")
        ax.set_ylabel("独立虚拟账户")
        ax.set_title(title,fontsize=12)
        ax.axvline(0,color="#777777",linewidth=.7)
        ax.margins(x=.23)
        for bar,value in zip(bars,comparison[col]):
            ax.text(value,bar.get_y()+bar.get_height()/2,f" {value:+.3f}%",va="center",fontsize=10)
        ax.grid(axis="x",alpha=.15)
    fig.suptitle("六策略纸面交易：单日收益不等于累计收益",fontsize=15)
    fig.text(.5,.025,f"来源：{pm['date']}至{date}封存账本｜含税前分红，未模拟红利税｜累计收益已计首日简化佣金；非实盘业绩",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94))
    fig.savefig(out/"daily_vs_cumulative.png",dpi=170)
    plt.close(fig)
    if not details.empty:
        fig, axes = plt.subplots(1,len(details),figsize=(6*len(details),5.8),squeeze=False)
        values = details[["price_pnl", "cash_gross", "holding_total_pnl"]].to_numpy()
        lower, upper = min(0,values.min()), max(0,values.max())
        margin = max(upper-lower,1)*.18
        for ax, rec in zip(axes[0], details.to_dict("records")):
            vals=[rec["price_pnl"],rec["cash_gross"],rec["holding_total_pnl"]]
            bars=ax.bar(["价格变动损益","税前现金分红","含分红持仓损益"],vals,color=["#a86e63","#356b89","#6c7378"])
            ax.axhline(0,color="#777777",linewidth=.8)
            ax.set_ylim(lower-margin,upper+margin)
            ax.set_title(f'{rec["label"]}：{rec["symbol"]}，{rec["eligible_shares"]:,.0f}股')
            ax.set_ylabel("该笔持仓当日损益（元）")
            ax.set_xlabel("价格损益 + 分红 = 持仓总损益")
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,v,f"{v:+,.2f}",ha="center",va="bottom" if v>=0 else "top",fontsize=12)
            ax.grid(axis="y",alpha=.15)
        fig.suptitle("分红不能漏记，也不等于这只股票今天赚钱了",fontsize=15)
        fig.text(.5,.025,f"来源：{date}封存持仓、行情与分红事件｜金额为对应股票持仓，不是整个策略账户｜税前研究口径",ha="center",fontsize=10)
        fig.tight_layout(rect=(0,.07,1,.94))
        fig.savefig(out/"dividend_price_pnl.png",dpi=170)
        plt.close(fig)
    return facts


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",type=Path,required=True)
    p.add_argument("--date",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.session,a.date,a.output),ensure_ascii=False,indent=2))
