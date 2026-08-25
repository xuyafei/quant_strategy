#!/usr/bin/env python3
"""Compare raw close and adjusted close in a lightweight momentum backtest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.backtest_utils import long_to_wide
from config import get_settings
from live.data_feed import load_prices_from_csv
from storage.price_adjustment import add_adjusted_close
from storage.warehouse import load_prices_daily


def _max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return float("nan")
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _annual_return(nav: pd.Series, trading_days_per_year: int = 252) -> float:
    nav = nav.dropna()
    if len(nav) < 2:
        return float("nan")
    years = len(nav) / float(trading_days_per_year)
    if years <= 0:
        return float("nan")
    return float(nav.iloc[-1] ** (1.0 / years) - 1.0)


def _stats(nav: pd.Series, name: str) -> dict[str, float | str]:
    returns = nav.pct_change().dropna()
    ann_vol = float(returns.std(ddof=0) * (252 ** 0.5)) if not returns.empty else float("nan")
    ann_return = _annual_return(nav)
    sharpe = ann_return / ann_vol if ann_vol and pd.notna(ann_vol) else float("nan")
    return {
        "strategy": name,
        "start": nav.index.min().strftime("%Y-%m-%d") if not nav.empty else "",
        "end": nav.index.max().strftime("%Y-%m-%d") if not nav.empty else "",
        "trading_days": float(len(nav)),
        "final_nav": float(nav.iloc[-1]) if not nav.empty else float("nan"),
        "total_return": float(nav.iloc[-1] - 1.0) if not nav.empty else float("nan"),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(nav),
    }


def run_monthly_momentum_topk(
    prices_wide: pd.DataFrame,
    *,
    top_k: int = 5,
    lookback: int = 20,
    rebalance_freq: str = "ME",
) -> tuple[pd.Series, pd.DataFrame]:
    """A small close-to-close monthly Top-K momentum backtest."""
    prices = prices_wide.sort_index().dropna(how="all")
    returns = prices.pct_change().fillna(0.0)
    momentum = prices / prices.shift(int(lookback)) - 1.0
    rebalance_dates = list(
        prices.groupby(pd.Grouper(freq=rebalance_freq)).tail(1).index.unique()
    )
    if prices.index.max() not in rebalance_dates:
        rebalance_dates.append(prices.index.max())
    rebalance_dates = sorted(pd.Index(rebalance_dates).unique())

    target_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    logs: list[dict[str, object]] = []
    for dt in rebalance_dates:
        scores = momentum.loc[dt].dropna().sort_values(ascending=False)
        picks = list(scores.head(int(top_k)).index)
        if picks:
            target_weights.loc[dt, picks] = 1.0 / len(picks)
        logs.append(
            {
                "trade_date": dt,
                "selected_count": len(picks),
                "selected_symbols": ",".join(picks),
            }
        )
    weights = target_weights.reindex(prices.index).ffill().fillna(0.0)
    portfolio_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    nav = (1.0 + portfolio_returns).cumprod()
    return nav, pd.DataFrame(logs)


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.database:
        return load_prices_daily(args.database, start=args.start or None, end=args.end or None)
    prices_csv = Path(args.prices_csv).expanduser()
    prices = load_prices_from_csv(prices_csv)
    if args.start:
        prices = prices[pd.to_datetime(prices["trade_date"]) >= pd.to_datetime(args.start)]
    if args.end:
        prices = prices[pd.to_datetime(prices["trade_date"]) <= pd.to_datetime(args.end)]
    return prices


def build_report(args: argparse.Namespace) -> dict[str, Path]:
    prices = _load_prices(args)
    if prices.empty:
        raise ValueError("没有可用于对比的行情数据")
    if "adj_close" not in prices.columns or prices["adj_close"].isna().all():
        prices = add_adjusted_close(prices, mode=args.adjustment_mode)

    raw_wide = long_to_wide(prices, "close")
    adj_wide = long_to_wide(prices, "adj_close")
    common_index = raw_wide.index.intersection(adj_wide.index)
    common_cols = raw_wide.columns.intersection(adj_wide.columns)
    raw_wide = raw_wide.loc[common_index, common_cols]
    adj_wide = adj_wide.loc[common_index, common_cols]

    raw_nav, raw_log = run_monthly_momentum_topk(
        raw_wide,
        top_k=args.top_k,
        lookback=args.lookback,
        rebalance_freq=args.rebalance_freq,
    )
    adj_nav, adj_log = run_monthly_momentum_topk(
        adj_wide,
        top_k=args.top_k,
        lookback=args.lookback,
        rebalance_freq=args.rebalance_freq,
    )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    nav_path = out_dir / "raw_vs_adjusted_nav.csv"
    nav_compare = pd.DataFrame({"raw_close_nav": raw_nav, "adjusted_close_nav": adj_nav})
    nav_compare.to_csv(
        nav_path,
        date_format="%Y-%m-%d",
    )
    paths["nav"] = nav_path
    plot_path = out_dir / "raw_vs_adjusted_nav.png"
    plot_nav_compare(nav_compare, plot_path)
    paths["nav_plot"] = plot_path
    summary_path = out_dir / "raw_vs_adjusted_summary.csv"
    pd.DataFrame(
        [
            _stats(raw_nav, "RAW_CLOSE_MOMENTUM_TOPK"),
            _stats(adj_nav, "ADJUSTED_CLOSE_MOMENTUM_TOPK"),
        ]
    ).to_csv(summary_path, index=False)
    paths["summary"] = summary_path
    raw_log.to_csv(out_dir / "raw_close_rebalance.csv", index=False, date_format="%Y-%m-%d")
    adj_log.to_csv(out_dir / "adjusted_close_rebalance.csv", index=False, date_format="%Y-%m-%d")
    paths["raw_rebalance"] = out_dir / "raw_close_rebalance.csv"
    paths["adjusted_rebalance"] = out_dir / "adjusted_close_rebalance.csv"
    return paths


def plot_nav_compare(nav_compare: pd.DataFrame, save_path: Path) -> None:
    """Plot raw close vs adjusted close NAV comparison when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ax = nav_compare.plot(figsize=(10, 5), linewidth=1.8)
    ax.set_title("Raw Close vs Adjusted Close Momentum Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.25)
    ax.legend(["Raw close", "Adjusted close"])
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-csv", default=settings.tushare_price_cache_path)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir / "adjusted_price_comparison")
    parser.add_argument("--adjustment-mode", choices=["qfq", "hfq"], default=settings.adjustment_mode)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--lookback", type=int, default=settings.momentum_lookback)
    parser.add_argument("--rebalance-freq", default=settings.rebalance_freq)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = build_report(args)
    for name, path in paths.items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
