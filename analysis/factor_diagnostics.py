"""
因子评价诊断。

该模块用于回答两个直接问题：
1）某个因子排名靠前的 Top-K 多头组合，相对股票池等权基准是否产生超额收益；
2）把股票按因子分成多组后，收益是否大致随因子分数升高而升高。
它不是完整交易回测的替代品，而是 IC 与完整回测之间的一层因子有效性检查。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from analysis.benchmark import equal_weight_benchmark_nav, summarize_excess
from analysis.performance import summarize
from backtest.backtest_utils import prices_to_wide_close


def _resample_freq_alias(freq: str) -> str:
    return {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(freq, freq)


def _last_trading_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if index.empty:
        return pd.DatetimeIndex([], name=index.name)
    s = pd.Series(index=index, data=index)
    out = pd.DatetimeIndex(s.resample(_resample_freq_alias(freq)).last().dropna())
    return out.sort_values()


def _factor_slice(factor_values: pd.Series, date: pd.Timestamp) -> pd.Series:
    try:
        xs = factor_values.xs(date, level="date")
    except KeyError:
        return pd.Series(dtype=float)
    return xs.astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def factor_long_only_nav(
    factor_values: pd.Series,
    prices: pd.DataFrame,
    *,
    top_k: int = 5,
    rebalance_freq: str = "ME",
    price_col: str = "close",
    name: str = "LONG_ONLY",
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """
    构造因子 Top-K 多头等权组合净值。

    约定：在再平衡日收盘后按当日因子排序建仓，因此新权重从下一交易日收益开始生效。
    该函数不计手续费、不做优化配权，用于观察因子多头腿本身是否具备解释力。
    """
    if not isinstance(factor_values.index, pd.MultiIndex) or factor_values.index.nlevels != 2:
        raise TypeError("factor_values 须为二级 MultiIndex (date, symbol)")
    if int(top_k) <= 0:
        raise ValueError("top_k 必须为正整数")

    wide = prices_to_wide_close(prices, close_col=price_col).sort_index().sort_index(axis=1)
    if wide.empty:
        return pd.Series(dtype=float, name=name), []

    factor = factor_values.copy()
    factor.index = factor.index.set_names(["date", "symbol"])
    factor = factor.sort_index()

    returns = wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    dates = pd.DatetimeIndex(wide.index)
    rebal_dates = _last_trading_dates(dates, rebalance_freq)
    if len(dates) and (len(rebal_dates) == 0 or dates[0] < rebal_dates[0]):
        rebal_dates = pd.DatetimeIndex([dates[0]]).append(rebal_dates)
    rebal_set = set(pd.Timestamp(x) for x in rebal_dates)

    nav_values: list[float] = []
    nav_index: list[pd.Timestamp] = []
    log: list[dict[str, Any]] = []
    current_weights = pd.Series(dtype=float)
    nav = 1.0

    for i, date in enumerate(dates):
        dt = pd.Timestamp(date)
        if i > 0 and not current_weights.empty:
            day_ret = returns.loc[dt].reindex(current_weights.index)
            port_ret = float((current_weights * day_ret.fillna(0.0)).sum())
            nav *= 1.0 + port_ret
        nav_index.append(dt)
        nav_values.append(nav)

        if dt not in rebal_set:
            continue
        scores = _factor_slice(factor, dt)
        scores = scores.reindex(wide.columns).dropna()
        scores = scores[scores.index.isin(wide.columns)]
        picks = list(scores.sort_values(ascending=False).head(int(top_k)).index)
        if not picks:
            current_weights = pd.Series(dtype=float)
            log.append({"date": dt, "picks": [], "weights": [], "n_candidates": int(scores.shape[0])})
            continue
        current_weights = pd.Series(1.0 / len(picks), index=pd.Index(picks, name="symbol"))
        log.append(
            {
                "date": dt,
                "picks": [str(x) for x in picks],
                "weights": [float(x) for x in current_weights.tolist()],
                "n_candidates": int(scores.shape[0]),
            }
        )

    out = pd.Series(nav_values, index=pd.DatetimeIndex(nav_index, name="date"), name=name)
    if not out.empty:
        out = out / float(out.iloc[0])
    return out, log


def factor_long_excess_summary(
    factor_values: pd.Series,
    prices: pd.DataFrame,
    *,
    factor_name: str,
    top_k: int = 5,
    rebalance_freq: str = "ME",
    price_col: str = "close",
    periods: int = 252,
    benchmark_prices: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """返回单个因子的多头净值与相对股票池等权基准的摘要指标。"""
    nav, log = factor_long_only_nav(
        factor_values,
        prices,
        top_k=top_k,
        rebalance_freq=rebalance_freq,
        price_col=price_col,
        name=factor_name,
    )
    stats = summarize(nav, periods=periods)
    benchmark_source = benchmark_prices if benchmark_prices is not None else prices
    benchmark = equal_weight_benchmark_nav(benchmark_source, dates=nav.index, price_col=price_col)
    excess_stats = summarize_excess(nav, benchmark, periods=periods) if not benchmark.empty else {}
    stats.update(excess_stats)
    stats["factor"] = factor_name
    stats["n_rebalances"] = len(log)
    if log:
        candidates = [float(rec.get("n_candidates", np.nan)) for rec in log]
        stats["avg_candidates"] = float(np.nanmean(candidates)) if candidates else np.nan
    else:
        stats["avg_candidates"] = np.nan
    return nav, stats


def batch_factor_long_excess(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factors: Iterable[str] | None = None,
    top_k: int = 5,
    rebalance_freq: str = "ME",
    price_col: str = "close",
    periods: int = 252,
    benchmark_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """批量计算多列因子的 Top-K 多头超额摘要。"""
    cols = list(factors) if factors is not None else list(panel.columns)
    rows: list[Mapping[str, Any]] = []
    navs: dict[str, pd.Series] = {}
    for col in cols:
        if col not in panel.columns:
            continue
        ser = panel[col]
        if ser.notna().sum() == 0:
            continue
        nav, stats = factor_long_excess_summary(
            ser,
            prices,
            factor_name=str(col),
            top_k=top_k,
            rebalance_freq=rebalance_freq,
            price_col=price_col,
            periods=periods,
            benchmark_prices=benchmark_prices,
        )
        if not nav.empty:
            navs[str(col)] = nav
        rows.append(stats)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        first_cols = ["factor", "ann_return", "excess_ann_return", "tracking_error", "information_ratio"]
        rest = [c for c in frame.columns if c not in first_cols]
        frame = frame[[c for c in first_cols if c in frame.columns] + rest]
        frame = frame.sort_values("factor").reset_index(drop=True)
    return frame, navs


def _annualize_period_return(mean_period_return: float, periods_per_year: float) -> float:
    if not np.isfinite(mean_period_return) or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return float("nan")
    if mean_period_return <= -1.0:
        return -1.0
    return float((1.0 + mean_period_return) ** periods_per_year - 1.0)


def _infer_periods_per_year(rebalance_dates: pd.DatetimeIndex, trading_days_per_year: int) -> float:
    if len(rebalance_dates) < 2:
        return float("nan")
    gaps = pd.Series(rebalance_dates).diff().dropna().dt.days.astype(float)
    avg_gap = float(gaps.mean()) if not gaps.empty else float("nan")
    if not np.isfinite(avg_gap) or avg_gap <= 0:
        return float("nan")
    return float(365.25 / avg_gap)


def _assign_rank_groups(scores: pd.Series, group_count: int) -> pd.Series:
    """
    将因子截面按值从低到高分组，1 为低分组，group_count 为高分组。

    使用排序位置而不是 qcut，避免大量重复值时直接失败；当有效股票少于组数时，
    实际只会填充部分组，汇总时据此反映样本不足。
    """
    clean = scores.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.Series(dtype=int)
    ordered = clean.sort_values(ascending=True, kind="mergesort")
    n = len(ordered)
    gcount = int(group_count)
    groups = (np.floor(np.arange(n, dtype=float) * gcount / n) + 1).astype(int)
    groups = np.clip(groups, 1, gcount)
    return pd.Series(groups, index=ordered.index, name="group")


def factor_group_return_detail(
    factor_values: pd.Series,
    prices: pd.DataFrame,
    *,
    factor_name: str,
    group_count: int = 5,
    rebalance_freq: str = "ME",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    计算单个因子的分组持有期收益明细。

    每个调仓日按当日因子从低到高分组，使用从该调仓日收盘到下一调仓日收盘的简单收益。
    group=1 表示低分组，group=group_count 表示高分组。
    """
    if not isinstance(factor_values.index, pd.MultiIndex) or factor_values.index.nlevels != 2:
        raise TypeError("factor_values 须为二级 MultiIndex (date, symbol)")
    gcount = int(group_count)
    if gcount < 2:
        raise ValueError("group_count 必须至少为 2")

    wide = prices_to_wide_close(prices, close_col=price_col).sort_index().sort_index(axis=1)
    if wide.empty:
        return pd.DataFrame()

    factor = factor_values.copy()
    factor.index = factor.index.set_names(["date", "symbol"])
    factor = factor.sort_index()

    dates = pd.DatetimeIndex(wide.index)
    rebal_dates = _last_trading_dates(dates, rebalance_freq)
    rebal_dates = rebal_dates.intersection(dates).sort_values()
    rows: list[dict[str, Any]] = []
    if len(rebal_dates) < 2:
        return pd.DataFrame(
            columns=[
                "factor",
                "date",
                "forward_end",
                "group",
                "n_symbols",
                "period_return",
                "group_count",
            ]
        )

    for cur, nxt in zip(rebal_dates[:-1], rebal_dates[1:]):
        cur_dt = pd.Timestamp(cur)
        nxt_dt = pd.Timestamp(nxt)
        scores = _factor_slice(factor, cur_dt).reindex(wide.columns).dropna()
        px0 = wide.loc[cur_dt].reindex(scores.index).astype(float)
        px1 = wide.loc[nxt_dt].reindex(scores.index).astype(float)
        valid = px0.notna() & px1.notna() & (px0 > 0)
        scores = scores.loc[valid]
        if scores.empty:
            continue
        groups = _assign_rank_groups(scores, gcount)
        ret = (px1.loc[groups.index] / px0.loc[groups.index] - 1.0).replace([np.inf, -np.inf], np.nan)
        by_group = pd.DataFrame({"group": groups, "period_return": ret}).dropna()
        if by_group.empty:
            continue
        for g, sub in by_group.groupby("group"):
            rows.append(
                {
                    "factor": str(factor_name),
                    "date": cur_dt,
                    "forward_end": nxt_dt,
                    "group": int(g),
                    "n_symbols": int(sub.shape[0]),
                    "period_return": float(sub["period_return"].mean()),
                    "group_count": gcount,
                }
            )

    return pd.DataFrame(rows)


def summarize_group_returns(
    detail: pd.DataFrame,
    *,
    group_count: int = 5,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """由分组收益明细汇总每个因子的分组收益、Top-Bottom 与单调性。"""
    if detail.empty:
        return pd.DataFrame()
    gcount = int(group_count)
    rows: list[dict[str, Any]] = []

    for factor, fdf in detail.groupby("factor"):
        piv = fdf.pivot_table(index="date", columns="group", values="period_return", aggfunc="mean")
        dates = pd.DatetimeIndex(sorted(fdf["date"].dropna().unique()))
        periods_per_year = _infer_periods_per_year(dates, trading_days_per_year)
        group_means = piv.mean(axis=0, skipna=True)
        if len(group_means) >= 2:
            inc = np.diff(group_means.reindex(sorted(group_means.index)).to_numpy(dtype=float))
            inc = inc[np.isfinite(inc)]
            monotonicity = float((inc > 0).sum() / len(inc)) if len(inc) else float("nan")
        else:
            monotonicity = float("nan")

        top_bottom = pd.Series(dtype=float)
        top_minus_bottom_mean = float("nan")
        top_minus_bottom_ann = float("nan")
        top_minus_bottom_hit_rate = float("nan")
        if 1 in piv.columns and gcount in piv.columns:
            top_bottom = (piv[gcount] - piv[1]).dropna()
            if not top_bottom.empty:
                top_minus_bottom_mean = float(top_bottom.mean())
                top_minus_bottom_ann = _annualize_period_return(top_minus_bottom_mean, periods_per_year)
                top_minus_bottom_hit_rate = float((top_bottom > 0).mean())

        symbol_counts = fdf.groupby("group")["n_symbols"].mean()
        for group in range(1, gcount + 1):
            sub = fdf[fdf["group"] == group]
            mean_ret = float(sub["period_return"].mean()) if not sub.empty else float("nan")
            rows.append(
                {
                    "factor": str(factor),
                    "group": group,
                    "group_label": "G%d" % group,
                    "mean_period_return": mean_ret,
                    "ann_return": _annualize_period_return(mean_ret, periods_per_year),
                    "hit_rate": float((sub["period_return"] > 0).mean()) if not sub.empty else float("nan"),
                    "n_periods": int(sub.shape[0]),
                    "avg_symbols": float(symbol_counts.get(group, np.nan)),
                    "top_minus_bottom_mean": top_minus_bottom_mean,
                    "top_minus_bottom_ann": top_minus_bottom_ann,
                    "top_minus_bottom_hit_rate": top_minus_bottom_hit_rate,
                    "monotonicity_score": monotonicity,
                    "periods_per_year": periods_per_year,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["factor", "group"]).reset_index(drop=True)
    return out


def batch_factor_group_returns(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factors: Iterable[str] | None = None,
    group_count: int = 5,
    rebalance_freq: str = "ME",
    price_col: str = "close",
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量计算多列因子的分组收益明细与汇总。"""
    cols = list(factors) if factors is not None else list(panel.columns)
    details: list[pd.DataFrame] = []
    for col in cols:
        if col not in panel.columns:
            continue
        ser = panel[col]
        if ser.notna().sum() == 0:
            continue
        frame = factor_group_return_detail(
            ser,
            prices,
            factor_name=str(col),
            group_count=group_count,
            rebalance_freq=rebalance_freq,
            price_col=price_col,
        )
        if not frame.empty:
            details.append(frame)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary = summarize_group_returns(
        detail,
        group_count=group_count,
        trading_days_per_year=trading_days_per_year,
    )
    return detail, summary
