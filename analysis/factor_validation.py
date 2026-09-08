"""
样本外验证与因子失效监控。

本模块把已有的 IC、Top-K 多头超额、分组收益诊断按时间切成训练段和验证段，
用于回答：训练期看起来有效的因子，在样本外是否仍然有效。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.factor_diagnostics import (
    _last_trading_dates,
    batch_factor_group_returns,
    batch_factor_long_excess,
)
from analysis.ic import daily_ic_spearman, ic_distribution_summary
from backtest.backtest_utils import prices_to_wide_close
from config import Settings


VALIDATION_COLUMNS = [
    "factor",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "train_ic_mean",
    "validation_ic_mean",
    "ic_mean_delta",
    "train_positive_rate",
    "validation_positive_rate",
    "positive_rate_delta",
    "train_excess_ann_return",
    "validation_excess_ann_return",
    "excess_ann_delta",
    "train_information_ratio",
    "validation_information_ratio",
    "train_top_minus_bottom_ann",
    "validation_top_minus_bottom_ann",
    "top_minus_bottom_delta",
    "train_monotonicity_score",
    "validation_monotonicity_score",
    "monotonicity_delta",
    "train_n_days",
    "validation_n_days",
]

MONITOR_COLUMNS = [
    "factor",
    "status",
    "severity",
    "reasons",
    "validation_ic_mean",
    "validation_positive_rate",
    "validation_excess_ann_return",
    "validation_top_minus_bottom_ann",
    "validation_monotonicity_score",
    "ic_mean_delta",
    "excess_ann_delta",
    "top_minus_bottom_delta",
]

ROLLING_VALIDATION_COLUMNS = [
    "window_id",
    "factor",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "train_ic_mean",
    "validation_ic_mean",
    "ic_mean_delta",
    "train_positive_rate",
    "validation_positive_rate",
    "positive_rate_delta",
    "train_excess_ann_return",
    "validation_excess_ann_return",
    "excess_ann_delta",
    "train_information_ratio",
    "validation_information_ratio",
    "train_top_minus_bottom_ann",
    "validation_top_minus_bottom_ann",
    "top_minus_bottom_delta",
    "train_monotonicity_score",
    "validation_monotonicity_score",
    "monotonicity_delta",
    "train_n_days",
    "validation_n_days",
]

ROLLING_SUMMARY_COLUMNS = [
    "factor",
    "n_windows",
    "avg_validation_ic_mean",
    "validation_ic_positive_window_rate",
    "avg_validation_positive_rate",
    "avg_validation_excess_ann_return",
    "excess_positive_window_rate",
    "avg_validation_information_ratio",
    "avg_validation_top_minus_bottom_ann",
    "top_minus_bottom_positive_window_rate",
    "avg_validation_monotonicity_score",
    "avg_ic_mean_delta",
    "avg_excess_ann_delta",
    "supportive_window_rate",
    "stable_window_rate",
    "status",
]

MULTI_HORIZON_COLUMNS = ["horizon", "horizon_days"] + VALIDATION_COLUMNS
MULTI_HORIZON_SUMMARY_COLUMNS = [
    "factor",
    "available_horizons",
    "supportive_horizons",
    "supportive_horizon_rate",
    "mean_validation_ic",
    "ic_1d",
    "ic_5d",
    "ic_20d",
    "ic_next_rebalance",
    "status",
]


def _next_rebalance_ic(
    factor: pd.Series,
    prices: pd.DataFrame,
    *,
    rebalance_freq: str,
    price_col: str,
    min_names: int = 3,
) -> pd.Series:
    """调仓日因子截面与持有至下一调仓日收益的 Spearman IC。"""
    wide = prices_to_wide_close(prices, close_col=price_col).sort_index().sort_index(axis=1)
    dates = _last_trading_dates(pd.DatetimeIndex(wide.index), rebalance_freq)
    values: dict[pd.Timestamp, float] = {}
    for start, end in zip(dates[:-1], dates[1:]):
        try:
            score = factor.xs(pd.Timestamp(start), level="date").astype(float).rename("factor")
        except KeyError:
            continue
        future_return = (wide.loc[end] / wide.loc[start] - 1.0).rename("forward_return")
        joined = pd.concat([score, future_return], axis=1).dropna()
        values[pd.Timestamp(start)] = (
            float(joined["factor"].corr(joined["forward_return"], method="spearman"))
            if len(joined) >= int(min_names)
            else np.nan
        )
    return pd.Series(values, name="ic").sort_index()


def split_train_validation_dates(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    train_ratio: float = 0.5,
) -> tuple[pd.Index, pd.Index]:
    """按日期交集切分训练段和样本外验证段。"""
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel 须为 MultiIndex(date, symbol)")
    panel_dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    dates = pd.Index(pd.to_datetime(prices.index)).intersection(panel_dates).sort_values()
    if len(dates) < 4:
        raise ValueError("可用于样本外验证的日期太少")
    ratio = min(max(float(train_ratio), 0.2), 0.8)
    split_pos = int(len(dates) * ratio)
    split_pos = min(max(split_pos, 2), len(dates) - 1)
    return dates[:split_pos], dates[split_pos:]


def _panel_on_dates(panel: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    mask = panel.index.get_level_values("date").isin(pd.Index(dates))
    return panel.loc[mask]


def _factor_level_group_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    if group_summary.empty:
        return pd.DataFrame(
            columns=["factor", "top_minus_bottom_ann", "monotonicity_score"]
        )
    out = (
        group_summary.sort_values(["factor", "group"])
        .groupby("factor", as_index=False)
        .tail(1)
        .copy()
    )
    keep = ["factor", "top_minus_bottom_ann", "monotonicity_score"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].reset_index(drop=True)


def _segment_metrics(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factors: list[str],
    settings: Settings,
    segment: str,
    benchmark_prices: pd.DataFrame | None = None,
    ic_forward_days: int | None = None,
    ic_to_next_rebalance: bool = False,
) -> pd.DataFrame:
    ic_by_name: dict[str, pd.Series] = {}
    for factor in factors:
        if factor not in panel.columns:
            continue
        ser = panel[factor]
        if ser.notna().sum() == 0:
            continue
        try:
            if ic_to_next_rebalance:
                ic_by_name[factor] = _next_rebalance_ic(
                    ser,
                    prices,
                    rebalance_freq=settings.rebalance_freq,
                    price_col=settings.price_col,
                )
            else:
                ic_by_name[factor] = daily_ic_spearman(
                    ser,
                    prices,
                    forward_days=int(
                        ic_forward_days if ic_forward_days is not None else settings.ic_forward_days
                    ),
                )
        except Exception:
            continue

    ic_summary = ic_distribution_summary(ic_by_name)
    if ic_summary.empty:
        ic_summary = pd.DataFrame(columns=["factor", "mean_ic", "positive_rate", "n_days"])
    ic_summary = ic_summary.rename(
        columns={
            "mean_ic": f"{segment}_ic_mean",
            "positive_rate": f"{segment}_positive_rate",
            "n_days": f"{segment}_n_days",
        }
    )

    long_summary, _ = batch_factor_long_excess(
        panel,
        prices,
        factors=factors,
        top_k=settings.top_k,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        periods=settings.trading_days_per_year,
        benchmark_prices=benchmark_prices,
    )
    if long_summary.empty:
        long_summary = pd.DataFrame(
            columns=["factor", "excess_ann_return", "information_ratio"]
        )
    long_summary = long_summary.rename(
        columns={
            "excess_ann_return": f"{segment}_excess_ann_return",
            "information_ratio": f"{segment}_information_ratio",
        }
    )

    _, group_summary = batch_factor_group_returns(
        panel,
        prices,
        factors=factors,
        group_count=settings.factor_group_count,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        trading_days_per_year=settings.trading_days_per_year,
    )
    group_level = _factor_level_group_summary(group_summary).rename(
        columns={
            "top_minus_bottom_ann": f"{segment}_top_minus_bottom_ann",
            "monotonicity_score": f"{segment}_monotonicity_score",
        }
    )

    out = pd.DataFrame({"factor": factors})
    out = out.merge(
        ic_summary[
            [
                c
                for c in ["factor", f"{segment}_ic_mean", f"{segment}_positive_rate", f"{segment}_n_days"]
                if c in ic_summary.columns
            ]
        ],
        on="factor",
        how="left",
    )
    out = out.merge(
        long_summary[
            [
                c
                for c in ["factor", f"{segment}_excess_ann_return", f"{segment}_information_ratio"]
                if c in long_summary.columns
            ]
        ],
        on="factor",
        how="left",
    )
    out = out.merge(group_level, on="factor", how="left")
    return out


def build_out_of_sample_validation(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    *,
    factors: list[str] | None = None,
    train_ratio: float | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    ic_forward_days: int | None = None,
    ic_to_next_rebalance: bool = False,
) -> pd.DataFrame:
    """生成训练段与样本外验证段的因子评价对照表。"""
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel 须为 MultiIndex(date, symbol)")
    factor_list = [str(x) for x in (factors or list(panel.columns)) if str(x) in panel.columns]
    factor_list = [x for x in factor_list if panel[x].notna().sum() > 0]
    if not factor_list:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)

    ratio = float(train_ratio if train_ratio is not None else settings.factor_weight_train_ratio)
    train_dates, validation_dates = split_train_validation_dates(panel, prices, train_ratio=ratio)
    train_panel = _panel_on_dates(panel[factor_list], train_dates)
    validation_panel = _panel_on_dates(panel[factor_list], validation_dates)
    train_prices = prices.loc[pd.Index(prices.index).isin(train_dates)]
    validation_prices = prices.loc[pd.Index(prices.index).isin(validation_dates)]
    train_benchmark = None
    validation_benchmark = None
    if benchmark_prices is not None:
        train_benchmark = benchmark_prices.loc[pd.Index(benchmark_prices.index).isin(train_dates)]
        validation_benchmark = benchmark_prices.loc[
            pd.Index(benchmark_prices.index).isin(validation_dates)
        ]

    train = _segment_metrics(
        train_panel,
        train_prices,
        factors=factor_list,
        settings=settings,
        segment="train",
        benchmark_prices=train_benchmark,
        ic_forward_days=ic_forward_days,
        ic_to_next_rebalance=ic_to_next_rebalance,
    )
    validation = _segment_metrics(
        validation_panel,
        validation_prices,
        factors=factor_list,
        settings=settings,
        segment="validation",
        benchmark_prices=validation_benchmark,
        ic_forward_days=ic_forward_days,
        ic_to_next_rebalance=ic_to_next_rebalance,
    )
    out = train.merge(validation, on="factor", how="outer")
    out.insert(1, "train_start", pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"))
    out.insert(2, "train_end", pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"))
    out.insert(3, "validation_start", pd.Timestamp(validation_dates[0]).strftime("%Y-%m-%d"))
    out.insert(4, "validation_end", pd.Timestamp(validation_dates[-1]).strftime("%Y-%m-%d"))

    pairs = [
        ("ic_mean", "train_ic_mean", "validation_ic_mean"),
        ("positive_rate", "train_positive_rate", "validation_positive_rate"),
        ("excess_ann", "train_excess_ann_return", "validation_excess_ann_return"),
        ("top_minus_bottom", "train_top_minus_bottom_ann", "validation_top_minus_bottom_ann"),
        ("monotonicity", "train_monotonicity_score", "validation_monotonicity_score"),
    ]
    for prefix, train_col, val_col in pairs:
        if train_col not in out.columns:
            out[train_col] = np.nan
        if val_col not in out.columns:
            out[val_col] = np.nan
        out[f"{prefix}_delta"] = out[val_col].astype(float) - out[train_col].astype(float)

    for col in VALIDATION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[VALIDATION_COLUMNS].reset_index(drop=True)


def _validation_windows(
    dates: pd.Index,
    *,
    train_days: int,
    validation_days: int,
    step_days: int,
    min_validation_days: int,
) -> list[tuple[pd.Index, pd.Index]]:
    if len(dates) < 4:
        return []
    train_n = max(2, int(train_days))
    validation_n = max(1, int(validation_days))
    step_n = max(1, int(step_days))
    min_val_n = max(1, int(min_validation_days))
    windows: list[tuple[pd.Index, pd.Index]] = []
    start = 0
    while start + train_n + min_val_n <= len(dates):
        train = dates[start : start + train_n]
        val_end = min(start + train_n + validation_n, len(dates))
        validation = dates[start + train_n : val_end]
        if len(train) >= train_n and len(validation) >= min_val_n:
            windows.append((train, validation))
        start += step_n
    return windows


def build_rolling_out_of_sample_validation(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    *,
    factors: list[str] | None = None,
    train_days: int | None = None,
    validation_days: int | None = None,
    step_days: int | None = None,
    min_validation_days: int | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    生成滚动样本外验证表。

    与一次性 train/validation 切分不同，本函数按时间滚动多个窗口：
    过去一段为训练窗口，紧随其后的一段为验证窗口，用来观察因子是否跨窗口稳定。
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel 须为 MultiIndex(date, symbol)")
    factor_list = [str(x) for x in (factors or list(panel.columns)) if str(x) in panel.columns]
    factor_list = [x for x in factor_list if panel[x].notna().sum() > 0]
    if not factor_list:
        return pd.DataFrame(columns=ROLLING_VALIDATION_COLUMNS)

    panel_dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    dates = pd.Index(pd.to_datetime(prices.index)).intersection(panel_dates).sort_values()
    windows = _validation_windows(
        dates,
        train_days=int(train_days if train_days is not None else settings.rolling_oos_train_days),
        validation_days=int(
            validation_days if validation_days is not None else settings.rolling_oos_validation_days
        ),
        step_days=int(step_days if step_days is not None else settings.rolling_oos_step_days),
        min_validation_days=int(
            min_validation_days
            if min_validation_days is not None
            else settings.rolling_oos_min_validation_days
        ),
    )
    if not windows:
        return pd.DataFrame(columns=ROLLING_VALIDATION_COLUMNS)

    rows: list[pd.DataFrame] = []
    for window_id, (train_dates, validation_dates) in enumerate(windows, start=1):
        train_panel = _panel_on_dates(panel[factor_list], train_dates)
        validation_panel = _panel_on_dates(panel[factor_list], validation_dates)
        train_prices = prices.loc[pd.Index(prices.index).isin(train_dates)]
        validation_prices = prices.loc[pd.Index(prices.index).isin(validation_dates)]
        train_benchmark = None
        validation_benchmark = None
        if benchmark_prices is not None:
            train_benchmark = benchmark_prices.loc[
                pd.Index(benchmark_prices.index).isin(train_dates)
            ]
            validation_benchmark = benchmark_prices.loc[
                pd.Index(benchmark_prices.index).isin(validation_dates)
            ]
        train = _segment_metrics(
            train_panel,
            train_prices,
            factors=factor_list,
            settings=settings,
            segment="train",
            benchmark_prices=train_benchmark,
        )
        validation = _segment_metrics(
            validation_panel,
            validation_prices,
            factors=factor_list,
            settings=settings,
            segment="validation",
            benchmark_prices=validation_benchmark,
        )
        out = train.merge(validation, on="factor", how="outer")
        out.insert(0, "window_id", int(window_id))
        out.insert(2, "train_start", pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"))
        out.insert(3, "train_end", pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"))
        out.insert(4, "validation_start", pd.Timestamp(validation_dates[0]).strftime("%Y-%m-%d"))
        out.insert(5, "validation_end", pd.Timestamp(validation_dates[-1]).strftime("%Y-%m-%d"))

        pairs = [
            ("ic_mean", "train_ic_mean", "validation_ic_mean"),
            ("positive_rate", "train_positive_rate", "validation_positive_rate"),
            ("excess_ann", "train_excess_ann_return", "validation_excess_ann_return"),
            ("top_minus_bottom", "train_top_minus_bottom_ann", "validation_top_minus_bottom_ann"),
            ("monotonicity", "train_monotonicity_score", "validation_monotonicity_score"),
        ]
        for prefix, train_col, val_col in pairs:
            if train_col not in out.columns:
                out[train_col] = np.nan
            if val_col not in out.columns:
                out[val_col] = np.nan
            out[f"{prefix}_delta"] = out[val_col].astype(float) - out[train_col].astype(float)
        for col in ROLLING_VALIDATION_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        rows.append(out[ROLLING_VALIDATION_COLUMNS])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ROLLING_VALIDATION_COLUMNS)


def summarize_rolling_out_of_sample_validation(rolling_validation: pd.DataFrame) -> pd.DataFrame:
    """按因子汇总滚动样本外验证结果。"""
    if rolling_validation.empty:
        return pd.DataFrame(columns=ROLLING_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for factor, g in rolling_validation.groupby("factor", sort=True):
        valid = g.copy()
        n_windows = int(valid["window_id"].nunique())
        val_ic = pd.to_numeric(valid["validation_ic_mean"], errors="coerce")
        val_pos = pd.to_numeric(valid["validation_positive_rate"], errors="coerce")
        val_excess = pd.to_numeric(valid["validation_excess_ann_return"], errors="coerce")
        val_ir = pd.to_numeric(valid["validation_information_ratio"], errors="coerce")
        val_tb = pd.to_numeric(valid["validation_top_minus_bottom_ann"], errors="coerce")
        val_mono = pd.to_numeric(valid["validation_monotonicity_score"], errors="coerce")
        ic_delta = pd.to_numeric(valid["ic_mean_delta"], errors="coerce")
        excess_delta = pd.to_numeric(valid["excess_ann_delta"], errors="coerce")

        stable_mask = (
            (val_ic.fillna(-np.inf) > 0.0)
            & (val_excess.fillna(-np.inf) > 0.0)
            & (val_tb.fillna(-np.inf) > 0.0)
        )
        stable_rate = float(stable_mask.mean()) if n_windows else np.nan
        supportive_count = (
            (val_ic.fillna(-np.inf) > 0.0).astype(int)
            + (val_excess.fillna(-np.inf) > 0.0).astype(int)
            + (val_tb.fillna(-np.inf) > 0.0).astype(int)
        )
        supportive_rate = float((supportive_count >= 2).mean()) if n_windows else np.nan
        if np.isfinite(supportive_rate) and supportive_rate >= 0.6:
            status = "ROBUST"
        elif np.isfinite(supportive_rate) and supportive_rate >= 0.3:
            status = "WATCH"
        else:
            status = "UNSTABLE"

        rows.append(
            {
                "factor": str(factor),
                "n_windows": n_windows,
                "avg_validation_ic_mean": float(val_ic.mean()) if val_ic.notna().any() else np.nan,
                "validation_ic_positive_window_rate": float((val_ic > 0).mean()) if len(val_ic) else np.nan,
                "avg_validation_positive_rate": float(val_pos.mean()) if val_pos.notna().any() else np.nan,
                "avg_validation_excess_ann_return": float(val_excess.mean()) if val_excess.notna().any() else np.nan,
                "excess_positive_window_rate": float((val_excess > 0).mean()) if len(val_excess) else np.nan,
                "avg_validation_information_ratio": float(val_ir.mean()) if val_ir.notna().any() else np.nan,
                "avg_validation_top_minus_bottom_ann": float(val_tb.mean()) if val_tb.notna().any() else np.nan,
                "top_minus_bottom_positive_window_rate": float((val_tb > 0).mean()) if len(val_tb) else np.nan,
                "avg_validation_monotonicity_score": float(val_mono.mean()) if val_mono.notna().any() else np.nan,
                "avg_ic_mean_delta": float(ic_delta.mean()) if ic_delta.notna().any() else np.nan,
                "avg_excess_ann_delta": float(excess_delta.mean()) if excess_delta.notna().any() else np.nan,
                "supportive_window_rate": supportive_rate,
                "stable_window_rate": stable_rate,
                "status": status,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=ROLLING_SUMMARY_COLUMNS)
    order = {"ROBUST": 0, "WATCH": 1, "UNSTABLE": 2}
    out["_order"] = out["status"].map(order).fillna(3).astype(int)
    out = out.sort_values(
        ["_order", "supportive_window_rate", "avg_validation_excess_ann_return", "factor"],
        ascending=[True, False, False, True],
    ).drop(columns=["_order"])
    return out[ROLLING_SUMMARY_COLUMNS].reset_index(drop=True)


def build_multi_horizon_out_of_sample_validation(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    *,
    factors: list[str] | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
    include_next_rebalance: bool = True,
    train_ratio: float | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按多个固定期限及“持有至下一调仓日”生成同口径样本外验证。"""
    frames: list[pd.DataFrame] = []
    for horizon in dict.fromkeys(int(x) for x in horizons if int(x) > 0):
        result = build_out_of_sample_validation(
            panel,
            prices,
            settings,
            factors=factors,
            train_ratio=train_ratio,
            benchmark_prices=benchmark_prices,
            ic_forward_days=horizon,
        )
        result.insert(0, "horizon", f"{horizon}D")
        result.insert(1, "horizon_days", horizon)
        frames.append(result[MULTI_HORIZON_COLUMNS])
    if include_next_rebalance:
        result = build_out_of_sample_validation(
            panel,
            prices,
            settings,
            factors=factors,
            train_ratio=train_ratio,
            benchmark_prices=benchmark_prices,
            ic_to_next_rebalance=True,
        )
        result.insert(0, "horizon", "NEXT_REBALANCE")
        result.insert(1, "horizon_days", np.nan)
        frames.append(result[MULTI_HORIZON_COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MULTI_HORIZON_COLUMNS)


def summarize_multi_horizon_validation(validation: pd.DataFrame) -> pd.DataFrame:
    """汇总多期限IC支持度，避免单一期限决定因子生死。"""
    if validation.empty:
        return pd.DataFrame(columns=MULTI_HORIZON_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for factor, group in validation.groupby("factor", sort=True):
        metrics = group[["horizon", "validation_ic_mean", "validation_positive_rate"]].copy()
        metrics["validation_ic_mean"] = pd.to_numeric(
            metrics["validation_ic_mean"], errors="coerce"
        )
        metrics["validation_positive_rate"] = pd.to_numeric(
            metrics["validation_positive_rate"], errors="coerce"
        )
        available = metrics[metrics["validation_ic_mean"].notna()].copy()
        supportive = available[
            (available["validation_ic_mean"] > 0.0)
            & (available["validation_positive_rate"] >= 0.5)
        ]
        n_available = int(len(available))
        n_supportive = int(len(supportive))
        rate = float(n_supportive / n_available) if n_available else np.nan
        next_rebalance_supportive = bool(
            (
                (supportive["horizon"] == "NEXT_REBALANCE")
                if not supportive.empty
                else pd.Series(dtype=bool)
            ).any()
        )
        if np.isfinite(rate) and rate >= 0.75 and next_rebalance_supportive:
            status = "ROBUST"
        elif np.isfinite(rate) and rate >= 0.5:
            status = "SUPPORTIVE"
        elif np.isfinite(rate) and rate > 0.0:
            status = "WEAK"
        else:
            status = "FAILED"
        by_horizon = metrics.set_index("horizon")["validation_ic_mean"]
        rows.append(
            {
                "factor": str(factor),
                "available_horizons": n_available,
                "supportive_horizons": n_supportive,
                "supportive_horizon_rate": rate,
                "mean_validation_ic": float(available["validation_ic_mean"].mean())
                if n_available
                else np.nan,
                "ic_1d": float(by_horizon.get("1D", np.nan)),
                "ic_5d": float(by_horizon.get("5D", np.nan)),
                "ic_20d": float(by_horizon.get("20D", np.nan)),
                "ic_next_rebalance": float(by_horizon.get("NEXT_REBALANCE", np.nan)),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=MULTI_HORIZON_SUMMARY_COLUMNS)


def build_factor_decay_monitor(
    validation: pd.DataFrame,
    *,
    min_validation_ic: float = 0.0,
    min_positive_rate: float = 0.5,
    min_excess_ann_return: float = 0.0,
    min_top_minus_bottom_ann: float = 0.0,
    min_monotonicity: float = 0.5,
) -> pd.DataFrame:
    """
    根据样本外验证结果生成失效监控状态。

    status:
    - OK：样本外主要指标仍然为正；
    - WATCH：有指标转弱，但未全面失效；
    - DEGRADED：训练期为正、验证期明显转弱；
    - FAILED：IC、多头超额和 Top-Bottom 同时为负。
    """
    if validation.empty:
        return pd.DataFrame(columns=MONITOR_COLUMNS)

    rows: list[dict[str, Any]] = []
    for rec in validation.to_dict("records"):
        factor = str(rec.get("factor", ""))
        val_ic = float(rec.get("validation_ic_mean", np.nan))
        val_pos = float(rec.get("validation_positive_rate", np.nan))
        val_excess = float(rec.get("validation_excess_ann_return", np.nan))
        val_tb = float(rec.get("validation_top_minus_bottom_ann", np.nan))
        val_mono = float(rec.get("validation_monotonicity_score", np.nan))
        ic_delta = float(rec.get("ic_mean_delta", np.nan))
        excess_delta = float(rec.get("excess_ann_delta", np.nan))
        tb_delta = float(rec.get("top_minus_bottom_delta", np.nan))

        reasons: list[str] = []
        if np.isfinite(val_ic) and val_ic < min_validation_ic:
            reasons.append("validation_ic_below_threshold")
        if np.isfinite(val_pos) and val_pos < min_positive_rate:
            reasons.append("positive_rate_below_threshold")
        if np.isfinite(val_excess) and val_excess < min_excess_ann_return:
            reasons.append("long_excess_below_threshold")
        if np.isfinite(val_tb) and val_tb < min_top_minus_bottom_ann:
            reasons.append("top_bottom_below_threshold")
        if np.isfinite(val_mono) and val_mono < min_monotonicity:
            reasons.append("monotonicity_below_threshold")
        if np.isfinite(ic_delta) and ic_delta < -0.02:
            reasons.append("ic_deteriorated")
        if np.isfinite(excess_delta) and excess_delta < -0.20:
            reasons.append("long_excess_deteriorated")
        if np.isfinite(tb_delta) and tb_delta < -0.20:
            reasons.append("top_bottom_deteriorated")

        negative_core = sum(
            [
                np.isfinite(val_ic) and val_ic < min_validation_ic,
                np.isfinite(val_excess) and val_excess < min_excess_ann_return,
                np.isfinite(val_tb) and val_tb < min_top_minus_bottom_ann,
            ]
        )
        if negative_core >= 3:
            status, severity = "FAILED", 3
        elif len(reasons) >= 4 or (
            np.isfinite(excess_delta) and excess_delta < -0.30 and np.isfinite(tb_delta) and tb_delta < -0.30
        ):
            status, severity = "DEGRADED", 2
        elif reasons:
            status, severity = "WATCH", 1
        else:
            status, severity = "OK", 0

        rows.append(
            {
                "factor": factor,
                "status": status,
                "severity": severity,
                "reasons": ";".join(reasons),
                "validation_ic_mean": val_ic,
                "validation_positive_rate": val_pos,
                "validation_excess_ann_return": val_excess,
                "validation_top_minus_bottom_ann": val_tb,
                "validation_monotonicity_score": val_mono,
                "ic_mean_delta": ic_delta,
                "excess_ann_delta": excess_delta,
                "top_minus_bottom_delta": tb_delta,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["severity", "factor"], ascending=[False, True]).reset_index(drop=True)
    return out[MONITOR_COLUMNS]


def save_factor_validation_outputs(
    settings: Settings,
    validation: pd.DataFrame,
    monitor: pd.DataFrame,
    rolling_validation: pd.DataFrame | None = None,
    rolling_summary: pd.DataFrame | None = None,
    multi_horizon_validation: pd.DataFrame | None = None,
    multi_horizon_summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """写入 output/factor_validation/ 下的样本外验证和失效监控表。"""
    base = settings.output_dir / "factor_validation"
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "out_of_sample_validation": base / "out_of_sample_validation.csv",
        "factor_decay_monitor": base / "factor_decay_monitor.csv",
    }
    validation.to_csv(paths["out_of_sample_validation"], index=False)
    monitor.to_csv(paths["factor_decay_monitor"], index=False)
    if rolling_validation is not None:
        paths["rolling_out_of_sample_validation"] = base / "rolling_out_of_sample_validation.csv"
        rolling_validation.to_csv(paths["rolling_out_of_sample_validation"], index=False)
    if rolling_summary is not None:
        paths["rolling_out_of_sample_summary"] = base / "rolling_out_of_sample_summary.csv"
        rolling_summary.to_csv(paths["rolling_out_of_sample_summary"], index=False)
    if multi_horizon_validation is not None:
        paths["multi_horizon_validation"] = base / "multi_horizon_validation.csv"
        multi_horizon_validation.to_csv(paths["multi_horizon_validation"], index=False)
    if multi_horizon_summary is not None:
        paths["multi_horizon_summary"] = base / "multi_horizon_summary.csv"
        multi_horizon_summary.to_csv(paths["multi_horizon_summary"], index=False)
    return paths
