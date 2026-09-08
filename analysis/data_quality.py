"""
数据质量与覆盖率报告。

本模块只做研究诊断，不改变因子、回测或调仓逻辑。它回答：
- 价格数据每只股票覆盖了多少交易日；
- 每个因子的非空比例是多少；
- 调仓日截面里每个因子有多少有效股票可选。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def _ensure_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为二级 MultiIndex (date, symbol)")
    out = panel.copy()
    out.index = out.index.set_names(["date", "symbol"])
    return out.sort_index()


def price_coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """统计每只股票价格覆盖天数与覆盖率。"""
    px = prices.sort_index().sort_index(axis=1).astype(float)
    n_days = int(len(px))
    rows: list[dict[str, Any]] = []
    for symbol in px.columns:
        s = px[symbol]
        valid = int(s.notna().sum())
        rows.append(
            {
                "symbol": str(symbol),
                "total_days": n_days,
                "valid_days": valid,
                "missing_days": n_days - valid,
                "coverage": float(valid / n_days) if n_days else float("nan"),
                "first_valid_date": s.dropna().index.min() if valid else pd.NaT,
                "last_valid_date": s.dropna().index.max() if valid else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def _aligned_eligible_mask(
    panel: pd.DataFrame,
    eligible_mask: pd.Series | None,
) -> pd.Series:
    if eligible_mask is None:
        return pd.Series(True, index=panel.index, dtype=bool)
    mask = eligible_mask.copy()
    if not isinstance(mask.index, pd.MultiIndex) or mask.index.nlevels != 2:
        raise TypeError("eligible_mask 须为 MultiIndex(date, symbol) 的 Series")
    mask.index = mask.index.set_names(["date", "symbol"])
    return mask.reindex(panel.index).fillna(False).astype(bool)


def factor_coverage(
    panel: pd.DataFrame,
    *,
    eligible_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """统计每个因子在当时有效股票池内的非空覆盖率。"""
    p = _ensure_panel(panel)
    eligible = _aligned_eligible_mask(p, eligible_mask)
    total = int(eligible.sum())
    rows: list[dict[str, Any]] = []
    for factor in p.columns:
        s = p.loc[eligible, factor]
        valid = int(s.notna().sum())
        rows.append(
            {
                "factor": str(factor),
                "total_cells": total,
                "valid_cells": valid,
                "missing_cells": total - valid,
                "coverage": float(valid / total) if total else float("nan"),
                "valid_dates": int(s.dropna().index.get_level_values("date").nunique()) if valid else 0,
                "valid_symbols": int(s.dropna().index.get_level_values("symbol").nunique()) if valid else 0,
            }
        )
    return pd.DataFrame(rows)


def factor_daily_coverage(
    panel: pd.DataFrame,
    *,
    eligible_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """统计每天在当时有效股票池内的因子覆盖率。"""
    p = _ensure_panel(panel)
    eligible = _aligned_eligible_mask(p, eligible_mask)
    symbols_total = eligible.groupby(level="date").sum().astype(int).rename("total_rows")
    frames: list[pd.DataFrame] = []
    for factor in p.columns:
        valid = (p[factor].notna() & eligible).groupby(level="date").sum().rename("valid_symbols")
        df = pd.concat([symbols_total, valid], axis=1).reset_index()
        df["factor"] = str(factor)
        df["coverage"] = df["valid_symbols"] / df["total_rows"].replace(0, pd.NA)
        frames.append(df[["date", "factor", "total_rows", "valid_symbols", "coverage"]])
    if not frames:
        return pd.DataFrame(columns=["date", "factor", "total_rows", "valid_symbols", "coverage"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "factor"]).reset_index(drop=True)


def rebalance_coverage(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    *,
    factors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """统计调仓日有效价格与有效因子截面规模。"""
    p = _ensure_panel(panel)
    px = prices.sort_index().sort_index(axis=1).astype(float)
    use_factors = [f for f in (list(factors) if factors is not None else list(p.columns)) if f in p.columns]
    rows: list[dict[str, Any]] = []

    for dt0 in rebalance_dates:
        dt = pd.Timestamp(dt0)
        row: dict[str, Any] = {"date": dt}
        if dt in px.index:
            price_valid_symbols = set(px.loc[dt].dropna().index.astype(str))
        else:
            price_valid_symbols = set()
        row["price_valid_symbols"] = int(len(price_valid_symbols))

        try:
            day_panel = p.xs(dt, level="date")
        except KeyError:
            day_panel = pd.DataFrame(columns=p.columns)

        all_valid_sets: list[set[str]] = []
        for factor in use_factors:
            if day_panel.empty or factor not in day_panel.columns:
                valid_symbols: set[str] = set()
            else:
                valid_symbols = set(day_panel[factor].dropna().index.astype(str))
            tradable_symbols = valid_symbols & price_valid_symbols
            row["%s_valid_symbols" % factor] = int(len(valid_symbols))
            row["%s_tradable_symbols" % factor] = int(len(tradable_symbols))
            all_valid_sets.append(tradable_symbols)

        row["all_factor_tradable_symbols"] = (
            int(len(set.intersection(*all_valid_sets))) if all_valid_sets else 0
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
