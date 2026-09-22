"""
因子清洗与横截面标准化。

输入为 MultiIndex(date, symbol) × 因子列的原始面板；输出保持同样索引和列。
支持普通横截面标准化，也支持按行业分组后的行业内标准化。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def winsorize_series(
    s: pd.Series,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """按分位数对单条 Series 去极值，NaN 保持 NaN。"""
    x = s.astype(float)
    valid = x.dropna()
    if valid.empty:
        return x
    lo = float(valid.quantile(float(lower_q)))
    hi = float(valid.quantile(float(upper_q)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        return x
    return x.clip(lower=lo, upper=hi)


def cross_sectional_zscore(
    panel: pd.DataFrame,
    *,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_count: int = 2,
    constant_fill: Optional[float] = 0.0,
) -> pd.DataFrame:
    """
    对每个交易日、每一列因子，在当日股票池上做去极值与横截面 z-score。

    当日有效样本少于 min_count，或标准差过小时，若 constant_fill 非 None 则填该值，否则保留 NaN。
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为 MultiIndex(date, symbol) × 因子列")

    values = panel.astype(float)
    dates = values.index.get_level_values(0)
    out = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for col in values.columns:
        s = values[col]
        grouped = s.groupby(dates, sort=False)
        if winsorize:
            lower = grouped.transform("quantile", q=float(lower_q))
            upper = grouped.transform("quantile", q=float(upper_q))
            x = s.clip(lower=lower, upper=upper)
        else:
            x = s
        x_grouped = x.groupby(dates, sort=False)
        count = x_grouped.transform("count")
        mean = x_grouped.transform("mean")
        sigma = x_grouped.transform("std", ddof=0)
        valid_group = (count >= int(min_count)) & sigma.notna() & (sigma >= 1e-12)
        z = (x - mean) / sigma
        if constant_fill is not None:
            z = z.mask(~valid_group, float(constant_fill))
        else:
            z = z.where(valid_group)
        out[col] = z
    out.index = out.index.set_names(["date", "symbol"])
    return out


def _industry_series_for_panel(
    panel: pd.DataFrame,
    industry: pd.Series | pd.DataFrame,
    *,
    industry_col: str = "industry",
) -> pd.Series:
    if isinstance(industry, pd.DataFrame):
        if industry_col in industry.columns:
            ser = industry[industry_col]
        elif industry.shape[1] == 1:
            ser = industry.iloc[:, 0]
        else:
            raise ValueError("industry DataFrame 缺少行业列 %r" % industry_col)
    else:
        ser = industry
    if not isinstance(ser.index, pd.MultiIndex) or ser.index.nlevels != 2:
        raise TypeError("industry 须为 MultiIndex(date, symbol) 的 Series/DataFrame")
    out = ser.copy()
    out.index = out.index.set_names(["date", "symbol"])
    return out.reindex(panel.index)


def industry_neutral_zscore(
    panel: pd.DataFrame,
    industry: pd.Series | pd.DataFrame,
    *,
    industry_col: str = "industry",
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_count: int = 2,
    min_industry_count: int = 3,
    constant_fill: Optional[float] = 0.0,
) -> pd.DataFrame:
    """
    对每个交易日、每一列因子做行业内 z-score。

    行业内有效样本不少于 `min_industry_count` 时，在行业内部去极值并标准化；
    样本不足或缺行业的数据，回退到当日全股票池 z-score，避免小行业被全部置零。
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为 MultiIndex(date, symbol) × 因子列")

    industry_ser = _industry_series_for_panel(panel, industry, industry_col=industry_col)
    global_z = cross_sectional_zscore(
        panel,
        winsorize=winsorize,
        lower_q=lower_q,
        upper_q=upper_q,
        min_count=min_count,
        constant_fill=constant_fill,
    )
    values = panel.astype(float)
    out = global_z.copy()
    dates = values.index.get_level_values(0)
    min_industry_count = max(int(min_industry_count), int(min_count))
    industries = industry_ser.fillna("").astype(str).str.strip()
    has_industry = industries.ne("")

    for col in values.columns:
        s = values[col]
        grouped = s.groupby([dates, industries], sort=False, dropna=False)
        if winsorize:
            lower = grouped.transform("quantile", q=float(lower_q))
            upper = grouped.transform("quantile", q=float(upper_q))
            x = s.clip(lower=lower, upper=upper)
        else:
            x = s
        x_grouped = x.groupby([dates, industries], sort=False, dropna=False)
        count = x_grouped.transform("count")
        mean = x_grouped.transform("mean")
        sigma = x_grouped.transform("std", ddof=0)
        enough = has_industry & (count >= min_industry_count)
        stable = sigma.notna() & (sigma >= 1e-12)
        replacement = enough & stable
        out.loc[replacement, col] = ((x - mean) / sigma).loc[replacement]
        if constant_fill is not None:
            constant_group = enough & ~stable
            out.loc[constant_group, col] = float(constant_fill)
    out.index = out.index.set_names(["date", "symbol"])
    return out


def preprocess_factor_panel(
    panel: pd.DataFrame,
    *,
    industry: pd.Series | pd.DataFrame | None = None,
    industry_col: str = "industry",
    by_industry: bool = False,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_count: int = 2,
    min_industry_count: int = 3,
) -> pd.DataFrame:
    """生成清洗后的横截面 z-score 因子面板。"""
    if by_industry and industry is not None:
        return industry_neutral_zscore(
            panel,
            industry,
            industry_col=industry_col,
            winsorize=winsorize,
            lower_q=lower_q,
            upper_q=upper_q,
            min_count=min_count,
            min_industry_count=min_industry_count,
            constant_fill=0.0,
        )
    return cross_sectional_zscore(
        panel,
        winsorize=winsorize,
        lower_q=lower_q,
        upper_q=upper_q,
        min_count=min_count,
        constant_fill=0.0,
    )
