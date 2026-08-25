"""Price adjustment helpers for research price series."""
from __future__ import annotations

import pandas as pd


def add_adjusted_close(
    prices: pd.DataFrame,
    *,
    mode: str = "qfq",
    close_col: str = "close",
    factor_col: str = "adj_factor",
    adjusted_col: str = "adj_close",
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
) -> pd.DataFrame:
    """
    Add an adjusted close column from raw close and adjustment factor.

    `mode="qfq"` normalizes by the latest factor in each symbol, so the newest
    adjusted close equals the newest raw close. `mode="hfq"` keeps the direct
    `close * adj_factor` series.
    """
    if prices.empty:
        return prices.copy()
    if adjusted_col in prices.columns and prices[adjusted_col].notna().any():
        return prices.copy()

    required = {date_col, symbol_col, close_col, factor_col}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError("计算复权价缺少列: %s" % ", ".join(sorted(missing)))

    mode = str(mode or "qfq").lower()
    if mode not in {"qfq", "hfq"}:
        raise ValueError("复权模式仅支持 qfq / hfq，当前: %s" % mode)

    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[symbol_col] = df[symbol_col].astype(str).str.strip()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")
    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

    raw_adjusted = df[close_col] * df[factor_col]
    if mode == "hfq":
        df[adjusted_col] = raw_adjusted
        return df

    latest_factor = df.groupby(symbol_col, sort=False)[factor_col].transform(_last_valid)
    latest_factor = latest_factor.where(latest_factor != 0)
    df[adjusted_col] = raw_adjusted / latest_factor
    return df


def _last_valid(values: pd.Series) -> float | pd.NA:
    valid = values.dropna()
    if valid.empty:
        return pd.NA
    return float(valid.iloc[-1])
