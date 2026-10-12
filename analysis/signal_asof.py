"""As-of momentum definitions and execution-aligned descriptive diagnostics.

Pure research helpers. Never fetch data, read credentials, or alter live strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signals_asof(prices, date, symbols, specs, *, minimum_history_fraction=.9):
    """Same available-at-date sample for every signal; skip is NOT execution lag."""
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates):
        raise ValueError("invalid price axes")
    if (pd.Timestamp(date) not in prices.index or not len(symbols)
            or len(set(symbols)) != len(symbols) or not set(symbols).issubset(prices.columns)):
        raise ValueError("unknown signal date or duplicate symbols")
    if not 0 < minimum_history_fraction <= 1 or not specs:
        raise ValueError("invalid history rule")
    for lookback, skip, direction in specs.values():
        if not isinstance(lookback, int) or not isinstance(skip, int) or lookback < 1 or skip < 0 or direction not in [-1, 1]:
            raise ValueError("invalid signal definition")
    length = max(v[0]+v[1] for v in specs.values()) + 1
    hist = prices.loc[:pd.Timestamp(date)].tail(length).reindex(columns=symbols)
    hist = hist.where(np.isfinite(hist) & hist.gt(0))
    if len(hist) < length:
        raise ValueError("insufficient full-market warmup")
    frame = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    for name, (lookback, skip, direction) in specs.items():
        frame[name] = direction * (hist.iloc[-1-skip] / hist.iloc[-1-skip-lookback] - 1)
    valid = np.isfinite(frame).all(axis=1) & hist.iloc[-1].notna()
    valid &= hist.notna().sum().ge(int(np.ceil(length*minimum_history_fraction)))
    frame = frame.loc[valid].copy()
    return frame, {"base_names":len(symbols), "common_names":len(frame),
                   "excluded_asof":len(symbols)-len(frame), "history_sessions":length,
                   "history_start":hist.index[0], "history_end":hist.index[-1]}
