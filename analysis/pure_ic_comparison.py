"""Research-only global positive-mean-IC weights and explicit cash targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pure_ic_weights(ic, outcome_end, signal, *, lookback=252, minimum=120):
    """No equal fallback, category budgets, sign reversal, or weight floors."""
    if minimum < 1 or lookback < minimum or not ic.index.is_monotonic_increasing or ic.index.has_duplicates:
        raise ValueError('invalid window or IC calendar')
    signal = pd.Timestamp(signal)
    end = pd.to_datetime(outcome_end.reindex(ic.index))
    known = ic.loc[(ic.index < signal) & end.lt(signal)].tail(lookback)
    count, mean = known.count(), known.mean()
    positive = mean.fillna(0).clip(lower=0).where(count.ge(minimum), 0)
    weights = positive / positive.sum() if positive.sum() > 1e-15 else positive * 0
    records = [{'signal_date': signal, 'factor': c, 'valid_ic_days': int(count[c]),
        'mean_ic': mean[c], 'weight': weights[c],
        'reason': 'insufficient_history' if count[c] < minimum else ('positive_ic' if weights[c] > 0 else 'nonpositive_ic'),
        'window_start': known.index.min(), 'window_end': known.index.max(),
        'latest_outcome_end': end.reindex(known.index).max()} for c in ic.columns]
    return weights, records


def pure_score(standardized, weights, raw_count=None, minimum_components=1):
    """One factor-level blend, with available-weight renormalization only."""
    if weights.isna().any() or (weights < 0).any():
        raise ValueError('invalid weights')
    w = weights[weights.gt(0)]
    if w.empty:
        return pd.Series(dtype=float, index=standardized.index[:0], name='score')
    x = standardized[w.index]
    denom = x.notna().astype(float).dot(w)
    score = x.mul(w).sum(axis=1, min_count=1).div(denom.where(denom.gt(0)))
    if raw_count is not None:
        score = score.where(raw_count.ge(minimum_components))
    return score.replace([np.inf, -np.inf], np.nan).dropna().rename('score')


def replay_actions(prices, execution_indices, selections, fee=.0003, marks=None):
    """None=HOLD, []=CASH, nonempty=equal-weight target. Missing quotes block fills."""
    px = np.asarray(prices, dtype=float)
    ex = np.asarray(execution_indices, dtype=int)
    if len(ex) != len(selections) or (np.diff(ex) <= 0).any() or (ex < 0).any() or (ex >= len(px)).any() or not 0 <= fee < 1:
        raise ValueError('invalid replay inputs')
    marks = pd.DataFrame(px).ffill().fillna(0).to_numpy() if marks is None else marks
    shares, cash = np.zeros(px.shape[1]), 1.
    nav, log = np.ones(len(px)), []
    for i, day in enumerate(ex):
        selection = selections[i]
        names = None if selection is None else np.asarray(selection, dtype=int)
        if names is not None and (len(np.unique(names)) != len(names) or (names < 0).any() or (names >= px.shape[1]).any()):
            raise ValueError('invalid target names')
        quoted = np.isfinite(px[day]) & (px[day] > 0)
        pre = cash + shares @ marks[day]
        turnover = fees = 0.
        action = 'HOLD' if names is None else ('CASH' if not len(names) else 'TOPK')
        if names is not None and (not len(names) or quoted[names].any()):
            quote = np.where(quoted, px[day], 0.)
            target = np.zeros(px.shape[1])
            if len(names):
                target[names] = pre / len(names)
            sell = np.maximum(shares * quote - target, 0)
            sell[sell <= 1e-9] = 0
            shares -= np.divide(sell, quote, out=np.zeros_like(sell), where=quote > 0)
            cash += sell.sum() * (1-fee)
            buy = np.maximum(target - shares * quote, 0) * quoted
            buy[buy <= 1e-9] = 0
            cost = buy.sum() * (1+fee)
            buy *= min(1., cash/cost) if cost > 0 else 1.
            shares += np.divide(buy, quote, out=np.zeros_like(buy), where=quote > 0)
            cash -= buy.sum() * (1+fee)
            turnover = (sell.sum() + buy.sum()) / pre
            fees = (sell.sum() + buy.sum()) * fee
        post = cash + shares @ marks[day]
        if cash < -1e-9 or not np.isclose(post, pre-fees, atol=1e-10, rtol=1e-10):
            raise AssertionError('cash / fee reconciliation failed')
        log.append({'execution_index': int(day), 'action': action, 'pre': pre, 'post': post,
            'two_way_turnover': turnover, 'fees': fees, 'cash_fraction': cash/post,
            'holdings': int((shares > 1e-12).sum()),
            'blocked_unquoted_value': float((shares * marks[day] * ~quoted).sum())})
        stop = ex[i+1] if i+1 < len(ex) else len(px)
        active = np.flatnonzero(shares)
        nav[day:stop] = marks[day:stop, active] @ shares[active] + cash
    stranded = float((shares * marks[-1] * ~np.isfinite(px[-1])).sum())
    return nav, log, stranded
