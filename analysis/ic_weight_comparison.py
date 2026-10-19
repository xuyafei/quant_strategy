"""Research-only, matured-label IC weights. Does not alter forward trading."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.preprocess import cross_sectional_zscore


def execution_labels(prices: pd.DataFrame, holding_days: int = 5):
    """Signal d -> entry d+1 close -> exit d+1+h close; no filling quotes."""
    if holding_days < 1:
        raise ValueError("holding_days must be positive")
    label = prices.shift(-(holding_days + 1)) / prices.shift(-1) - 1
    end = pd.Series(prices.index, index=prices.index).shift(-(holding_days + 1))
    return label.replace([np.inf, -np.inf], np.nan), end


def daily_rank_ic(panel, labels, minimum_names=30):
    """Pairwise-complete Spearman on the contemporaneously eligible universe."""
    y = labels.reindex(panel.index)
    days = panel.index.get_level_values('date')
    values, counts = {}, {}
    for col in panel:
        x = panel[col].where(y.notna())
        yy = y.where(x.notna())
        rx, ry = x.groupby(days).rank(), yy.groupby(days).rank()
        a = rx - rx.groupby(days).transform('mean')
        b = ry - ry.groupby(days).transform('mean')
        n = x.groupby(days).count()
        denom = ((a*a).groupby(days).sum() * (b*b).groupby(days).sum()) ** .5
        values[col] = ((a*b).groupby(days).sum() / denom).where(n.ge(minimum_names) & denom.gt(0))
        counts[col] = n
    return pd.DataFrame(values), pd.DataFrame(counts)


def rolling_weights(ic, outcome_end, signal, families, *, lookback=252, minimum=120, fixed_share=.5):
    """Use only labels ending STRICTLY before signal. Equal fallback per family."""
    signal = pd.Timestamp(signal)
    if not 0 <= fixed_share <= 1 or minimum < 1 or lookback < minimum:
        raise ValueError('invalid weight configuration')
    end = pd.to_datetime(outcome_end.reindex(ic.index))
    known = ic.loc[(ic.index < signal) & end.lt(signal)].tail(lookback)
    records = []
    weights = {}
    for family, columns in families.items():
        sub = known.reindex(columns=columns)
        count, mean = sub.count(), sub.mean()
        equal = pd.Series(1/len(columns), index=columns)
        positive = mean.fillna(0).clip(lower=0)
        if len(columns) == 1:
            w, reason = equal, 'single_component'
        elif (count < minimum).any():
            w, reason = equal, 'insufficient_history'
        elif positive.sum() <= 1e-15:
            w, reason = equal, 'no_positive_mean_ic'
        else:
            w = fixed_share*equal + (1-fixed_share)*positive/positive.sum()
            reason = 'positive_mean_ic_with_fixed_shrinkage'
        if not np.isclose(w.sum(), 1) or (w <= 0).any():
            raise AssertionError('invalid nonzero family weights')
        weights.update(w.to_dict())
        for col in columns:
            records.append({'signal_date': signal, 'family': family, 'factor': col,
                'valid_ic_days': int(count[col]), 'mean_ic': mean[col],
                'within_family_weight': w[col], 'family_budget': 1/len(families),
                'nominal_weight': w[col]/len(families), 'reason': reason,
                'window_start': known.index.min() if len(known) else pd.NaT,
                'window_end': known.index.max() if len(known) else pd.NaT,
                'latest_outcome_end': end.reindex(known.index).max() if len(known) else pd.NaT})
    return pd.Series(weights), records


def combine_scores(standardized, families, weights, *, focused=False, raw_count=None, minimum_components=1):
    """Same existing family restandardization, with available-weight renormalization."""
    def average(cols):
        x = standardized[cols]
        w = weights.reindex(cols)
        denom = x.notna().astype(float).dot(w)
        return x.mul(w).sum(axis=1, min_count=1).div(denom.where(denom.gt(0)))
    if focused:
        cols = next(iter(families.values()))
        value = average(cols).where(raw_count.ge(minimum_components))
        return cross_sectional_zscore(value.to_frame('score'))['score'].dropna()
    family = pd.DataFrame({name: average(cols) for name, cols in families.items()})
    family = cross_sectional_zscore(family)
    return family.mean(axis=1).where(family.notna().sum(axis=1).eq(len(families))).dropna().rename('score')


def replay_universe(prices, execution_indices, selections, fee=.0003):
    """Variable-size equal-weight PIT benchmark, same fractional sell-first model."""
    px = np.asarray(prices, dtype=float)
    marks = pd.DataFrame(px).ffill().fillna(0).to_numpy()
    shares, cash = np.zeros(px.shape[1]), 1.
    nav = np.ones(len(px))
    for w, day in enumerate(execution_indices):
        target_names = np.asarray(selections[w], dtype=int)
        if not len(target_names) or len(np.unique(target_names)) != len(target_names) or (target_names < 0).any():
            raise ValueError('invalid benchmark selection')
        quoted = np.isfinite(px[day])
        if quoted[target_names].any():
            pre = cash + shares @ marks[day]
            target = np.zeros(px.shape[1]); target[target_names] = pre/len(target_names)
            quote = np.where(quoted, px[day], 0.)
            sell = np.maximum(shares*quote-target, 0); sell[sell <= 1e-9] = 0
            shares -= np.divide(sell, quote, out=np.zeros_like(sell), where=quote > 0)
            cash += sell.sum()*(1-fee)
            buy = np.maximum(target-shares*quote, 0)*quoted; buy[buy <= 1e-9] = 0
            cost = buy.sum()*(1+fee)
            buy *= min(1., cash/cost) if cost > 0 else 1.
            shares += np.divide(buy, quote, out=np.zeros_like(buy), where=quote > 0)
            cash -= buy.sum()*(1+fee)
            if cash < -1e-9 or not np.isclose(cash+shares@marks[day], pre-(sell.sum()+buy.sum())*fee):
                raise AssertionError('benchmark reconciliation failed')
        stop = execution_indices[w+1] if w+1 < len(execution_indices) else len(px)
        active = np.flatnonzero(shares)
        nav[day:stop] = marks[day:stop, active] @ shares[active]+cash
    return nav
