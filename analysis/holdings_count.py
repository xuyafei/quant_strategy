"""Research-only batch execution and coverage adapter; no live defaults changed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from universe.dynamic import _stock_master, _st_intervals


def eligible_base(close, amount, master, names, decisions):
    """Vectorized BASE_V1; emit only passing rows for the existing risk overlay."""
    master = _stock_master(master).reindex(close.columns)
    valid = close.notna().rolling(20, min_periods=1).sum()
    adv = amount.where(close.notna()).rolling(20, min_periods=1).mean()
    positions = np.searchsorted(close.index.values, master.list_date.values.astype('datetime64[ns]'))
    intervals = _st_intervals(names)
    frames = []
    for date in decisions:
        age = close.index.get_loc(date) + 1 - positions
        st = intervals[intervals.effective_from.le(date) & (intervals.effective_to.isna() | intervals.effective_to.ge(date))]
        mask = (master.list_date.le(date) & (master.delist_date.isna() | master.delist_date.gt(date))
                & ~master.index.isin(st.symbol) & close.loc[date].notna() & (age >= 250)
                & valid.loc[date].ge(15) & adv.loc[date].ge(50_000_000))
        symbols = master.index[mask]
        frames.append(pd.DataFrame({'date': date, 'symbol': symbols, 'eligible': True,
            'exclude_reason': '', 'listed_sessions': age[mask],
            'valid_days_20': valid.loc[date, symbols].to_numpy(),
            'adv20_yuan': adv.loc[date, symbols].to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def replay_batch(prices, execution_indices, plans, fee=.0003):
    """Replay W x R x K fixed selections; return daily POST-trade NAV and audit.

    Same sell-first/cash-scaled fractional matching as _rebalance_to_target_weights.
    Missing execution quotes never cause a replacement or target renormalization.
    The complete target is formed at signal close. No daily weight rebalancing.
    """
    px = np.asarray(prices, dtype=float)
    plans = np.asarray(plans)
    ex = np.asarray(execution_indices, dtype=int)
    if px.ndim != 2 or plans.ndim != 3 or len(ex) != len(plans) or not len(ex):
        raise ValueError('invalid shapes')
    if not np.issubdtype(plans.dtype, np.integer) or (plans < 0).any() or (plans >= px.shape[1]).any():
        raise ValueError('invalid selections')
    if (np.diff(ex) <= 0).any() or ex[0] < 1 or ex[-1] >= len(px):
        raise ValueError('invalid execution dates')
    if np.isinf(px).any() or (px[np.isfinite(px)] <= 0).any() or not 0 <= fee < 1:
        raise ValueError('invalid prices/cost')
    if (np.diff(np.sort(plans, axis=2), axis=2) == 0).any():
        raise ValueError('duplicate selection')
    marks = pd.DataFrame(px).ffill().fillna(0).to_numpy()
    _, replicates, k = plans.shape
    shares = np.zeros((replicates, px.shape[1]))
    cash = np.ones(replicates)
    nav = np.ones((len(px), replicates))
    audits = []
    for w, day in enumerate(ex):
        active = np.union1d(np.flatnonzero((shares != 0).any(axis=0)), plans[w].ravel())
        held = shares[:, active].copy()
        quote, mark = px[day, active], marks[day, active]
        quoted = np.isfinite(quote)
        safe = np.where(quoted, quote, 0.)
        pre = cash + held @ mark
        target = np.zeros_like(held)
        pick_idx = np.searchsorted(active, plans[w])
        target[np.arange(replicates)[:, None], pick_idx] = pre[:, None] / k
        cur = held * safe
        sell = np.maximum(cur - target, 0.)
        sell[sell <= 1e-9] = 0.
        cash += sell.sum(axis=1) * (1-fee)
        held -= np.divide(sell, safe, out=np.zeros_like(sell), where=safe > 0)
        need = np.maximum(target - held * safe, 0.) * quoted
        need[need <= 1e-9] = 0.
        cost = need.sum(axis=1) * (1+fee)
        scale = np.minimum(1., np.divide(cash, cost, out=np.ones_like(cash), where=cost > 0))
        buy = need * scale[:, None]
        cash -= buy.sum(axis=1) * (1+fee)
        held += np.divide(buy, safe, out=np.zeros_like(buy), where=safe > 0)
        # Existing helper holds everything when ALL targets have missing quotes.
        all_missing = ~np.isfinite(px[day, plans[w]]).any(axis=1)
        if all_missing.any():
            cash[all_missing] = pre[all_missing] - shares[all_missing][:, active] @ mark
            held[all_missing] = shares[all_missing][:, active]
            buy[all_missing] = 0.; sell[all_missing] = 0.
        shares[:, active] = held
        paid = (buy.sum(axis=1) + sell.sum(axis=1)) * fee
        post = cash + held @ mark
        if not np.allclose(post, pre-paid, atol=1e-10, rtol=1e-10) or cash.min() < -1e-8:
            raise AssertionError('cash/fee reconciliation failed')
        stop = ex[w+1] if w+1 < len(ex) else len(px)
        nav[day:stop] = marks[day:stop, active] @ held.T + cash
        audits.append({'execution_index': int(day), 'pre': pre, 'post': post,
            'two_way_turnover': (buy.sum(axis=1)+sell.sum(axis=1))/pre,
            'fees': paid, 'cash_fraction': cash/post,
            'holdings': (held*mark > 1e-10).sum(axis=1),
            'missing_targets': (~np.isfinite(px[day, plans[w]])).sum(axis=1),
            'blocked_exit_value': ((held*mark)*(~quoted)*(target == 0)).sum(axis=1)/post})
    stranded = (shares * marks[-1] * ~np.isfinite(px[-1])).sum(axis=1)
    return nav, audits, stranded


def metrics(nav):
    """Daily post-fee returns, 252 sessions/year, zero risk-free-rate Sharpe."""
    a = np.asarray(nav, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if len(a) < 3 or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('invalid NAV')
    r = a[1:]/a[:-1]-1
    vol = r.std(axis=0, ddof=1)*np.sqrt(252)
    return {'total_return': a[-1]/a[0]-1,
        'cagr': (a[-1]/a[0])**(252/(len(a)-1))-1,
        'volatility': vol,
        'sharpe': np.divide(r.mean(axis=0)*252, vol, out=np.zeros_like(vol), where=vol > 0),
        'max_drawdown': (a/np.maximum.accumulate(a, axis=0)-1).min(axis=0)}


def nested_random_plans(ranked_pools, k_values, repeats, seed):
    """Same random ordering for every K; no realized outcomes enter selection."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in k_values}
    for pool in ranked_pools:
        pool = np.asarray(pool, dtype=int)
        if len(pool) < max(k_values) or len(np.unique(pool)) != len(pool):
            raise ValueError('insufficient/duplicate pool')
        random = np.stack([rng.permutation(pool) for _ in range(repeats)])
        for k in k_values:
            out[k].append(np.vstack([pool[:k], random[:, :k]]))
    return {k: np.stack(v) for k, v in out.items()}
