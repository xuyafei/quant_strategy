"""Explicit research-only target replay; no broker calls and no live defaults.

Fractional adjusted-close matching, single proportional fee, no lots/limits/tax.
None target = HOLD; empty Series = CASH; weights sum <= 1 reserve cash.
Quotes are never forward-filled for fills. Stale marks value existing holdings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.holding_buffer import ordered_scores, select_buffer


def replay(prices, schedule, target_factory, *, fee=.0003):
    """Call factory(signal_date, actual_held_symbols) after signal-close valuation.

    Schedule index is signal_date, column execution_date must be next market day.
    All-missing nonempty targets explicitly HOLD (legacy research parity).
    Target factory is trusted caller code and must enforce universe/risk rules.
    """
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.empty or
            prices.index.hasnans or prices.index.has_duplicates or not prices.index.is_monotonic_increasing or
            prices.columns.has_duplicates or prices.columns.isna().any()):
        raise ValueError('invalid price axes')
    px = prices.to_numpy(float)
    if np.isinf(px).any() or (px[np.isfinite(px)] <= 0).any() or not np.isfinite(fee) or not 0 <= fee < 1:
        raise ValueError('invalid prices or fee')
    schedule = schedule.copy()
    schedule.index = pd.to_datetime(schedule.index, errors='raise')
    schedule['execution_date'] = pd.to_datetime(schedule.execution_date, errors='raise')
    if schedule.index.has_duplicates or schedule.index.hasnans or schedule.execution_date.isna().any() or schedule.execution_date.duplicated().any():
        raise ValueError('duplicate/missing schedule')
    for signal, row in schedule.iterrows():
        pos = prices.index.get_indexer([signal])[0]
        if pos < 0 or pos+1 >= len(prices) or row.execution_date != prices.index[pos+1]:
            raise ValueError('execution must be next market day')
    marks = prices.ffill().fillna(0).to_numpy(float)
    names = prices.columns
    shares, cash = np.zeros(len(names)), 1.
    pending, daily, executions, trades, plans = {}, [], [], [], []
    for day, date in enumerate(prices.index):
        quote, mark = px[day], marks[day]
        valid = np.isfinite(quote)
        safe = np.where(valid, quote, 0.)
        pre = float(cash + shares @ mark)
        if date in pending:
            signal, weights = pending.pop(date)
            before, old_cash = shares.copy(), cash
            buy, sell = np.zeros(len(names)), np.zeros(len(names))
            action = 'HOLD' if weights is None else ('CASH' if weights.empty else 'TARGET')
            target = np.zeros(len(names))
            positions = np.array([], dtype=int) if weights is None else names.get_indexer(weights.index)
            if weights is not None:
                target[positions] = weights.to_numpy(float)
            all_missing = action == 'TARGET' and not valid[positions].any()
            if action != 'HOLD' and not all_missing:
                dollars = pre * target
                sell = np.maximum(shares*safe-dollars, 0.)
                sell[sell <= 1e-9] = 0.
                cash += sell.sum()*(1-fee)
                shares -= np.divide(sell, safe, out=np.zeros_like(sell), where=safe > 0)
                need = np.maximum(dollars-shares*safe, 0.)*valid
                need[need <= 1e-9] = 0.
                total = need.sum()*(1+fee)
                buy = need*(min(1., cash/total) if total > 0 else 1.)
                cash -= buy.sum()*(1+fee)
                shares += np.divide(buy, safe, out=np.zeros_like(buy), where=safe > 0)
            paid = float((buy.sum()+sell.sum())*fee)
            post = float(cash + shares @ mark)
            if cash < -1e-9 or not np.isclose(post, pre-paid, rtol=1e-10, atol=1e-10):
                raise AssertionError('NAV reconciliation failed')
            if not np.isclose(cash, old_cash+sell.sum()-buy.sum()-paid, atol=1e-10):
                raise AssertionError('cash reconciliation failed')
            blocked = (before*mark > 1e-10) & ~valid & (target == 0)
            executions.append(dict(date=date, signal_date=signal, action=action, pre_nav=pre, post_nav=post,
                cash=cash, buy_notional=float(buy.sum()), sell_notional=float(sell.sum()), fees=paid,
                two_way_turnover=float((buy.sum()+sell.sum())/pre), cash_fraction=cash/post,
                holdings=int((shares*mark > 1e-10).sum()), all_targets_missing_hold=all_missing,
                missing_targets=int((~valid[positions]).sum()), blocked_exit_value=float((before*mark*blocked).sum())))
            for j in np.flatnonzero(buy+sell > 0):
                trades.append(dict(date=date, symbol=names[j], side='BUY' if buy[j] else 'SELL',
                    notional=float(buy[j]+sell[j]), fee=float((buy[j]+sell[j])*fee),
                    shares_before=float(before[j]), shares_after=float(shares[j]), price=float(quote[j]), pre_nav=pre))
        daily.append(dict(date=date, pre_nav=pre, post_nav=float(cash+shares@mark), cash=cash,
                          unquoted_value=float((shares*mark*~valid).sum())))
        if date in schedule.index:
            held = list(names[shares*mark > 1e-10])
            target = target_factory(date, held)
            if target is not None:
                target = pd.Series(target, dtype=float)
                if (target.index.has_duplicates or target.index.isna().any() or
                    not target.index.isin(names).all() or not np.isfinite(target).all() or
                    (target <= 0).any() or target.sum() > 1+1e-12):
                    raise ValueError('invalid target; positive unique weights summing <= 1 required')
            execution = schedule.loc[date, 'execution_date']
            pending[execution] = (date, target)
            for symbol, weight in ([] if target is None else target.items()):
                plans.append(dict(date=date, execution_date=execution, symbol=symbol, weight=weight,
                                  was_actually_held=symbol in held))
    return dict(daily=pd.DataFrame(daily).set_index('date'), executions=pd.DataFrame(executions),
                trades=pd.DataFrame(trades, columns=['date','symbol','side','notional','fee','shares_before','shares_after','price','pre_nav']),
                targets=pd.DataFrame(plans, columns=['date','execution_date','symbol','weight','was_actually_held']))


def replay_ranked(prices, scores, schedule, *, top_k, exit_rank=None, fee=.0003):
    """No buffer unless exit_rank is explicit. Scores must be pre-filtered PIT scores.

    Missing schedule signals HOLD, but a present insufficient cross-section errors.
    """
    if not isinstance(scores.index, pd.MultiIndex) or scores.index.nlevels != 2 or scores.index.has_duplicates:
        raise ValueError('unique (signal_date, symbol) scores required')
    groups = {pd.Timestamp(d): g.droplevel(0) for d,g in scores.groupby(level=0)}
    if not set(groups).issubset(set(pd.to_datetime(schedule.index))):
        raise ValueError('scores outside declared schedule')
    decisions = []
    def factory(date, held):
        if date not in groups:
            return None
        ranked = ordered_scores(groups[date])
        chosen = select_buffer(ranked, held, top_k, top_k if exit_rank is None else exit_rank)
        rank = {s:i+1 for i,s in enumerate(ranked.index)}
        for s in chosen:
            decisions.append(dict(date=date, symbol=s, rank=rank[s], was_actually_held=s in held,
                                  reason='retained' if s in held else 'entry'))
        return pd.Series(1/top_k, index=chosen)
    result = replay(prices, schedule, factory, fee=fee)
    result['decisions'] = pd.DataFrame(decisions)
    return result
