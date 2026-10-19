"""Read-only execution and allocation diagnostics; never an admission decision."""
from itertools import combinations
import numpy as np
import pandas as pd


def cost_summary(trades):
    """Canonical fill records, not broker orders. Monetary units must be consistent."""
    required = ['date','symbol','side','notional','fee','shares_before','shares_after','price','pre_nav']
    if not set(required).issubset(trades):
        raise ValueError('missing canonical trade fields')
    t = trades.copy()
    t['date'] = pd.to_datetime(t.date, errors='raise')
    if t.date.isna().any() or t.symbol.isna().any() or not t.side.isin(['BUY','SELL']).all():
        raise ValueError('invalid trade keys/side')
    nums = ['notional','fee','shares_before','shares_after','price','pre_nav']
    if (not np.isfinite(t[nums]).all().all() or (t.fee < 0).any()
            or (t[['shares_before','shares_after']] < -1e-12).any().any()
            or (t[['notional','price','pre_nav']] <= 0).any().any()):
        raise ValueError('invalid trade amounts')
    delta = t.shares_after-t.shares_before
    expected = np.where(t.side.eq('BUY'), 1., -1.)*t.notional
    if not np.allclose(delta*t.price, expected, atol=1e-9, rtol=1e-8):
        raise ValueError('share/notional/side mismatch')
    if t.groupby('date').pre_nav.nunique().gt(1).any():
        raise ValueError('one pretrade NAV per date required')
    t['category'] = np.where(t.shares_before.le(1e-12), 'entry', np.where(t.shares_after.le(1e-12), 'exit', 'rebalance'))
    categories = {name:dict(records=int(len(g)), notional=float(g.notional.sum()), fees=float(g.fee.sum()))
                  for name,g in t.groupby('category')}
    return dict(status='ASSESSED', fill_records=len(t), order_count=None,
                order_count_status='NOT_ASSESSED_FROM_FILLS',
                buy_notional=float(t.loc[t.side.eq('BUY'),'notional'].sum()),
                sell_notional=float(t.loc[t.side.eq('SELL'),'notional'].sum()), fees=float(t.fee.sum()),
                cumulative_two_way_turnover=float((t.notional/t.pre_nav).sum()), categories=categories,
                cost_scope='only supplied fees; no inferred minimum commission/tax/impact')


def portfolio_diagnostics(weights):
    """date,strategy,symbol,weight (fractions of TOTAL NAV), optional industry.

    Single snapshot kind per call: target or actual, explicitly declared by caller.
    Cash is inferred as 1 - stock weights; all-cash portfolios have no rows and
    are therefore NOT_ASSESSED, not silently inferred to have zero concentration.
    """
    w = weights.copy()
    required = ['date','strategy','symbol','weight']
    if not set(required).issubset(w) or w.empty:
        raise ValueError('nonempty allocation fields required')
    w.date = pd.to_datetime(w.date, errors='raise')
    if w[required[:3]].isna().any().any() or w.duplicated(required[:3]).any():
        raise ValueError('duplicate/missing allocation keys')
    if not np.isfinite(w.weight).all() or (w.weight <= 0).any():
        raise ValueError('positive finite stock weights required')
    summaries, pairs = [], []
    for (date,strategy),g in w.groupby(['date','strategy']):
        budget = float(g.weight.sum())
        if budget > 1+1e-10:
            raise ValueError('stock budget exceeds NAV')
        normalized = g.weight/budget
        industries = None
        if 'industry' in g:
            industries = g.assign(industry=g.industry.fillna('UNKNOWN').replace('', 'UNKNOWN')).groupby('industry').weight.sum().to_dict()
        summaries.append(dict(date=str(date.date()), strategy=strategy, stock_count=len(g), stock_budget=budget,
            cash_weight=max(0.,1-budget), effective_stock_count=float(1/(normalized**2).sum()),
            max_stock_weight=float(g.weight.max()), industry_weights=industries))
    strategies = sorted(w.strategy.unique())
    for date,g in w.groupby('date'):
        books = {s: z.set_index('symbol').weight for s,z in g.groupby('strategy')}
        for a,b in combinations(strategies,2):
            row = dict(date=str(date.date()), left=a, right=b)
            if a not in books or b not in books:
                pairs.append(dict(**row, status='NOT_ASSESSED', reason='missing allocation snapshot (possibly cash)'))
                continue
            x,y=books[a],books[b]; names=x.index.union(y.index)
            pairs.append(dict(**row,status='ASSESSED',overlap_names=len(x.index.intersection(y.index)),
                common_capital_weight=float(np.minimum(x.reindex(names,fill_value=0),y.reindex(names,fill_value=0)).sum())))
    return dict(status='ASSESSED', snapshots=summaries, overlap=pairs,
                convention='weights/total NAV; effective count normalized within equity; no automatic account netting')


def return_correlations(nav, pairs):
    from analysis.research_audit import validate_nav
    validate_nav(nav)
    if any(s not in nav or b not in nav for s,b in pairs.items()):
        raise ValueError('missing strategy/benchmark NAV')
    r=nav.pct_change(fill_method=None).iloc[1:]
    active=pd.DataFrame({s:r[s]-r[b] for s,b in pairs.items()})
    records=[]
    for a,b in combinations(pairs,2):
        raw=r[a].corr(r[b]); excess=active[a].corr(active[b])
        records.append(dict(left=a,right=b,raw_correlation=float(raw) if np.isfinite(raw) else None,
                            active_correlation=float(excess) if np.isfinite(excess) else None))
    return records
