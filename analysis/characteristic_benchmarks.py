"""Local characteristic-matched comparator targets, not original DGTW CS/CT/AS."""
import numpy as np
import pandas as pd

FEATURES = ['total_mv_yuan', 'VALUE_BP', 'MOM_12_1']
LEVELS = ['SBM', 'SB', 'S', 'ALL']


def bins_for(values, bins):
    """Equal-count deterministic bins; ties broken by symbol, no return labels."""
    if bins < 2 or int(bins) != bins or values.empty or values.index.has_duplicates:
        raise ValueError('invalid bin input')
    if not np.isfinite(values).all():
        raise ValueError('bins require finite values')
    order = values.rename('value').rename_axis('symbol').reset_index().sort_values(
        ['value', 'symbol'], kind='stable').symbol
    return pd.Series(np.floor(np.arange(len(order))*bins/len(order)).astype(int)+1,
                     index=order).reindex(values.index)


def memberships(frame, bins):
    if frame.empty or frame.index.has_duplicates or not set(FEATURES).issubset(frame):
        raise ValueError('invalid characteristic sample')
    f = frame.copy().sort_index()
    if not np.isfinite(f[FEATURES[:2]]).all().all() or not f[FEATURES[:2]].gt(0).all().all():
        raise ValueError('size and BP must be finite positive for every original stock')
    if np.isinf(f.MOM_12_1).any():
        raise ValueError('infinite momentum')
    f['size_bin'] = bins_for(f.total_mv_yuan, bins)
    f['bp_bin'] = 0
    for _, g in f.groupby('size_bin'):
        f.loc[g.index, 'bp_bin'] = bins_for(g.VALUE_BP, bins)
    f['momentum_bin'] = 0  # zero explicitly means unavailable, not bottom-ranked.
    for _, g in f.groupby(['size_bin', 'bp_bin']):
        g = g[g.MOM_12_1.notna()]
        if not g.empty:
            f.loc[g.index, 'momentum_bin'] = bins_for(g.MOM_12_1, bins)
    f['peer_eligible'] = f.MOM_12_1.notna()
    for col in FEATURES:
        f[col+'_rank'] = f[col].rank(method='average', pct=True)
    return f


def matching_plan(classified, target, *, minimum_peers=20):
    """Exclude each target itself, but not all other original portfolio names.

    Returns a comparator weight Series plus per-target coverage/mismatch audit.
    Never receives prices after signal or future return labels.
    """
    f = classified
    w = pd.Series(target, dtype=float)
    if (minimum_peers < 1 or int(minimum_peers) != minimum_peers or w.empty
            or w.index.has_duplicates or not set(w.index).issubset(f.index)
            or not np.isfinite(w).all() or not w.gt(0).all()
            or not np.isclose(w.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError('invalid source weights or peer count')
    if f.index.has_duplicates:
        raise ValueError('duplicate classified symbol')
    cols = ['size_bin', 'bp_bin', 'momentum_bin']
    peer = f[f.peer_eligible]
    cache = {}
    for level, groupcols in [('SBM',cols), ('SB',cols[:2]), ('S',cols[:1])]:
        for key, g in peer.groupby(groupcols):
            key = key if isinstance(key, tuple) else (key,)
            cache[(level, key)] = f.index.get_indexer(g.index)
    cache[('ALL', ())] = f.index.get_indexer(peer.index)
    ranks = f[[c+'_rank' for c in FEATURES]].to_numpy(float)
    rank_sums = {k: ranks[pos].sum(axis=0) for k,pos in cache.items()}
    result = np.zeros(len(f)); audits = []
    for symbol, weight in w.items():
        own = f.index.get_loc(symbol); row = f.loc[symbol]
        choices = [('SBM',tuple(int(row[c]) for c in cols)),
                   ('SB',tuple(int(row[c]) for c in cols[:2])),
                   ('S',(int(row.size_bin),)), ('ALL',())]
        for level, key in choices:
            pos = cache.get((level,key), np.array([], dtype=int))
            included = bool(row.peer_eligible) and own in pos
            count = len(pos)-int(included)
            if count >= minimum_peers:
                break
        else:
            raise ValueError('insufficient peers even after global fallback')
        result[pos] += weight/count
        if included:
            result[own] -= weight/count
        means = (rank_sums[(level,key)]-(ranks[own] if included else 0))/count
        audit = dict(symbol=symbol, weight=weight, level=level,
                     cell=':'.join(map(str,key)), peers=count,
                     self_excluded=included, missing_target_momentum=not bool(row.peer_eligible))
        for j,col in enumerate(FEATURES):
            audit[col+'_target_rank'] = ranks[own,j]
            audit[col+'_peer_rank'] = means[j]
            audit[col+'_abs_rank_gap'] = abs(means[j]-ranks[own,j])
        audits.append(audit)
    if result.min() < -1e-12 or not np.isclose(result.sum(), 1, rtol=0, atol=1e-10):
        raise AssertionError('comparator budget mismatch')
    output = pd.Series(result, index=f.index, name='weight')
    output = output[output.gt(1e-15)]
    if not np.isclose(output.sum(),1,rtol=0,atol=1e-10):
        raise AssertionError('dust removal lost budget')
    return output, pd.DataFrame(audits)


def arithmetic_decomposition(strategy, matched, market):
    if not strategy.index.equals(matched.index) or not strategy.index.equals(market.index):
        raise ValueError('unaligned NAVs')
    nav = pd.DataFrame({'strategy':strategy,'matched':matched,'market':market})
    if not np.isfinite(nav).all().all() or not nav.gt(0).all().all():
        raise ValueError('invalid NAV values')
    r = nav.pct_change(fill_method=None).iloc[1:]
    d = pd.DataFrame({'total_active':r.strategy-r.market,
                      'matched_residual':r.strategy-r.matched,
                      'matched_style':r.matched-r.market})
    if not np.allclose(d.total_active,d.matched_residual+d.matched_style,atol=1e-14,rtol=0):
        raise AssertionError('daily arithmetic identity failed')
    return d


def build_matched_targets(features, targets, *, bins=3, minimum_peers=20):
    """PIT feature snapshots + frozen fully-invested plans -> comparator plans.

    Canonical keys: date, symbol; targets additionally strategy, execution_date,
    weight. No prices/outcomes used. Reject incomplete snapshots, never shrink
    the target universe. Disclosure/vintage checks belong in feature preparation.
    """
    f, t = features.copy(), targets.copy()
    for frame in [f, t]:
        frame['date'] = pd.to_datetime(frame.date, errors='raise')
        if frame[['date', 'symbol']].isna().any().any():
            raise ValueError('missing snapshot keys')
    t['execution_date'] = pd.to_datetime(t.execution_date, errors='raise')
    if (f.duplicated(['date', 'symbol']).any() or t.duplicated(['date', 'strategy', 'symbol']).any()
            or t.strategy.isna().any() or t.execution_date.isna().any()
            or not t.execution_date.gt(t.date).all()):
        raise ValueError('invalid target dates/keys')
    plans, audits = [], []
    for date, day in t.groupby('date'):
        if day.execution_date.nunique() != 1:
            raise ValueError('ambiguous execution date')
        snapshot = f.loc[f.date.eq(date)].set_index('symbol')
        classified = memberships(snapshot, bins)
        for strategy, group in day.groupby('strategy'):
            weights, audit = matching_plan(classified, group.set_index('symbol').weight, minimum_peers=minimum_peers)
            plan = weights.rename_axis('symbol').reset_index()
            plan['date'], plan['execution_date'], plan['strategy'] = date, group.execution_date.iloc[0], strategy
            audit['date'], audit['strategy'] = date, strategy
            plans.append(plan); audits.append(audit)
    if not plans:
        raise ValueError('empty targets')
    return pd.concat(plans, ignore_index=True), pd.concat(audits, ignore_index=True)
