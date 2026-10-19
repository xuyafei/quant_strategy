#!/usr/bin/env python3
"""Long-history K sensitivity of the existing six fixed-score MVP strategies.

Run --stage prepare, then --stage backtest. Checkpoints are integrity checked;
never writes to the forward accounts or changes production defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from analysis.holdings_count import eligible_base, replay_batch, metrics, nested_random_plans
from scripts.build_dynamic_universe_comparison import build_weekly_raw_factor_panel
from strategies.style import build_fixed_family_score, build_focused_style_score
from universe.dynamic import eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import (build_strategy_universe_report, profiles_from_config,
    eligibility_for_profile, point_in_time_industry, daily_basic_for_index)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n')


def load_matrix(path, fields, dates, symbols, multipliers=None):
    matrices = {f: np.full((len(dates), len(symbols)), np.nan) for f in fields}
    for chunk in pd.read_csv(path, usecols=['trade_date', 'ts_code']+fields,
            dtype={'trade_date': str, 'ts_code': str}, chunksize=300000):
        di = dates.get_indexer(pd.to_datetime(chunk.trade_date, format='%Y%m%d'))
        si = symbols.get_indexer(chunk.ts_code)
        ok = (di >= 0) & (si >= 0)
        for f in fields:
            matrices[f][di[ok], si[ok]] = pd.to_numeric(chunk.loc[ok, f], errors='coerce')*(multipliers or {}).get(f, 1.)
    return {f: pd.DataFrame(v, index=dates, columns=symbols) for f, v in matrices.items()}


def code_hashes():
    names = ['scripts/run_mvp_holdings_count.py', 'analysis/holdings_count.py',
        'scripts/build_dynamic_universe_comparison.py', 'strategies/style.py',
        'factors/preprocess.py', 'universe/dynamic.py', 'universe/risk.py', 'universe/strategy.py']
    return {n: sha(ROOT/n) for n in names}


def prepare(protocol, out):
    p = json.loads(protocol.read_text())
    source = ROOT/p['data_source']
    parent = json.loads((ROOT/p['parent_protocol']).read_text())
    integrity = json.loads((source/'aggregate_validation.json').read_text())
    if json.loads((source/'completion.json').read_text())['status'] != 'DOWNLOAD_COMPLETE':
        raise ValueError('incomplete source')
    hashes = {n: sha(source/n) for n in integrity['files']}
    if any(hashes[n] != v['sha256'] for n, v in integrity['files'].items()):
        raise ValueError('raw source checksum mismatch')
    stamp = {'protocol': sha(protocol), 'parent': sha(ROOT/p['parent_protocol']),
             'raw': hashes, 'financial_supplement': sha(ROOT/p['financial_supplement']), 'code': code_hashes()}
    if (out/'started.json').exists():
        old = json.loads((out/'started.json').read_text())
        if old != stamp:
            raise ValueError('checkpoint inputs/code changed: use new output')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'started.json', stamp)
    write_json(out/'protocol.json', p)
    basic = pd.read_csv(source/'stock_basic.csv.gz', dtype=str)
    basic = basic[basic.ts_code.str.match(r'^(?:[036]\d{5}\.S[HZ])$') & ~basic.ts_code.isin(p['excluded_transition_codes'])]
    symbols = pd.Index(sorted(basic.ts_code.unique()), name='symbol')
    calendar = pd.read_csv(source/'trade_cal.csv.gz', dtype={'cal_date': str})
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.eq(1), 'cal_date'], format='%Y%m%d')).unique().sort_values()
    dates = dates[dates <= p['evaluation_end']]
    matrices = load_matrix(source/'daily.csv.gz', ['close', 'amount'], dates, symbols, {'amount': 1000.})
    close, amount = matrices['close'], matrices['amount']
    adj = load_matrix(source/'adj_factor.csv.gz', ['adj_factor'], dates, symbols)['adj_factor']
    adjusted = close*adj
    del adj, matrices
    if (adjusted.notna() != close.notna()).any().any():
        raise ValueError('unmatched adjustment')
    weeks = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max()
    weeks = weeks[(weeks.index.end_time.normalize() <= dates[-1]) & (weeks >= p['evaluation_start']) & (weeks < dates[-1])]
    decisions = pd.DatetimeIndex(weeks.to_numpy())
    names = pd.read_csv(source/'namechange.csv.gz')
    base = eligible_base(close, amount, basic, names, decisions)
    del amount
    print(f'raw matrices {adjusted.shape}; base rows {len(base)}; weeks {len(decisions)}', flush=True)
    fin_cols = ['ts_code','ann_date','end_date','eps','roe','grossprofit_margin','netprofit_margin',
                'debt_to_assets','or_yoy','netprofit_yoy','ocfps','ocf_to_profit']
    finance = pd.read_csv(ROOT/p['financial_supplement'], usecols=fin_cols)
    balance = pd.read_csv(source/'balancesheet_vip.csv.gz', usecols=lambda x: x in
        ['ts_code','ann_date','f_ann_date','end_date','report_type','update_flag','total_hldr_eqy_exc_min_int'])
    audit = pd.read_csv(source/'fina_audit.csv.gz')
    db = pd.read_csv(source/'daily_basic.csv.gz', usecols=['ts_code','trade_date','pe_ttm','pb','circ_mv','total_mv'], dtype={'trade_date': str})
    industry = pd.read_csv(source/'index_member_all.csv.gz')
    profiles = profiles_from_config(parent['strategy_universe_profiles'])
    families = parent['candidate_families']
    for year, base_year in base.groupby(base.date.dt.year):
        dest = out/f'inputs_{year}'
        if (dest/'complete.json').exists():
            record = json.loads((dest/'complete.json').read_text())
            if any(sha(dest/n) != v for n,v in record.items()):
                raise ValueError('checkpoint output changed')
            print(f'reuse verified {year}', flush=True)
            continue
        dest.mkdir(exist_ok=True)
        d = decisions[decisions.year == year]
        start = max(0, dates.get_loc(d[0])-90)
        stop = dates.get_loc(d[-1])+1
        long = close.iloc[start:stop].stack().rename('close').reset_index()
        long.columns = ['trade_date','ts_code','close']
        idx = pd.MultiIndex.from_frame(long[['trade_date','ts_code']])
        adjstack = adjusted.iloc[start:stop].stack()
        long['adj_close'] = adjstack.reindex(idx).to_numpy()
        raw = build_weekly_raw_factor_panel(long, finance, d)
        risk = build_hard_risk_universe_report(base_year, balance, audit, names)
        # Date-limited market data retains the complete seven-day as-of window.
        db_year = db[db.trade_date.ge((d[0]-pd.Timedelta(days=7)).strftime('%Y%m%d')) & db.trade_date.le(d[-1].strftime('%Y%m%d'))]
        report = build_strategy_universe_report(risk, db_year, raw, industry, profiles)
        masks = {'BASE_V2': eligibility_from_report(risk), **{v.name: eligibility_for_profile(report,v) for v in profiles}}
        ind = point_in_time_industry(raw.index, industry)
        market = daily_basic_for_index(raw.index, db_year)
        pe, pb = market.pe_ttm, market.pb
        raw['EARNINGS_YIELD'] = (1/pe).where(pe.gt(0))
        raw['BOOK_TO_PRICE'] = (1/pb).where(pb.gt(0))
        rankings, coverage = [], []
        for spec in parent['strategies']:
            mask = masks[spec['universe']]
            if spec['score'] == 'fixed_equal_five_family_multi_factor':
                score, _ = build_fixed_family_score(raw, mask, ind, families)
            else:
                score, _ = build_focused_style_score(raw, mask, ind, spec['factors'], minimum_components=spec['minimum_components'])
            frame = score.reset_index().sort_values(['date','score','symbol'], ascending=[True,False,True], kind='stable')
            frame['rank'] = frame.groupby('date').cumcount()+1
            frame['strategy'] = spec['name'].replace('_TOP50','')
            top = frame[frame['rank'] <= 100].copy()
            topidx = pd.MultiIndex.from_frame(top[['date','symbol']])
            top['industry'] = ind.reindex(topidx).to_numpy()
            rankings.append(top)
            counts = frame.groupby('date').size()
            aligned = mask.reindex(raw.index).fillna(False).astype(bool)
            for date in d:
                day_mask = aligned.loc[date]
                sub = raw.loc[date].loc[day_mask]
                ds = ind.loc[date].reindex(sub.index).fillna('')
                row = {'date': date, 'strategy': spec['name'].replace('_TOP50',''),
                    'eligible_names': int(day_mask.sum()), 'scored_names': int(counts.get(date,0)),
                    'industry_coverage': float(ds.ne('').mean())}
                row.update({f'valid_{c}': float(sub[c].notna().mean()) for c in raw.columns})
                coverage.append(row)
        pd.concat(rankings).to_csv(dest/'ranked_top100.csv.gz', index=False)
        pd.DataFrame(coverage).to_csv(dest/'coverage.csv', index=False)
        report.to_csv(dest/'universe.csv.gz', index=False)
        write_json(dest/'complete.json', {f.name: sha(f) for f in dest.iterdir() if f.is_file()})
        print(f'prepared {year}; scored min={min(r["scored_names"] for r in coverage)}', flush=True)
    np.savez_compressed(out/'prices.npz', values=adjusted.to_numpy(), dates=dates.to_numpy(), symbols=symbols.to_numpy(dtype=str))
    write_json(out/'prepared.json', {'finished': datetime.now(timezone.utc).isoformat(),
        'prices_sha256': sha(out/'prices.npz'), 'missing_financial_source_columns': sorted(set(fin_cols)-set(finance.columns)),
        'protocol_sha256': sha(protocol), 'parent_sha256': sha(ROOT/p['parent_protocol']), 'source_snapshot_not_vintage': True})
    print('PREPARATION COMPLETE', flush=True)


def backtest(protocol, out):
    p = json.loads(protocol.read_text())
    ready = json.loads((out/'prepared.json').read_text())
    if sha(protocol) != ready['protocol_sha256'] or sha(out/'prices.npz') != ready['prices_sha256']:
        raise ValueError('input integrity mismatch')
    ranks, cover = [], []
    for dest in sorted(out.glob('inputs_*')):
        record = json.loads((dest/'complete.json').read_text())
        if any(sha(dest/n) != v for n,v in record.items()):
            raise ValueError('year input integrity mismatch')
        ranks.append(pd.read_csv(dest/'ranked_top100.csv.gz', parse_dates=['date']))
        cover.append(pd.read_csv(dest/'coverage.csv', parse_dates=['date']))
    ranks = pd.concat(ranks, ignore_index=True)
    cover = pd.concat(cover, ignore_index=True)
    cover.to_csv(out/'coverage.csv', index=False)
    z = np.load(out/'prices.npz')
    dates, symbols = pd.DatetimeIndex(z['dates']), pd.Index(z['symbols'])
    px = z['values']
    # One common calendar interval; all arms start in cash. Insufficient-name
    # dates are reported and skipped equally across K within that strategy.
    first = dates.searchsorted(pd.Timestamp(p['evaluation_start']))
    dates, px = dates[first:], px[first:]
    rows, random_rows, period_rows, yearly_rows, audits, exposures = [], [], [], [], [], []
    navs, stress_navs, mc_navs = {}, {}, {}
    for sn, (strategy, frame) in enumerate(ranks.groupby('strategy', sort=True)):
        groups = [g.sort_values('rank') for _,g in frame.groupby('date', sort=True) if len(g) >= 50]
        signal = pd.DatetimeIndex([g.date.iloc[0] for g in groups])
        ex = dates.get_indexer(signal)+1
        if (ex < 1).any() or (ex >= len(dates)).any():
            raise ValueError('invalid signal lag')
        pools = [symbols.get_indexer(g.symbol) for g in groups]
        plans = nested_random_plans(pools, p['k_values'], p['random_subset']['repeats'], p['random_subset']['seed']+sn)
        for k in p['k_values']:
            nav, log, stranded = replay_batch(px, ex, plans[k], p['commission_rate'])
            stress, _, _ = replay_batch(px, ex, plans[k][:,:1], p['stress_one_way_cost'])
            key = f'{strategy}_K{k}'
            navs[key], stress_navs[key] = nav[:,0], stress[:,0]
            mc_navs[key] = nav[:,1:]
            m = metrics(nav)
            annual_turn = sum(a['two_way_turnover'][0] for a in log)*252/(len(dates)-1)
            row = {'strategy': strategy, 'k': k, 'start': str(dates[0].date()), 'end': str(dates[-1].date()),
                **{f: float(v[0]) for f,v in m.items()}, 'annual_two_way_turnover': float(annual_turn),
                'mean_cash_fraction': float(np.mean([a['cash_fraction'][0] for a in log])),
                'mean_actual_holdings': float(np.mean([a['holdings'][0] for a in log])),
                'terminal_unquoted_value_fraction': float(stranded[0]/nav[-1,0]),
                'terminal_zero_unquoted_total_return': float((nav[-1,0]-stranded[0])/nav[0,0]-1),
                **{'stress_'+f: float(v[0]) for f,v in metrics(stress).items()}}
            rows.append(row)
            for r in range(1, nav.shape[1]):
                random_rows.append({'strategy':strategy, 'k':k,'replicate':r-1, **{f:float(v[r]) for f,v in m.items()}})
            for (lo, hi) in p['subperiods']:
                idx = np.flatnonzero((dates >= lo)&(dates <= hi))
                if len(idx) > 2:
                    a = max(0,idx[0]-1); b = idx[-1]+1
                    period_rows.append({'strategy':strategy,'k':k,'period':lo[:4]+'-'+hi[:4], **{f:float(v[0]) for f,v in metrics(nav[a:b,:1]).items()}})
            for year in sorted(set(dates.year)):
                idx = np.flatnonzero(dates.year == year)
                a, b = max(0,idx[0]-1), idx[-1]+1
                yearly_rows.append({'strategy':strategy,'k':k,'year':year, **{f:float(v[0]) for f,v in metrics(nav[a:b,:1]).items()}})
            for w, a in enumerate(log):
                audits.append({'strategy':strategy,'k':k,'signal_date':signal[w], 'execution_date':dates[a['execution_index']],
                    **{n:float(v[0]) for n,v in a.items() if n != 'execution_index'}})
                counts = groups[w].head(k).industry.fillna('UNKNOWN').replace('', 'UNKNOWN').value_counts()/k
                exposures.append({'strategy':strategy,'k':k,'signal_date':signal[w],
                    'max_industry_target_weight': float(counts.max()), 'industry_hhi': float((counts**2).sum()),
                    'industry_count':len(counts), 'unknown_weight':float(counts.get('UNKNOWN',0))})
            print(f'{key}: CAGR {row["cagr"]:.2%}, Sharpe {row["sharpe"]:.3f}, MDD {row["max_drawdown"]:.2%}', flush=True)
    for name,data in [('performance',rows),('random_paths_metrics',random_rows),('subperiods',period_rows),
                      ('annual',yearly_rows),('execution_audit',audits),('concentration',exposures)]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'), index=False)
    pd.DataFrame(navs,index=dates).to_csv(out/'nav.csv',index_label='date')
    pd.DataFrame(stress_navs,index=dates).to_csv(out/'stress_nav.csv',index_label='date')
    np.savez_compressed(out/'random_daily_nav.npz', dates=dates.to_numpy(), **mc_navs)
    random = pd.DataFrame(random_rows)
    summaries=[]
    for (strategy,k),g in random.groupby(['strategy','k']):
        rec={'strategy':strategy,'k':k}
        for f in ['cagr','sharpe','max_drawdown','volatility']:
            q=g[f].quantile([.05,.5,.95])
            rec.update({f+'_p05':q.loc[.05], f+'_median':q.loc[.5], f+'_p95':q.loc[.95],f+'_width':q.loc[.95]-q.loc[.05]})
        summaries.append(rec)
    pd.DataFrame(summaries).to_csv(out/'random_summary.csv',index=False)
    write_json(out/'completed.json', {'completed':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(protocol),'code_hashes':code_hashes(),
        'output_hashes':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.suffix in ['.csv','.npz']},
        'live_approval':False,'strategies':6,'deterministic_paths':30,'random_paths':3000})
    print('BACKTEST COMPLETE',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare','backtest'], required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'research_protocols/20261017_mvp_holdings_count_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/mvp_holdings_count_20261017_complete_factors')
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else backtest)(args.protocol.resolve(),args.output.resolve())
