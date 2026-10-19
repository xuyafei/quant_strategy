#!/usr/bin/env python3
"""Frozen, offline six-strategy buffer comparison plus actual audit CLI output.

Writes ONLY a fresh output directory; never mutates paper ledgers or source data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from backtest.research_replay import replay_ranked
from analysis.holdings_count import metrics
from analysis.portfolio_diagnostics import cost_summary
from scripts.run_research_audit import run as run_audit


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def account_hashes():
    root=ROOT/'output/six_strategy_forward_20260928'
    return {str(f.relative_to(root)):sha(f) for folder in ['initial','plans','days']
            for f in sorted((root/folder).rglob('*')) if f.is_file()}


def run(protocol, output):
    p=json.loads(protocol.read_text());out=output.resolve()
    if out.exists():raise FileExistsError('fresh output required')
    if p['top_k']!=50 or p['exit_rank']!=75:raise ValueError('use a new reviewed protocol for changed parameters')
    source=ROOT/p['source'];pricefile=ROOT/p['prior_prices']
    inputfiles=[protocol,pricefile,source/'rankings.csv.gz',source/'nav.csv',source/'benchmark_nav.csv',source/'calendar.json',source/'completed.json']
    complete=json.loads((source/'completed.json').read_text())
    for name in ['rankings.csv.gz','nav.csv','benchmark_nav.csv']:
        if sha(source/name)!=complete['files'][name]:raise ValueError('archive checksum mismatch '+name)
    prepared=json.loads((pricefile.parent/'prepared.json').read_text())
    if sha(pricefile)!=prepared['prices_sha256']:raise ValueError('price checksum mismatch')
    inputfiles.append(pricefile.parent/'prepared.json')
    inputs={str(f.resolve()):sha(f) for f in inputfiles};before=account_hashes()
    code=['backtest/research_replay.py','strategies/holding_buffer.py','analysis/portfolio_diagnostics.py',
          'analysis/research_audit.py','analysis/holdings_count.py','scripts/run_research_audit.py','scripts/verify_season02_integration.py']
    code_before={n:sha(ROOT/n) for n in code}
    out.mkdir(parents=True);write_json(out/'protocol.json',p);write_json(out/'started.json',{'inputs':inputs,'code':code_before,'forward_before':before})
    z=np.load(pricefile,allow_pickle=False)
    prices=pd.DataFrame(z['values'],index=pd.DatetimeIndex(z['dates'],name='date'),columns=z['symbols'])
    old=pd.read_csv(source/'nav.csv',index_col='date',parse_dates=['date'])
    prices=prices.loc[old.index]
    ranks=pd.read_csv(source/'rankings.csv.gz',parse_dates=['date'])
    ranks=ranks[ranks.variant.eq('FIXED')].copy()
    calendar=json.loads((source/'calendar.json').read_text())
    signals=pd.DatetimeIndex(calendar['signals'])
    ex=prices.index.get_indexer(signals)+1
    if (ex<=0).any() or (ex>=len(prices)).any():raise ValueError('invalid signal calendar')
    schedule=pd.DataFrame({'execution_date':prices.index[ex]},index=signals)
    bench=pd.read_csv(source/'benchmark_nav.csv',index_col='date',parse_dates=['date'])
    if not prices.index.equals(bench.index):raise ValueError('benchmark calendar differs')
    navs={};stress_navs={};perf=[];periods=[];parity=[];trades=[];targets=[];pairs={}
    for strategy,g in ranks.groupby('strategy',sort=True):
        scores=g.set_index(['date','symbol']).score
        for variant,exit_rank in [('FIXED',None),('BUFFER',p['exit_rank'])]:
            name=f'{strategy}__{variant}'
            print('Replay',name,flush=True)
            result=replay_ranked(prices,scores,schedule,top_k=p['top_k'],exit_rank=exit_rank,fee=p['fee'])
            nav=result['daily'].post_nav;navs[name]=nav
            if variant=='FIXED':
                err=float((nav-old[f'{strategy}__FIXED__K50']).abs().max())
                if err>1e-8:raise AssertionError(f'baseline changed {strategy}: {err}')
                parity.append({'strategy':strategy,'maximum_nav_error':err})
            summary=cost_summary(result['trades']);m=metrics(nav)
            e=result['executions'];post=result['daily'].post_nav
            perf.append(dict(strategy=strategy,variant=variant,**{k:float(v[0]) for k,v in m.items()},
                cumulative_two_way_turnover=summary['cumulative_two_way_turnover'],fees=summary['fees'],
                fill_records=summary['fill_records'],mean_rebalance_cash=float(e.cash_fraction.mean()),
                mean_holdings=float(e.holdings.mean()),missing_targets=int(e.missing_targets.sum()),
                terminal_unquoted_fraction=float(result['daily'].unquoted_value.iloc[-1]/post.iloc[-1]),
                terminal_zero_unquoted_return=float(post.iloc[-1]-result['daily'].unquoted_value.iloc[-1]-1)))
            for start,end in p['subperiods']:
                loc=nav.index.searchsorted(pd.Timestamp(start));stop=nav.index.searchsorted(pd.Timestamp(end),side='right')
                sub=nav.iloc[max(0,loc-1):stop]
                periods.append(dict(strategy=strategy,variant=variant,start=start,end=end,**{k:float(v[0]) for k,v in metrics(sub).items()}))
            for key in ['executions','decisions']:
                result[key].to_csv(out/f'{name}_{key}.csv',index=False)
            result['trades']['strategy']=name;trades.append(result['trades'])
            result['targets']['strategy']=name
            result['targets']=result['targets'].merge(g[['date','symbol','industry']],on=['date','symbol'],how='left',validate='one_to_one')
            targets.append(result['targets'])
            pairs[name]=f'{strategy}__BENCHMARK'
            navs[pairs[name]]=bench[strategy]
            stress=replay_ranked(prices,scores,schedule,top_k=p['top_k'],exit_rank=exit_rank,fee=p['stress_fee'])
            stress_navs[name]=stress['daily'].post_nav
            perf[-1]['stress_total_return']=float(stress['daily'].post_nav.iloc[-1]-1)
            print('done',name,'return',round(perf[-1]['total_return'],6),flush=True)
    nav=pd.DataFrame(navs);nav.to_csv(out/'nav.csv',index_label='date')
    pd.DataFrame(stress_navs).to_csv(out/'stress_nav.csv',index_label='date')
    pd.DataFrame(perf).to_csv(out/'performance.csv',index=False)
    pd.DataFrame(periods).to_csv(out/'subperiods.csv',index=False)
    pd.DataFrame(parity).to_csv(out/'baseline_parity.csv',index=False)
    pd.concat(trades,ignore_index=True).to_csv(out/'trades.csv.gz',index=False)
    pd.concat(targets,ignore_index=True).to_csv(out/'targets.csv.gz',index=False)
    # Audit loader supports plain CSV: hash the exact uncompressed reader inputs.
    pd.concat(trades,ignore_index=True).to_csv(out/'audit_trades.csv',index=False)
    pd.concat(targets,ignore_index=True).to_csv(out/'audit_targets.csv',index=False)
    def entry(name):return dict(path=name,sha256=sha(out/name))
    spec=dict(nav=entry('nav.csv'),pairs=pairs,frequency='daily',return_units='decimal',
        family_description='six fixed versus buffered pairs, all twelve declared; already-seen history',
        cost_convention='fractional close, proportional fee 3bp only, no lots/tax/impact',
        diagnostics=dict(trades=entry('audit_trades.csv'),allocations=entry('audit_targets.csv'),allocation_kind='target',return_correlations=True))
    write_json(out/'audit_spec.json',spec);run_audit(out/'audit_spec.json',out/'audit')
    paired=dict(spec)
    paired.pop('diagnostics')
    paired['pairs']={s+'__BUFFER':s+'__FIXED' for s in sorted(ranks.strategy.unique())}
    paired['family_description']='all six preregistered BUFFER minus FIXED comparisons; no winner filtering'
    write_json(out/'paired_spec.json',paired);run_audit(out/'paired_spec.json',out/'paired_audit')
    if before!=account_hashes():raise AssertionError('forward accounts changed')
    if inputs!={name:sha(name) for name in inputs}:raise AssertionError('inputs changed during run')
    if code_before!={n:sha(ROOT/n) for n in code}:raise AssertionError('code changed during validation')
    write_json(out/'complete.json',dict(status='PASS',baseline_pairs=len(parity),forward_unchanged=True,
        independent_holdout=False,live_approval=False,inputs=inputs,code_hashes=code_before,
        outputs={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    return pd.DataFrame(perf)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'research_protocols/20261019_season02_integration_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.protocol,args.output)
