#!/usr/bin/env python3
"""Supplement sealed six-strategy experiment with pure IC; never edit source runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.pure_ic_comparison import pure_ic_weights,pure_score,replay_actions
from analysis.holdings_count import metrics,replay_batch
from scripts.run_six_strategy_ic_comparison import recipe,strategy_name,verify,load_prices
from scripts.run_mvp_holdings_count import sha,write_json

PROTOCOL=ROOT/'research_protocols/20261018_six_strategy_pure_ic_v1.json'
OUTPUT=ROOT/'output/six_strategy_pure_ic_20261018'


def freeze(out):
    p=json.loads(PROTOCOL.read_text());source=ROOT/p['source_experiment']
    sp=json.loads((source/'protocol.json').read_text());f=json.loads((source/'frozen.json').read_text())
    if sha(source/'protocol.json') != sha(ROOT/'research_protocols/20261018_six_strategy_ic_comparison_v1.json'):
        # Source JSON can differ only in indentation; compare its actual content.
        if sp != json.loads((ROOT/'research_protocols/20261018_six_strategy_ic_comparison_v1.json').read_text()):
            raise ValueError('source protocol changed')
    if sha(ROOT/'research_protocols/20261018_six_strategy_ic_comparison_v1.json')!=f['protocol']:
        raise ValueError('source protocol seal mismatch')
    for name,digest in f['code'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source code changed: '+name)
    if sha(ROOT/sp['parent_protocol'])!=f['parent']:raise ValueError('parent protocol changed')
    for name,digest in f['raw'].items():
        if sha(ROOT/sp['data_source']/name)!=digest:raise ValueError('raw source changed: '+name)
    if sha(ROOT/sp['financial_supplement'])!=f['financial_supplement']:raise ValueError('finance source changed')
    for block in sorted(source.glob('inputs_*')):verify(block)
    completed=json.loads((source/'completed.json').read_text())
    for name,digest in completed['files'].items():
        if sha(source/name)!=digest:raise ValueError('source result changed: '+name)
    originals={str(x.relative_to(ROOT)):sha(x) for x in source.rglob('*') if x.is_file()}
    session=ROOT/'output/six_strategy_forward_20260928'
    originals.update({str(x.relative_to(ROOT)):sha(x) for folder in ['initial','plans','days'] for x in (session/folder).rglob('*') if x.is_file()})
    stamp={'protocol':sha(PROTOCOL),'source_manifest':sha(source/'completed.json'),
        'code':{n:sha(ROOT/n) for n in ['analysis/pure_ic_comparison.py','scripts/run_six_strategy_pure_ic.py']},
        'source_and_accounts':originals}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists() and json.loads((out/'frozen.json').read_text())!=stamp:
        raise ValueError('frozen experiment changed; use a new output directory')
    write_json(out/'frozen.json',stamp);write_json(out/'protocol.json',p)
    return p,sp,json.loads((ROOT/sp['parent_protocol']).read_text()),source


def rank(out,p,sp,parent,source):
    dates=pd.DatetimeIndex(json.loads((source/'calendar.json').read_text())['sessions'])
    controls=pd.read_csv(source/'rankings.csv.gz',parse_dates=['date'])
    blocks=sorted(source.glob('inputs_*'));ranks=[];weights=[];actions=[]
    for spec in parent['strategies']:
        name=strategy_name(spec);families,focused=recipe(spec,parent)
        columns=[c for cs in families.values() for c in cs]
        family={c:f for f,cs in families.items() for c in cs}
        ic=pd.concat([pd.read_csv(b/f'{name}_ic.csv',parse_dates=['date','outcome_end']).set_index('date') for b in blocks if (b/f'{name}_ic.csv').exists()]).sort_index().reindex(dates)
        end=ic.pop('outcome_end');ic=ic[columns]
        weekly=pd.concat([pd.read_csv(b/f'{name}_weekly.csv.gz',parse_dates=['date']) for b in blocks if (b/f'{name}_weekly.csv.gz').exists()]).set_index(['date','symbol']).sort_index()
        active=set(controls.loc[controls.strategy.eq(name),'date'])
        for signal,g in weekly.groupby(level='date',sort=True):
            w,logs=pure_ic_weights(ic,end,signal,lookback=p['lookback'],minimum=p['minimum_ic_days'])
            for row in logs:row.update(strategy=name,family=family[row['factor']])
            weights.extend(logs)
            score=pure_score(g[columns],w,g.raw_count if focused else None,spec.get('minimum_components',1))
            action='HOLD' if signal not in active else ('CASH' if w.sum()==0 else ('TOPK' if len(score)>=50 else 'HOLD'))
            reason='original_calendar_unavailable' if signal not in active else ('no_positive_eligible_ic' if w.sum()==0 else ('ready' if len(score)>=50 else 'fewer_than_50_scores'))
            ready=all(r['valid_ic_days']>=p['minimum_ic_days'] for r in logs) and action=='TOPK'
            actions.append({'date':signal,'strategy':name,'action':action,'reason':reason,'all_factors_ready':ready,
                'positive_factors':int(w.gt(0).sum()),'weight_sum':w.sum(),'scored_names':len(score),
                'max_factor_weight':w.max(),'effective_factor_count':1/(w*w).sum() if w.sum()>0 else 0.})
            if action=='TOPK':
                top=score.reset_index().sort_values(['score','symbol'],ascending=[False,True],kind='stable').head(100)
                top['rank']=np.arange(1,len(top)+1);top['strategy']=name;top['variant']='PURE_IC'
                top['industry']=g.industry.droplevel('date').reindex(top.symbol).to_numpy();ranks.append(top)
        print('Pure IC ranked:',name,flush=True)
    actions=pd.DataFrame(actions);w=pd.DataFrame(weights)
    readiness=actions.pivot(index='date',columns='strategy',values='all_factors_ready').fillna(False).all(axis=1)
    common=readiness.index[readiness][0]
    pd.concat(ranks,ignore_index=True).to_csv(out/'rankings.csv.gz',index=False)
    w.to_csv(out/'weights.csv.gz',index=False);actions.to_csv(out/'actions.csv',index=False)
    write_json(out/'ready.json',{'common_signal':str(common.date()),'protocol_sha256':sha(PROTOCOL),
        'files':{n:sha(out/n) for n in ['rankings.csv.gz','weights.csv.gz','actions.csv']}})
    print('COMMON POST-WARMUP SIGNAL:',common.date(),flush=True)


def simulate(out,p,sp,parent,source):
    ready=json.loads((out/'ready.json').read_text())
    for n,digest in ready['files'].items():
        if sha(out/n)!=digest:raise ValueError('ranking changed')
    price=load_prices(sp).loc[sp['evaluation_start']:sp['evaluation_end']]
    controls=pd.read_csv(source/'rankings.csv.gz',parse_dates=['date'])
    pure=pd.read_csv(out/'rankings.csv.gz',parse_dates=['date'])
    actions=pd.read_csv(out/'actions.csv',parse_dates=['date'])
    members=pd.read_csv(source/'benchmark_members.csv.gz',parse_dates=['date'])
    rows=[];periods=[];audits=[];parity=[];navs={};stressnav={};benchmarks={}
    for scope,start in [('full',price.index[0]),('common',pd.Timestamp(ready['common_signal']))]:
        prices=price.loc[start:];dates,symbols=prices.index,prices.columns
        px=prices.to_numpy();marks=prices.ffill().fillna(0).to_numpy()
        scope_nav={};scope_stress={};scope_bench={}
        for spec in parent['strategies']:
            name=strategy_name(spec)
            c=controls[controls.strategy.eq(name)&controls.date.ge(start)]
            groups={(v,d):g.sort_values('rank') for (v,d),g in c.groupby(['variant','date'])}
            signals=pd.DatetimeIndex(sorted(c.date.unique()));ex=dates.get_indexer(signals)+1
            if (ex<=0).any() or (ex>=len(dates)).any():raise ValueError('invalid execution dates')
            pg={d:g.sort_values('rank') for d,g in pure[pure.strategy.eq(name)].groupby('date')}
            act=actions[actions.strategy.eq(name)].set_index('date')
            mg={d:g for d,g in members[members.strategy.eq(name)].groupby('date')}
            bench,_,_=replay_actions(px,ex,[symbols.get_indexer(mg[d].symbol) for d in signals],p['cost'],marks)
            scope_bench[name]=bench;bm=metrics(bench)['cagr'][0];br=bench[1:]/bench[:-1]-1
            for k in p['k_values']:
                # Supplementary full period reuses sealed controls; main window resets all three.
                variants=['PURE_IC'] if scope=='full' else p['variants']
                for v in variants:
                    plans=[(None if act.loc[d,'action']=='HOLD' else ([] if act.loc[d,'action']=='CASH' else symbols.get_indexer(pg[d].head(k).symbol))) if v=='PURE_IC' else symbols.get_indexer(groups[v,d].head(k).symbol) for d in signals]
                    nav,log,stranded=replay_actions(px,ex,plans,p['cost'],marks)
                    stress,_,_=replay_actions(px,ex,plans,p['stress_cost'],marks)
                    key=f'{name}__{v}__K{k}';scope_nav[key]=nav;scope_stress[key]=stress
                    mm={f:float(z[0]) for f,z in metrics(nav).items()}
                    sm={'stress_'+f:float(z[0]) for f,z in metrics(stress).items()}
                    active=nav[1:]/nav[:-1]-1-br;te=active.std(ddof=1)*np.sqrt(252)
                    rows.append({'scope':scope,'strategy':name,'variant':v,'k':k,'start':str(dates[0].date()),'end':str(dates[-1].date()),**mm,**sm,
                        'benchmark_cagr':bm,'excess_cagr_difference':mm['cagr']-bm,
                        'information_ratio':active.mean()*252/te if te>0 else 0,
                        'annual_two_way_turnover':sum(a['two_way_turnover'] for a in log)*252/(len(dates)-1),
                        'cash_target_signals':sum(a['action']=='CASH' for a in log),
                        'mean_cash_fraction_at_rebalance':np.mean([a['cash_fraction'] for a in log]),
                        'terminal_unquoted_fraction':stranded/nav[-1],
                        'terminal_zero_unquoted_total_return':nav[-1]-stranded-1})
                    for lo,hi in sp['subperiods']:
                        idx=np.flatnonzero((dates>=lo)&(dates<=hi))
                        if len(idx)<2:continue
                        a=max(0,idx[0]-1);b=idx[-1]+1
                        periods.append({'scope':scope,'strategy':name,'variant':v,'k':k,'period':lo[:4]+'-'+hi[:4],
                            'actual_start':str(dates[a].date()),'actual_end':str(dates[b-1].date()),
                            **{f:float(z[0]) for f,z in metrics(nav[a:b]).items()}})
                    for a in log:
                        audits.append({'scope':scope,'strategy':name,'variant':v,'k':k,'date':dates[a['execution_index']],**a})
                    # Independent original engine comparison on non-cash real-data segments.
                    if scope=='common':
                        normal=[i for i,x in enumerate(plans) if x is not None and len(x)]
                        for year in [2011,2018,2026]:
                            use=[i for i in normal if signals[i].year==year][:4]
                            if not use:continue
                            a=ex[use[0]]-1;b=min(len(px),ex[use[-1]]+3)
                            pp=[plans[i] for i in use];ee=ex[use]-a
                            local,_,_=replay_actions(px[a:b],ee,pp,p['cost'])
                            ref,_,_=replay_batch(px[a:b],ee,np.stack(pp)[:,None,:],p['cost'])
                            err=float(np.max(np.abs(local-ref[:,0])))
                            if err>1e-10:raise AssertionError('original engine parity failed')
                            parity.append({'strategy':name,'variant':v,'k':k,'year':year,'max_nav_error':err})
            print('Replayed',scope,name,flush=True)
        navs[scope]=pd.DataFrame(scope_nav,index=dates);stressnav[scope]=pd.DataFrame(scope_stress,index=dates)
        benchmarks[scope]=pd.DataFrame(scope_bench,index=dates)
    # Preserve original full-calendar baselines exactly, not rebased post-hoc.
    for n in ['nav','stress_nav']:
        old=pd.read_csv(source/f'{n}.csv',index_col='date',parse_dates=True)
        mapping=navs if n=='nav' else stressnav
        mapping['full']=pd.concat([old,mapping['full']],axis=1)
    prior=pd.read_csv(source/'performance.csv');prior['scope']='full'
    perf=pd.concat([prior,pd.DataFrame(rows)],ignore_index=True)
    oldsub=pd.read_csv(source/'subperiods.csv');oldsub['scope']='full'
    pd.concat([oldsub,pd.DataFrame(periods)],ignore_index=True).to_csv(out/'subperiods.csv',index=False)
    perf.to_csv(out/'performance.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'execution_audit.csv',index=False)
    for scope in ['full','common']:
        navs[scope].to_csv(out/f'{scope}_nav.csv',index_label='date')
        stressnav[scope].to_csv(out/f'{scope}_stress_nav.csv',index_label='date')
        benchmarks[scope].to_csv(out/f'{scope}_benchmark_nav.csv',index_label='date')
    w=pd.read_csv(out/'weights.csv.gz',parse_dates=['signal_date','latest_outcome_end'])
    if not (w.latest_outcome_end.isna()|w.latest_outcome_end.lt(w.signal_date)).all():raise AssertionError('lookahead in labels')
    sums=w.groupby(['strategy','signal_date']).weight.sum()
    if not (np.isclose(sums,1)|np.isclose(sums,0)).all() or (w.weight<0).any():raise AssertionError('invalid weights')
    if len(perf)!=180 or perf[['scope','strategy','variant','k']].duplicated().any():raise AssertionError('expected two 90-path comparisons')
    maxerr=0.
    for r in perf.itertuples():
        val=metrics(navs[r.scope][f'{r.strategy}__{r.variant}__K{r.k}'].to_numpy())
        for key,value in val.items():maxerr=max(maxerr,abs(value[0]-getattr(r,key)))
    if maxerr>1e-9:raise AssertionError('NAV metric audit failed')
    frozen=json.loads((out/'frozen.json').read_text())
    for name,digest in frozen['source_and_accounts'].items():
        if sha(ROOT/name)!=digest:raise AssertionError('original file changed: '+name)
    result={'status':'PASS','common_signal':ready['common_signal'],'common_execution':str(price.index[price.index.get_loc(pd.Timestamp(ready['common_signal']))+1].date()),
        'new_paths':120,'new_stress_paths':120,'reused_control_paths':60,'main_common_paths':90,
        'original_engine_parity_checks':len(parity),'max_engine_nav_error':max(x['max_nav_error'] for x in parity),
        'max_metric_error':float(maxerr),'labels_strictly_matured':True,'original_results_and_accounts_unchanged':True,
        'parity':parity,'files':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.suffix in ['.csv','.gz']}}
    write_json(out/'completed.json',result)
    print('COMPLETED: 90 common-start paths + 90 full-calendar paths; stress and parity PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--stage',choices=['all','rank','backtest'],default='all');args=parser.parse_args()
    p,sp,parent,source=freeze(args.output)
    if args.stage in ['all','rank']:rank(args.output,p,sp,parent,source)
    if args.stage in ['all','backtest']:simulate(args.output,p,sp,parent,source)


if __name__=='__main__':main()
