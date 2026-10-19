#!/usr/bin/env python3
"""Frozen six-strategy IC/K experiment; checkpointed and isolated from paper ledgers."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from analysis.holdings_count import eligible_base, replay_batch, metrics
from analysis.ic_weight_comparison import execution_labels, daily_rank_ic, rolling_weights, combine_scores, replay_universe
from factors.preprocess import preprocess_factor_panel
from scripts.run_mvp_holdings_count import sha, write_json, load_matrix
from scripts.build_dynamic_universe_comparison import build_weekly_raw_factor_panel
from strategies.style import build_fixed_family_score, build_focused_style_score
from universe.dynamic import eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import build_strategy_universe_report, profiles_from_config, eligibility_for_profile, point_in_time_industry, daily_basic_for_index

DEFAULT_PROTOCOL = ROOT/'research_protocols/20261018_six_strategy_ic_comparison_v1.json'
DEFAULT_OUTPUT = ROOT/'output/six_strategy_ic_comparison_20261018'


def strategy_name(spec):
    return spec['name'].removesuffix('_TOP50')


def recipe(spec, parent):
    focused = spec['score'] != 'fixed_equal_five_family_multi_factor'
    families = {'FOCUSED': spec['factors']} if focused else parent['candidate_families']
    return families, focused


def seal(directory):
    write_json(directory/'complete.json', {f.name: sha(f) for f in sorted(directory.iterdir()) if f.is_file() and f.name != 'complete.json'})


def verify(directory):
    record = json.loads((directory/'complete.json').read_text())
    for name, digest in record.items():
        if sha(directory/name) != digest:
            raise ValueError(f'checkpoint tampered: {directory/name}')


def load_prices(p):
    prior = ROOT/p['prior_replay']
    ready = json.loads((prior/'prepared.json').read_text())
    if sha(prior/'prices.npz') != ready['prices_sha256']:
        raise ValueError('prior price cache checksum mismatch')
    z = np.load(prior/'prices.npz')
    return pd.DataFrame(z['values'], index=pd.DatetimeIndex(z['dates'],name='date'), columns=pd.Index(z['symbols'],name='symbol'))


def freeze(protocol, out):
    p=json.loads(protocol.read_text()); parent=ROOT/p['parent_protocol']; source=ROOT/p['data_source']
    prior=ROOT/p['prior_replay']; old=json.loads((prior/'started.json').read_text())
    raw={name:sha(source/name) for name in old['raw']}
    if raw != old['raw'] or sha(ROOT/p['financial_supplement']) != old['financial_supplement']:
        raise ValueError('source differs from previous validated history')
    names=['analysis/ic_weight_comparison.py','scripts/run_six_strategy_ic_comparison.py',
        'analysis/holdings_count.py','scripts/run_mvp_holdings_count.py','scripts/build_dynamic_universe_comparison.py',
        'strategies/style.py','factors/preprocess.py','universe/risk.py','universe/strategy.py','universe/dynamic.py']
    stamp={'protocol':sha(protocol),'parent':sha(parent),'raw':raw,
        'financial_supplement':sha(ROOT/p['financial_supplement']), 'code':{n:sha(ROOT/n) for n in names}}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists() and json.loads((out/'frozen.json').read_text()) != stamp:
        raise ValueError('frozen source/code changed; use a new output directory')
    write_json(out/'frozen.json',stamp);write_json(out/'protocol.json',p)
    # Exclude reports, which can be legitimately edited. Hash sealed account inputs and ledgers.
    session=ROOT/'output/six_strategy_forward_20260928'
    if not (out/'forward_before.json').exists():
        write_json(out/'forward_before.json',{str(f.relative_to(session)):sha(f)
            for folder in ['initial','plans','days'] for f in sorted((session/folder).rglob('*')) if f.is_file()})
    return p,json.loads(parent.read_text())


def prepare(protocol,out):
    p,parent=freeze(protocol,out)
    source=ROOT/p['data_source']; prior=ROOT/p['prior_replay']
    adjusted=load_prices(p); dates,symbols=adjusted.index,adjusted.columns
    matrices=load_matrix(source/'daily.csv.gz',['close','amount'],dates,symbols,{'amount':1000.})
    close,amount=matrices['close'],matrices['amount']
    labels,outcome=execution_labels(adjusted,p['ic']['holding_sessions'])
    old_protocol=json.loads((prior/'protocol.json').read_text())
    basic=pd.read_csv(source/'stock_basic.csv.gz',dtype=str)
    basic=basic[basic.ts_code.isin(symbols)]
    names=pd.read_csv(source/'namechange.csv.gz')
    base=eligible_base(close,amount,basic,names,dates)
    del amount,matrices
    base['quarter']=base.date.dt.to_period('Q').astype(str)
    fincols=['ts_code','ann_date','end_date','eps','roe','grossprofit_margin','netprofit_margin','debt_to_assets','or_yoy','netprofit_yoy','ocfps','ocf_to_profit']
    finance=pd.read_csv(ROOT/p['financial_supplement'],usecols=fincols)
    balance=pd.read_csv(source/'balancesheet_vip.csv.gz',usecols=lambda x:x in ['ts_code','ann_date','f_ann_date','end_date','report_type','update_flag','total_hldr_eqy_exc_min_int'])
    audit=pd.read_csv(source/'fina_audit.csv.gz')
    db=pd.read_csv(source/'daily_basic.csv.gz',usecols=['ts_code','trade_date','pe_ttm','pb','circ_mv','total_mv'],dtype={'trade_date':str})
    industry=pd.read_csv(source/'index_member_all.csv.gz')
    profiles=profiles_from_config(parent['strategy_universe_profiles'])
    weeks=pd.Series(dates,index=dates).groupby(dates.to_period('W-FRI')).max()
    weeks=weeks[(weeks.index.end_time.normalize()<=dates[-1])&(weeks>=p['evaluation_start'])&(weeks<dates[-1])]
    decisions=pd.DatetimeIndex(weeks.to_numpy())
    write_json(out/'calendar.json',{'sessions':[str(d.date()) for d in dates], 'signals':[str(d.date()) for d in decisions]})
    print(f'Loaded {len(dates)} sessions, {len(symbols)} symbols, {len(base)} eligible-day rows; {len(decisions)} weekly signals',flush=True)
    for quarter,b in base.groupby('quarter',sort=True):
        dest=out/f'inputs_{quarter}'
        if (dest/'complete.json').exists():
            verify(dest);print('Verified reuse',quarter,flush=True);continue
        dest.mkdir(exist_ok=True)
        d=pd.DatetimeIndex(sorted(b.date.unique())); weekly=decisions.intersection(d)
        start=max(0,dates.get_loc(d[0])-90);stop=dates.get_loc(d[-1])+1
        long=close.iloc[start:stop].stack().rename('close').reset_index()
        long.columns=['trade_date','ts_code','close']
        idx=pd.MultiIndex.from_frame(long[['trade_date','ts_code']])
        long['adj_close']=adjusted.iloc[start:stop].stack().reindex(idx).to_numpy()
        raw=build_weekly_raw_factor_panel(long,finance,d)
        risk=build_hard_risk_universe_report(b.drop(columns='quarter'),balance,audit,names)
        dbcut=db[db.trade_date.between((d[0]-pd.Timedelta(days=7)).strftime('%Y%m%d'),d[-1].strftime('%Y%m%d'))]
        report=build_strategy_universe_report(risk,dbcut,raw,industry,profiles)
        masks={'BASE_V2':eligibility_from_report(risk),**{v.name:eligibility_for_profile(report,v) for v in profiles}}
        ind=point_in_time_industry(raw.index,industry)
        market=daily_basic_for_index(raw.index,dbcut)
        raw['EARNINGS_YIELD']=(1/market.pe_ttm).where(market.pe_ttm.gt(0))
        raw['BOOK_TO_PRICE']=(1/market.pb).where(market.pb.gt(0))
        label_long=labels.loc[d].stack().rename('label')
        label_long.index=label_long.index.set_names(['date','symbol'])
        coverage=[];parity=[]
        for spec in parent['strategies']:
            name=strategy_name(spec); families,focused=recipe(spec,parent)
            cols=[f for fs in families.values() for f in fs]
            mask=masks[spec['universe']].reindex(raw.index).fillna(False).astype(bool)
            # Removing excluded rows is equivalent to the production masked preprocessing;
            # production weekly score parity is checked below on every signal.
            selected=raw.loc[mask,cols]
            if selected.empty:
                continue
            standard=preprocess_factor_panel(selected,industry=ind.reindex(selected.index),by_industry=True,min_industry_count=3)
            ic,n=daily_rank_ic(standard.where(selected.notna()),label_long,p['ic']['minimum_cross_section_names'])
            ic=ic.reindex(d);n=n.reindex(d)
            ic['outcome_end']=outcome.reindex(d)
            ic.to_csv(dest/f'{name}_ic.csv',index_label='date')
            n.to_csv(dest/f'{name}_ic_names.csv',index_label='date')
            sw=standard.loc[standard.index.get_level_values('date').isin(weekly)].copy()
            sw['raw_count']=selected.notna().sum(axis=1).reindex(sw.index)
            sw['industry']=ind.reindex(sw.index).fillna('')
            sw.to_csv(dest/f'{name}_weekly.csv.gz',index_label=['date','symbol'])
            if len(weekly):
                wr=raw.loc[raw.index.get_level_values('date').isin(weekly)]
                wm=mask.reindex(wr.index);wi=ind.reindex(wr.index)
                if focused:
                    official,_=build_focused_style_score(wr,wm,wi,cols,minimum_components=spec['minimum_components'])
                else:
                    official,_=build_fixed_family_score(wr,wm,wi,families)
                eq=pd.Series({c:1/len(fs) for fs in families.values() for c in fs})
                ours=combine_scores(sw[cols],families,eq,focused=focused,raw_count=sw.raw_count,minimum_components=spec.get('minimum_components',1))
                if not official.index.equals(ours.index):
                    official,ours=official.sort_index(),ours.sort_index()
                if not official.index.equals(ours.index) or not np.allclose(official,ours,atol=1e-11,rtol=1e-11):
                    raise AssertionError(f'production fixed score mismatch {quarter} {name}')
                parity.append({'strategy':name,'rows':len(official),'max_error':float(np.max(np.abs(official-ours)))})
                off=official.rename('score').reset_index().sort_values(['date','score','symbol'],ascending=[True,False,True],kind='stable')
                off['rank']=off.groupby('date').cumcount()+1
                off.loc[off['rank']<=100].to_csv(dest/f'{name}_fixed_top100.csv.gz',index=False)
                # Raw component coverage is explicit; configured factors are not necessarily available per stock.
                for day in weekly:
                    if day not in selected.index.get_level_values('date'):continue
                    dayraw=selected.loc[day]
                    coverage.append({'date':day,'strategy':name,'eligible_names':len(dayraw),
                        **{f'valid_{c}':float(dayraw[c].notna().mean()) for c in cols}})
        pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
        write_json(dest/'production_score_parity.json',parity)
        seal(dest)
        print(f'Prepared {quarter}: daily IC + {len(weekly)} weekly signals; production score parity PASS',flush=True)
        del raw,report,risk,standard,selected,long,market;gc.collect()
    write_json(out/'prepared.json',{'finished':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol),'quarters':len(list(out.glob('inputs_*')))})
    print('PREPARATION COMPLETE',flush=True)


def rankings(protocol,out):
    p,parent=freeze(protocol,out)
    calendar=json.loads((out/'calendar.json').read_text())
    dates=pd.DatetimeIndex(calendar['sessions']); decisions=pd.DatetimeIndex(calendar['signals'])
    blocks=sorted(out.glob('inputs_*'))
    for b in blocks:verify(b)
    allranks=[];allweights=[];skips=[];bench=[];parity=[]
    for spec in parent['strategies']:
        name=strategy_name(spec);families,focused=recipe(spec,parent)
        cols=[c for fs in families.values() for c in fs]
        parts=[pd.read_csv(b/f'{name}_ic.csv',parse_dates=['date','outcome_end']).set_index('date') for b in blocks if (b/f'{name}_ic.csv').exists()]
        ic=pd.concat(parts).sort_index().reindex(dates)
        outcome=ic.pop('outcome_end')
        parts=[pd.read_csv(b/f'{name}_weekly.csv.gz',parse_dates=['date'],keep_default_na=True) for b in blocks if (b/f'{name}_weekly.csv.gz').exists()]
        weekly=pd.concat(parts).set_index(['date','symbol']).sort_index()
        fixed=pd.concat([pd.read_csv(b/f'{name}_fixed_top100.csv.gz',parse_dates=['date']) for b in blocks if (b/f'{name}_fixed_top100.csv.gz').exists()])
        fixed_groups={d:g for d,g in fixed.groupby('date')}
        for signal,g in weekly.groupby(level='date',sort=True):
            weights,log=rolling_weights(ic,outcome,signal,families,
                lookback=p['ic']['lookback_matured_sessions'],minimum=p['ic']['minimum_valid_ic_days_per_factor'],fixed_share=p['ic']['fixed_share'])
            for row in log:row['strategy']=name
            allweights.extend(log)
            score=combine_scores(g[cols],families,weights,focused=focused,raw_count=g.raw_count,minimum_components=spec.get('minimum_components',1))
            dynamic=score.rename('score').reset_index().sort_values(['score','symbol'],ascending=[False,True],kind='stable')
            dynamic['rank']=np.arange(1,len(dynamic)+1)
            control=fixed_groups.get(signal,pd.DataFrame())
            if len(dynamic)<50 or len(control)<50:
                skips.append({'date':signal,'strategy':name,'fixed_names':len(control),'ic_names':len(dynamic),'action':'ALL_ARMS_HOLD'})
                continue
            for variant,frame in [('FIXED',control),('IC_SHRUNK',dynamic)]:
                top=frame.head(100).copy();top['strategy']=name;top['variant']=variant
                top['industry']=g.industry.droplevel('date').reindex(top.symbol).fillna('').to_numpy()
                allranks.append(top)
            targets=g.reset_index()[['date','symbol']];targets['strategy']=name;bench.append(targets)
        # Confirm all fixed selections still match the prior long-history run, independently of IC.
        prior=ROOT/p['prior_replay']
        for year,group in fixed.groupby(fixed.date.dt.year):
            old=pd.read_csv(prior/f'inputs_{year}'/'ranked_top100.csv.gz',parse_dates=['date'])
            old=old[old.strategy.eq(name)]
            a=group.sort_values(['date','rank']);b=old.sort_values(['date','rank'])
            if not a[['date','symbol','rank']].reset_index(drop=True).equals(b[['date','symbol','rank']].reset_index(drop=True)):
                raise AssertionError(f'prior fixed ranking mismatch {name} {year}')
            parity.append({'strategy':name,'year':int(year),'rows':len(a),'status':'EXACT_RANK_MATCH'})
        print('Ranked both variants:',name,flush=True)
    rank=pd.concat(allranks,ignore_index=True)
    rank.to_csv(out/'rankings.csv.gz',index=False)
    pd.concat(bench,ignore_index=True).to_csv(out/'benchmark_members.csv.gz',index=False)
    weights=pd.DataFrame(allweights)
    if not weights.loc[weights.latest_outcome_end.notna(),'latest_outcome_end'].lt(weights.loc[weights.latest_outcome_end.notna(),'signal_date']).all():
        raise AssertionError('unmatured labels used')
    weights.to_csv(out/'weights.csv.gz',index=False)
    pd.DataFrame(skips,columns=['date','strategy','fixed_names','ic_names','action']).to_csv(out/'skipped_signals.csv',index=False)
    write_json(out/'prior_fixed_rank_parity.json',parity)
    write_json(out/'ranked.json',{'protocol_sha256':sha(protocol),'rankings_sha256':sha(out/'rankings.csv.gz'),
        'benchmark_sha256':sha(out/'benchmark_members.csv.gz'),'weights_sha256':sha(out/'weights.csv.gz'),
        'completed':datetime.now(timezone.utc).isoformat()})


def backtest(protocol,out):
    p,parent=freeze(protocol,out)
    sealed=json.loads((out/'ranked.json').read_text())
    for key,file in [('rankings_sha256','rankings.csv.gz'),('benchmark_sha256','benchmark_members.csv.gz'),('weights_sha256','weights.csv.gz')]:
        if sha(out/file)!=sealed[key]:raise ValueError('ranked inputs changed')
    adjusted=load_prices(p)
    adjusted=adjusted.loc[p['evaluation_start']:p['evaluation_end']]
    dates,symbols=adjusted.index,adjusted.columns;px=adjusted.to_numpy()
    ranks=pd.read_csv(out/'rankings.csv.gz',parse_dates=['date'])
    members=pd.read_csv(out/'benchmark_members.csv.gz',parse_dates=['date'])
    rows=[];periods=[];annual=[];audit=[];concentration=[];navs={};stressnav={};benchmarks={}
    for spec in parent['strategies']:
        name=strategy_name(spec)
        rg=ranks[ranks.strategy.eq(name)]
        groups={(v,d):g.sort_values('rank') for (v,d),g in rg.groupby(['variant','date'])}
        signals=pd.DatetimeIndex(sorted(rg.date.unique()))
        ex=dates.get_indexer(signals)+1
        if (ex<1).any() or (ex>=len(dates)).any():raise ValueError('invalid execution calendar')
        mg={d:g for d,g in members[members.strategy.eq(name)].groupby('date')}
        bench=replay_universe(px,ex,[symbols.get_indexer(mg[d].symbol) for d in signals],p['commission_rate'])
        benchmarks[name]=bench
        br=bench[1:]/bench[:-1]-1
        bm=metrics(bench)
        for k in p['k_values']:
            plans=np.stack([np.stack([symbols.get_indexer(groups[v,d].head(k).symbol) for v in p['variants']]) for d in signals])
            nav,log,stranded=replay_batch(px,ex,plans,p['commission_rate'])
            stress,_,_=replay_batch(px,ex,plans,p['stress_one_way_cost'])
            mm=metrics(nav);sm=metrics(stress)
            for j,v in enumerate(p['variants']):
                key=f'{name}__{v}__K{k}'
                navs[key]=nav[:,j];stressnav[key]=stress[:,j]
                active=nav[1:,j]/nav[:-1,j]-1-br
                te=active.std(ddof=1)*np.sqrt(252)
                rows.append({'strategy':name,'variant':v,'k':k,'start':str(dates[0].date()),'end':str(dates[-1].date()),
                    **{f:float(a[j]) for f,a in mm.items()},'benchmark_cagr':float(bm['cagr'][0]),
                    'excess_cagr_difference':float(mm['cagr'][j]-bm['cagr'][0]),
                    'tracking_error':te,'information_ratio':float(active.mean()*252/te) if te>0 else 0.,
                    'annual_two_way_turnover':sum(a['two_way_turnover'][j] for a in log)*252/(len(dates)-1),
                    'mean_cash_fraction_at_rebalance':float(np.mean([a['cash_fraction'][j] for a in log])),
                    'mean_actual_holdings_at_rebalance':float(np.mean([a['holdings'][j] for a in log])),
                    'terminal_unquoted_fraction':float(stranded[j]/nav[-1,j]),
                    'terminal_zero_unquoted_total_return':float(nav[-1,j]-stranded[j]-1),
                    **{'stress_'+f:float(a[j]) for f,a in sm.items()}})
                for lo,hi in p['subperiods']:
                    idx=np.flatnonzero((dates>=lo)&(dates<=hi))
                    a=max(0,idx[0]-1);b=idx[-1]+1
                    metrics_sub=metrics(nav[a:b,j]);benchmark_sub=metrics(bench[a:b])
                    periods.append({'strategy':name,'variant':v,'k':k,'period':lo[:4]+'-'+hi[:4],
                        **{f:float(z[0]) for f,z in metrics_sub.items()},
                        'benchmark_cagr':float(benchmark_sub['cagr'][0]),
                        'excess_cagr_difference':float(metrics_sub['cagr'][0]-benchmark_sub['cagr'][0])})
                for year in sorted(set(dates.year)):
                    idx=np.flatnonzero(dates.year==year);a=max(0,idx[0]-1);b=idx[-1]+1
                    annual.append({'strategy':name,'variant':v,'k':k,'year':int(year),**{f:float(z[0]) for f,z in metrics(nav[a:b,j]).items()}})
                for d in signals:
                    top=groups[v,d].head(k);industry=top.industry.fillna('UNKNOWN').replace('','UNKNOWN').value_counts()/k
                    concentration.append({'date':d,'strategy':name,'variant':v,'k':k,
                        'max_industry_weight':float(industry.max()),'industry_hhi':float((industry**2).sum()),
                        'unknown_industry_weight':float(industry.get('UNKNOWN',0.))})
                for a in log:
                    audit.append({'date':dates[a['execution_index']],'strategy':name,'variant':v,'k':k,
                        **{f:float(z[j]) for f,z in a.items() if f!='execution_index'}})
            print('Replayed',name,'K',k,'fixed / IC',flush=True)
    perf=pd.DataFrame(rows)
    if len(perf)!=60 or perf[['strategy','variant','k']].duplicated().any():raise AssertionError('expected exactly 60 paths')
    perf.to_csv(out/'performance.csv',index=False)
    pd.DataFrame(periods).to_csv(out/'subperiods.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'annual.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'execution_audit.csv',index=False)
    pd.DataFrame(concentration).to_csv(out/'concentration.csv',index=False)
    pd.DataFrame(navs,index=dates).to_csv(out/'nav.csv',index_label='date')
    pd.DataFrame(stressnav,index=dates).to_csv(out/'stress_nav.csv',index_label='date')
    pd.DataFrame(benchmarks,index=dates).to_csv(out/'benchmark_nav.csv',index_label='date')
    fixed=perf[perf.variant.eq('FIXED')].set_index(['strategy','k'])
    dynamic=perf[perf.variant.eq('IC_SHRUNK')].set_index(['strategy','k'])
    cols=['total_return','cagr','sharpe','max_drawdown','excess_cagr_difference','information_ratio','annual_two_way_turnover','stress_cagr']
    (dynamic[cols]-fixed[cols]).add_prefix('ic_minus_fixed_').reset_index().to_csv(out/'paired_differences.csv',index=False)
    prior=pd.read_csv(ROOT/p['prior_replay']/'performance.csv').set_index(['strategy','k'])
    errors=(fixed[['total_return','cagr','sharpe','max_drawdown']]-prior[['total_return','cagr','sharpe','max_drawdown']]).abs()
    if errors.to_numpy().max()>1e-10:raise AssertionError('prior fixed results changed')
    session=ROOT/'output/six_strategy_forward_20260928'
    old=json.loads((out/'forward_before.json').read_text())
    if any(sha(session/n)!=v for n,v in old.items()):raise AssertionError('forward account changed')
    write_json(out/'completed.json',{'finished':datetime.now(timezone.utc).isoformat(),'paths':60,'stress_paths':60,'benchmarks':6,
        'fixed_prior_max_metric_error':float(errors.to_numpy().max()),'paper_account_unchanged':True,
        'files':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.suffix in {'.csv','.gz'}}})
    print('ALL 60 MAIN + 60 STRESS + 6 BENCHMARK PATHS COMPLETE; prior fixed parity PASS; paper account unchanged',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--stage',choices=['prepare','rank','backtest','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare(args.protocol,args.output)
    if args.stage in ['rank','all']:rankings(args.protocol,args.output)
    if args.stage in ['backtest','all']:backtest(args.protocol,args.output)


if __name__=='__main__':main()
