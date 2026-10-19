#!/usr/bin/env python3
"""Audit actual historical selections against the unchanged production matcher."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.holdings_count import replay_batch
from backtest.backtest_single import _rebalance_to_target_weights


def validate(out):
    z=np.load(out/'prices.npz')
    dates=pd.DatetimeIndex(z['dates']);symbols=pd.Index(z['symbols'])
    rank=pd.read_csv(out/'inputs_2011/ranked_top100.csv.gz',parse_dates=['date'])
    records=[]
    for strategy,frame in rank.groupby('strategy'):
        groups=[g.sort_values('rank') for _,g in frame.groupby('date') if len(g)>=50][:12]
        signal=pd.DatetimeIndex([g.date.iloc[0] for g in groups])
        start=dates.get_indexer([signal[0]])[0]
        end=dates.get_indexer([signal[-1]])[0]+6
        d=dates[start:end]
        ex=d.get_indexer(signal)+1
        for k in [10,15,20,30,50]:
            chosen=sorted(set(s for g in groups for s in g.head(k).symbol))
            columns=pd.Index(chosen)
            prices=pd.DataFrame(z['values'][start:end][:,symbols.get_indexer(columns)],index=d,columns=columns)
            plans=np.stack([columns.get_indexer(g.head(k).symbol) for g in groups])[:,None,:]
            nav,_,_=replay_batch(prices,ex,plans)
            mark=prices.ffill()
            cash,shares=1.,pd.Series(0.,index=columns)
            actual=[]
            for i,date in enumerate(d):
                if i in ex:
                    w=list(ex).index(i)
                    target=[s for s in groups[w].head(k).symbol if pd.notna(prices.loc[date,s])]
                    cash,shares=_rebalance_to_target_weights(cash,shares,prices.loc[date],target,
                        [1/k]*len(target),.0003,valuation_px=mark.loc[date])
                actual.append(cash+(shares*mark.loc[date]).sum())
            error=float(np.max(np.abs(np.asarray(actual)-nav[:,0])))
            if error>1e-10:
                raise AssertionError(f'production matcher mismatch: {strategy} K{k} error={error}')
            records.append({'strategy':strategy,'k':k,'signals':len(groups),'days':len(d),'maximum_absolute_nav_error':error})
    result={'status':'PASS','scope':'30 actual historical TopK paths, first 12 feasible signals per strategy, unchanged production fractional matcher',
        'maximum_absolute_nav_error':max(r['maximum_absolute_nav_error'] for r in records),'checks':records}
    (out/'real_data_engine_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'], 'max NAV difference',result['maximum_absolute_nav_error'])


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=ROOT/'output/mvp_holdings_count_20261017_complete_factors')
    validate(a.parse_args().output)
