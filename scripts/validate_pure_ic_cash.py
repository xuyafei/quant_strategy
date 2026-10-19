#!/usr/bin/env python3
"""Independent full-history growth replay: production matcher plus cash exits."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_six_strategy_pure_ic import OUTPUT
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_mvp_holdings_count import write_json,sha
from backtest.backtest_single import _rebalance_to_target_weights


def main(out=OUTPUT):
    p=json.loads((out/'protocol.json').read_text())
    source=ROOT/p['source_experiment'];sp=json.loads((source/'protocol.json').read_text())
    ready=json.loads((out/'ready.json').read_text());strategy='GROWTH_FOCUSED'
    prices=load_prices(sp).loc[ready['common_signal']:sp['evaluation_end']]
    actions=pd.read_csv(out/'actions.csv',parse_dates=['date'])
    actions=actions[actions.strategy.eq(strategy)&actions.date.ge(ready['common_signal'])].set_index('date')
    ranks=pd.read_csv(out/'rankings.csv.gz',parse_dates=['date'])
    ranks=ranks[ranks.strategy.eq(strategy)&ranks.date.ge(ready['common_signal'])]
    groups={d:g.sort_values('rank') for d,g in ranks.groupby('date')}
    records=[]
    for k in [20,50]:
        targets={d:g.head(k).symbol.tolist() for d,g in groups.items()}
        names=pd.Index(sorted({s for selected in targets.values() for s in selected}))
        px=prices.reindex(columns=names);marks=px.ffill()
        scheduled={prices.index[prices.index.get_loc(d)+1]:d for d in actions.index if prices.index.get_loc(d)+1<len(prices)}
        for label,fee in [('nav',p['cost']),('stress_nav',p['stress_cost'])]:
            shares=pd.Series(dtype=float);cash=1.;values=[];cash_events=0
            for d in prices.index:
                if d in scheduled:
                    signal=scheduled[d];action=actions.loc[signal,'action']
                    if action=='CASH':
                        cash_events+=1
                        for stock in list(shares.index):
                            quote=px.at[d,stock]
                            if pd.notna(quote) and quote>0 and shares[stock]*quote>1e-9:
                                cash+=float(shares[stock])*float(quote)*(1-fee)
                                shares[stock]=0.
                    elif action=='TOPK':
                        selected=[s for s in targets[signal] if pd.notna(px.at[d,s])]
                        union=shares.index.union(pd.Index(selected))
                        shares=shares.reindex(union,fill_value=0.)
                        cash,shares=_rebalance_to_target_weights(cash,shares,px.loc[d].reindex(union),selected,
                            [1/k]*len(selected),fee,valuation_px=marks.loc[d].reindex(union))
                    shares=shares[shares.abs()>1e-16]
                values.append(cash+float(shares.mul(marks.loc[d].reindex(shares.index)).sum()))
            expected=pd.read_csv(out/f'common_{label}.csv',index_col='date',parse_dates=True)[f'{strategy}__PURE_IC__K{k}']
            error=float(np.max(np.abs(np.asarray(values)-expected.to_numpy())))
            if error>1e-9:raise AssertionError(f'independent cash path mismatch: {k} {fee} {error}')
            records.append({'strategy':strategy,'k':k,'fee':fee,'sessions':len(values),'cash_target_events':cash_events,'max_nav_error':error})
            print('Production matcher + independent cash exits PASS',records[-1],flush=True)
    write_json(out/'cash_path_validation.json',{'status':'PASS','code_sha256':sha(Path(__file__)),
        'full_history_paths':len(records),'max_nav_error':max(r['max_nav_error'] for r in records),'checks':records})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args().output)
