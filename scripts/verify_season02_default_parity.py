#!/usr/bin/env python3
"""Compare disabled-buffer core backtest against its committed pre-integration code."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from config import get_settings
from backtest.backtest_single import run_single_backtest
from scripts.verify_season02_integration import sha,write_json


def run(output):
    if output.exists():raise FileExistsError('fresh output required')
    old=types.ModuleType('prior_backtest')
    source=subprocess.check_output(['git','show','HEAD:backtest/backtest_single.py'],cwd=ROOT,text=True)
    exec(compile(source,'HEAD:backtest/backtest_single.py','exec'),old.__dict__)
    rng=np.random.default_rng(20261019);dates=pd.bdate_range('2024-01-01',periods=130);symbols=list('ABCDEFGH')
    prices=pd.DataFrame(10*np.cumprod(1+rng.normal(.0002,.02,(len(dates),len(symbols))),axis=0),index=dates,columns=symbols)
    scores=pd.DataFrame(rng.normal(size=prices.shape),index=dates,columns=symbols).stack()
    base=replace(get_settings(),backtest_start=str(dates[0].date()),backtest_end=str(dates[-1].date()),
        rebalance_freq='W-FRI',force_final_rebalance=False,portfolio_weighting='equal',max_position_weight=0.,
        max_industry_weight=0.,max_tech_growth_weight=0.,min_positions=0,target_volatility=0.,
        max_rebalance_turnover=0.,min_avg_volume=0.,min_avg_amount=0.,enable_trade_status_filter=False)
    rows=[]
    for name,settings in [('equal',base),('turnover_cap',replace(base,max_rebalance_turnover=.2)),
                           ('risk_parity',replace(base,portfolio_weighting='risk_parity')),
                           ('position_cap',replace(base,max_position_weight=.4))]:
        a,ma=old.run_single_backtest('parity',prices=prices,factor_values=scores,settings=settings,top_k=3)
        b,mb=run_single_backtest('parity',prices=prices,factor_values=scores,settings=settings,top_k=3)
        pd.testing.assert_series_equal(a,b,check_exact=True)
        if json.dumps(ma,sort_keys=True,default=str)!=json.dumps(mb,sort_keys=True,default=str):raise AssertionError('metadata changed '+name)
        rows.append(dict(case=name,nav_exact=True,decisions_exact=True))
    output.mkdir(parents=True)
    write_json(output/'result.json',dict(status='PASS',cases=rows,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        current_code_hash=sha(ROOT/'backtest/backtest_single.py'),seed=20261019,
        scope='synthetic core-engine regression; six real historical baselines separately checked'))
    print('Default core parity PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.output)
