#!/usr/bin/env python3
"""Verify migrated matching code against all 86 archived paper11 signal dates.

Uses six paper quality portfolios, NOT the six daily MVP strategies. Also checks
archived paper12 fit boundaries; does not refit models or certify all raw data.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.verify_season02_integration import sha,write_json,account_hashes
from scripts.run_characteristic_audit import run as run_matching
from analysis.training_time import validate_training_folds


def run(paper_root,output):
    out=output.resolve()
    if out.exists():raise FileExistsError('fresh output required')
    p10=paper_root/'output/paper10_quality_20261014';p11=paper_root/'output/paper11_characteristic_benchmarks_20261015'
    price=ROOT/'data/all_a_dynamic_20230101_20260904/prices_all_a_qfq.csv.gz'
    folds=paper_root/'docs/paper_lab/season02/results/paper12_long/long/folds.csv'
    paths=[p11/'memberships_3.csv.gz',p10/'target_weights.csv.gz',p11/'matched_targets_3.csv.gz',
           p11/'matched_nav_3_0.0003.csv',p10/'nav_0.0003.csv',price,folds]
    for folder,names in [(p10,['target_weights.csv.gz','nav_0.0003.csv']),
                         (p11,['memberships_3.csv.gz','matched_targets_3.csv.gz','matched_nav_3_0.0003.csv'])]:
        manifest=json.loads((folder/'manifest.json').read_text());paths.append(folder/'manifest.json')
        for name in names:
            if sha(folder/name)!=manifest['output_hashes'][name]:raise ValueError('archive mismatch '+name)
        if sha(price)!=manifest['input_hashes'][str(price)]:raise ValueError('price archive mismatch')
    inputs={str(f):sha(f) for f in paths};before=account_hashes()
    code=['analysis/characteristic_benchmarks.py','analysis/training_time.py','backtest/research_replay.py',
          'scripts/run_characteristic_audit.py','scripts/verify_season02_archives.py']
    code_before={n:sha(ROOT/n) for n in code}
    out.mkdir(parents=True);write_json(out/'started.json',{'inputs':inputs,'forward_before':before})
    features=pd.read_csv(paths[0],usecols=['signal_date','symbol','total_mv_yuan','VALUE_BP','MOM_12_1']).rename(columns={'signal_date':'date'})
    targets=pd.read_csv(paths[1]).rename(columns={'signal_date':'date','arm':'strategy'})
    targets=targets[targets.strategy.ne('BENCHMARK')][['date','execution_date','strategy','symbol','weight']]
    features.to_csv(out/'features.csv.gz',index=False);targets.to_csv(out/'targets.csv.gz',index=False)
    def entry(path):return dict(path=str(path.resolve()),sha256=sha(path))
    spec=dict(features=entry(out/'features.csv.gz'),targets=entry(out/'targets.csv.gz'),prices=entry(price),
        start='2025-01-03',end='2026-09-04',bins=3,minimum_peers=20,fee=.0003)
    write_json(out/'spec.json',spec);nav=run_matching(out/'spec.json',out/'matching')
    matched=pd.read_csv(out/'matching/matched_targets.csv.gz',parse_dates=['date'])
    old=pd.read_csv(paths[2],parse_dates=['signal_date']).rename(columns={'signal_date':'date','arm':'strategy'})
    old=old[old.strategy.ne('BENCHMARK')]
    keys=['date','strategy','symbol']
    joined=matched.merge(old[keys+['weight']],on=keys,how='outer',suffixes=('_new','_old'),indicator=True,validate='one_to_one')
    if not joined._merge.eq('both').all():raise AssertionError('matched constituent mismatch')
    target_error=float((joined.weight_new-joined.weight_old).abs().max())
    if target_error>1e-12:raise AssertionError('matched weight mismatch')
    checks=[]
    for kind,path in [('MATCHED',paths[3]),('ORIGINAL',paths[4])]:
        ref=pd.read_csv(path,index_col='date',parse_dates=['date'])
        if not nav.index.equals(ref.index):raise AssertionError('calendar mismatch')
        for strategy in sorted(targets.strategy.unique()):
            error=float((nav[strategy+'__'+kind+'__PRETRADE']-ref[strategy]).abs().max())
            if error>1e-9:raise AssertionError('NAV changed '+strategy)
            checks.append(dict(kind=kind,strategy=strategy,max_nav_error=error))
    f=pd.read_csv(folds,parse_dates=['fit_date','train_first','train_last_label','validation_first','validation_last_label'])
    time_check=validate_training_folds(f,minimum_train_periods=156)
    if before!=account_hashes():raise AssertionError('forward accounts changed')
    if inputs!={name:sha(name) for name in inputs}:raise AssertionError('inputs changed')
    if code_before!={n:sha(ROOT/n) for n in code}:raise AssertionError('code changed during validation')
    pd.DataFrame(checks).to_csv(out/'nav_parity.csv',index=False)
    write_json(out/'complete.json',dict(status='PASS',matched_rows=len(joined),signals=int(matched.date.nunique()),
        strategies=int(targets.strategy.nunique()),target_max_error=target_error,nav_checks=checks,
        fit_boundary_checks=len(f),time_check=time_check,models_refitted=False,forward_unchanged=True,live_approval=False,
        inputs=inputs,code_hashes=code_before,outputs={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}))
    print('ARCHIVE VALIDATION PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.paper_root,a.output)
