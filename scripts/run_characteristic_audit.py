#!/usr/bin/env python3
"""Generate/replay local characteristic comparators from hash-declared snapshots.

Input spec: features (date,symbol,total_mv_yuan,VALUE_BP,MOM_12_1), targets
(date,execution_date,strategy,symbol,weight), prices (trade_date,ts_code,adj_close).
All three entries require path,sha256. Explicit start,end,bins,minimum_peers,fee.
Not a full DGTW replication; no production decisions or live account writes.
"""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from analysis.characteristic_benchmarks import build_matched_targets
from backtest.research_replay import replay
from scripts.verify_season02_integration import sha, write_json


def run(spec_path,output):
    out=Path(output).resolve();spec_path=Path(spec_path).resolve()
    if out.exists():raise FileExistsError('new output required')
    spec=json.loads(spec_path.read_text());inputs={str(spec_path):sha(spec_path)}
    def load(key):
        entry=spec[key];path=(spec_path.parent/entry['path']).resolve()
        if sha(path)!=entry['sha256']:raise ValueError('hash mismatch '+key)
        inputs[str(path)]=entry['sha256']
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as f:
            header=next(csv.reader(f))
        if len(header)!=len(set(header)):raise ValueError('duplicate headers')
        return pd.read_csv(path)
    features,targets,raw=load('features'),load('targets'),load('prices')
    raw.trade_date=pd.to_datetime(raw.trade_date,errors='raise')
    raw=raw[raw.trade_date.between(spec['start'],spec['end'])]
    if raw.duplicated(['trade_date','ts_code']).any():raise ValueError('duplicate prices')
    prices=raw.pivot(index='trade_date',columns='ts_code',values='adj_close').sort_index()
    matched,audits=build_matched_targets(features,targets,bins=spec['bins'],minimum_peers=spec['minimum_peers'])
    targets=targets.copy();targets.date=pd.to_datetime(targets.date);targets.execution_date=pd.to_datetime(targets.execution_date)
    navs={};executions=[]
    for kind,plans in [('ORIGINAL',targets),('MATCHED',matched)]:
        for strategy,g in plans.groupby('strategy'):
            schedule=g[['date','execution_date']].drop_duplicates().set_index('date')
            groups={date:z.set_index('symbol').weight for date,z in g.groupby('date')}
            result=replay(prices,schedule,lambda date,held:groups[date],fee=spec['fee'])
            name=strategy+'__'+kind
            navs[name]=result['daily'].post_nav
            navs[name+'__PRETRADE']=result['daily'].pre_nav
            e=result['executions'];e['strategy']=strategy;e['kind']=kind;executions.append(e)
            print('matched audit replay',name,flush=True)
    if inputs!={name:sha(name) for name in inputs}:raise ValueError('inputs changed')
    out.mkdir(parents=True)
    matched.to_csv(out/'matched_targets.csv.gz',index=False)
    audits.to_csv(out/'matching_quality.csv.gz',index=False)
    pd.DataFrame(navs).to_csv(out/'nav.csv',index_label='date')
    pd.concat(executions).to_csv(out/'executions.csv',index=False)
    write_json(out/'spec.json',spec)
    write_json(out/'manifest.json',dict(inputs=inputs,diagnostic_only=True,live_approval=False,
        point_in_time_data_check='CALLER_PROVIDED_SNAPSHOTS_NOT_VINTAGE_CERTIFICATION',
        code_hashes={n:sha(ROOT/n) for n in ['analysis/characteristic_benchmarks.py','backtest/research_replay.py','scripts/run_characteristic_audit.py']},
        outputs={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    return pd.DataFrame(navs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.spec,a.output)
