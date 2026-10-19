#!/usr/bin/env python3
"""Explicit MVP financial fields missing from Tushare's default VIP field set.

Read credential only from environment or a hidden interactive prompt. Separate
supplement: never rewrites the verified long-history provider aggregates.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
import tushare as ts

FIELDS = 'ts_code,ann_date,end_date,eps,roe,grossprofit_margin,netprofit_margin,debt_to_assets,or_yoy,netprofit_yoy,ocfps,ocf_to_profit'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/mvp_history_finance_20261017'))
    args = parser.parse_args()
    token = os.environ.get('TUSHARE_TOKEN','').strip() or getpass.getpass('Tushare token (hidden): ')
    pro = ts.pro_api(token, timeout=40)
    args.output.mkdir(parents=True, exist_ok=True)
    frames=[]
    periods=pd.date_range('2009-03-31','2026-09-30',freq='QE')
    for i,period in enumerate(periods):
        key=period.strftime('%Y%m%d')
        path=args.output/(key+'.csv.gz')
        frame=pd.read_csv(path) if path.exists() else None
        if frame is None or len(frame) == 12000:
            for attempt in range(3):
                try:
                    pages=[]
                    offset=0
                    while True:
                        page=pro.fina_indicator_vip(period=key, fields=FIELDS, limit=12000, offset=offset)
                        pages.append(page)
                        if len(page) < 12000:
                            break
                        offset += len(page)
                    frame=pd.concat(pages,ignore_index=True)
                    if frame.empty or 'ocf_to_profit' not in frame:
                        raise ValueError('empty response or missing required field')
                    frame.to_csv(path,index=False)
                    break
                except Exception as error:
                    if attempt==2:
                        # Provider messages could contain request credentials.
                        raise RuntimeError(f'finance request failed: {key}, {type(error).__name__}') from None
                    time.sleep(1+attempt)
            time.sleep(.2)
        frames.append(frame)
        print(f'{i+1}/{len(periods)} {key}: rows={len(frame)}, cash_quality_valid={frame.ocf_to_profit.notna().sum()}',flush=True)
    total=pd.concat(frames,ignore_index=True).drop_duplicates(['ts_code','ann_date','end_date'],keep='last')
    path=args.output/'fina_indicator_mvp.csv.gz'
    total.to_csv(path,index=False)
    manifest={'status':'COMPLETE','endpoint':'fina_indicator_vip','fields':FIELDS,'rows':len(total),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'periods':len(periods),
        'cash_quality_valid':int(total.ocf_to_profit.notna().sum())}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
