#!/usr/bin/env python3
"""Invoke the unchanged frozen daily workflow, then export reading documents."""
import argparse
import getpass
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_forward_daily import run as frozen_daily_run
from scripts.export_forward_daily_records import export


def run(root, data_dir, date, blacklist=None):
    try:
        result=frozen_daily_run(root,data_dir,date,blacklist)
    except Exception:
        # Weekly preparation can fail after the day is sealed. Still expose
        # committed holdings, but propagate the failure (no false success).
        if (Path(root)/'days'/pd.Timestamp(date).strftime('%Y%m%d')).is_dir():
            export(root)
        raise
    if result['status']=='DAY_COMMITTED':
        result['daily_records']=export(root)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--date',required=True)
    p.add_argument('--blacklist',type=Path)
    p.add_argument('--prompt-token',action='store_true')
    a=p.parse_args()
    if a.prompt_token:
        os.environ['TUSHARE_TOKEN']=getpass.getpass('Tushare token (hidden): ')
    print(run(a.session,a.data_dir,a.date,a.blacklist))
