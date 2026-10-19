#!/usr/bin/env python3
"""Check full date coverage, market identity and daily/adjustment joins, streaming."""
import argparse
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.fetch_long_history_dataset import json_write, audit, sha


def validate_aggregates(root):
    """Verify every assembled file against freshly audited part row counts/schema."""
    root = Path(root)
    coverage = audit(root)
    errors, files = [], {}
    if not coverage['core_partition_download_complete']:
        errors.append('Expected core partitions are missing')
    for endpoint, record in coverage['datasets'].items():
        path = root/f'{endpoint}.csv.gz'
        if not path.exists():
            errors.append(f'{endpoint}: aggregate missing')
            continue
        part_columns = set()
        for receipt in (root/'parts'/endpoint).glob('*.csv.json'):
            part_columns.update(json.loads(receipt.read_text())['columns'])
        columns = pd.read_csv(path, nrows=0).columns
        if set(columns) != part_columns:
            errors.append(f'{endpoint}: aggregate schema mismatch')
        rows = sum(len(chunk) for chunk in pd.read_csv(path, dtype=str, usecols=[columns[0]], chunksize=100000))
        if rows != record['rows']:
            errors.append(f'{endpoint}: aggregate rows {rows} != partition rows {record["rows"]}')
        files[path.name] = {'rows':rows, 'bytes':path.stat().st_size, 'sha256':sha(path)}
        print(f'aggregate verified {endpoint} rows={rows}', flush=True)
    for stage in ['market','finance','industry','audit_opinions']:
        path = root/f'{stage}_failures.json'
        if not path.exists() or json.loads(path.read_text()):
            errors.append(f'{stage}: failure manifest missing or nonempty')
    report = {'status':'PASS' if not errors else 'FAIL', 'files':files, 'errors':errors,
              'scope':'Download/assembly integrity only; not PIT or strategy certification'}
    json_write(root/'aggregate_validation.json', report)
    return not errors


def summarize_source_quality(root):
    """Keep raw provider returns; expose records that need downstream universe review."""
    root = Path(root)
    master = set(pd.read_csv(root/'stock_basic.csv.gz', usecols=['ts_code'], dtype=str).ts_code)
    alias_path = root/'identity_aliases.json'
    if alias_path.exists():
        master.update(item['old_code'] for item in json.loads(alias_path.read_text())['aliases'])
    datasets = {}
    for path in sorted(root.glob('*.csv.gz')):
        if 'ts_code' not in pd.read_csv(path, nrows=0).columns:
            continue
        b_codes, unknown_codes, b_rows, unknown_rows = set(), set(), 0, 0
        for chunk in pd.read_csv(path, usecols=['ts_code'], dtype=str, chunksize=200000):
            codes = chunk.ts_code.fillna('')
            b_mask = codes.str.match(r'^(?:20\d{4}\.SZ|900\d{3}\.SH)$')
            unknown_mask = ~codes.isin(master) & ~b_mask
            b_rows += int(b_mask.sum())
            unknown_rows += int(unknown_mask.sum())
            b_codes.update(codes[b_mask])
            unknown_codes.update(codes[unknown_mask])
        datasets[path.name] = {'b_share_rows':b_rows, 'b_share_codes':sorted(b_codes),
                              'non_b_codes_outside_master_rows':unknown_rows,
                              'non_b_codes_outside_master':sorted(unknown_codes)}
    report = {'status':'RAW_SNAPSHOT_REQUIRES_RESEARCH_PREPARATION', 'datasets':datasets,
              'raw_records_modified':False,
              'note':'Exchange suffix filtering does not guarantee A-share identity. Exclude B shares and resolve unmatched identities, announcement timing, revisions and code continuity before research.'}
    json_write(root/'source_quality.json', report)
    return report


def wait_for_download(root, timeout_seconds=10800):
    """Finite local post-download job; no credentials and no network calls."""
    deadline = time.monotonic() + timeout_seconds
    required = ['daily.csv.gz','adj_factor.csv.gz','daily_basic.csv.gz',
                'fina_indicator_vip.csv.gz','income_vip.csv.gz','balancesheet_vip.csv.gz',
                'cashflow_vip.csv.gz','fina_audit.csv.gz','index_member_all.csv.gz']
    while time.monotonic() < deadline:
        for stage in ['market','finance','industry','audit_opinions']:
            path = root/f'{stage}_failures.json'
            if path.exists() and json.loads(path.read_text()):
                raise RuntimeError(f'{stage} has failed partitions; resume downloader first')
        if all((root/name).exists() for name in required):
            result = audit(root)
            if not result['core_partition_download_complete']:
                raise RuntimeError('Aggregates exist but expected core partitions are missing')
            return
        time.sleep(15)
    raise TimeoutError('Download not finished within watcher window; can resume later')


def validate(root, aliases_path=None):
    calendar = pd.read_csv(root/'trade_cal.csv.gz', dtype={'cal_date':str})
    dates = sorted(calendar.loc[calendar.is_open.eq(1),'cal_date'])
    if not dates or len(dates) != len(set(dates)):
        raise ValueError('Empty or duplicated open-session calendar')
    basic = pd.read_csv(root/'stock_basic.csv.gz', dtype=str)
    master = set(basic.ts_code)
    aliases = {}
    if aliases_path is not None:
        for item in json.loads(Path(aliases_path).read_text())['aliases']:
            old = item['old_code']
            pd.to_datetime(item['effective_date'], format='%Y%m%d', errors='raise')
            if old in aliases or old in master or item['new_code'] not in master or not item['source_url'].startswith('https://'):
                raise ValueError(f'Invalid historical identity alias: {old}')
            aliases[old] = item
    alias_rows = {old:0 for old in aliases}
    delisted = set(basic.loc[basic.list_status.eq('D'),'ts_code'])
    errors, yearly, symbols, covered_delisted, unknown = [], {}, set(), set(), set()
    for n,date in enumerate(dates,1):
        rawpath = root/'parts'/'daily'/f'{date}.csv.gz'
        adjpath = root/'parts'/'adj_factor'/f'{date}.csv.gz'
        if not rawpath.exists() or not adjpath.exists():
            errors.append({'date':date,'reason':'missing daily/adjustment partition'})
            continue
        raw = pd.read_csv(rawpath, dtype={'ts_code':str,'trade_date':str})
        adj = pd.read_csv(adjpath, dtype={'ts_code':str,'trade_date':str})
        if raw.empty or adj.empty:
            errors.append({'date':date,'reason':'empty daily/adjustment partition'})
            continue
        if not raw.trade_date.eq(date).all() or not adj.trade_date.eq(date).all():
            errors.append({'date':date,'reason':'partition contains wrong trade date'})
            continue
        if raw.duplicated(['ts_code','trade_date']).any() or adj.duplicated(['ts_code','trade_date']).any():
            errors.append({'date':date,'reason':'duplicate daily keys'})
            continue
        merged = raw.merge(adj,on=['ts_code','trade_date'],how='left',validate='one_to_one')
        factors = pd.to_numeric(merged.adj_factor, errors='coerce')
        prices = merged[['open','high','low','close']].apply(pd.to_numeric, errors='coerce')
        bad_adj = ~np.isfinite(factors) | factors.le(0)
        bad_price = (~np.isfinite(prices) | prices.le(0)).any(axis=1)
        bad_price |= prices.high.lt(prices[['open','low','close']].max(axis=1))
        bad_price |= prices.low.gt(prices[['open','high','close']].min(axis=1))
        if bad_adj.any() or bad_price.any():
            errors.append({'date':date,'reason':'invalid price/adjustment','bad_adj':int(bad_adj.sum()),'bad_price':int(bad_price.sum())})
        codes = set(raw.ts_code)
        symbols.update(codes)
        covered_delisted.update(codes & delisted)
        unknown.update(codes-master-set(aliases))
        for old in codes & set(aliases):
            alias_rows[old] += int(raw.ts_code.eq(old).sum())
            if date >= aliases[old]['effective_date']:
                errors.append({'date':date,'reason':'old identity code after official change date','symbol':old})
        year = date[:4]
        record = yearly.setdefault(year,{'sessions':0,'rows':0,'min_daily_stocks':len(raw),'max_daily_stocks':0})
        record['sessions'] += 1
        record['rows'] += len(raw)
        record['min_daily_stocks'] = min(record['min_daily_stocks'],len(raw))
        record['max_daily_stocks'] = max(record['max_daily_stocks'],len(raw))
        if n % 500 == 0:
            print(f'validated {n}/{len(dates)}',flush=True)
    if unknown:
        errors.append({'reason':'price symbols absent from identity master','symbols':sorted(unknown)})
    report = {'status':'PASS' if not errors else 'FAIL','scope':'market coverage and joins only; not strategy/PIT certification',
              'expected_sessions':len(dates),'first_date':dates[0],'last_date':dates[-1],
              'price_symbols':len(symbols),'delisted_symbols_with_prices':len(covered_delisted),
              'identity_alias_rows':alias_rows,
              'identity_alias_source_sha256':sha(aliases_path) if aliases_path else None,
              'yearly':yearly,'errors':errors}
    json_write(root/'market_validation.json',report)
    print({k:v for k,v in report.items() if k not in ['yearly','errors']},'errors=',len(errors),flush=True)
    return not errors


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--wait-for-download',action='store_true')
    parser.add_argument('--verify-aggregates',action='store_true')
    parser.add_argument('--identity-aliases',type=Path)
    args=parser.parse_args()
    try:
        if args.wait_for_download:
            wait_for_download(args.data_dir)
        passed = validate(args.data_dir, args.identity_aliases)
        if args.verify_aggregates:
            passed = validate_aggregates(args.data_dir) and passed
            summarize_source_quality(args.data_dir)
    except Exception as exc:
        json_write(args.data_dir/'market_validation.json',{'status':'INCOMPLETE','reason':str(exc)})
        raise
    raise SystemExit(0 if passed else 1)
