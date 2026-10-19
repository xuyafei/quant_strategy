#!/usr/bin/env python3
"""Isolated, paginated, resumable Tushare history snapshot; never a vintage archive.

Credentials are read from TUSHARE_TOKEN or an echo-disabled terminal prompt.
All partitions are compressed, checksummed and bound to their request parameters.
Existing production data are never modified. Run --audit-only without credentials.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def csv_write(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False, compression={'method': 'gzip', 'mtime': 0})
    temp.replace(path)


def sh_sz(frame):
    if 'ts_code' not in frame:
        return frame
    return frame[frame.ts_code.astype(str).str.endswith(('.SH', '.SZ'))].copy()


def paginated(call, endpoint, params, page_size=3000, known_page_cap=False):
    """Advance by actual rows, and require an empty terminal page.

    Even a short page is followed: provider caps may be below requested limit.
    Repeated pages fail closed; exact duplicate source rows are retained/audited.
    """
    frames, pages, signatures = [], [], set()
    offset = 0
    for _ in range(200):
        frame = call(endpoint, **params, limit=page_size, offset=offset)
        pages.append({'offset': offset, 'rows': len(frame)})
        if frame.empty:
            if not frames:
                return frame, pages
            result = pd.concat(frames, ignore_index=True)
            return result, pages
        signature = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        if signature in signatures:
            raise ValueError(f'{endpoint}: repeated page; offset may be ignored')
        signatures.add(signature)
        frames.append(frame)
        offset += len(frame)
        if known_page_cap and len(frame) < page_size:
            return pd.concat(frames, ignore_index=True), pages
    raise ValueError(f'{endpoint}: pagination bound exceeded')


class Downloader:
    def __init__(self, root, token, workers=4, interval=0.15):
        self.root, self.token, self.workers = Path(root), token, workers
        self.local, self.lock = threading.local(), threading.Lock()
        self.next_call, self.interval = 0.0, interval

    def call(self, endpoint, **params):
        import tushare as ts
        if not hasattr(self.local, 'pro'):
            self.local.pro = ts.pro_api(self.token, timeout=40)
        for attempt in range(6):
            with self.lock:
                delay = max(0, self.next_call - time.monotonic())
                self.next_call = time.monotonic() + delay + self.interval
            time.sleep(delay)
            try:
                result = self.local.pro.query(endpoint, **params)
                if result is None:
                    raise ValueError('None API result')
                return result
            except Exception as exc:
                message = str(exc)
                if any(s in message for s in ['权限', '积分', '每天最多']):
                    raise RuntimeError(f'{endpoint}: permission/quota unavailable') from None
                if attempt == 5:
                    # Never emit request bodies, URLs, credentials or provider exception text.
                    raise RuntimeError(f'{endpoint}: failed ({type(exc).__name__})') from None
                time.sleep(55 if '频' in message or 'rate' in message.lower() else 2 ** attempt)

    def fetch(self, endpoint, key, params, required=(), nonempty=False):
        path = self.root / 'parts' / endpoint / f'{key}.csv.gz'
        receipt = path.with_suffix('.json')
        request = {'endpoint': endpoint, 'params': params}
        if path.exists() and receipt.exists():
            meta = json.loads(receipt.read_text())
            if meta['request'] != request:
                raise ValueError(f'{path}: request mismatch')
            if sha(path) != meta['sha256']:
                raise ValueError(f'{path}: checksum mismatch')
            return path
        if shutil.disk_usage(self.root).free < 2 * 1024 ** 3:
            raise RuntimeError('Disk safety stop: less than 2 GiB free')
        # Official daily/daily_basic docs specify a 6000-row cap.
        # Other endpoints use an explicit empty terminal page instead.
        market_endpoint = endpoint in {'daily', 'daily_basic'}
        frame, pages = paginated(self.call, endpoint, params,
                                 page_size=6000 if market_endpoint else 3000,
                                 known_page_cap=market_endpoint)
        if not frame.empty and not set(required).issubset(frame.columns):
            raise ValueError(f'{endpoint}: missing required columns')
        frame = sh_sz(frame)
        if nonempty and frame.empty:
            raise ValueError(f'{endpoint} {key}: unexpectedly empty')
        if frame.empty and len(frame.columns) == 0:
            frame = pd.DataFrame(columns=list(required))
        if 'trade_date' in params and not frame.empty:
            if not frame.trade_date.astype(str).eq(params['trade_date']).all():
                raise ValueError(f'{endpoint}: wrong trade date')
            if frame.ts_code.duplicated().any():
                raise ValueError(f'{endpoint}: duplicate daily stock')
        if 'period' in params and not frame.empty:
            if not frame.end_date.astype(str).eq(params['period']).all():
                raise ValueError(f'{endpoint}: wrong report period')
        csv_write(path, frame)
        json_write(receipt, {'request': request, 'rows': len(frame), 'columns': list(frame.columns),
                            'pages': pages, 'duplicate_rows': int(frame.duplicated().sum()), 'sha256': sha(path),
                            'fetched_utc': datetime.now(timezone.utc).isoformat()})
        return path

    def batch(self, name, jobs):
        failures = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch, *job): job[:2] for job in jobs}
            for n, f in enumerate(as_completed(futures), 1):
                try:
                    f.result()
                except Exception as exc:
                    failures.append({'endpoint': futures[f][0], 'key': futures[f][1], 'error': str(exc)})
                if n % 50 == 0 or n == len(futures):
                    print(f'{name} {n}/{len(futures)} failures={len(failures)}', flush=True)
        json_write(self.root / f'{name}_failures.json', failures)
        if failures:
            raise RuntimeError(f'{name}: {len(failures)} failed partitions; see failure manifest')


def quarter_ends(start, end):
    return [p.end_time.strftime('%Y%m%d') for p in pd.period_range(start, end, freq='Q')
            if p.end_time.normalize() <= pd.Timestamp(end)]


def audit_symbols(root, basic, cutoff):
    """Include documented retired codes without changing the provider identity master."""
    symbols = set(basic.loc[basic.list_date.le(cutoff), 'ts_code'].unique())
    path = Path(root)/'identity_aliases.json'
    if path.exists():
        for item in json.loads(path.read_text())['aliases']:
            if item['new_code'] not in set(basic.ts_code):
                raise ValueError('Historical alias target absent from identity master')
            symbols.add(item['old_code'])
    return sorted(symbols)


def merge_parts(paths, output):
    """Stream CSVs using a union schema (financial API schemas can change)."""
    paths = sorted(paths)
    columns = list(dict.fromkeys(c for p in paths for c in pd.read_csv(p, nrows=0).columns))
    temp = output.with_name(output.name + '.tmp')
    rows = 0
    with gzip.open(temp, 'wt') as target:
        pd.DataFrame(columns=columns).to_csv(target, index=False)
        for p in paths:
            df = pd.read_csv(p, dtype=str).reindex(columns=columns)
            df.to_csv(target, index=False, header=False)
            rows += len(df)
    temp.replace(output)
    return rows


def audit(root):
    root = Path(root)
    spec = json.loads((root/'download_spec.json').read_text())
    cutoff = pd.Timestamp(spec['end']).strftime('%Y%m%d')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'datasets': {},
              'snapshot_is_not_vintage_archive': True,
              'production_inputs_changed': False}
    for folder in sorted((root / 'parts').iterdir()):
        if not folder.is_dir():
            continue
        rows, symbols, dates, periods, missing, duplicates = 0, set(), set(), set(), 0, 0
        announcements, after_cutoff, empty_parts = set(), 0, 0
        parts = sorted(folder.glob('*.csv.gz'))
        for p in parts:
            meta = json.loads(p.with_suffix('.json').read_text())
            if sha(p) != meta['sha256']:
                raise ValueError(f'{p}: checksum failed')
            cols = pd.read_csv(p, nrows=0).columns
            selected = [c for c in ['ts_code', 'trade_date', 'end_date', 'ann_date'] if c in cols]
            df = pd.read_csv(p, usecols=selected, dtype=str) if selected else pd.read_csv(p, dtype=str)
            if len(df) != meta['rows']:
                raise ValueError(f'{p}: row count failed')
            rows += len(df)
            empty_parts += int(df.empty)
            duplicates += meta.get('duplicate_rows', 0)
            if 'ts_code' in df:
                symbols.update(df.ts_code.dropna())
            if 'trade_date' in df:
                dates.update(df.trade_date.dropna())
            if 'end_date' in df:
                periods.update(df.end_date.dropna())
            if 'ann_date' in df:
                missing += int(df.ann_date.isna().sum())
                announcements.update(df.ann_date.dropna())
                after_cutoff += int(df.ann_date.dropna().gt(cutoff).sum())
        result['datasets'][folder.name] = {'partitions': len(parts), 'rows': rows, 'symbols': len(symbols),
            'trade_date_min': min(dates) if dates else None, 'trade_date_max': max(dates) if dates else None,
            'trade_dates': len(dates), 'period_min': min(periods) if periods else None,
            'period_max': max(periods) if periods else None, 'missing_ann_date': missing,
            'empty_partitions': empty_parts,
            'ann_date_min': min(announcements) if announcements else None,
            'ann_date_max': max(announcements) if announcements else None,
            'ann_date_after_snapshot_end_rows': after_cutoff,
            'source_duplicate_rows': duplicates}
    cal_path = root/'trade_cal.csv.gz'
    expected, missing_parts = {}, {}
    if cal_path.exists():
        calendar = pd.read_csv(cal_path, dtype={'cal_date': str})
        dates = sorted(calendar.loc[calendar.is_open.eq(1), 'cal_date'].tolist())
        weeks = pd.DataFrame({'date': pd.to_datetime(dates)})
        weekly = weeks.groupby(weeks.date.dt.to_period('W-FRI')).date.max().dt.strftime('%Y%m%d').tolist()
        expected = {'daily': dates, 'adj_factor': dates, 'daily_basic': weekly}
        for ep in ['fina_indicator_vip','income_vip','balancesheet_vip','cashflow_vip']:
            expected[ep] = quarter_ends(spec['finance_start'], spec['end'])
        basic_path = root/'stock_basic.csv.gz'
        if basic_path.exists():
            basic = pd.read_csv(basic_path, dtype=str)
            cutoff = pd.Timestamp(spec['end']).strftime('%Y%m%d')
            expected['fina_audit'] = [s.replace('.', '_') for s in audit_symbols(root, basic, cutoff)]
        expected.update({'stock_basic':['L','D','P'], 'trade_cal':['SSE'], 'namechange':['all'],
                         'index_classify':['SW2014','SW2021']})
        classification_paths = [root/'parts'/'index_classify'/f'{v}.csv.gz' for v in ['SW2014','SW2021']]
        if all(p.exists() for p in classification_paths):
            classification = pd.concat([pd.read_csv(p, dtype=str) for p in classification_paths])
            expected['index_member_all'] = [code.replace('.', '_')+'_'+flag
                for code in sorted(classification.index_code.unique()) for flag in ['Y','N']]
        for ep, keys in expected.items():
            missing_parts[ep] = [k for k in keys if not (root/'parts'/ep/f'{k}.csv.gz').exists()]
    result['expected_partition_counts'] = {ep: len(keys) for ep,keys in expected.items()}
    result['missing_partitions'] = missing_parts
    result['core_partition_download_complete'] = bool(expected) and not any(missing_parts.values())
    result['research_ready'] = False
    result['research_ready_reason'] = 'Requires PIT financial/industry quality review and new research protocol; does not change production.'
    json_write(root / 'coverage_audit.json', result)
    return result


def print_audit_summary(result):
    summary = {k:v for k,v in result.items() if k != 'missing_partitions'}
    summary['missing_partition_counts'] = {k:len(v) for k,v in result.get('missing_partitions', {}).items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stages', default='reference,market,finance,audit_opinions,industry,assemble')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--identity-aliases', type=Path, help='Evidence-backed old codes for audit-opinion coverage')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.audit_only:
        print_audit_summary(audit(root))
        return
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    today = pd.Timestamp.now(tz='Asia/Shanghai').tz_localize(None).normalize()
    if start > end or end >= today:
        raise ValueError('Use a completed prior date, with start <= end')
    # Restrict writes to a dedicated new snapshot, never the production caches.
    if not root.name.startswith('all_a_history_'):
        raise ValueError('output name must start with all_a_history_')
    root.mkdir(parents=True, exist_ok=True)
    spec = {'version': 1, 'start': args.start, 'end': args.end, 'exchanges': ['SH', 'SZ'],
            'finance_start': '2009-01-01', 'daily_basic_frequency': 'weekly_last_open',
            'statement_report_type': '1', 'financial_snapshot_not_vintage_archive': True}
    spec_path = root / 'download_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('snapshot specification mismatch')
    json_write(spec_path, spec)
    if args.identity_aliases:
        aliases = json.loads(args.identity_aliases.read_text())
        aliases_path = root/'identity_aliases.json'
        if aliases_path.exists() and json.loads(aliases_path.read_text()) != aliases:
            raise ValueError('identity alias specification mismatch')
        json_write(aliases_path, aliases)
    token = '' if args.stages == 'assemble' else (
        os.environ.get('TUSHARE_TOKEN', '').strip() or getpass.getpass('Tushare token: '))
    dl = Downloader(root, token, args.workers)
    compact = lambda d: pd.Timestamp(d).strftime('%Y%m%d')
    stages = args.stages.split(',')
    if 'reference' in stages:
        for status in ['L', 'D', 'P']:
            dl.fetch('stock_basic', status, {'exchange': '', 'list_status': status,
                'fields': 'ts_code,symbol,name,industry,market,exchange,list_status,list_date,delist_date'},
                ['ts_code', 'list_date'])
        dl.fetch('trade_cal', 'SSE', {'exchange': 'SSE', 'start_date': compact(start), 'end_date': compact(end)}, ['cal_date', 'is_open'], True)
        dl.fetch('namechange', 'all', {'fields': 'ts_code,name,start_date,end_date,ann_date,change_reason'}, ['ts_code', 'start_date'], True)
        for endpoint in ['stock_basic', 'trade_cal', 'namechange']:
            merge_parts(list((root/'parts'/endpoint).glob('*.csv.gz')), root/f'{endpoint}.csv.gz')
    calendar = pd.read_csv(root/'trade_cal.csv.gz', dtype={'cal_date': str})
    dates = sorted(calendar.loc[calendar.is_open.eq(1), 'cal_date'].tolist())
    week_frame = pd.DataFrame({'date': pd.to_datetime(dates)})
    weekly = week_frame.groupby(week_frame.date.dt.to_period('W-FRI')).date.max().dt.strftime('%Y%m%d').tolist()
    if 'market' in stages:
        jobs = [(ep, d, {'trade_date': d}, ['ts_code', 'trade_date', field], True)
                for d in dates for ep, field in [('daily', 'close'), ('adj_factor', 'adj_factor')]]
        jobs += [('daily_basic', d, {'trade_date': d}, ['ts_code', 'trade_date', 'total_mv'], True) for d in weekly]
        dl.batch('market', jobs)
    if 'finance' in stages:
        jobs = []
        for period in quarter_ends('2009-01-01', args.end):
            for ep in ['fina_indicator_vip', 'income_vip', 'balancesheet_vip', 'cashflow_vip']:
                params = {'period': period}
                if ep != 'fina_indicator_vip':
                    params['report_type'] = '1'
                jobs.append((ep, period, params, ['ts_code', 'ann_date', 'end_date'], False))
        dl.batch('finance', jobs)
    if 'audit_opinions' in stages:
        basic = pd.read_csv(root/'stock_basic.csv.gz', dtype=str)
        symbols = audit_symbols(root, basic, compact(end))
        jobs = [('fina_audit', s.replace('.', '_'), {'ts_code': s, 'start_date': '20090101',
                 'end_date': compact(end)}, ['ts_code', 'ann_date', 'end_date'], False) for s in symbols]
        dl.batch('audit_opinions', jobs)
    if 'industry' in stages:
        for version in ['SW2014', 'SW2021']:
            dl.fetch('index_classify', version, {'level': 'L1', 'src': version}, ['index_code'], True)
        classify = pd.concat([pd.read_csv(p) for p in (root/'parts'/'index_classify').glob('*.csv.gz')])
        jobs = [('index_member_all', code.replace('.', '_')+'_'+flag, {'l1_code': code, 'is_new': flag},
                 ['ts_code', 'in_date'], False) for code in sorted(classify.index_code.unique()) for flag in ['Y','N']]
        dl.batch('industry', jobs)
    if 'assemble' in stages:
        for folder in sorted((root/'parts').iterdir()):
            if folder.is_dir():
                count = merge_parts(list(folder.glob('*.csv.gz')), root/f'{folder.name}.csv.gz')
                print(f'assembled {folder.name} rows={count}', flush=True)
        print_audit_summary(audit(root))


if __name__ == '__main__':
    main()
