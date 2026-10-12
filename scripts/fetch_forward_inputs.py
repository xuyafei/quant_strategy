#!/usr/bin/env python3
"""Refresh research inputs, with explicit retrieval receipts (never credentials).

Copy the historical dataset for the first forward run. Subsequent refreshes use
the same destination and reuse daily market parts, but refresh financial/risk
announcements. This is data preparation only: it never generates broker orders.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from config import get_tushare_token
from scripts.fetch_all_a_dynamic_universe_dataset import (
    run as refresh_base, _pro, _call_with_retry, _atomic_csv, _quarter_ends,
)
from scripts.fetch_strategy_universe_data import run as refresh_styles


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(source: Path, output: Path, end: str, workers: int = 6) -> None:
    token = get_tushare_token()
    if not output.exists():
        shutil.copytree(source, output)
    # Remove the success receipt before refresh; partial failure is not fresh data.
    receipt = output / "forward_input_receipt.json"
    if receipt.exists():
        receipt.replace(output / "forward_input_receipt.previous.json")
    refresh_base("20230101", end, output, workers=workers)
    calendar = _call_with_retry(lambda: _pro(token).trade_cal(
        exchange="SSE", start_date="20230101",
        end_date=(pd.Timestamp(end) + pd.Timedelta(days=100)).strftime("%Y%m%d")), "forward calendar")
    _atomic_csv(calendar, output / "trade_calendar.csv")
    refresh_styles(output, "20230101", end, workers=workers, refresh_industry=True)

    # Recent announcements can revise an older report: refresh every report period.
    balances = []
    for period in _quarter_ends(pd.Timestamp("2023-01-01"), pd.Timestamp(end)):
        frame = _call_with_retry(lambda p=period: _pro(token).balancesheet_vip(
            period=p, fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_hldr_eqy_exc_min_int,update_flag"),
            "balance " + period)
        if not frame.empty:
            balances.append(frame)
    balance = pd.concat(balances, ignore_index=True)
    balance = balance[balance.ts_code.str.endswith((".SH", ".SZ"))]
    _atomic_csv(balance.drop_duplicates(), output / "balance_sheet_all_a.csv.gz", compression="gzip")

    basic = pd.read_csv(output / "stock_basic_all_a.csv")
    symbols = sorted(basic.ts_code.unique())
    # Fresh audit query for each symbol; do not label the old cached audit as refreshed.
    def audit(symbol: str) -> pd.DataFrame:
        return _call_with_retry(lambda: _pro(token).fina_audit(
            ts_code=symbol, start_date="20220101", end_date=pd.Timestamp(end).strftime("%Y%m%d")), "audit " + symbol)
    audits = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(audit, symbols), 1):
            if not frame.empty:
                audits.append(frame)
            if i % 250 == 0 or i == len(symbols):
                print("fresh_audit=%d/%d" % (i, len(symbols)), flush=True)
    if not audits:
        raise RuntimeError("audit refresh returned no records")
    _atomic_csv(pd.concat(audits, ignore_index=True).drop_duplicates(),
                output / "fina_audit_all_a.csv.gz", compression="gzip")
    files = ["prices_all_a_qfq.csv.gz", "prices_all_a_raw.csv.gz", "adj_factor_all_a.csv.gz",
             "fina_indicator_all_a.csv.gz", "balance_sheet_all_a.csv.gz", "fina_audit_all_a.csv.gz",
             "stock_basic_all_a.csv", "namechange_all_a.csv", "trade_calendar.csv",
             "daily_basic_weekly.csv.gz", "sw2021_l1_membership.csv.gz"]
    daily = pd.read_csv(output / "parts" / "daily" / (pd.Timestamp(end).strftime("%Y%m%d") + ".csv"))
    if daily.ts_code.duplicated().any() or len(daily) < 4000:
        raise RuntimeError("incomplete/duplicate all-market signal-date prices")
    receipt.write_text(json.dumps({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "through_date": str(pd.Timestamp(end).date()),
        "provider": "Tushare Pro", "signal_daily_rows": len(daily),
        "fresh_audit_symbols": len(symbols),
        "files": {name: digest(output / name) for name in files},
        "limitations": ["vendor revisions are not historical vintage data", "missing per-stock financial data remains explicitly missing"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("forward_inputs_ready=" + str(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prompt-token", action="store_true")
    args = parser.parse_args()
    if args.prompt_token:
        os.environ["TUSHARE_TOKEN"] = getpass.getpass("Tushare token (hidden): ")
    run(args.source, args.output, args.end, args.workers)
