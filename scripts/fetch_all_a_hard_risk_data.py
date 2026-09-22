#!/usr/bin/env python3
"""Fetch resumable point-in-time balance-sheet and audit-opinion risk data."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_tushare_token


_THREAD_LOCAL = threading.local()


def _pro(token: str):
    import tushare as ts

    if not hasattr(_THREAD_LOCAL, "pro"):
        _THREAD_LOCAL.pro = ts.pro_api(token)
    return _THREAD_LOCAL.pro


def _call_with_retry(fn: Callable[[], pd.DataFrame], label: str, attempts: int = 8) -> pd.DataFrame:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            frame = fn()
            return frame if frame is not None else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - network behaviour
            last = exc
            if attempt >= attempts:
                break
            message = str(exc)
            if "频率" in message or "rate" in message.lower():
                time.sleep(62.0)
            else:
                time.sleep(delay)
                delay = min(delay * 2.0, 20.0)
    raise RuntimeError("Tushare request failed for %s: %s" % (label, last))


def _atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, **kwargs)
    temp.replace(path)


def _quarter_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    first = (start - pd.DateOffset(years=1)).to_period("Q")
    last = end.to_period("Q")
    return [period.end_time.strftime("%Y%m%d") for period in pd.period_range(first, last, freq="Q")]


def _fetch_balance_period(token: str, period: str, part_dir: Path) -> tuple[str, int]:
    path = part_dir / "balance_sheet" / (period + ".csv")
    if path.is_file():
        return period, -1
    fields = (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
        "total_hldr_eqy_exc_min_int,update_flag"
    )
    frame = _call_with_retry(
        lambda: _pro(token).balancesheet_vip(period=period, fields=fields),
        "balancesheet_vip %s" % period,
    )
    frame = frame[frame["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
    _atomic_csv(frame, path)
    return period, int(len(frame))


def _symbol_part_name(symbol: str) -> str:
    return symbol.replace(".", "_") + ".csv"


def _fetch_audit_symbol(
    token: str,
    symbol: str,
    start_text: str,
    end_text: str,
    part_dir: Path,
) -> tuple[str, int]:
    path = part_dir / "fina_audit" / _symbol_part_name(symbol)
    if path.is_file():
        return symbol, -1
    frame = _call_with_retry(
        lambda: _pro(token).fina_audit(
            ts_code=symbol,
            start_date=start_text,
            end_date=end_text,
        ),
        "fina_audit %s" % symbol,
    )
    _atomic_csv(frame, path)
    return symbol, int(len(frame))


def _assemble(paths: list[Path], output: Path, subset: list[str]) -> pd.DataFrame:
    if not paths:
        raise RuntimeError("no parts found for %s" % output)
    frames = []
    for path in sorted(paths):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        out = pd.DataFrame(columns=subset)
    else:
        out = pd.concat(frames, ignore_index=True)
        available = [column for column in subset if column in out.columns]
        out = out.drop_duplicates(available, keep="last").sort_values(available)
    _atomic_csv(out, output, compression="gzip")
    return out


def run(
    data_dir: Path,
    start_text: str,
    end_text: str,
    *,
    audit_workers: int = 12,
) -> dict[str, Path]:
    token = get_tushare_token()
    start = pd.Timestamp(start_text).normalize()
    end = pd.Timestamp(end_text).normalize()
    if end < start:
        raise ValueError("end must be on or after start")
    basic_path = data_dir / "stock_basic_all_a.csv"
    if not basic_path.is_file():
        raise FileNotFoundError(basic_path)
    basic = pd.read_csv(basic_path, dtype={"ts_code": str})
    basic["list_date_parsed"] = pd.to_datetime(
        basic["list_date"].astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    )
    symbols = sorted(
        basic.loc[
            basic["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
            & basic["list_date_parsed"].le(end),
            "ts_code",
        ].astype(str).unique()
    )
    part_dir = data_dir / "risk_parts"
    balance_output = data_dir / "balance_sheet_all_a.csv.gz"
    audit_output = data_dir / "fina_audit_all_a.csv.gz"

    periods = _quarter_ends(start, end)
    for index, period in enumerate(periods, start=1):
        _period, rows = _fetch_balance_period(token, period, part_dir)
        print(
            "balance_checkpoint=%d/%d period=%s rows=%s"
            % (index, len(periods), period, "cached" if rows < 0 else rows),
            flush=True,
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(audit_workers))) as executor:
        futures = {
            executor.submit(
                _fetch_audit_symbol,
                token,
                symbol,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                part_dir,
            ): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(futures):
            symbol, _rows = future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(symbols):
                print(
                    "audit_checkpoint=%d/%d latest=%s"
                    % (completed, len(symbols), symbol),
                    flush=True,
                )

    balance = _assemble(
        list((part_dir / "balance_sheet").glob("*.csv")),
        balance_output,
        ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type"],
    )
    audit = _assemble(
        list((part_dir / "fina_audit").glob("*.csv")),
        audit_output,
        ["ts_code", "ann_date", "end_date", "audit_result"],
    )
    print(
        "risk_dataset_complete symbols=%d balance_rows=%d audit_rows=%d"
        % (len(symbols), len(balance), len(audit)),
        flush=True,
    )
    return {"balance_sheet": balance_output, "fina_audit": audit_output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--audit-workers", type=int, default=12)
    args = parser.parse_args()
    paths = run(
        args.data_dir,
        args.start,
        args.end,
        audit_workers=args.audit_workers,
    )
    for name, path in paths.items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
