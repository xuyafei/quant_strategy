#!/usr/bin/env python3
"""Fetch a resumable all-Shanghai/Shenzhen-A point-in-time research dataset."""
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
from live.data_feed import merge_adj_factor


_THREAD_LOCAL = threading.local()


def _pro(token: str):
    import tushare as ts

    if not hasattr(_THREAD_LOCAL, "pro"):
        _THREAD_LOCAL.pro = ts.pro_api(token)
    return _THREAD_LOCAL.pro


def _call_with_retry(fn: Callable[[], pd.DataFrame], label: str, attempts: int = 6) -> pd.DataFrame:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            value = fn()
            return value if value is not None else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - network behaviour
            last = exc
            if attempt >= attempts:
                break
            message = str(exc)
            if "频率超限" in message or "rate" in message.lower():
                time.sleep(65.0)
                delay = 1.0
            else:
                time.sleep(delay)
                delay = min(delay * 2.0, 20.0)
    raise RuntimeError("Tushare request failed for %s: %s" % (label, last))


def _atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, **kwargs)
    temp.replace(path)


def _fetch_stock_basic(token: str) -> pd.DataFrame:
    fields = "ts_code,symbol,name,industry,market,exchange,list_status,list_date,delist_date"
    frames = []
    for status in ("L", "D", "P"):
        frame = _call_with_retry(
            lambda s=status: _pro(token).stock_basic(
                exchange="", list_status=s, fields=fields
            ),
            "stock_basic %s" % status,
        )
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ts_code", "list_status"], keep="last"
    )
    return out[out["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()


def _fetch_date(token: str, date_text: str, part_dir: Path) -> tuple[str, int]:
    daily_path = part_dir / "daily" / (date_text + ".csv")
    adj_path = part_dir / "adj_factor" / (date_text + ".csv")
    if daily_path.is_file() and adj_path.is_file():
        return date_text, 0
    pro = _pro(token)
    daily = _call_with_retry(
        lambda: pro.daily(trade_date=date_text), "daily %s" % date_text
    )
    adj = _call_with_retry(
        lambda: pro.adj_factor(trade_date=date_text), "adj_factor %s" % date_text
    )
    if daily.empty:
        raise RuntimeError("daily returned no rows for open date %s" % date_text)
    daily = daily[daily["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
    daily = daily.rename(columns={"vol": "volume"})
    daily["amount_yuan"] = pd.to_numeric(daily["amount"], errors="coerce") * 1000.0
    adj = adj[adj["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
    _atomic_csv(daily, daily_path)
    _atomic_csv(adj, adj_path)
    return date_text, int(len(daily))


def _quarter_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    first = (start - pd.DateOffset(years=1)).to_period("Q")
    last = end.to_period("Q")
    return [period.end_time.strftime("%Y%m%d") for period in pd.period_range(first, last, freq="Q")]


def _assemble_parts(paths: list[Path], output: Path, sort_cols: list[str]) -> pd.DataFrame:
    if not paths:
        raise RuntimeError("no part files found for %s" % output)
    frames = [pd.read_csv(path) for path in sorted(paths)]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(sort_cols, keep="last")
    out = out.sort_values(sort_cols).reset_index(drop=True)
    _atomic_csv(out, output, compression="gzip")
    return out


def run(
    start_text: str,
    end_text: str,
    output_dir: Path,
    *,
    workers: int = 6,
) -> dict[str, Path]:
    token = get_tushare_token()
    start = pd.Timestamp(start_text).normalize()
    end = pd.Timestamp(end_text).normalize()
    if end < start:
        raise ValueError("end must be on or after start")
    output_dir.mkdir(parents=True, exist_ok=True)
    part_dir = output_dir / "parts"
    paths = {
        "stock_basic": output_dir / "stock_basic_all_a.csv",
        "trade_calendar": output_dir / "trade_calendar.csv",
        "name_history": output_dir / "namechange_all_a.csv",
        "prices_raw": output_dir / "prices_all_a_raw.csv.gz",
        "adj_factor": output_dir / "adj_factor_all_a.csv.gz",
        "prices": output_dir / "prices_all_a_qfq.csv.gz",
        "fina": output_dir / "fina_indicator_all_a.csv.gz",
    }

    basic = _fetch_stock_basic(token)
    _atomic_csv(basic, paths["stock_basic"])
    calendar = _call_with_retry(
        lambda: _pro(token).trade_cal(
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        ),
        "trade_cal",
    )
    _atomic_csv(calendar, paths["trade_calendar"])
    open_dates = calendar.loc[
        pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"
    ].astype(str).tolist()

    names = _call_with_retry(
        lambda: _pro(token).namechange(
            fields="ts_code,name,start_date,end_date,ann_date,change_reason"
        ),
        "namechange",
    )
    names = names[names["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
    _atomic_csv(names, paths["name_history"])

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_fetch_date, token, date, part_dir): date
            for date in open_dates
        }
        for future in concurrent.futures.as_completed(futures):
            date, _rows = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(open_dates):
                print("market_data_checkpoint=%d/%d latest=%s" % (completed, len(open_dates), date), flush=True)

    raw = _assemble_parts(
        list((part_dir / "daily").glob("*.csv")),
        paths["prices_raw"],
        ["ts_code", "trade_date"],
    )
    adj = _assemble_parts(
        list((part_dir / "adj_factor").glob("*.csv")),
        paths["adj_factor"],
        ["ts_code", "trade_date"],
    )
    # Tushare returns YYYYMMDD strings, but CSV inference may turn them into
    # integers.  Explicit format parsing prevents pandas from interpreting an
    # integer such as 20230103 as nanoseconds after the Unix epoch.
    raw["trade_date"] = pd.to_datetime(
        raw["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="raise",
    )
    adj["trade_date"] = pd.to_datetime(
        adj["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="raise",
    )
    adjusted = merge_adj_factor(raw, adj, adjustment_mode="qfq")
    adjusted = adjusted.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    _atomic_csv(adjusted, paths["prices"], compression="gzip", date_format="%Y-%m-%d")

    fina_parts: list[pd.DataFrame] = []
    fina_fields = (
        "ts_code,ann_date,end_date,eps,roe,grossprofit_margin,netprofit_margin,"
        "debt_to_assets,or_yoy,netprofit_yoy,ocfps,ocf_to_profit"
    )
    for index, period in enumerate(_quarter_ends(start, end), start=1):
        frame = _call_with_retry(
            lambda p=period: _pro(token).fina_indicator_vip(
                period=p, fields=fina_fields
            ),
            "fina_indicator_vip %s" % period,
        )
        if not frame.empty:
            frame = frame[frame["ts_code"].astype(str).str.endswith((".SH", ".SZ"))]
            fina_parts.append(frame)
        print("finance_checkpoint=%d/%d period=%s rows=%d" % (index, len(_quarter_ends(start, end)), period, len(frame)), flush=True)
    fina = pd.concat(fina_parts, ignore_index=True).drop_duplicates(
        ["ts_code", "ann_date", "end_date"], keep="last"
    )
    fina = fina.sort_values(["ts_code", "ann_date", "end_date"]).reset_index(drop=True)
    _atomic_csv(fina, paths["fina"], compression="gzip")

    print(
        "dataset_complete open_dates=%d symbols=%d price_rows=%d fina_rows=%d"
        % (len(open_dates), raw["ts_code"].nunique(), len(adjusted), len(fina)),
        flush=True,
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for name, path in run(args.start, args.end, args.output_dir, workers=args.workers).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
