#!/usr/bin/env python3
"""Fetch a resumable point-in-time CSI 300 research dataset from Tushare Pro."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_tushare_token
from live.data_feed import merge_adj_factor


INDEX_CODE = "399300.SZ"


def _call_with_retry(fn: Callable[[], pd.DataFrame], label: str, attempts: int = 5) -> pd.DataFrame:
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
            return result if result is not None else pd.DataFrame()
        except Exception:
            if attempt >= attempts:
                raise RuntimeError(f"Tushare request failed after {attempts} attempts: {label}")
            time.sleep(delay)
            delay = min(delay * 2.0, 12.0)
    return pd.DataFrame()


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    months = pd.period_range(start=start.to_period("M") - 1, end=end.to_period("M"), freq="M")
    return [(period.start_time.strftime("%Y%m%d"), period.end_time.strftime("%Y%m%d")) for period in months]


def _parse_tushare_dates(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, format="mixed", errors="coerce")
    return compact.fillna(fallback)


def _membership_intervals(
    snapshots: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    work = snapshots.copy()
    work["trade_date"] = _parse_tushare_dates(work["trade_date"])
    dates = sorted(work["trade_date"].dropna().unique())
    rows: list[dict[str, object]] = []
    for i, snapshot_date in enumerate(dates):
        effective_from = max(start, pd.Timestamp(snapshot_date) + pd.Timedelta(days=1))
        next_snapshot = pd.Timestamp(dates[i + 1]) if i + 1 < len(dates) else end
        effective_to = min(end, next_snapshot)
        if effective_from > effective_to:
            continue
        members = work.loc[
            work["trade_date"] == pd.Timestamp(snapshot_date), "con_code"
        ].astype(str)
        for symbol in members:
            rows.append(
                {
                    "ts_code": symbol,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                }
            )
    raw = pd.DataFrame(rows).sort_values(["ts_code", "effective_from"])
    merged: list[dict[str, object]] = []
    for symbol, group in raw.groupby("ts_code", sort=True):
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        for rec in group.itertuples(index=False):
            rec_start = pd.Timestamp(rec.effective_from)
            rec_end = pd.Timestamp(rec.effective_to)
            if current_start is None:
                current_start, current_end = rec_start, rec_end
            elif rec_start <= current_end + pd.Timedelta(days=1):
                current_end = max(current_end, rec_end)
            else:
                merged.append({"ts_code": symbol, "effective_from": current_start, "effective_to": current_end})
                current_start, current_end = rec_start, rec_end
        if current_start is not None:
            merged.append({"ts_code": symbol, "effective_from": current_start, "effective_to": current_end})
    return pd.DataFrame(merged).sort_values(["effective_from", "ts_code"]).reset_index(drop=True)


def _resume_frames(path: Path, key: str) -> tuple[list[pd.DataFrame], set[str]]:
    if not path.is_file():
        return [], set()
    frame = pd.read_csv(path)
    done = set(frame[key].dropna().astype(str)) if key in frame.columns else set()
    return [frame], done


def _checkpoint(frames: list[pd.DataFrame], path: Path, sort_cols: list[str]) -> None:
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    for column in ("ann_date", "end_date"):
        if column not in out.columns:
            continue
        out[column] = _parse_tushare_dates(out[column])
    out = out.sort_values(sort_cols)
    out.to_csv(path, index=False)
    frames[:] = [out]


def run(start_text: str, end_text: str, output_dir: Path, sleep_seconds: float = 0.12) -> dict[str, Path]:
    import tushare as ts

    start = pd.Timestamp(start_text)
    end = pd.Timestamp(end_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = get_tushare_token()
    pro = ts.pro_api(token)
    paths = {
        "snapshots": output_dir / f"csi300_weight_snapshots_{start:%Y%m%d}_{end:%Y%m%d}.csv",
        "membership": output_dir / f"csi300_membership_intervals_{start:%Y%m%d}_{end:%Y%m%d}.csv",
        "stock_pool": output_dir / f"stock_pool_csi300_pit_union_{start:%Y%m%d}_{end:%Y%m%d}.csv",
        "prices_raw": output_dir / f"prices_csi300_pit_union_{start:%Y%m%d}_{end:%Y%m%d}_raw.csv",
        "adj_factor": output_dir / f"adj_factor_csi300_pit_union_{start:%Y%m%d}_{end:%Y%m%d}.csv",
        "prices": output_dir / f"prices_csi300_pit_union_{start:%Y%m%d}_{end:%Y%m%d}_adj.csv",
        "fina": output_dir / f"fina_indicator_csi300_pit_union_{start:%Y%m%d}_{end:%Y%m%d}.csv",
    }

    if paths["snapshots"].is_file():
        snapshots = pd.read_csv(paths["snapshots"], dtype={"con_code": str})
    else:
        parts: list[pd.DataFrame] = []
        for month_start, month_end in _month_ranges(start, end):
            frame = _call_with_retry(
                lambda a=month_start, b=month_end: pro.index_weight(
                    index_code=INDEX_CODE,
                    start_date=a,
                    end_date=b,
                ),
                f"index_weight {month_start}-{month_end}",
            )
            if not frame.empty:
                parts.append(frame)
            time.sleep(sleep_seconds)
        snapshots = pd.concat(parts, ignore_index=True).drop_duplicates()
        snapshots.to_csv(paths["snapshots"], index=False)
    snapshots["con_code"] = snapshots["con_code"].astype(str)
    membership = _membership_intervals(snapshots, start, end)
    membership.to_csv(paths["membership"], index=False, date_format="%Y-%m-%d")
    symbols = sorted(membership["ts_code"].unique())

    basic = _call_with_retry(
        lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,list_date"),
        "stock_basic",
    )
    basic["ts_code"] = basic["ts_code"].astype(str)
    pool = pd.DataFrame({"股票代码": symbols}).merge(basic, left_on="股票代码", right_on="ts_code", how="left")
    pool = pd.DataFrame(
        {
            "分类": pool["industry"].fillna("未知"),
            "股票代码": pool["股票代码"],
            "股票简称": pool["name"].fillna(""),
            "主题": f"CSI 300 point-in-time union {start:%Y-%m-%d}..{end:%Y-%m-%d}",
            "是否启用": 1,
        }
    )
    pool.to_csv(paths["stock_pool"], index=False)

    price_frames, price_done = _resume_frames(paths["prices_raw"], "ts_code")
    adj_frames, adj_done = _resume_frames(paths["adj_factor"], "ts_code")
    fina_frames, fina_done = _resume_frames(paths["fina"], "ts_code")
    start_api, end_api = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    fina_start = (start - pd.DateOffset(years=2)).strftime("%Y%m%d")

    for i, symbol in enumerate(symbols, start=1):
        if symbol not in price_done:
            frame = _call_with_retry(
                lambda s=symbol: pro.daily(ts_code=s, start_date=start_api, end_date=end_api),
                f"daily {symbol}",
            )
            if not frame.empty:
                frame = frame.rename(columns={"vol": "volume"})
                price_frames.append(frame[["trade_date", "ts_code", "open", "high", "low", "close", "volume"]])
            price_done.add(symbol)
            time.sleep(sleep_seconds)
        if symbol not in adj_done:
            frame = _call_with_retry(
                lambda s=symbol: pro.adj_factor(ts_code=s, start_date=start_api, end_date=end_api),
                f"adj_factor {symbol}",
            )
            if not frame.empty:
                adj_frames.append(frame[["trade_date", "ts_code", "adj_factor"]])
            adj_done.add(symbol)
            time.sleep(sleep_seconds)
        if symbol not in fina_done:
            frame = _call_with_retry(
                lambda s=symbol: pro.fina_indicator(ts_code=s, start_date=fina_start, end_date=end_api),
                f"fina_indicator {symbol}",
            )
            if not frame.empty:
                fina_frames.append(frame)
            fina_done.add(symbol)
            time.sleep(sleep_seconds)
        if i % 25 == 0:
            _checkpoint(price_frames, paths["prices_raw"], ["ts_code", "trade_date"])
            _checkpoint(adj_frames, paths["adj_factor"], ["ts_code", "trade_date"])
            _checkpoint(fina_frames, paths["fina"], ["ts_code", "ann_date"])
            print(f"checkpoint={i}/{len(symbols)}")

    _checkpoint(price_frames, paths["prices_raw"], ["ts_code", "trade_date"])
    _checkpoint(adj_frames, paths["adj_factor"], ["ts_code", "trade_date"])
    _checkpoint(fina_frames, paths["fina"], ["ts_code", "ann_date"])
    prices = pd.read_csv(paths["prices_raw"], parse_dates=["trade_date"])
    adj = pd.read_csv(paths["adj_factor"], parse_dates=["trade_date"])
    adjusted = merge_adj_factor(prices, adj, adjustment_mode="qfq")
    adjusted.to_csv(paths["prices"], index=False, date_format="%Y-%m-%d")
    print(f"symbols={len(symbols)} price_rows={len(adjusted)} membership_rows={len(membership)}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    args = parser.parse_args()
    for name, path in run(args.start, args.end, args.output_dir, args.sleep_seconds).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
