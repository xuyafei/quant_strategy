#!/usr/bin/env python3
"""Fetch resumable market-cap and historical SW-industry data for strategy pools."""
from __future__ import annotations

import argparse
import concurrent.futures
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
            value = fn()
            return value if value is not None else pd.DataFrame()
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


def _parse_dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(raw, format="%Y%m%d", errors="coerce")


def weekly_decision_dates(
    calendar: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[str]:
    required = {"cal_date", "is_open"}
    if missing := required - set(calendar.columns):
        raise ValueError("trade calendar missing columns: %s" % sorted(missing))
    dates = _parse_dates(
        calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1), "cal_date"]
    ).dropna()
    dates = dates[(dates >= start) & (dates <= end)]
    if dates.empty:
        return []
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    return frame.groupby("week")["date"].max().dt.strftime("%Y%m%d").tolist()


def _fetch_daily_basic_date(token: str, date_text: str, part_dir: Path) -> tuple[str, int]:
    path = part_dir / "daily_basic" / (date_text + ".csv")
    if path.is_file():
        return date_text, -1
    fields = "ts_code,trade_date,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv"
    frame = _call_with_retry(
        lambda: _pro(token).daily_basic(ts_code="", trade_date=date_text, fields=fields),
        "daily_basic %s" % date_text,
    )
    if frame.empty:
        raise RuntimeError("daily_basic returned no rows for %s" % date_text)
    frame = frame[frame["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
    _atomic_csv(frame, path)
    return date_text, int(len(frame))


def _fetch_industry_membership(token: str, data_dir: Path) -> tuple[Path, Path]:
    classify_path = data_dir / "sw2021_l1_classify.csv"
    membership_path = data_dir / "sw2021_l1_membership.csv.gz"
    part_dir = data_dir / "strategy_parts" / "sw2021_membership"
    classify = _call_with_retry(
        lambda: _pro(token).index_classify(level="L1", src="SW2021"),
        "index_classify SW2021 L1",
    )
    if classify.empty:
        raise RuntimeError("index_classify returned no SW2021 L1 industries")
    code_column = "index_code" if "index_code" in classify.columns else "industry_code"
    if code_column not in classify.columns:
        raise RuntimeError("index_classify missing index_code/industry_code")
    classify[code_column] = classify[code_column].astype(str).str.strip()
    classify = classify[classify[code_column].ne("")].drop_duplicates(code_column)
    _atomic_csv(classify, classify_path)

    total_calls = len(classify) * 2
    completed = 0
    for code in sorted(classify[code_column].astype(str).unique()):
        for is_new in ("Y", "N"):
            path = part_dir / (code.replace(".", "_") + "_" + is_new + ".csv")
            if not path.is_file():
                frame = _call_with_retry(
                    lambda c=code, flag=is_new: _pro(token).index_member_all(
                        l1_code=c, is_new=flag
                    ),
                    "index_member_all %s %s" % (code, is_new),
                )
                _atomic_csv(frame, path)
            completed += 1
            if completed % 10 == 0 or completed == total_calls:
                print(
                    "industry_checkpoint=%d/%d code=%s status=%s"
                    % (completed, total_calls, code, is_new),
                    flush=True,
                )

    frames = []
    for path in sorted(part_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path, dtype={"ts_code": str})
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no SW2021 industry membership rows were fetched")
    membership = pd.concat(frames, ignore_index=True)
    required = [column for column in ("l1_code", "ts_code", "in_date", "out_date") if column in membership.columns]
    membership = membership.drop_duplicates(required, keep="last")
    sort_columns = [column for column in ("l1_code", "ts_code", "in_date") if column in membership.columns]
    membership = membership.sort_values(sort_columns).reset_index(drop=True)
    _atomic_csv(membership, membership_path, compression="gzip")
    return classify_path, membership_path


def _assemble_daily_basic(part_dir: Path, output: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype={"ts_code": str}) for path in sorted(part_dir.glob("*.csv"))]
    if not frames:
        raise RuntimeError("no daily_basic part files found")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    out = out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    _atomic_csv(out, output, compression="gzip")
    return out


def run(
    data_dir: Path,
    start_text: str,
    end_text: str,
    *,
    workers: int = 4,
) -> dict[str, Path]:
    token = get_tushare_token()
    start = pd.Timestamp(start_text).normalize()
    end = pd.Timestamp(end_text).normalize()
    if end < start:
        raise ValueError("end must be on or after start")
    calendar_path = data_dir / "trade_calendar.csv"
    if not calendar_path.is_file():
        raise FileNotFoundError(calendar_path)
    calendar = pd.read_csv(calendar_path, dtype={"cal_date": str})
    dates = weekly_decision_dates(calendar, start, end)
    if not dates:
        raise RuntimeError("no weekly decision dates found")
    daily_part_dir = data_dir / "strategy_parts" / "daily_basic"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_fetch_daily_basic_date, token, date, data_dir / "strategy_parts"): date
            for date in dates
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            date, rows = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                print(
                    "daily_basic_checkpoint=%d/%d latest=%s rows=%s"
                    % (completed, len(futures), date, "cached" if rows < 0 else rows),
                    flush=True,
                )
    daily_output = data_dir / "daily_basic_weekly.csv.gz"
    daily = _assemble_daily_basic(daily_part_dir, daily_output)
    classify_path, membership_path = _fetch_industry_membership(token, data_dir)
    print(
        "strategy_universe_data_complete dates=%d daily_rows=%d industries=%s"
        % (len(dates), len(daily), membership_path),
        flush=True,
    )
    return {
        "daily_basic": daily_output,
        "industry_classify": classify_path,
        "industry_membership": membership_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for name, path in run(args.data_dir, args.start, args.end, workers=args.workers).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
