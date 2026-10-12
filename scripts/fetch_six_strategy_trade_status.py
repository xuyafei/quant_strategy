#!/usr/bin/env python3
"""Build strict suspension and price-limit status for six paper accounts."""
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


def _call_with_retry(fn: Callable[[], pd.DataFrame], label: str) -> pd.DataFrame:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, 7):
        try:
            value = fn()
            return value if value is not None else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - network behaviour
            last = exc
            if attempt == 6:
                break
            if "频率" in str(exc) or "rate" in str(exc).lower():
                time.sleep(62.0)
            else:
                time.sleep(delay)
                delay = min(delay * 2.0, 20.0)
    raise RuntimeError("Tushare request failed for %s: %s" % (label, last))


def _target_symbols(output_dir: Path, trade_date: pd.Timestamp) -> set[str]:
    symbols: set[str] = set()
    for path in sorted((output_dir / "rebalance_logs").glob("*.csv")):
        frame = pd.read_csv(path, dtype={"symbol": str})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        active = frame[frame["date"].eq(trade_date)]
        symbols.update(active["symbol"].dropna().astype(str))
    for path in sorted((output_dir / "paper_account").glob("*/positions.csv")):
        frame = pd.read_csv(path, dtype={"symbol": str})
        if "shares" in frame.columns:
            frame = frame[pd.to_numeric(frame["shares"], errors="coerce").fillna(0).gt(0)]
        symbols.update(frame.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
    return symbols


def _fetch_symbol_daily(token: str, symbol: str, date_text: str) -> pd.DataFrame:
    frame = _call_with_retry(
        lambda: _pro(token).daily(
            ts_code=symbol,
            start_date=date_text,
            end_date=date_text,
        ),
        "daily %s %s" % (symbol, date_text),
    )
    if frame.empty:
        return frame
    return frame[frame["trade_date"].astype(str).eq(date_text)].copy()


def run(
    data_dir: Path,
    output_dir: Path,
    trade_date: str,
    *,
    workers: int = 8,
) -> dict[str, Path]:
    date = pd.Timestamp(trade_date).normalize()
    date_text = date.strftime("%Y%m%d")
    targets = _target_symbols(output_dir, date)
    if not targets:
        raise RuntimeError("no frozen target symbols found for %s" % date_text)

    token = get_tushare_token()
    frames: list[pd.DataFrame] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_fetch_symbol_daily, token, symbol, date_text): symbol
            for symbol in sorted(targets)
        }
        for future in concurrent.futures.as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(
                    "execution_price_checkpoint=%d/%d"
                    % (completed, len(futures)),
                    flush=True,
                )
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if daily.empty:
        raise RuntimeError("no target execution prices were returned for %s" % date_text)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["amount_yuan"] = pd.to_numeric(daily["amount"], errors="coerce") * 1000.0
    price_path = output_dir / "cache" / ("execution_prices_%s.csv" % date_text)
    price_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(price_path, index=False)

    limits = _call_with_retry(
        lambda: _pro(token).stk_limit(trade_date=date_text),
        "stk_limit %s" % date_text,
    )
    if limits.empty:
        raise RuntimeError("stk_limit returned no rows for %s" % date_text)
    limits["ts_code"] = limits["ts_code"].astype(str)

    market = daily[["ts_code", "close"]].drop_duplicates("ts_code", keep="last")
    market = market.merge(
        limits[["ts_code", "up_limit", "down_limit"]].drop_duplicates(
            "ts_code", keep="last"
        ),
        on="ts_code",
        how="outer",
    ).set_index("ts_code")
    rows = []
    missing_limits = []
    for symbol in sorted(targets):
        if symbol not in market.index:
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "is_suspended": True,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "close": float("nan"),
                    "up_limit": float("nan"),
                    "down_limit": float("nan"),
                    "status_source": "missing_daily_row",
                }
            )
            continue
        rec = market.loc[symbol]
        close = pd.to_numeric(pd.Series([rec.get("close")]), errors="coerce").iloc[0]
        up = pd.to_numeric(pd.Series([rec.get("up_limit")]), errors="coerce").iloc[0]
        down = pd.to_numeric(pd.Series([rec.get("down_limit")]), errors="coerce").iloc[0]
        suspended = bool(pd.isna(close))
        if not suspended and (pd.isna(up) or pd.isna(down)):
            missing_limits.append(symbol)
        tolerance = 1e-6
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "is_suspended": suspended,
                "is_limit_up": bool(not suspended and close >= up - tolerance),
                "is_limit_down": bool(not suspended and close <= down + tolerance),
                "close": close,
                "up_limit": up,
                "down_limit": down,
                "status_source": "tushare_daily_and_stk_limit",
            }
        )
    if missing_limits:
        raise RuntimeError(
            "price-limit coverage missing for target symbols: %s"
            % ", ".join(missing_limits[:10])
        )
    out = pd.DataFrame(rows)
    path = output_dir / "cache" / ("trade_status_%s.csv" % date_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(
        "trade_status_complete date=%s symbols=%d suspended=%d limit_up=%d limit_down=%d"
        % (
            date.strftime("%Y-%m-%d"),
            len(out),
            int(out["is_suspended"].sum()),
            int(out["is_limit_up"].sum()),
            int(out["is_limit_down"].sum()),
        )
    )
    return {"execution_prices": price_path, "trade_status": path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name, path in run(
        args.data_dir,
        args.output,
        args.trade_date,
        workers=args.workers,
    ).items():
        print("%s=%s" % (name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
