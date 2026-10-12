#!/usr/bin/env python3
"""Fetch a sealed exact-date execution/valuation snapshot for the new ledger.

Missing daily rows are not suspension evidence. Suspensions must be confirmed
by suspend_d. Held stocks with corporate actions are stopped by the ledger.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from config import get_tushare_token
from live.forward_ledger import (calendar_dates, latest_day, locked, next_session, required_symbols,
                                 seal, staging, utc_now, verify)
from live.risk_blacklist import RISK_BLACKLIST_COLUMNS
from scripts.fetch_all_a_dynamic_universe_dataset import _pro, _call_with_retry


def build_market(daily: pd.DataFrame, limits: pd.DataFrame, adj: pd.DataFrame,
                 suspensions: pd.DataFrame, symbols: set[str], date: str,
                 previous_prices: pd.DataFrame) -> pd.DataFrame:
    compact = pd.Timestamp(date).strftime("%Y%m%d")
    frames = []
    for frame, name in [(daily, "daily"), (limits, "limits"), (adj, "adjustment")]:
        if not {"ts_code", "trade_date"}.issubset(frame.columns):
            raise RuntimeError("missing schema: " + name)
        frame = frame[frame.ts_code.isin(symbols)].copy()
        if frame.ts_code.duplicated().any() or not frame.trade_date.astype(str).eq(compact).all():
            raise RuntimeError("duplicate or wrong-date " + name)
        frames.append(frame.set_index("ts_code"))
    prices, price_limits, factors = frames
    suspended = set()
    if not suspensions.empty:
        for rec in suspensions.to_dict("records"):
            if str(rec.get("trade_date")) != compact:
                raise RuntimeError("wrong-date suspension data")
            if rec.get("suspend_type") == "S":
                suspended.add(rec["ts_code"])
    old = previous_prices.set_index("symbol") if not previous_prices.empty else pd.DataFrame()
    rows = []
    for symbol in sorted(symbols):
        missing = symbol not in prices.index
        if missing and symbol not in suspended:
            raise RuntimeError("missing daily row without suspension confirmation: " + symbol)
        if symbol not in factors.index:
            raise RuntimeError("missing exact-date adjustment factor: " + symbol)
        if missing:
            if symbol not in old.index:
                raise RuntimeError("no prior valuation for confirmed suspension: " + symbol)
            close = float(old.loc[symbol, "close"])
            price_date = str(old.loc[symbol, "price_date"])
            up, down = False, False
        else:
            close = float(prices.loc[symbol, "close"])
            price_date = date
            if symbol not in price_limits.index:
                raise RuntimeError("missing price limits: " + symbol)
            upper, lower = float(price_limits.loc[symbol, "up_limit"]), float(price_limits.loc[symbol, "down_limit"])
            if not 0 < lower <= upper or not lower - 0.011 <= close <= upper + 0.011:
                raise RuntimeError("invalid price/limits: " + symbol)
            up, down = close >= upper - 1e-6, close <= lower + 1e-6
        rows.append({"date": date, "symbol": symbol, "close": close, "price_date": price_date,
                     "adj_factor": float(factors.loc[symbol, "adj_factor"]),
                     "is_suspended": symbol in suspended, "is_limit_up": up, "is_limit_down": down})
    return pd.DataFrame(rows, columns=["date", "symbol", "close", "price_date", "adj_factor",
                                       "is_suspended", "is_limit_up", "is_limit_down"])


def run(root: Path, trade_date: str, blacklist: Path | None = None) -> Path:
    with locked(root):
        date = pd.Timestamp(trade_date)
        if utc_now().tz_convert("Asia/Shanghai") < date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
            raise RuntimeError("cannot retrieve a future close for execution")
        prior, meta = latest_day(root)
        sessions = calendar_dates(root / "initial" / "calendar.csv")
        if date != next_session(sessions, meta["date"]):
            raise RuntimeError("market snapshot must be for the next unprocessed session")
        dest = root / "market" / date.strftime("%Y%m%d")
        if dest.exists():
            verify(dest)
            return dest
        symbols = required_symbols(root, trade_date)
        pro = _pro(get_tushare_token())
        compact = date.strftime("%Y%m%d")
        daily = _call_with_retry(lambda: pro.daily(trade_date=compact), "daily " + compact)
        limits = _call_with_retry(lambda: pro.stk_limit(trade_date=compact), "limits " + compact)
        adj = _call_with_retry(lambda: pro.adj_factor(trade_date=compact), "adjustment " + compact)
        suspensions = _call_with_retry(lambda: pro.suspend_d(trade_date=compact), "suspensions " + compact)
        if (prior / "market.csv").exists():
            previous = pd.read_csv(prior / "market.csv")
        else:
            plan = root / "plans" / date.strftime("%Y%m%d")
            verify(plan)
            previous = pd.read_csv(plan / "signal_market.csv.gz")[["ts_code", "trade_date", "close"]]
            previous = previous.rename(columns={"ts_code": "symbol", "trade_date": "price_date"})
        market = build_market(daily, limits, adj, suspensions, symbols, trade_date, previous)
        if blacklist is not None and not blacklist.is_file():
            raise FileNotFoundError(blacklist)
        with staging(root / "market") as stage:
            market.to_csv(stage / "market.csv", index=False)
            for frame, name in [(daily, "daily"), (limits, "limits"), (adj, "adj_factor"), (suspensions, "suspensions")]:
                frame.to_csv(stage / (name + ".csv.gz"), index=False, compression="gzip")
            if blacklist is None:
                pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS).to_csv(stage / "risk_blacklist.csv", index=False)
            else:
                shutil.copy2(blacklist, stage / "risk_blacklist.csv")
            seal(stage, {"date": trade_date, "retrieved_at_utc": utc_now().isoformat(), "provider": "Tushare Pro",
                         "required_symbols": len(symbols), "blacklist": "EXPLICIT_EMPTY_NOT_CONFIGURED" if blacklist is None else "PROVIDED",
                         "news_risk": "NOT_ASSESSED"})
            stage.rename(dest)
        return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--blacklist", type=Path)
    p.add_argument("--prompt-token", action="store_true")
    a = p.parse_args()
    if a.prompt_token:
        os.environ["TUSHARE_TOKEN"] = getpass.getpass("Tushare token (hidden): ")
    print(run(a.session, a.date, a.blacklist))
