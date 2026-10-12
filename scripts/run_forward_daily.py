#!/usr/bin/env python3
"""Run one completed trading day, then prepare next week's targets if due.

This is an on-demand command, not an installed scheduler. Re-running a completed
day never trades twice; it may finish a previously failed weekly preparation.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from live.forward_ledger import (advance, calendar_dates, latest_day, next_session, prepare, utc_now, verify, session_config)
from live.cash_dividends import POLICY
from scripts.fetch_forward_actions import run as fetch_actions
from scripts.fetch_forward_market import run as fetch_market
from scripts.fetch_forward_inputs import run as refresh_inputs


def run(root: Path, data_dir: Path, date: str, blacklist: Path | None = None) -> dict:
    sessions = calendar_dates(root / "initial" / "calendar.csv")
    day = pd.Timestamp(date)
    if day not in sessions:
        return {"status": "MARKET_CLOSED_NO_MUTATION", "date": date}
    if utc_now().tz_convert("Asia/Shanghai") < day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
        raise RuntimeError("session has not closed; do not simulate future prices")
    _, meta = latest_day(root)
    if pd.Timestamp(meta["date"]) < day:
        if day != next_session(sessions, meta["date"]):
            raise RuntimeError("process the missing earlier trading day first")
        market = fetch_market(root, date, blacklist)
        if session_config(root)["corporate_actions"] == POLICY:
            fetch_actions(root, market, date)
        advance(root, market, date)
    elif pd.Timestamp(meta["date"]) > day:
        raise RuntimeError("cannot rewrite an older day")
    following = next_session(sessions, day)
    result = {"status": "DAY_COMMITTED", "date": date, "next_session": str(following.date()), "live_approval": False}
    if day.to_period("W-FRI") != following.to_period("W-FRI"):
        plan = root / "plans" / following.strftime("%Y%m%d")
        if plan.exists():
            verify(plan)
        else:
            if utc_now().tz_convert("Asia/Shanghai") >= following.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30):
                raise RuntimeError("weekly preparation deadline missed; old day retained but no backdated plan allowed")
            refresh_inputs(data_dir, data_dir, day.strftime("%Y%m%d"))
            prepare(root, data_dir, date, str(following.date()))
        result["next_action"] = "EXECUTE_FROZEN_WEEKLY_PLAN"
    else:
        result["next_action"] = "HOLD_MARK"
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--blacklist", type=Path)
    p.add_argument("--prompt-token", action="store_true")
    a = p.parse_args()
    if a.prompt_token:
        os.environ["TUSHARE_TOKEN"] = getpass.getpass("Tushare token (hidden): ")
    print(run(a.session, a.data_dir, a.date, a.blacklist))
