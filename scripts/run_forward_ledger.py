#!/usr/bin/env python3
"""Strict six-strategy paper ledger: init, prepare, day, status (no real orders)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from live.forward_ledger import (initialize, prepare, advance, latest_day, session_config,
                                 calendar_dates, next_session, verify)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["init", "prepare", "day", "status"])
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--signal-date")
    p.add_argument("--execution-date")
    p.add_argument("--date")
    p.add_argument("--market-dir", type=Path)
    a = p.parse_args()
    if a.command in {"init", "prepare"}:
        if not all([a.data_dir, a.signal_date, a.execution_date]):
            p.error("init/prepare require --data-dir --signal-date --execution-date")
        if a.command == "init":
            if not a.protocol:
                p.error("init requires --protocol")
            result = initialize(a.session, a.data_dir, a.protocol, a.signal_date, a.execution_date)
        else:
            result = prepare(a.session, a.data_dir, a.signal_date, a.execution_date)
    elif a.command == "day":
        if not a.date:
            p.error("day requires --date")
        market = a.market_dir or a.session / "market" / pd.Timestamp(a.date).strftime("%Y%m%d")
        result = advance(a.session, market, a.date)
    else:
        config = session_config(a.session)
        state, meta = latest_day(a.session)
        sessions = calendar_dates(a.session / "initial" / "calendar.csv")
        date = next_session(sessions, meta["date"])
        plan = a.session / "plans" / date.strftime("%Y%m%d")
        if plan.exists():
            verify(plan)
        weekly = pd.Timestamp(meta["date"]).to_period("W-FRI") != date.to_period("W-FRI")
        result = json.dumps({"version": config["version"], "last_committed_date": meta["date"],
                             "last_record_kind": meta["kind"], "next_open_session": str(date.date()),
                             "next_action": "EXECUTE_FROZEN_PLAN" if plan.exists() else ("MISSING_WEEKLY_PLAN" if weekly else "HOLD_MARK"),
                             "days_completed": len(list((a.session / "days").glob("????????"))),
                             "live_approval": False, "auto_schedule_installed": False}, ensure_ascii=False, indent=2)
    print(result)


if __name__ == "__main__":
    main()
