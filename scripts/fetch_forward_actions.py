#!/usr/bin/env python3
"""Seal dividend evidence separately; never mutate an existing market snapshot."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import get_tushare_token
from live.account_state import load_account_state
from live.cash_dividends import FIELDS, POLICY, normalize_events
from live.forward_ledger import (locked, session_config, session_settings, latest_day, verify,
                                 sha, staging, seal, utc_now, next_session, calendar_dates)
from scripts.fetch_all_a_dynamic_universe_dataset import _pro, _call_with_retry


def run(root: Path, market_dir: Path, date: str) -> Path:
    with locked(root):
        config = session_config(root)
        if config["corporate_actions"] != POLICY:
            raise RuntimeError("cash dividend policy has not been explicitly enabled")
        prior, meta = latest_day(root)
        if str(next_session(calendar_dates(root / "initial/calendar.csv"), meta["date"]).date()) != date:
            raise RuntimeError("action snapshot requires next unprocessed session")
        market_meta = verify(market_dir)
        if market_meta["date"] != date or utc_now() < pd.Timestamp(date, tz="Asia/Shanghai") + pd.Timedelta(hours=15):
            raise RuntimeError("invalid/unclosed corporate action date")
        dest = root / "corporate_actions" / date.replace("-", "")
        expected = {"date": date, "prior_manifest_sha256": sha(prior / "manifest.json"),
                    "market_manifest_sha256": sha(market_dir / "manifest.json"),
                    "revision_sha256": config["revision_sha256"]}
        if dest.exists():
            saved = verify(dest)
            if any(saved[k] != v for k, v in expected.items()):
                raise RuntimeError("corporate action snapshot belongs to different inputs")
            return dest
        held = set()
        for strategy in config["strategies"]:
            _, pos = load_account_state(session_settings(prior, config), strategy=strategy)
            held.update(pos.symbol)
        pro = _pro(get_tushare_token())
        raw = _call_with_retry(lambda: pro.dividend(ex_date=date.replace("-", ""), fields=FIELDS), "cash dividend " + date)
        if len(raw) >= 2000:
            raise RuntimeError("possible truncated dividend response; review pagination")
        # A successful empty provider result may have no columns.
        if raw.empty and not len(raw.columns):
            raw = pd.DataFrame(columns=FIELDS.split(","))
        previous = pd.read_csv(prior / "market.csv").set_index("symbol")
        market = pd.read_csv(market_dir / "market.csv").set_index("symbol")
        events = normalize_events(raw, held, previous, market, meta["date"], date)
        with staging(dest.parent) as stage:
            raw.to_csv(stage / "dividend_raw.csv", index=False)
            events.to_csv(stage / "events.csv", index=False)
            seal(stage, {**expected, "retrieved_at_utc": utc_now().isoformat(), "provider": "Tushare Pro dividend",
                         "held_symbols": sorted(held), "policy": POLICY, "tax_policy": "GROSS_DIVIDEND_TAX_NOT_MODELED"})
            stage.rename(dest)
        return dest
