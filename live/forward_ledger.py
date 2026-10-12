"""Immutable, close-model paper sessions. No real broker access.

Plans and days are committed by one directory rename under an exclusive lock.
The previous committed day is the only account source. A failed six-account
run leaves no committed day, and can be retried without duplicating fills.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from config import get_settings
from live.account_state import load_account_state, save_account_state
from live.board_rules import board
from live.order_builder import build_order_plan
from live.paper_guard import validate_daily_result, raise_on_guard_errors
from live.paper_runner import run_daily_paper_trade
from live.risk_blacklist import load_risk_blacklist
from live.six_strategy_forward import build_forward_signals

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ["live/forward_ledger.py", "live/six_strategy_forward.py", "live/board_rules.py",
              "live/order_builder.py", "live/order_precheck.py", "live/paper_runner.py", "live/broker.py",
              "strategies/style.py", "universe/dynamic.py", "universe/risk.py", "universe/strategy.py",
              "scripts/build_dynamic_universe_comparison.py"]
LABELS = {"BASE_V2_MULTI_FACTOR_TOP50": "基础多因子", "LARGE_CAP_MULTI_FACTOR_TOP50": "大盘多因子",
          "MID_SMALL_CAP_MULTI_FACTOR_TOP50": "中小盘多因子", "VALUE_FOCUSED_TOP50": "价值",
          "GROWTH_FOCUSED_TOP50": "成长", "TECH_GROWTH_THEME_MULTI_FACTOR_TOP50": "科技成长主题"}


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seal(directory: Path, meta: dict) -> None:
    meta = dict(meta)
    meta["files"] = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*"))
                     if p.is_file() and p.name not in {"manifest.json", "manifest.sha256"}}
    write_json(directory / "manifest.json", meta)
    (directory / "manifest.sha256").write_text(sha(directory / "manifest.json") + "\n")


def verify(directory: Path) -> dict:
    manifest = directory / "manifest.json"
    if sha(manifest) != (directory / "manifest.sha256").read_text().strip():
        raise RuntimeError("manifest changed: " + str(directory))
    meta = read_json(manifest)
    for name, expected in meta["files"].items():
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("frozen file changed: " + str(path))
    return meta


@contextmanager
def locked(root: Path):
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / (root.name + ".lock")
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError("another operation holds the session lock; inspect it before recovery") from None
    try:
        yield
    finally:
        lock.rmdir()


@contextmanager
def staging(parent: Path):
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)  # Only the exact directory created above.


def calendar_dates(path: Path) -> pd.DatetimeIndex:
    df = pd.read_csv(path, dtype={"cal_date": str})
    values = pd.to_datetime(df.cal_date, format="%Y%m%d", errors="raise")
    if values.duplicated().any() or not df.is_open.isin([0, 1]).all():
        raise RuntimeError("invalid calendar")
    full = pd.date_range(values.min(), values.max())
    if len(values) != len(full):
        raise RuntimeError("calendar contains gaps")
    return pd.DatetimeIndex(values[df.is_open.eq(1)]).sort_values()


def next_session(sessions: pd.DatetimeIndex, date: Any) -> pd.Timestamp:
    following = sessions[sessions > pd.Timestamp(date)]
    if following.empty:
        raise RuntimeError("calendar horizon exhausted; extend through a reviewed session version")
    return following[0]


def session_settings(root: Path, config: dict):
    return replace(get_settings(), output_dir=root, paper_initial_cash=config["initial_cash"],
                   commission_rate=config["commission_rate"], portfolio_weighting="equal",
                   order_lot_size=100, order_board_rules=True, min_order_amount=0.0, order_cash_buffer=0.0,
                   broker_mode="simulated", broker_provider="", broker_account_id="")


def session_config(root: Path) -> dict:
    from live.forward_revision import effective_config
    verify(root / "initial")
    config = effective_config(root, read_json(root / "initial" / "session.json"))
    for name, expected in config["code_hashes"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("execution/research code changed; review and version the session: " + name)
    return config


def latest_day(root: Path) -> tuple[Path, dict]:
    days = sorted(p for p in (root / "days").glob("????????") if p.is_dir())
    previous = root / "initial"
    meta = verify(previous)
    for day in days:
        current = verify(day)
        if current["previous_manifest_sha256"] != sha(previous / "manifest.json"):
            raise RuntimeError("broken daily account chain")
        if current.get("revision_sha256"):
            revision = root / "revisions/cash_dividend_v1"
            verify(revision)
            if current["revision_sha256"] != sha(revision / "manifest.json"):
                raise RuntimeError("daily session revision changed")
        previous, meta = day, current
    return previous, meta


def _receipt(data_dir: Path, signal: pd.Timestamp) -> dict:
    receipt = read_json(data_dir / "forward_input_receipt.json")
    if receipt["through_date"] != str(signal.date()):
        raise RuntimeError("fresh input receipt does not cover the requested signal date")
    retrieved = pd.Timestamp(receipt["retrieved_at_utc"])
    if retrieved > utc_now() or retrieved < signal.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
        raise RuntimeError("invalid input retrieval timestamp")
    for name, expected in receipt["files"].items():
        if sha(data_dir / name) != expected:
            raise RuntimeError("data changed since successful refresh: " + name)
    return receipt


def initialize(root: Path, data_dir: Path, protocol_path: Path, signal_date: str,
               execution_date: str, initial_cash: float = 1_000_000.0) -> Path:
    """Create a new, separate session; never reset the old replay accounts."""
    with locked(root):
        if root.exists():
            raise RuntimeError("session already exists; use prepare/day/status, never reset")
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial cash must be finite and positive")
        expected_path = protocol_path.with_suffix(".sha256")
        if expected_path.exists() and sha(protocol_path) != expected_path.read_text().split()[0]:
            raise RuntimeError("parent strategy protocol hash mismatch")
        with staging(root.parent) as stage:
            initial = stage / "initial"
            initial.mkdir()
            shutil.copy2(data_dir / "trade_calendar.csv", initial / "calendar.csv")
            shutil.copy2(protocol_path, initial / "strategy_protocol.json")
            protocol = read_json(protocol_path)
            config = {"version": "SIX_STRATEGY_FORWARD_LEDGER_V2", "created_at_utc": utc_now().isoformat(),
                      "initial_cash": initial_cash, "commission_rate": 0.0003,
                      "strategies": sorted(x["name"] for x in protocol["strategies"]),
                      "execution_model": "NEXT_SESSION_CLOSE_IDEALIZED_NOT_LIVE_FILL",
                      "live_approval": False, "auto_schedule_installed": False,
                      "quantity_rules": "MAIN_CHINEXT_100_STAR_MIN200_INCREMENT1",
                      "fees": "commission only, 0.03% each side; excludes stamp duty, transfer fees, minimum fees, slippage",
                      "risk_overlays": "parent strategy unchanged; blacklist and execution gates only; stress/news unassessed",
                      "corporate_actions": "block on held-stock adjustment-factor change; no fabricated dividend cash",
                      "account_permissions": "research SH/SZ main+ChiNext+STAR; NOT the user's brokerage permissions",
                      "code_hashes": {name: sha(ROOT / name) for name in CODE_FILES}}
            write_json(initial / "session.json", config)
            settings = session_settings(initial, config)
            for strategy in config["strategies"]:
                save_account_state(settings, strategy=strategy, cash=initial_cash,
                                   positions={}, trade_date=signal_date,
                                   snapshot={"cash": initial_cash, "market_value": 0, "total_asset": initial_cash, "n_positions": 0})
            seal(initial, {"date": signal_date, "kind": "INITIAL_CASH_ONLY", "live_approval": False})
            _prepare(stage, data_dir, signal_date, execution_date, first=True)
            stage.rename(root)
    return root


def prepare(root: Path, data_dir: Path, signal_date: str, execution_date: str) -> Path:
    with locked(root):
        return _prepare(root, data_dir, signal_date, execution_date)


def _prepare(root: Path, data_dir: Path, signal_date: str, execution_date: str, first: bool = False) -> Path:
    config = session_config(root)
    sessions = calendar_dates(root / "initial" / "calendar.csv")
    signal, execution = pd.Timestamp(signal_date), pd.Timestamp(execution_date)
    if signal not in sessions or execution != next_session(sessions, signal):
        raise RuntimeError("signal/execution must be consecutive open sessions")
    now = utc_now().tz_convert("Asia/Shanghai")
    if now < signal.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
        raise RuntimeError("signal session has not closed")
    if now >= execution.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30):
        raise RuntimeError("too late to freeze: execution session has already opened")
    if not first and signal.to_period("W-FRI") == execution.to_period("W-FRI"):
        raise RuntimeError("not a weekly rebalance signal session")
    prior, prior_meta = latest_day(root)
    if pd.Timestamp(prior_meta["date"]) != signal:
        raise RuntimeError("mark the signal session before preparing its next-session targets")
    for p in (root / "plans").glob("????????"):
        if p.is_dir() and not (root / "days" / p.name).exists():
            raise RuntimeError("unexecuted frozen plan already exists")
    dest = root / "plans" / execution.strftime("%Y%m%d")
    if dest.exists():
        raise RuntimeError("frozen plan already exists")
    receipt = _receipt(data_dir, signal)
    bundle = build_forward_signals(data_dir, root / "initial" / "strategy_protocol.json", signal_date=signal)
    if sorted(bundle["targets"]) != config["strategies"]:
        raise RuntimeError("strategy set changed")
    last = bundle["prices"][bundle["prices"].trade_date.eq(signal)]
    reference = last.set_index("ts_code")["close"].astype(float)
    if (prior / "market.csv").exists():
        held_marks = pd.read_csv(prior / "market.csv").set_index("symbol")
        reference = reference.combine_first(held_marks.loc[held_marks.is_suspended, "close"])
    source_report = bundle["report"].drop_duplicates("symbol").set_index("symbol")
    with staging(root / "plans") as stage:
        combined, summary = [], []
        for strategy in config["strategies"]:
            target = bundle["targets"][strategy].copy()
            for sym in target.symbol:
                board(sym)
            target["name"] = target.symbol.map(source_report["name"]) if "name" in source_report else target.symbol
            target["industry"] = target.symbol.map(source_report["industry_l1"])
            target["board"] = target.symbol.map(board)
            target["reference_price"] = target.symbol.map(reference)
            if not target.reference_price.gt(0).all():
                raise RuntimeError("missing exact signal-date target price")
            target["signal_date"] = signal_date
            target["execution_date"] = execution_date
            cash, positions = load_account_state(session_settings(prior, config), strategy=strategy)
            orders = build_order_plan(target.set_index("symbol").weight, positions, reference,
                                      cash=cash, trade_date=execution, include_holds=True, board_rules=True)
            # Preview only. Future suspension/limits are UNKNOWN, never marked PASS.
            preview = run_daily_paper_trade(session_settings(prior, config), strategy=strategy,
                                            target_weights=target.set_index("symbol").weight,
                                            latest_prices=reference, trade_date=execution,
                                            execution_mode="simulated_broker", persist_outputs=False)
            estimates = orders.set_index("symbol")
            target["estimated_target_shares"] = target.symbol.map(estimates.target_shares)
            target["estimated_order_shares"] = target.symbol.map(estimates.delta_shares)
            target["estimate_reason"] = target.symbol.map(estimates.trade_reason)
            prechecks = preview["order_checks"].set_index("symbol")
            target["reference_quantity_check"] = target.symbol.map(prechecks.check_reason).fillna("no_order")
            target["execution_day_status"] = "UNKNOWN_UNTIL_EXECUTION_DAY"
            target["minimum_buy_shares"] = target.board.map({"STAR": 200, "MAIN": 100, "CHINEXT": 100})
            target.to_csv(stage / (strategy + ".csv"), index=False)
            combined.append(target)
            proposed_value = float(preview["account_snapshot"]["market_value"])
            preview_trades = preview["paper_trades"]
            preview_fees = float(preview_trades.commission.sum()) if not preview_trades.empty else 0.0
            summary.append({"strategy": strategy, "label": LABELS.get(strategy, strategy), "targets": len(target),
                            "estimated_positions": len(preview["positions"]),
                            "estimated_stock_value": proposed_value, "estimated_commission": preview_fees,
                            "estimated_cash": float(preview["cash"]),
                            "star_targets": int(target.board.eq("STAR").sum()), "chinext_targets": int(target.board.eq("CHINEXT").sum()),
                            "status": "FROZEN_TARGETS_NOT_FILLS"})
        pd.concat(combined, ignore_index=True).to_csv(stage / "targets_combined.csv", index=False)
        pd.DataFrame(summary).to_csv(stage / "summary.csv", index=False)
        bundle["report"].to_csv(stage / "universe_audit.csv.gz", index=False, compression="gzip")
        last.to_csv(stage / "signal_market.csv.gz", index=False, compression="gzip")
        # Save raw adjustment factors for corporate-action guards on the first day.
        adj = pd.read_csv(data_dir / "adj_factor_all_a.csv.gz", dtype={"trade_date": str})
        adj = adj[adj.trade_date.eq(signal.strftime("%Y%m%d"))]
        if adj.empty:
            raise RuntimeError("missing signal-date adjustment factors")
        adj.to_csv(stage / "adj_factor.csv", index=False)
        write_json(stage / "input_receipt.json", receipt)
        seal(stage, {"kind": "FROZEN_PLAN", "created_at_utc": utc_now().isoformat(), "signal_date": signal_date,
                     "execution_date": execution_date, "prior_state_sha256": sha(prior / "manifest.json"),
                     "session_sha256": sha(root / "initial" / "manifest.json"),
                     "status": "AWAITING_FUTURE_SESSION", "live_approval": False,
                     "revision_sha256": config.get("revision_sha256")})
        stage.rename(dest)
    return dest


def required_symbols(root: Path, date: str) -> set[str]:
    config = session_config(root)
    prior, _ = latest_day(root)
    symbols = set()
    for strategy in config["strategies"]:
        _, positions = load_account_state(session_settings(prior, config), strategy=strategy)
        symbols.update(positions.symbol)
    plan = root / "plans" / pd.Timestamp(date).strftime("%Y%m%d")
    if plan.exists():
        verify(plan)
        symbols.update(pd.read_csv(plan / "targets_combined.csv").symbol)
    return symbols


def _market_check(frame: pd.DataFrame, symbols: set[str], date: str) -> pd.DataFrame:
    required = {"date", "symbol", "close", "adj_factor", "is_suspended", "is_limit_up", "is_limit_down", "price_date"}
    if required - set(frame):
        raise RuntimeError("market snapshot schema incomplete")
    if frame.symbol.duplicated().any() or set(frame.symbol) != symbols or not frame.date.eq(date).all():
        raise RuntimeError("market snapshot date/symbol coverage mismatch")
    for col in ["is_suspended", "is_limit_up", "is_limit_down"]:
        if not frame[col].isin([True, False, 0, 1]).all():
            raise RuntimeError("unknown trade status")
    for col in ["close", "adj_factor"]:
        if not frame[col].map(lambda x: math.isfinite(float(x)) and float(x) > 0).all():
            raise RuntimeError("invalid market " + col)
    if not frame.loc[~frame.is_suspended.astype(bool), "price_date"].eq(date).all():
        raise RuntimeError("active stock has stale close")
    return frame.set_index("symbol")


def advance(root: Path, market_dir: Path, trade_date: str) -> Path:
    """One daily entry: execute a frozen plan or hold/mark the previous day."""
    with locked(root):
        config = session_config(root)
        sessions = calendar_dates(root / "initial" / "calendar.csv")
        date = pd.Timestamp(trade_date)
        if utc_now().tz_convert("Asia/Shanghai") < date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
            raise RuntimeError("requested session has not closed")
        prior, prior_meta = latest_day(root)
        if config.get("revision_effective_date") and trade_date < config["revision_effective_date"]:
            raise RuntimeError("session revision is not effective for this day")
        if date != next_session(sessions, prior_meta["date"]):
            raise RuntimeError("must process exactly the next open session; duplicate/backdated/skipped day blocked")
        dest = root / "days" / date.strftime("%Y%m%d")
        if dest.exists():
            raise RuntimeError("day already committed")
        plan = root / "plans" / date.strftime("%Y%m%d")
        if plan.exists():
            meta = verify(plan)
            if meta.get("revision_sha256") != config.get("revision_sha256"):
                raise RuntimeError("frozen plan belongs to another execution revision")
            if meta["prior_state_sha256"] != sha(prior / "manifest.json"):
                raise RuntimeError("account changed after plan freeze")
            if pd.Timestamp(meta["created_at_utc"]).tz_convert("Asia/Shanghai") >= date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30):
                raise RuntimeError("late plan is not forward evidence")
        else:
            previous_date = pd.Timestamp(prior_meta["date"])
            if previous_date.to_period("W-FRI") != date.to_period("W-FRI") or prior == root / "initial":
                raise RuntimeError("scheduled rebalance plan is missing; do not silently hold")
        market_meta = verify(market_dir)
        if market_meta["date"] != trade_date:
            raise RuntimeError("wrong market snapshot day")
        observed = pd.Timestamp(market_meta["retrieved_at_utc"])
        if observed > utc_now() or observed < date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
            raise RuntimeError("market snapshot was not retrieved after close")
        market = _market_check(pd.read_csv(market_dir / "market.csv"), required_symbols(root, trade_date), trade_date)
        # Carrying prices is permitted only for explicitly confirmed suspensions.
        for sym, rec in market.iterrows():
            if rec.is_suspended and pd.Timestamp(rec.price_date) > date:
                raise RuntimeError("future suspended-stock valuation")
        previous_market = pd.read_csv(prior / "market.csv").set_index("symbol") if (prior / "market.csv").exists() else None
        carry_source = previous_market
        if carry_source is None and plan.exists():
            carry_source = pd.read_csv(plan / "signal_market.csv.gz").rename(
                columns={"ts_code": "symbol", "trade_date": "price_date"}).set_index("symbol")
        for symbol, rec in market.iterrows():
            if rec.price_date != trade_date:
                if carry_source is None or symbol not in carry_source.index:
                    raise RuntimeError("no traceable suspended-stock carry price")
                old = carry_source.loc[symbol]
                if rec.price_date != old.price_date or not math.isclose(float(rec.close), float(old.close), abs_tol=1e-8):
                    raise RuntimeError("suspended-stock price changed without a trade")
        from live.cash_dividends import POLICY, EVENT_COLUMNS, PAYMENT_COLUMNS, normalize_events, payments_for
        events = pd.DataFrame(columns=EVENT_COLUMNS)
        action_dir = None
        held = set()
        if previous_market is not None:
            held = set()
            for strategy in config["strategies"]:
                _, pos = load_account_state(session_settings(prior, config), strategy=strategy)
                held.update(pos.symbol)
            for symbol in held:
                if symbol not in market.index or symbol not in previous_market.index:
                    raise RuntimeError("held stock missing from price history")
                if config["corporate_actions"] != POLICY and not math.isclose(float(market.loc[symbol, "adj_factor"]), float(previous_market.loc[symbol, "adj_factor"]), rel_tol=1e-10):
                    raise RuntimeError("corporate action needs explicit cash/share processing: " + symbol)
        if config["corporate_actions"] == POLICY:
            if previous_market is None:
                raise RuntimeError("cash dividend revision requires a prior daily account")
            action_dir = root / "corporate_actions" / date.strftime("%Y%m%d")
            action_meta = verify(action_dir)
            expected = {"date": trade_date, "prior_manifest_sha256": sha(prior / "manifest.json"),
                        "market_manifest_sha256": sha(market_dir / "manifest.json"),
                        "revision_sha256": config["revision_sha256"], "policy": POLICY}
            if any(action_meta.get(k) != v for k, v in expected.items()):
                raise RuntimeError("corporate action evidence does not match daily inputs")
            retrieved = pd.Timestamp(action_meta["retrieved_at_utc"])
            if retrieved > utc_now() or retrieved < date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15):
                raise RuntimeError("invalid corporate action retrieval timestamp")
            raw = pd.read_csv(action_dir / "dividend_raw.csv", dtype=str)
            events = normalize_events(raw, held, previous_market, market, prior_meta["date"], trade_date)
            saved_events = pd.read_csv(action_dir / "events.csv")
            pd.testing.assert_frame_equal(events, saved_events, check_dtype=False)
        dividend_history = (pd.read_csv(prior / "dividends.csv") if (prior / "dividends.csv").exists()
                            else pd.DataFrame(columns=PAYMENT_COLUMNS))
        if dividend_history.duplicated(["strategy", "event_id"]).any():
            raise RuntimeError("duplicate dividend history")
        with staging(root / "days") as stage:
            shutil.copytree(prior / "paper_account", stage / "paper_account")
            market.reset_index().to_csv(stage / "market.csv", index=False)
            shutil.copytree(market_dir, stage / "market_source")
            if action_dir is not None:
                shutil.copytree(action_dir, stage / "corporate_action_source")
            settings = session_settings(stage, config)
            prices = market.close
            blacklist_path = market_dir / "risk_blacklist.csv"
            if not blacklist_path.exists():
                raise RuntimeError("explicit blacklist snapshot required, even when empty")
            blacklist = load_risk_blacklist(blacklist_path)
            summaries = []
            for strategy in config["strategies"]:
                cash, positions = load_account_state(settings, strategy=strategy)
                credited = set(dividend_history.loc[dividend_history.strategy.eq(strategy), "event_id"])
                payments = payments_for(events, positions, strategy, credited)
                dividend_cash = float(payments.cash_gross.sum()) if not payments.empty else 0.0
                if not payments.empty:
                    dividend_history = pd.concat([dividend_history, payments], ignore_index=True)
                cash += dividend_cash
                # Stage the adjusted starting cash for the existing trading runner.
                pd.DataFrame([{"cash": cash, "updated_at": trade_date}]).to_csv(
                    stage / "paper_account" / strategy / "account.csv", index=False)
                # Each run is a new open session: yesterday's purchases are now sellable.
                positions["available_shares"] = positions.shares
                positions.to_csv(stage / "paper_account" / strategy / "positions.csv", index=False)
                if plan.exists():
                    target = pd.read_csv(plan / (strategy + ".csv"))
                    result = run_daily_paper_trade(settings, strategy=strategy, target_weights=target.set_index("symbol").weight,
                                                  latest_prices=prices, trade_date=date, trade_status=market.reset_index(),
                                                  risk_blacklist=blacklist, execution_mode="simulated_broker", persist_outputs=False)
                    raise_on_guard_errors(validate_daily_result(result))
                    cash, positions = result["cash"], result["positions"]
                    positions["available_shares"] = positions.shares
                    buys = result["paper_trades"]
                    if not buys.empty:
                        buys = buys[(buys.side == "BUY") & (buys.fill_status == "FILLED")].set_index("symbol").qty
                        positions["available_shares"] -= positions.symbol.map(buys).fillna(0).to_numpy()
                    for key in ["orders", "order_checks", "paper_trades"]:
                        folder = stage / key
                        folder.mkdir(exist_ok=True)
                        result[key].to_csv(folder / (strategy + ".csv"), index=False)
                    filled = result["paper_trades"]
                    fees = float(filled.commission.sum()) if not filled.empty else 0.0
                else:
                    result, fees = None, 0.0
                mv = float((positions.set_index("symbol").shares * prices).sum()) if not positions.empty else 0.0
                snapshot = {"cash": float(cash), "market_value": mv, "total_asset": float(cash) + mv, "n_positions": len(positions)}
                if not all(math.isfinite(v) and v >= 0 for v in snapshot.values()):
                    raise RuntimeError("invalid account reconciliation")
                # Reconcile cash independently against filled trade flows.
                old_cash, _ = load_account_state(session_settings(prior, config), strategy=strategy)
                flow = float(result["paper_trades"].net_cash_flow.sum()) if result is not None and not result["paper_trades"].empty else 0.0
                if not math.isclose(cash, old_cash + dividend_cash + flow, rel_tol=0, abs_tol=1e-6):
                    raise RuntimeError("cash reconciliation failed")
                _, old_positions = load_account_state(session_settings(prior, config), strategy=strategy)
                expected = old_positions.set_index("symbol").shares.to_dict()
                if result is not None:
                    for trade in result["paper_trades"].to_dict("records"):
                        if trade["fill_status"] == "FILLED":
                            expected[trade["symbol"]] = expected.get(trade["symbol"], 0) + trade["qty"] * (1 if trade["side"] == "BUY" else -1)
                expected = {k: v for k, v in expected.items() if v > 0}
                if expected != positions.set_index("symbol").shares.to_dict():
                    raise RuntimeError("position reconciliation failed")
                save_account_state(settings, strategy=strategy, cash=cash, positions=positions, snapshot=snapshot, trade_date=date)
                summaries.append({"date": trade_date, "strategy": strategy, "action": "REBALANCE" if plan.exists() else "HOLD_MARK",
                                  **snapshot, "commission": fees, "net_value": snapshot["total_asset"] / config["initial_cash"],
                                  "dividend_cash_gross": dividend_cash,
                                  "dividend_tax_policy": config.get("dividend_tax_policy", "NOT_ENABLED"),
                                  "filled_orders": int(result["paper_trades"].fill_status.eq("FILLED").sum()) if result is not None and not result["paper_trades"].empty else 0})
            dividend_history.to_csv(stage / "dividends.csv", index=False)
            pd.DataFrame(summaries).to_csv(stage / "summary.csv", index=False)
            seal(stage, {"kind": "DAILY_CLOSE_MODEL", "date": trade_date, "completed_at_utc": utc_now().isoformat(),
                         "previous_manifest_sha256": sha(prior / "manifest.json"),
                         "plan_sha256": sha(plan / "manifest.json") if plan.exists() else None,
                         "revision_sha256": config.get("revision_sha256"),
                         "corporate_action_manifest_sha256": sha(action_dir / "manifest.json") if action_dir is not None else None,
                         "dividend_tax_policy": config.get("dividend_tax_policy", "NOT_ENABLED"),
                         "live_approval": False, "news_risk": "NOT_ASSESSED", "stress_gate": "NOT_ENABLED_IN_PARENT_PROTOCOL"})
            stage.rename(dest)
    return dest
