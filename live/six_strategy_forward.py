"""Six-account forward paper-trading workflow for strategy universes.

The signal layer is deliberately separated from the execution layer.  A signal
known after one session closes is saved with the *next* trading session as its
execution date.  The execution entry point refuses to fill it until an exact
price row for that date exists, preventing accidental same-close look-ahead.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Settings, get_settings
from live.account_state import account_state_dir, load_account_state, save_account_state
from live.daily_paper_cli import load_latest_prices, run_daily_paper_from_outputs
from live.paper_run_control import has_paper_snapshot
from live.strategy_universe import latest_universe_snapshot
from scripts.build_dynamic_universe_comparison import build_weekly_raw_factor_panel
from strategies.style import build_fixed_family_score, build_focused_style_score
from universe.dynamic import build_dynamic_universe_report, eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import (
    build_strategy_universe_report,
    daily_basic_for_index,
    eligibility_for_profile,
    point_in_time_industry,
    profiles_from_config,
)


FORWARD_MANIFEST = "forward_manifest.json"
PENDING_STATUS = "PENDING_EXECUTION_PRICE"
EXECUTED_STATUS = "EXECUTED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    hash_path = path.with_suffix(".sha256")
    if hash_path.exists():
        expected = hash_path.read_text(encoding="utf-8").split()[0]
        if digest != expected:
            raise RuntimeError("strategy protocol hash mismatch")
    return protocol, digest


def _read_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "prices": ("prices_all_a_qfq.csv.gz", {"parse_dates": ["trade_date"]}),
        "finance": ("fina_indicator_all_a.csv.gz", {}),
        "balance": ("balance_sheet_all_a.csv.gz", {}),
        "audit": ("fina_audit_all_a.csv.gz", {}),
        "basic": ("stock_basic_all_a.csv", {"dtype": {"ts_code": str}}),
        "names": ("namechange_all_a.csv", {"dtype": {"ts_code": str}}),
        "calendar": ("trade_calendar.csv", {"dtype": {"cal_date": str}}),
        "daily_basic": ("daily_basic_weekly.csv.gz", {"dtype": {"ts_code": str}}),
        "industry_membership": (
            "sw2021_l1_membership.csv.gz",
            {"dtype": {"ts_code": str}},
        ),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (filename, kwargs) in required.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frames[name] = pd.read_csv(path, **kwargs)
    return frames


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return ""


def _score_targets(
    protocol: dict[str, Any],
    raw_panel: pd.DataFrame,
    base_eligible: pd.Series,
    profile_eligibility: dict[str, pd.Series],
    industry: pd.Series,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    families = {
        str(name): [str(factor) for factor in factors]
        for name, factors in protocol["candidate_families"].items()
    }
    targets: dict[str, pd.DataFrame] = {}
    universe_by_strategy: dict[str, str] = {}
    for spec in protocol["strategies"]:
        name = str(spec["name"])
        universe = str(spec["universe"])
        eligible = (
            base_eligible if universe == "BASE_V2" else profile_eligibility[universe]
        )
        if spec["score"] == "fixed_equal_five_family_multi_factor":
            score, _ = build_fixed_family_score(
                raw_panel,
                eligible,
                industry,
                families,
            )
        else:
            score, _ = build_focused_style_score(
                raw_panel,
                eligible,
                industry,
                [str(value) for value in spec["factors"]],
                minimum_components=int(spec["minimum_components"]),
            )
        if score.empty:
            raise RuntimeError("strategy %s produced no score" % name)
        latest_date = pd.Timestamp(score.index.get_level_values("date").max())
        ranked = (
            score.xs(latest_date, level="date")
            .dropna()
            .sort_values(ascending=False)
            .head(int(spec["top_k"]))
        )
        if len(ranked) != int(spec["top_k"]):
            raise RuntimeError(
                "strategy %s produced %d targets, expected %d"
                % (name, len(ranked), int(spec["top_k"]))
            )
        frame = pd.DataFrame(
            {
                "symbol": ranked.index.astype(str),
                "score": ranked.to_numpy(dtype=float),
                "rank": np.arange(1, len(ranked) + 1),
                "weight": 1.0 / float(len(ranked)),
            }
        )
        frame["strategy"] = name
        frame["universe"] = universe
        targets[name] = frame
        universe_by_strategy[name] = universe
    return targets, universe_by_strategy


def build_forward_signals(
    data_dir: Path,
    protocol_path: Path,
    *,
    signal_date: Any,
) -> dict[str, Any]:
    """Build six point-in-time targets using information known by signal_date."""
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    frames = _read_inputs(data_dir)
    signal_dt = pd.Timestamp(signal_date).normalize()
    prices = frames["prices"].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="raise").dt.normalize()
    latest_price_date = pd.Timestamp(prices["trade_date"].max())
    if signal_dt > latest_price_date:
        raise ValueError(
            "signal date %s exceeds latest complete price date %s"
            % (signal_dt.strftime("%Y-%m-%d"), latest_price_date.strftime("%Y-%m-%d"))
        )
    if not bool(prices["trade_date"].eq(signal_dt).any()):
        raise ValueError("signal date is not present in complete market data")
    prices = prices[prices["trade_date"].le(signal_dt)].copy()
    decision_dates = pd.DatetimeIndex([signal_dt])

    base_cfg = protocol["base_universe"]
    tradable_report = build_dynamic_universe_report(
        prices,
        frames["basic"],
        frames["names"],
        frames["calendar"],
        decision_dates,
        min_listing_sessions=int(base_cfg["minimum_listing_open_sessions"]),
        liquidity_window=int(base_cfg["liquidity_lookback_open_sessions"]),
        min_valid_days=int(base_cfg["minimum_valid_trading_days"]),
        min_adv_yuan=float(base_cfg["minimum_adv20_cny"]),
        amount_col="amount_yuan",
    )
    risk_report = build_hard_risk_universe_report(
        tradable_report,
        frames["balance"],
        frames["audit"],
        frames["names"],
    )
    base_eligible = eligibility_from_report(risk_report)
    raw_panel = build_weekly_raw_factor_panel(
        prices,
        frames["finance"],
        decision_dates,
    )
    profiles = profiles_from_config(protocol["strategy_universe_profiles"])
    report = build_strategy_universe_report(
        risk_report,
        frames["daily_basic"],
        raw_panel,
        frames["industry_membership"],
        profiles,
        max_daily_basic_staleness_days=int(
            protocol["data"]["maximum_daily_basic_staleness_calendar_days"]
        ),
    )
    profile_eligibility = {
        profile.name: eligibility_for_profile(report, profile) for profile in profiles
    }
    industry = point_in_time_industry(
        raw_panel.index,
        frames["industry_membership"],
    )
    market = daily_basic_for_index(
        raw_panel.index,
        frames["daily_basic"],
        max_staleness_days=int(
            protocol["data"]["maximum_daily_basic_staleness_calendar_days"]
        ),
    )
    pe_ttm = pd.to_numeric(market["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(market["pb"], errors="coerce")
    raw_panel = raw_panel.copy()
    raw_panel["EARNINGS_YIELD"] = (1.0 / pe_ttm).where(pe_ttm.gt(0))
    raw_panel["BOOK_TO_PRICE"] = (1.0 / pb).where(pb.gt(0))
    targets, universe_by_strategy = _score_targets(
        protocol,
        raw_panel,
        base_eligible,
        profile_eligibility,
        industry,
    )
    return {
        "signal_date": signal_dt,
        "latest_price_date": latest_price_date,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha256,
        "report": report,
        "targets": targets,
        "universe_by_strategy": universe_by_strategy,
        "prices": prices,
    }


def refresh_runtime_market_cache(
    data_dir: Path,
    output_dir: Path,
    *,
    through_date: Any,
    execution_prices_path: Path | None = None,
    calendar_sessions: int = 160,
    liquidity_sessions: int = 60,
) -> dict[str, Path]:
    """Write execution-price and liquidity caches from complete raw market rows."""
    source = data_dir / "prices_all_a_qfq.csv.gz"
    prices = pd.read_csv(
        source,
        usecols=["trade_date", "ts_code", "close", "amount_yuan"],
        parse_dates=["trade_date"],
    )
    if execution_prices_path is not None:
        extra_path = Path(execution_prices_path)
        if not extra_path.exists():
            raise FileNotFoundError(extra_path)
        extra = pd.read_csv(extra_path, parse_dates=["trade_date"])
        required = {"trade_date", "ts_code", "close", "amount_yuan"}
        if missing := required - set(extra.columns):
            raise ValueError("execution prices missing columns: %s" % sorted(missing))
        prices = pd.concat(
            [prices, extra[list(required)]],
            ignore_index=True,
        ).drop_duplicates(["trade_date", "ts_code"], keep="last")
    through = pd.Timestamp(through_date).normalize()
    prices["trade_date"] = prices["trade_date"].dt.normalize()
    prices = prices[prices["trade_date"].le(through)].copy()
    sessions = pd.DatetimeIndex(prices["trade_date"].dropna().unique()).sort_values()
    if sessions.empty or pd.Timestamp(sessions[-1]) < through:
        raise RuntimeError("complete execution prices are unavailable for %s" % through.date())
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    wide_dates = sessions[-max(1, int(calendar_sessions)) :]
    wide = (
        prices[prices["trade_date"].isin(wide_dates)]
        .pivot(index="trade_date", columns="ts_code", values="close")
        .sort_index()
    )
    # Execution valuation carries the last close through short suspensions.  A
    # separate trade-status file must still block orders in suspended names.
    wide = wide.ffill(limit=5)
    wide.index.name = "date"
    wide_path = cache_dir / "prices_wide_close.csv"
    wide.to_csv(wide_path)

    long_dates = sessions[-max(1, int(liquidity_sessions)) :]
    long = prices[prices["trade_date"].isin(long_dates)][
        ["trade_date", "ts_code", "close", "amount_yuan"]
    ].rename(
        columns={"trade_date": "date", "ts_code": "symbol", "amount_yuan": "amount"}
    )
    long_path = cache_dir / "prices_long.csv"
    long.to_csv(long_path, index=False)
    return {"prices_wide": wide_path, "prices_long": long_path}


def _initialize_accounts(
    settings: Settings,
    strategies: list[str],
    *,
    initial_cash: float,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        base = account_state_dir(settings, strategy)
        account_path = base / "account.csv"
        positions_path = base / "positions.csv"
        if account_path.exists() or positions_path.exists():
            status = "EXISTING_PRESERVED"
        else:
            save_account_state(
                settings,
                strategy=strategy,
                cash=float(initial_cash),
                positions=pd.DataFrame(columns=["symbol", "shares"]),
                snapshot={
                    "cash": float(initial_cash),
                    "market_value": 0.0,
                    "total_asset": float(initial_cash),
                    "n_positions": 0,
                },
                trade_date=start_date,
            )
            status = "INITIALIZED"
        rows.append(
            {
                "strategy": strategy,
                "status": status,
                "initial_cash": float(initial_cash),
                "account_dir": str(base),
            }
        )
    return pd.DataFrame(rows)


def save_forward_bundle(
    bundle: dict[str, Any],
    output_dir: Path,
    *,
    execution_date: Any,
    initial_cash: float = 1_000_000.0,
    paper_protocol_path: Path | None = None,
) -> dict[str, Path]:
    """Persist target logs, universe audit, caches, manifest and six cash accounts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signal_date = pd.Timestamp(bundle["signal_date"]).normalize()
    execution_dt = pd.Timestamp(execution_date).normalize()
    if execution_dt <= signal_date:
        raise ValueError("execution date must be after signal date")
    rebalance_dir = output / "rebalance_logs"
    snapshot_dir = output / "universe_snapshots"
    signal_dir = output / "signals" / signal_date.strftime("%Y%m%d")
    rebalance_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)

    target_rows = []
    paths: dict[str, Path] = {}
    for strategy, target in bundle["targets"].items():
        frame = target.copy()
        frame.insert(0, "date", execution_dt.strftime("%Y-%m-%d"))
        frame["signal_date"] = signal_date.strftime("%Y-%m-%d")
        frame["selected"] = True
        frame["selected_rank"] = frame["rank"]
        frame["weighting"] = "equal"
        frame["execution_status"] = PENDING_STATUS
        log_path = rebalance_dir / (strategy + ".csv")
        if log_path.exists():
            previous = pd.read_csv(log_path, dtype={"symbol": str})
            previous["date"] = pd.to_datetime(previous["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
            previous = previous[
                previous["date"].ne(execution_dt.strftime("%Y-%m-%d"))
            ]
            history = pd.concat([previous, frame], ignore_index=True, sort=False)
        else:
            history = frame
        history = history.sort_values(["date", "rank", "symbol"]).reset_index(drop=True)
        history.to_csv(log_path, index=False)
        signal_path = signal_dir / (strategy + ".csv")
        frame.to_csv(signal_path, index=False)
        paths["rebalance_" + strategy] = log_path
        target_rows.append(frame)

    combined = pd.concat(target_rows, ignore_index=True)
    combined_path = signal_dir / "targets_combined.csv"
    combined.to_csv(combined_path, index=False)
    paths["targets_combined"] = combined_path

    report = bundle["report"]
    audit_path = signal_dir / "strategy_universe_audit.csv.gz"
    report.to_csv(audit_path, index=False, compression="gzip")
    paths["universe_audit"] = audit_path
    for universe in sorted(set(bundle["universe_by_strategy"].values())):
        snapshot = latest_universe_snapshot(report, universe=universe)
        path = snapshot_dir / (universe + ".csv")
        snapshot.to_csv(path, index=False)
        paths["universe_" + universe] = path

    industry = report[["symbol", "industry_l1"]].drop_duplicates("symbol", keep="last")
    industry = industry.rename(columns={"industry_l1": "industry"}).sort_values("symbol")
    industry_path = output / "cache" / "industry_latest.csv"
    industry_path.parent.mkdir(parents=True, exist_ok=True)
    industry.to_csv(industry_path, index=False)
    paths["industry"] = industry_path

    settings = replace(
        get_settings(),
        output_dir=output,
        paper_initial_cash=float(initial_cash),
        portfolio_weighting="equal",
    )
    accounts = _initialize_accounts(
        settings,
        sorted(bundle["targets"]),
        initial_cash=float(initial_cash),
        start_date=signal_date,
    )
    accounts_path = output / "paper_accounts.csv"
    accounts.to_csv(accounts_path, index=False)
    paths["paper_accounts"] = accounts_path

    paper_protocol_sha256 = ""
    if paper_protocol_path is not None:
        _paper_protocol, paper_protocol_sha256 = _load_protocol(paper_protocol_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(get_settings().project_root),
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "scheduled_execution_date": execution_dt.strftime("%Y-%m-%d"),
        "data_latest_complete_date": pd.Timestamp(bundle["latest_price_date"]).strftime(
            "%Y-%m-%d"
        ),
        "status": PENDING_STATUS,
        "strategy_protocol": str(bundle["protocol_path"]),
        "strategy_protocol_sha256": bundle["protocol_sha256"],
        "paper_protocol": str(paper_protocol_path) if paper_protocol_path is not None else "",
        "paper_protocol_sha256": paper_protocol_sha256,
        "execution_timing": "NEXT_SESSION_CLOSE",
        "accounts": {
            "count": int(len(bundle["targets"])),
            "initial_cash_each": float(initial_cash),
            "total_virtual_capital": float(initial_cash) * len(bundle["targets"]),
            "isolated_by_strategy": True,
        },
        "strategies": sorted(bundle["targets"]),
        "forward_evidence_start": execution_dt.strftime("%Y-%m-%d"),
        "bootstrap_note": (
            "Signal was computed after the signal-date close. No fill is permitted "
            "until exact scheduled-execution-date prices are available."
        ),
        "live_approval": False,
    }
    manifest_path = output / FORWARD_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths


def prepare_six_strategy_forward_paper(
    data_dir: Path,
    output_dir: Path,
    protocol_path: Path,
    *,
    signal_date: Any,
    execution_date: Any,
    initial_cash: float = 1_000_000.0,
    paper_protocol_path: Path | None = None,
) -> dict[str, Path]:
    bundle = build_forward_signals(
        data_dir,
        protocol_path,
        signal_date=signal_date,
    )
    paths = save_forward_bundle(
        bundle,
        output_dir,
        execution_date=execution_date,
        initial_cash=initial_cash,
        paper_protocol_path=paper_protocol_path,
    )
    paths.update(
        refresh_runtime_market_cache(
            data_dir,
            output_dir,
            through_date=signal_date,
        )
    )
    return paths


def execute_six_strategy_paper(
    data_dir: Path,
    output_dir: Path,
    *,
    trade_date: Any,
    trade_status_path: Path | None = None,
    allow_rerun: bool = False,
) -> pd.DataFrame:
    """Execute all frozen accounts only when exact trade-date prices exist."""
    output = Path(output_dir)
    manifest_path = output / FORWARD_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trade_dt = pd.Timestamp(trade_date).normalize()
    scheduled = pd.Timestamp(manifest["scheduled_execution_date"]).normalize()
    if trade_dt != scheduled:
        raise ValueError(
            "trade date %s does not match frozen execution date %s"
            % (trade_dt.date(), scheduled.date())
        )
    execution_prices_path = output / "cache" / (
        "execution_prices_%s.csv" % trade_dt.strftime("%Y%m%d")
    )
    refresh_runtime_market_cache(
        data_dir,
        output,
        through_date=trade_dt,
        execution_prices_path=execution_prices_path,
    )
    status_path = (
        Path(trade_status_path)
        if trade_status_path is not None
        else output / "cache" / ("trade_status_%s.csv" % trade_dt.strftime("%Y%m%d"))
    )
    if not status_path.exists():
        raise FileNotFoundError(
            "missing execution trade-status gate: %s" % status_path
        )
    price_path = output / "cache" / "prices_wide_close.csv"
    price_date, _ = load_latest_prices(price_path, trade_date=trade_dt)
    if pd.Timestamp(price_date).normalize() != trade_dt:
        raise RuntimeError("exact execution-date prices are unavailable")

    initial_cash = float(manifest["accounts"]["initial_cash_each"])
    settings = replace(
        get_settings(),
        output_dir=output,
        paper_initial_cash=initial_cash,
        portfolio_weighting="equal",
    )
    common = {
        "settings": settings,
        "trade_date": trade_dt,
        "prices_path": price_path,
        "trade_status_path": status_path,
        "execution_mode": "simulated_broker",
        "generate_manual_confirmation": False,
        "liquidity_history_path": output / "cache" / "prices_long.csv",
        "industry_path": output / "cache" / "industry_latest.csv",
        "allow_rerun": allow_rerun,
    }
    strategies = [str(value) for value in manifest["strategies"]]
    # Validate every account before mutating any account state.
    for strategy in strategies:
        run_daily_paper_from_outputs(
            strategy=strategy,
            rebalance_log_path=output / "rebalance_logs" / (strategy + ".csv"),
            persist_outputs=False,
            generate_report=False,
            **common,
        )

    rows = []
    for strategy in strategies:
        result = run_daily_paper_from_outputs(
            strategy=strategy,
            rebalance_log_path=output / "rebalance_logs" / (strategy + ".csv"),
            persist_outputs=True,
            generate_report=True,
            **common,
        )
        trades = result["paper_trades"]
        checks = result["order_checks"]
        snapshot = result["account_snapshot"]
        rows.append(
            {
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "strategy": strategy,
                "orders": int(len(result["orders"])),
                "passed_orders": int((checks["check_status"] == "PASS").sum())
                if not checks.empty
                else 0,
                "blocked_orders": int((checks["check_status"] == "BLOCK").sum())
                if not checks.empty
                else 0,
                "filled_orders": int((trades["fill_status"] == "FILLED").sum())
                if not trades.empty
                else 0,
                "cash": float(snapshot.get("cash", 0.0)),
                "market_value": float(snapshot.get("market_value", 0.0)),
                "total_asset": float(snapshot.get("total_asset", 0.0)),
                "positions": int(float(snapshot.get("n_positions", 0))),
            }
        )
    summary = pd.DataFrame(rows)
    run_dir = output / "parallel_runs" / trade_dt.strftime("%Y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_dir / "summary.csv", index=False)
    manifest["status"] = EXECUTED_STATUS
    manifest["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["execution_summary"] = str(run_dir / "summary.csv")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def mark_six_strategy_accounts(
    data_dir: Path,
    output_dir: Path,
    *,
    trade_date: Any,
    allow_rerun: bool = False,
) -> pd.DataFrame:
    """Mark all six accounts to market without creating rebalance orders."""
    output = Path(output_dir)
    manifest_path = output / FORWARD_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trade_dt = pd.Timestamp(trade_date).normalize()
    refresh_runtime_market_cache(data_dir, output, through_date=trade_dt)
    price_date, prices = load_latest_prices(
        output / "cache" / "prices_wide_close.csv",
        trade_date=trade_dt,
    )
    if pd.Timestamp(price_date).normalize() != trade_dt:
        raise RuntimeError("exact mark-to-market date prices are unavailable")
    initial_cash = float(manifest["accounts"]["initial_cash_each"])
    settings = replace(
        get_settings(),
        output_dir=output,
        paper_initial_cash=initial_cash,
        portfolio_weighting="equal",
    )
    rows = []
    for strategy in [str(value) for value in manifest["strategies"]]:
        if not allow_rerun and has_paper_snapshot(
            settings,
            strategy=strategy,
            trade_date=trade_dt,
        ):
            raise RuntimeError(
                "strategy %s already has a snapshot for %s"
                % (strategy, trade_dt.strftime("%Y-%m-%d"))
            )
        cash, positions = load_account_state(
            settings,
            strategy=strategy,
            default_cash=initial_cash,
        )
        if positions.empty:
            market_value = 0.0
        else:
            shares = pd.Series(
                pd.to_numeric(positions["shares"], errors="coerce").fillna(0.0).to_numpy(),
                index=positions["symbol"].astype(str),
                dtype=float,
            )
            missing = sorted(set(shares[shares.gt(0)].index) - set(prices.index.astype(str)))
            if missing:
                raise RuntimeError(
                    "mark-to-market prices missing for %s: %s"
                    % (strategy, ", ".join(missing[:10]))
                )
            market_value = float((shares * prices.reindex(shares.index).astype(float)).sum())
        snapshot = {
            "cash": float(cash),
            "market_value": market_value,
            "total_asset": float(cash) + market_value,
            "n_positions": int(len(positions[positions["shares"].astype(float).gt(0)])),
        }
        save_account_state(
            settings,
            strategy=strategy,
            cash=float(cash),
            positions=positions,
            snapshot=snapshot,
            trade_date=trade_dt,
        )
        rows.append({"trade_date": trade_dt.strftime("%Y-%m-%d"), "strategy": strategy, **snapshot})
    summary = pd.DataFrame(rows)
    run_dir = output / "parallel_marks" / trade_dt.strftime("%Y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_dir / "summary.csv", index=False)
    return summary
