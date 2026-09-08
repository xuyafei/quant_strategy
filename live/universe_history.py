"""Point-in-time universe helpers for index constituent backtests."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from live.stock_pool import normalize_ts_code


CHANGE_COLUMNS = ["effective_date", "action", "ts_code", "name", "source_url"]
MEMBERSHIP_COLUMNS = ["ts_code", "effective_from", "effective_to"]


def load_universe_changes(path: str | Path) -> pd.DataFrame:
    """Load and validate index constituent changes."""
    frame = pd.read_csv(Path(path).expanduser())
    missing = set(CHANGE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("universe changes missing columns: %s" % ", ".join(sorted(missing)))
    out = frame[CHANGE_COLUMNS].copy()
    out["effective_date"] = pd.to_datetime(out["effective_date"], errors="raise").dt.normalize()
    out["action"] = out["action"].astype(str).str.strip().str.upper()
    if not out["action"].isin({"ENTER", "EXIT"}).all():
        bad = sorted(out.loc[~out["action"].isin({"ENTER", "EXIT"}), "action"].unique())
        raise ValueError("unsupported universe change actions: %s" % bad)
    out["ts_code"] = out["ts_code"].map(normalize_ts_code)
    if (out["ts_code"] == "").any():
        raise ValueError("universe changes contain empty ts_code")
    if out.duplicated(["effective_date", "action", "ts_code"]).any():
        raise ValueError("universe changes contain duplicate effective_date/action/ts_code rows")
    return out.sort_values(["effective_date", "action", "ts_code"]).reset_index(drop=True)


def load_membership_intervals(path: str | Path) -> pd.DataFrame:
    """Load validated inclusive point-in-time membership intervals."""
    frame = pd.read_csv(Path(path).expanduser())
    missing = set(MEMBERSHIP_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("membership intervals missing columns: %s" % ", ".join(sorted(missing)))
    out = frame[MEMBERSHIP_COLUMNS].copy()
    out["ts_code"] = out["ts_code"].map(normalize_ts_code)
    out["effective_from"] = pd.to_datetime(out["effective_from"], errors="raise").dt.normalize()
    out["effective_to"] = pd.to_datetime(out["effective_to"], errors="raise").dt.normalize()
    if (out["ts_code"] == "").any():
        raise ValueError("membership intervals contain empty ts_code")
    if (out["effective_to"] < out["effective_from"]).any():
        raise ValueError("membership intervals contain effective_to before effective_from")
    return out.sort_values(["effective_from", "ts_code"]).reset_index(drop=True)


def membership_mask(index: pd.MultiIndex, membership: pd.DataFrame) -> pd.Series:
    """Build a boolean mask for a (date, symbol) MultiIndex."""
    if index.nlevels != 2:
        raise ValueError("membership mask index must have two levels")
    work_index = index.set_names(["date", "symbol"])
    dates = pd.DatetimeIndex(work_index.get_level_values("date")).normalize()
    symbols = pd.Index(work_index.get_level_values("symbol")).map(normalize_ts_code)
    allowed = pd.Series(False, index=work_index, dtype=bool)
    for rec in membership.to_dict("records"):
        active = (
            (symbols == normalize_ts_code(rec["ts_code"]))
            & (dates >= pd.Timestamp(rec["effective_from"]).normalize())
            & (dates <= pd.Timestamp(rec["effective_to"]).normalize())
        )
        if active.any():
            allowed.iloc[active] = True
    return allowed


def mask_factor_panel_by_membership(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Set factor values to NaN outside the contemporaneous index universe."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("factor panel index must be a two-level (date, symbol) MultiIndex")
    out = panel.copy()
    out.index = out.index.set_names(["date", "symbol"])
    return out.where(membership_mask(out.index, membership), other=float("nan"))


def mask_wide_prices_by_membership(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Mask a wide price frame for point-in-time benchmark construction only."""
    out = prices.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    for symbol in out.columns:
        intervals = membership[membership["ts_code"] == normalize_ts_code(symbol)]
        active = pd.Series(False, index=out.index)
        for rec in intervals.to_dict("records"):
            active |= (out.index >= pd.Timestamp(rec["effective_from"])) & (
                out.index <= pd.Timestamp(rec["effective_to"])
            )
        out.loc[~active.to_numpy(), symbol] = float("nan")
    return out


def build_membership_intervals(
    current_symbols: Iterable[str],
    changes: pd.DataFrame,
    *,
    start: Any,
    end: Any,
    as_of: Any | None = None,
    expected_size: int | None = None,
) -> pd.DataFrame:
    """Reconstruct inclusive membership intervals backwards from a known current snapshot."""
    start_dt = pd.Timestamp(start).normalize()
    end_dt = pd.Timestamp(end).normalize()
    as_of_dt = pd.Timestamp(as_of if as_of is not None else end).normalize()
    if end_dt < start_dt:
        raise ValueError("end must be on or after start")
    active = {normalize_ts_code(symbol) for symbol in current_symbols}
    active.discard("")
    target_size = int(expected_size) if expected_size is not None else len(active)
    if len(active) != target_size:
        raise ValueError("current universe size=%d expected=%d" % (len(active), target_size))

    work = changes.copy()
    work["effective_date"] = pd.to_datetime(work["effective_date"], errors="raise").dt.normalize()
    work["action"] = work["action"].astype(str).str.upper()
    work["ts_code"] = work["ts_code"].map(normalize_ts_code)
    work = work[work["effective_date"] <= as_of_dt]

    rows: list[dict[str, Any]] = []
    upper = end_dt
    effective_dates = sorted(work["effective_date"].unique(), reverse=True)
    for raw_date in effective_dates:
        effective = pd.Timestamp(raw_date).normalize()
        interval_start = max(start_dt, effective)
        if interval_start <= upper:
            rows.extend(
                {
                    "ts_code": symbol,
                    "effective_from": interval_start,
                    "effective_to": upper,
                }
                for symbol in sorted(active)
            )
        day_changes = work[work["effective_date"] == effective]
        entrants = set(day_changes.loc[day_changes["action"] == "ENTER", "ts_code"])
        exits = set(day_changes.loc[day_changes["action"] == "EXIT", "ts_code"])
        missing_entrants = entrants - active
        if missing_entrants:
            raise ValueError(
                "cannot reverse %s; entrants absent from later snapshot: %s"
                % (effective.date(), sorted(missing_entrants))
            )
        active.difference_update(entrants)
        active.update(exits)
        if len(active) != target_size:
            raise ValueError(
                "universe size after reversing %s is %d, expected %d"
                % (effective.date(), len(active), target_size)
            )
        upper = min(upper, effective - pd.Timedelta(days=1))
        if upper < start_dt:
            break

    if upper >= start_dt:
        rows.extend(
            {
                "ts_code": symbol,
                "effective_from": start_dt,
                "effective_to": upper,
            }
            for symbol in sorted(active)
        )
    return pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).sort_values(
        ["effective_from", "ts_code"]
    ).reset_index(drop=True)


def filter_prices_by_membership(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
) -> pd.DataFrame:
    """Keep only price rows whose symbol was an index member on that date."""
    required_prices = {date_col, symbol_col}
    required_membership = set(MEMBERSHIP_COLUMNS)
    if missing := required_prices - set(prices.columns):
        raise ValueError("prices missing columns: %s" % ", ".join(sorted(missing)))
    if missing := required_membership - set(membership.columns):
        raise ValueError("membership missing columns: %s" % ", ".join(sorted(missing)))

    left = prices.copy()
    left[date_col] = pd.to_datetime(left[date_col], errors="raise").dt.normalize()
    left[symbol_col] = left[symbol_col].map(normalize_ts_code)
    right = membership.copy()
    right["effective_from"] = pd.to_datetime(right["effective_from"], errors="raise").dt.normalize()
    right["effective_to"] = pd.to_datetime(right["effective_to"], errors="raise").dt.normalize()
    merged = left.merge(right, left_on=symbol_col, right_on="ts_code", how="inner", suffixes=("", "_member"))
    active = (merged[date_col] >= merged["effective_from"]) & (
        merged[date_col] <= merged["effective_to"]
    )
    out = merged.loc[active, prices.columns].copy()
    if out.duplicated([date_col, symbol_col]).any():
        raise ValueError("membership filtering produced duplicate trade_date/ts_code rows")
    return out.sort_values([symbol_col, date_col]).reset_index(drop=True)
