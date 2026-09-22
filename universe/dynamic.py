"""Build an auditable point-in-time A-share tradable universe.

The universe layer only answers whether a security is eligible to receive a new
position.  It deliberately avoids profitability, valuation, growth and market-
capitalisation filters because those belong to the alpha/risk layers.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


REPORT_COLUMNS = [
    "date",
    "symbol",
    "name",
    "industry",
    "exchange",
    "market",
    "eligible",
    "exclude_reason",
    "listed_sessions",
    "valid_days_20",
    "adv20_yuan",
    "is_st",
    "listed",
    "not_delisted",
    "price_available",
    "listing_age_pass",
    "trading_days_pass",
    "liquidity_pass",
]


def _date_series(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, format="mixed", errors="coerce")
    return compact.fillna(fallback).dt.normalize()


def _stock_master(stock_basic: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "list_date"}
    missing = required - set(stock_basic.columns)
    if missing:
        raise ValueError("stock_basic missing columns: %s" % sorted(missing))
    work = stock_basic.copy()
    work["ts_code"] = work["ts_code"].astype(str).str.strip()
    work = work[work["ts_code"].str.endswith((".SH", ".SZ"))].copy()
    work["list_date"] = _date_series(work["list_date"])
    if "delist_date" not in work.columns:
        work["delist_date"] = pd.NaT
    else:
        work["delist_date"] = _date_series(work["delist_date"])
    for column in ("name", "industry", "exchange", "market"):
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].fillna("").astype(str)
    status_rank = work.get("list_status", pd.Series("", index=work.index)).map(
        {"L": 0, "P": 1, "D": 2}
    ).fillna(9)
    work = work.assign(_status_rank=status_rank).sort_values(
        ["ts_code", "_status_rank", "list_date"]
    )
    work = work.drop_duplicates("ts_code", keep="first").drop(columns="_status_rank")
    return work.set_index("ts_code", drop=False).sort_index()


def _st_intervals(name_history: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["symbol", "effective_from", "effective_to"]
    if name_history is None or name_history.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "name", "start_date"}
    if missing := required - set(name_history.columns):
        raise ValueError("name_history missing columns: %s" % sorted(missing))
    work = name_history.copy()
    work["name"] = work["name"].fillna("").astype(str)
    work = work[work["name"].str.contains("ST", case=False, regex=False)].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["start_date"] = _date_series(work["start_date"])
    work["end_date"] = (
        _date_series(work["end_date"])
        if "end_date" in work.columns
        else pd.Series(pd.NaT, index=work.index)
    )
    work["ann_date"] = (
        _date_series(work["ann_date"])
        if "ann_date" in work.columns
        else pd.Series(pd.NaT, index=work.index)
    )
    # A state cannot be used before it was announced.  Usually ann_date is on
    # or before start_date; max() is the conservative point-in-time rule.
    work["effective_from"] = work[["start_date", "ann_date"]].max(axis=1)
    work["effective_to"] = work["end_date"]
    work["symbol"] = work["ts_code"].astype(str).str.strip()
    return work.dropna(subset=["effective_from"])[columns].sort_values(
        ["effective_from", "symbol"]
    )


def _open_dates(trade_calendar: pd.DataFrame | Iterable[Any]) -> pd.DatetimeIndex:
    if isinstance(trade_calendar, pd.DataFrame):
        date_col = "cal_date" if "cal_date" in trade_calendar.columns else "trade_date"
        if date_col not in trade_calendar.columns:
            raise ValueError("trade_calendar requires cal_date or trade_date")
        work = trade_calendar.copy()
        if "is_open" in work.columns:
            work = work[pd.to_numeric(work["is_open"], errors="coerce").eq(1)]
        values = _date_series(work[date_col])
    else:
        values = pd.to_datetime(list(trade_calendar), errors="coerce")
    return pd.DatetimeIndex(values.dropna().unique()).normalize().sort_values()


def build_dynamic_universe_report(
    prices: pd.DataFrame,
    stock_basic: pd.DataFrame,
    name_history: pd.DataFrame | None,
    trade_calendar: pd.DataFrame | Iterable[Any],
    decision_dates: Iterable[Any],
    *,
    min_listing_sessions: int = 250,
    liquidity_window: int = 20,
    min_valid_days: int = 15,
    min_adv_yuan: float = 50_000_000.0,
    amount_col: str = "amount_yuan",
) -> pd.DataFrame:
    """Return one auditable eligibility row per decision date and security.

    ``prices`` must contain unadjusted close and RMB turnover.  A missing close
    on a decision date makes a security ineligible for a new position, but this
    report does not instruct the execution layer to liquidate an existing,
    suspended position.
    """
    required = {"trade_date", "ts_code", "close", amount_col}
    if missing := required - set(prices.columns):
        raise ValueError("prices missing columns: %s" % sorted(missing))
    if min_listing_sessions < 1:
        raise ValueError("min_listing_sessions must be positive")
    if liquidity_window < 1 or min_valid_days < 1 or min_valid_days > liquidity_window:
        raise ValueError("invalid liquidity window/min_valid_days")
    if min_adv_yuan < 0:
        raise ValueError("min_adv_yuan cannot be negative")

    master = _stock_master(stock_basic)
    calendar = _open_dates(trade_calendar)
    if calendar.empty:
        raise ValueError("trade calendar has no open dates")
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).normalize()
    dates = dates[dates.isin(calendar)].unique().sort_values()
    if dates.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    px = prices[["trade_date", "ts_code", "close", amount_col]].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], errors="coerce").dt.normalize()
    px["ts_code"] = px["ts_code"].astype(str).str.strip()
    px = px[
        px["trade_date"].isin(calendar)
        & px["ts_code"].isin(master.index)
    ].drop_duplicates(["trade_date", "ts_code"], keep="last")
    close = px.pivot(index="trade_date", columns="ts_code", values="close").reindex(
        index=calendar, columns=master.index
    )
    amount = px.pivot(index="trade_date", columns="ts_code", values=amount_col).reindex(
        index=calendar, columns=master.index
    )
    valid_days = close.notna().rolling(liquidity_window, min_periods=1).sum()
    adv = amount.where(close.notna()).rolling(liquidity_window, min_periods=1).mean()
    intervals = _st_intervals(name_history)

    list_positions = np.searchsorted(
        calendar.values,
        master["list_date"].values.astype("datetime64[ns]"),
        side="left",
    )
    rows: list[pd.DataFrame] = []
    for date in dates:
        date_position = int(np.searchsorted(calendar.values, np.datetime64(date), side="right"))
        listed_sessions = date_position - list_positions
        listed = master["list_date"].notna() & master["list_date"].le(date)
        not_delisted = master["delist_date"].isna() | master["delist_date"].gt(date)
        price_available = close.loc[date].notna()
        age_pass = pd.Series(listed_sessions >= int(min_listing_sessions), index=master.index)
        valid_now = valid_days.loc[date].fillna(0.0)
        trading_pass = valid_now.ge(int(min_valid_days))
        adv_now = adv.loc[date]
        liquidity_pass = adv_now.ge(float(min_adv_yuan))
        active_st = intervals[
            intervals["effective_from"].le(date)
            & (intervals["effective_to"].isna() | intervals["effective_to"].ge(date))
        ]
        st_symbols = set(active_st["symbol"].astype(str))
        is_st = pd.Series(master.index.isin(st_symbols), index=master.index)

        frame = master[["ts_code", "name", "industry", "exchange", "market"]].copy()
        frame = frame.rename(columns={"ts_code": "symbol"})
        frame["date"] = date
        frame["listed_sessions"] = listed_sessions.astype(int)
        frame["valid_days_20"] = valid_now.to_numpy(dtype=float)
        frame["adv20_yuan"] = adv_now.to_numpy(dtype=float)
        frame["is_st"] = is_st.to_numpy(dtype=bool)
        frame["listed"] = listed.to_numpy(dtype=bool)
        frame["not_delisted"] = not_delisted.to_numpy(dtype=bool)
        frame["price_available"] = price_available.to_numpy(dtype=bool)
        frame["listing_age_pass"] = age_pass.to_numpy(dtype=bool)
        frame["trading_days_pass"] = trading_pass.to_numpy(dtype=bool)
        frame["liquidity_pass"] = liquidity_pass.to_numpy(dtype=bool)
        frame["eligible"] = (
            listed
            & not_delisted
            & ~is_st
            & price_available
            & age_pass
            & trading_pass
            & liquidity_pass
        ).to_numpy(dtype=bool)

        reasons: list[str] = []
        checks = [
            (~listed, "not_yet_listed"),
            (~not_delisted, "delisted"),
            (is_st, "st_or_star_st"),
            (~price_available, "no_price_on_decision_date"),
            (~age_pass, "listing_age_below_min"),
            (~trading_pass, "valid_trading_days_below_min"),
            (~liquidity_pass, "adv20_below_min"),
        ]
        for symbol in master.index:
            reasons.append(
                ";".join(label for mask, label in checks if bool(mask.loc[symbol]))
            )
        frame["exclude_reason"] = reasons
        rows.append(frame.reset_index(drop=True)[REPORT_COLUMNS])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def eligibility_from_report(report: pd.DataFrame) -> pd.Series:
    """Convert an audit report into a ``(date, symbol)`` boolean mask."""
    required = {"date", "symbol", "eligible"}
    if missing := required - set(report.columns):
        raise ValueError("report missing columns: %s" % sorted(missing))
    if report.duplicated(["date", "symbol"]).any():
        raise ValueError("report contains duplicate date/symbol rows")
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(report["date"]).values, report["symbol"].astype(str).values],
        names=["date", "symbol"],
    )
    return pd.Series(report["eligible"].astype(bool).values, index=index, name="eligible").sort_index()
