"""Reusable point-in-time strategy universes derived from the base universe.

The base universe decides whether a security is safe and tradable.  Strategy
universes are narrower, explicitly named mandates used by a strategy: size,
value-data, growth-data, or an industry/theme scope.  They never bypass the
base universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyUniverseProfile:
    """Configuration for one derived strategy universe."""

    name: str
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clean = str(self.name).strip().upper()
        if not clean:
            raise ValueError("strategy-universe profile name cannot be empty")
        if not clean.replace("_", "").isalnum():
            raise ValueError("profile name must contain only letters, numbers and underscores")
        supported = {"size_large", "size_mid_small", "value", "growth", "industry_theme"}
        if self.kind not in supported:
            raise ValueError("unsupported strategy-universe kind: %s" % self.kind)
        object.__setattr__(self, "name", clean)

    @property
    def eligibility_column(self) -> str:
        return "eligible_" + self.name

    @property
    def reason_column(self) -> str:
        return "reason_" + self.name


def profiles_from_config(values: Iterable[Mapping[str, Any]]) -> list[StrategyUniverseProfile]:
    profiles = [
        StrategyUniverseProfile(
            name=str(value["name"]),
            kind=str(value["kind"]),
            parameters=dict(value.get("parameters", {})),
        )
        for value in values
    ]
    names = [profile.name for profile in profiles]
    if len(set(names)) != len(names):
        raise ValueError("strategy-universe profile names must be unique")
    return profiles


def _dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, format="mixed", errors="coerce")
    return compact.fillna(fallback).dt.normalize()


def normalize_daily_basic(daily_basic: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize Tushare ``daily_basic`` fields used by strategy universes."""
    columns = [
        "symbol",
        "daily_basic_date",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv_yuan",
        "circ_mv_yuan",
    ]
    if daily_basic is None or daily_basic.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "trade_date", "circ_mv"}
    if missing := required - set(daily_basic.columns):
        raise ValueError("daily_basic missing columns: %s" % sorted(missing))
    work = daily_basic.copy()
    work["symbol"] = work["ts_code"].astype(str).str.strip()
    work = work[work["symbol"].str.endswith((".SH", ".SZ"))].copy()
    work["daily_basic_date"] = _dates(work["trade_date"])
    for column in ("pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"):
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    # Tushare daily_basic reports market value in CNY 10,000.
    work["total_mv_yuan"] = work["total_mv"] * 10_000.0
    work["circ_mv_yuan"] = work["circ_mv"] * 10_000.0
    work = work.dropna(subset=["daily_basic_date"])
    work = work.sort_values(["daily_basic_date", "symbol"]).drop_duplicates(
        ["daily_basic_date", "symbol"], keep="last"
    )
    return work[columns].reset_index(drop=True)


def normalize_industry_membership(membership: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize inclusive SW industry membership intervals."""
    columns = [
        "symbol",
        "industry_from",
        "industry_to",
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
    ]
    if membership is None or membership.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "in_date", "l1_name"}
    if missing := required - set(membership.columns):
        raise ValueError("industry membership missing columns: %s" % sorted(missing))
    work = membership.copy()
    work["symbol"] = work["ts_code"].astype(str).str.strip()
    work = work[work["symbol"].str.endswith((".SH", ".SZ"))].copy()
    work["industry_from"] = _dates(work["in_date"])
    work["industry_to"] = (
        _dates(work["out_date"])
        if "out_date" in work.columns
        else pd.Series(pd.NaT, index=work.index)
    )
    for column in ("l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name"):
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].fillna("").astype(str).str.strip()
    work = work.dropna(subset=["industry_from"])
    work = work.sort_values(["industry_from", "symbol", "l1_code"]).drop_duplicates(
        ["symbol", "industry_from", "l1_code"], keep="last"
    )
    return work[columns].reset_index(drop=True)


def point_in_time_industry(
    index: pd.MultiIndex,
    membership: pd.DataFrame | None,
    *,
    level: str = "l1_name",
) -> pd.Series:
    """Map each ``(date, symbol)`` row to its active historical industry."""
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise TypeError("index must be a two-level (date, symbol) MultiIndex")
    normalized = normalize_industry_membership(membership)
    if level not in normalized.columns:
        raise ValueError("unsupported industry level: %s" % level)
    work_index = index.set_names(["date", "symbol"])
    left = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(work_index.get_level_values("date")).normalize(),
            "symbol": work_index.get_level_values("symbol").astype(str),
        }
    )
    result = pd.Series("", index=np.arange(len(left)), dtype="object")
    if normalized.empty:
        return pd.Series(result.to_numpy(), index=work_index, name="industry")
    for date, positions in left.groupby("date", sort=False).groups.items():
        active = normalized[
            normalized["industry_from"].le(date)
            & (normalized["industry_to"].isna() | normalized["industry_to"].ge(date))
        ]
        if active.empty:
            continue
        lookup = (
            active.sort_values(["industry_from", "l1_code"])
            .drop_duplicates("symbol", keep="last")
            .set_index("symbol")[level]
        )
        loc = np.asarray(list(positions), dtype=int)
        result.iloc[loc] = left.iloc[loc]["symbol"].map(lookup).fillna("").to_numpy()
    return pd.Series(result.to_numpy(), index=work_index, name="industry")


def daily_basic_for_index(
    index: pd.MultiIndex,
    daily_basic: pd.DataFrame | None,
    *,
    max_staleness_days: int = 7,
) -> pd.DataFrame:
    """Attach only daily-basic rows known on or before each decision date."""
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise TypeError("index must be a two-level (date, symbol) MultiIndex")
    if max_staleness_days < 0:
        raise ValueError("max_staleness_days cannot be negative")
    work_index = index.set_names(["date", "symbol"])
    left = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(work_index.get_level_values("date"))
            .normalize()
            .astype("datetime64[ns]"),
            "symbol": work_index.get_level_values("symbol").astype(str),
            "_row": np.arange(len(work_index)),
        }
    )
    right = normalize_daily_basic(daily_basic)
    if right.empty:
        out = left.sort_values("_row").drop(columns="_row")
        for column in (
            "daily_basic_date",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "total_mv_yuan",
            "circ_mv_yuan",
        ):
            out[column] = pd.NaT if column == "daily_basic_date" else np.nan
    else:
        right = right.copy()
        right["daily_basic_date"] = right["daily_basic_date"].astype("datetime64[ns]")
        out = pd.merge_asof(
            left.sort_values(["date", "symbol"]),
            right.sort_values(["daily_basic_date", "symbol"]),
            left_on="date",
            right_on="daily_basic_date",
            by="symbol",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row")
        stale = (out["date"] - out["daily_basic_date"]).dt.days.gt(int(max_staleness_days))
        fields = ["pe_ttm", "pb", "ps_ttm", "total_mv_yuan", "circ_mv_yuan"]
        out.loc[stale, fields] = np.nan
    if bool((out["daily_basic_date"].notna() & out["daily_basic_date"].gt(out["date"])).any()):
        raise RuntimeError("daily-basic point-in-time violation")
    out["daily_basic_fresh"] = (
        out["daily_basic_date"].notna()
        & (out["date"] - out["daily_basic_date"]).dt.days.le(int(max_staleness_days))
    )
    out.index = work_index
    return out.drop(columns=["_row", "date", "symbol"], errors="ignore")


def _reason(base: pd.Series, eligible: pd.Series, failure: str) -> pd.Series:
    reason = pd.Series("", index=base.index, dtype="object")
    reason.loc[~base] = "base_universe_ineligible"
    reason.loc[base & ~eligible] = failure
    return reason


def build_strategy_universe_report(
    base_report: pd.DataFrame,
    daily_basic: pd.DataFrame | None,
    factor_panel: pd.DataFrame,
    industry_membership: pd.DataFrame | None,
    profiles: Iterable[StrategyUniverseProfile],
    *,
    max_daily_basic_staleness_days: int = 7,
) -> pd.DataFrame:
    """Build a wide auditable report for every configured strategy universe."""
    required = {"date", "symbol", "eligible"}
    if missing := required - set(base_report.columns):
        raise ValueError("base_report missing columns: %s" % sorted(missing))
    if base_report.duplicated(["date", "symbol"]).any():
        raise ValueError("base_report contains duplicate date/symbol rows")
    profile_list = list(profiles)
    if not profile_list:
        raise ValueError("at least one strategy-universe profile is required")
    if len({profile.name for profile in profile_list}) != len(profile_list):
        raise ValueError("strategy-universe profile names must be unique")

    out = base_report.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    index = pd.MultiIndex.from_arrays(
        [out["date"].to_numpy(), out["symbol"].to_numpy()], names=["date", "symbol"]
    )
    basics = daily_basic_for_index(
        index, daily_basic, max_staleness_days=max_daily_basic_staleness_days
    )
    for column in basics.columns:
        out[column] = basics[column].to_numpy()
    industry = point_in_time_industry(index, industry_membership, level="l1_name")
    out["industry_l1"] = industry.to_numpy()

    factors = factor_panel.copy()
    factors.index = factors.index.set_names(["date", "symbol"])
    factors = factors.reindex(index)
    for column in ("FREE_CASH_FLOW_YIELD", "REVENUE_GROWTH", "PROFIT_GROWTH"):
        out[column] = pd.to_numeric(factors.get(column), errors="coerce").to_numpy()

    base = out["eligible"].astype(bool)
    market_cap_valid = (
        out["daily_basic_fresh"].astype(bool)
        & pd.to_numeric(out["circ_mv_yuan"], errors="coerce").gt(0)
    )
    out["market_cap_percentile"] = np.nan
    for _date, positions in out.groupby("date", sort=False).groups.items():
        loc = np.asarray(list(positions), dtype=int)
        valid = base.iloc[loc].to_numpy() & market_cap_valid.iloc[loc].to_numpy()
        if not valid.any():
            continue
        selected = loc[valid]
        values = pd.to_numeric(out.loc[selected, "circ_mv_yuan"], errors="coerce")
        out.loc[selected, "market_cap_percentile"] = values.rank(
            pct=True, method="average"
        ).to_numpy()

    pe_valid = pd.to_numeric(out["pe_ttm"], errors="coerce").gt(0)
    pb_valid = pd.to_numeric(out["pb"], errors="coerce").gt(0)
    cash_valid = np.isfinite(pd.to_numeric(out["FREE_CASH_FLOW_YIELD"], errors="coerce"))
    out["value_component_count"] = (
        pe_valid.astype(int) + pb_valid.astype(int) + pd.Series(cash_valid, index=out.index).astype(int)
    )
    revenue_valid = np.isfinite(pd.to_numeric(out["REVENUE_GROWTH"], errors="coerce"))
    profit_valid = np.isfinite(pd.to_numeric(out["PROFIT_GROWTH"], errors="coerce"))
    out["growth_component_count"] = (
        pd.Series(revenue_valid, index=out.index).astype(int)
        + pd.Series(profit_valid, index=out.index).astype(int)
    )

    for profile in profile_list:
        params = dict(profile.parameters)
        if profile.kind == "size_large":
            split = float(params.get("minimum_percentile", 0.70))
            eligible = base & market_cap_valid & out["market_cap_percentile"].ge(split)
            failure = "outside_large_cap_band_or_market_cap_unavailable"
        elif profile.kind == "size_mid_small":
            split = float(params.get("maximum_percentile", 0.70))
            eligible = base & market_cap_valid & out["market_cap_percentile"].lt(split)
            failure = "outside_mid_small_cap_band_or_market_cap_unavailable"
        elif profile.kind == "value":
            minimum = int(params.get("minimum_components", 2))
            eligible = base & out["value_component_count"].ge(minimum)
            failure = "insufficient_value_data"
        elif profile.kind == "growth":
            minimum = int(params.get("minimum_components", 2))
            eligible = base & out["growth_component_count"].ge(minimum)
            failure = "insufficient_growth_data"
        else:
            names = {str(value).strip() for value in params.get("l1_names", []) if str(value).strip()}
            if not names:
                raise ValueError("industry_theme profile %s has no l1_names" % profile.name)
            eligible = base & out["industry_l1"].isin(names)
            failure = "outside_configured_industry_theme"
        out[profile.eligibility_column] = eligible.astype(bool)
        out[profile.reason_column] = _reason(base, eligible, failure)

    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def eligibility_for_profile(
    report: pd.DataFrame,
    profile: StrategyUniverseProfile | str,
) -> pd.Series:
    """Return a ``(date, symbol)`` eligibility mask for one profile."""
    name = profile.name if isinstance(profile, StrategyUniverseProfile) else str(profile).strip().upper()
    column = "eligible_" + name
    required = {"date", "symbol", column}
    if missing := required - set(report.columns):
        raise ValueError("strategy report missing columns: %s" % sorted(missing))
    index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(report["date"], errors="raise").to_numpy(),
            report["symbol"].astype(str).to_numpy(),
        ],
        names=["date", "symbol"],
    )
    return pd.Series(report[column].astype(bool).to_numpy(), index=index, name=name).sort_index()
