"""Point-in-time hard-risk overlays for an already tradable universe.

This layer deliberately blocks only explicit balance-sheet, audit and
delisting risks.  Valuation, profitability, growth and size preferences stay
in the alpha/strategy layers so the universe does not silently pre-select the
same characteristics later used by the factor model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SEVERE_AUDIT_PATTERN = r"否定意见|无法表示意见|拒绝表示意见"


def _dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(raw, format="mixed", errors="coerce")
    return compact.fillna(fallback).dt.normalize()


def _latest_asof(
    base: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_date: str,
    columns: list[str],
) -> pd.DataFrame:
    left = base[["date", "symbol"]].copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    left["symbol"] = left["symbol"].astype(str)
    left["_row"] = np.arange(len(left))
    right = source[["symbol", source_date, *columns]].copy()
    right["symbol"] = right["symbol"].astype(str)
    right[source_date] = (
        pd.to_datetime(right[source_date], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    right = right.dropna(subset=[source_date]).sort_values([source_date, "symbol"])
    merged = pd.merge_asof(
        left.sort_values(["date", "symbol"]),
        right,
        left_on="date",
        right_on=source_date,
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def normalize_balance_sheet(balance_sheet: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["symbol", "balance_effective_date", "balance_end_date", "net_assets"]
    if balance_sheet is None or balance_sheet.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "ann_date", "end_date", "total_hldr_eqy_exc_min_int"}
    if missing := required - set(balance_sheet.columns):
        raise ValueError("balance_sheet missing columns: %s" % sorted(missing))
    work = balance_sheet.copy()
    work["symbol"] = work["ts_code"].astype(str).str.strip()
    work = work[work["symbol"].str.endswith((".SH", ".SZ"))].copy()
    if "report_type" in work.columns:
        report_type = work["report_type"].astype("string").str.replace(r"\.0$", "", regex=True)
        primary = work[report_type.eq("1")]
        if not primary.empty:
            work = primary.copy()
    work["ann_date"] = _dates(work["ann_date"])
    if "f_ann_date" in work.columns:
        work["f_ann_date"] = _dates(work["f_ann_date"])
    else:
        work["f_ann_date"] = pd.NaT
    work["balance_effective_date"] = work[["ann_date", "f_ann_date"]].max(axis=1)
    work["balance_end_date"] = _dates(work["end_date"])
    work["net_assets"] = pd.to_numeric(
        work["total_hldr_eqy_exc_min_int"], errors="coerce"
    )
    update = (
        pd.to_numeric(work["update_flag"], errors="coerce").fillna(0)
        if "update_flag" in work.columns
        else pd.Series(0, index=work.index)
    )
    work = work.assign(_update=update).dropna(subset=["balance_effective_date"])
    work = work.sort_values(
        ["symbol", "balance_effective_date", "balance_end_date", "_update"]
    ).drop_duplicates(["symbol", "balance_effective_date"], keep="last")
    return work[columns].sort_values(["balance_effective_date", "symbol"]).reset_index(drop=True)


def normalize_audit_opinions(audit: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["symbol", "audit_effective_date", "audit_end_date", "audit_result"]
    if audit is None or audit.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "ann_date", "end_date", "audit_result"}
    if missing := required - set(audit.columns):
        raise ValueError("audit missing columns: %s" % sorted(missing))
    work = audit.copy()
    work["symbol"] = work["ts_code"].astype(str).str.strip()
    work = work[work["symbol"].str.endswith((".SH", ".SZ"))].copy()
    work["audit_effective_date"] = _dates(work["ann_date"])
    work["audit_end_date"] = _dates(work["end_date"])
    work["audit_result"] = work["audit_result"].fillna("").astype(str).str.strip()
    work = work.dropna(subset=["audit_effective_date"]).sort_values(
        ["symbol", "audit_effective_date", "audit_end_date"]
    )
    work = work.drop_duplicates(["symbol", "audit_effective_date"], keep="last")
    return work[columns].sort_values(["audit_effective_date", "symbol"]).reset_index(drop=True)


def _delisting_intervals(name_history: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["symbol", "delisting_from", "delisting_to", "delisting_name"]
    if name_history is None or name_history.empty:
        return pd.DataFrame(columns=columns)
    required = {"ts_code", "name", "start_date"}
    if missing := required - set(name_history.columns):
        raise ValueError("name_history missing columns: %s" % sorted(missing))
    work = name_history.copy()
    work["name"] = work["name"].fillna("").astype(str).str.strip()
    work = work[
        work["name"].str.contains("退市", regex=False)
        | work["name"].str.startswith("退")
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["start_date"] = _dates(work["start_date"])
    work["ann_date"] = _dates(work["ann_date"]) if "ann_date" in work.columns else pd.NaT
    work["delisting_from"] = work[["start_date", "ann_date"]].max(axis=1)
    work["delisting_to"] = (
        _dates(work["end_date"]) if "end_date" in work.columns else pd.NaT
    )
    work["symbol"] = work["ts_code"].astype(str).str.strip()
    work["delisting_name"] = work["name"]
    return work.dropna(subset=["delisting_from"])[columns].sort_values(
        ["delisting_from", "symbol"]
    )


def build_hard_risk_universe_report(
    base_report: pd.DataFrame,
    balance_sheet: pd.DataFrame | None,
    audit: pd.DataFrame | None,
    name_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Overlay explicit hard risks on a point-in-time tradable-universe report.

    Missing balance-sheet or audit data is recorded but does not automatically
    exclude a security.  This avoids turning provider coverage into a hidden
    size/listing-age filter.
    """
    required = {"date", "symbol", "eligible", "exclude_reason"}
    if missing := required - set(base_report.columns):
        raise ValueError("base_report missing columns: %s" % sorted(missing))
    if base_report.duplicated(["date", "symbol"]).any():
        raise ValueError("base_report contains duplicate date/symbol rows")

    out = base_report.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    balance = normalize_balance_sheet(balance_sheet)
    opinions = normalize_audit_opinions(audit)

    balance_asof = _latest_asof(
        out, balance, source_date="balance_effective_date", columns=["balance_end_date", "net_assets"]
    )
    audit_asof = _latest_asof(
        out, opinions, source_date="audit_effective_date", columns=["audit_end_date", "audit_result"]
    )
    for column in ("balance_effective_date", "balance_end_date", "net_assets"):
        out[column] = balance_asof[column].to_numpy()
    for column in ("audit_effective_date", "audit_end_date", "audit_result"):
        out[column] = audit_asof[column].to_numpy()

    out["balance_available"] = out["balance_effective_date"].notna()
    out["audit_available"] = out["audit_effective_date"].notna()
    out["negative_net_assets"] = pd.to_numeric(out["net_assets"], errors="coerce").le(0.0)
    audit_text = out["audit_result"].fillna("").astype(str)
    out["severe_audit_opinion"] = audit_text.str.contains(
        SEVERE_AUDIT_PATTERN, regex=True, na=False
    )
    out["qualified_audit_warning"] = (
        audit_text.str.contains("保留意见", regex=False, na=False)
        & ~audit_text.str.contains("无保留意见", regex=False, na=False)
        & ~out["severe_audit_opinion"]
    )

    intervals = _delisting_intervals(name_history)
    out["delisting_arrangement"] = False
    out["delisting_name"] = ""
    for date, positions in out.groupby("date", sort=False).groups.items():
        active = intervals[
            intervals["delisting_from"].le(date)
            & (intervals["delisting_to"].isna() | intervals["delisting_to"].ge(date))
        ]
        if active.empty:
            continue
        lookup = active.drop_duplicates("symbol", keep="last").set_index("symbol")["delisting_name"]
        symbols = out.loc[positions, "symbol"]
        names = symbols.map(lookup)
        mask = names.notna()
        out.loc[np.asarray(list(positions))[mask.to_numpy()], "delisting_arrangement"] = True
        out.loc[np.asarray(list(positions))[mask.to_numpy()], "delisting_name"] = names[mask].astype(str).to_numpy()

    hard_block = (
        out["negative_net_assets"]
        | out["severe_audit_opinion"]
        | out["delisting_arrangement"]
    )
    reasons = []
    for rec in out.to_dict("records"):
        current = [x for x in str(rec.get("exclude_reason", "") or "").split(";") if x]
        if bool(rec["negative_net_assets"]):
            current.append("negative_net_assets")
        if bool(rec["severe_audit_opinion"]):
            current.append("severe_audit_opinion")
        if bool(rec["delisting_arrangement"]):
            current.append("delisting_arrangement")
        reasons.append(";".join(dict.fromkeys(current)))
    out["exclude_reason"] = reasons
    out["base_eligible"] = out["eligible"].astype(bool)
    out["hard_risk_block"] = hard_block.astype(bool)
    out["eligible"] = out["base_eligible"] & ~out["hard_risk_block"]

    if bool(
        (
            out["balance_effective_date"].notna()
            & out["balance_effective_date"].gt(out["date"])
        ).any()
    ):
        raise RuntimeError("balance-sheet point-in-time violation")
    if bool(
        (
            out["audit_effective_date"].notna()
            & out["audit_effective_date"].gt(out["date"])
        ).any()
    ):
        raise RuntimeError("audit point-in-time violation")
    return out
