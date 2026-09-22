"""
因子相关性与冗余分析。

本模块用每日横截面相关性衡量因子之间是否表达了相近信号。它不判断
某个因子是否有效，而是回答：如果两个因子都进入候选池，它们是否过于重复。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REDUNDANCY_COLUMNS = [
    "factor_a",
    "factor_b",
    "correlation",
    "abs_correlation",
    "n_days",
    "decision_a",
    "decision_b",
    "score_a",
    "score_b",
    "recommended_keep",
    "recommended_drop",
    "reason",
]


def _ensure_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为二级 MultiIndex(date, symbol)")
    out = panel.copy()
    out.index = out.index.set_names(["date", "symbol"])
    return out.sort_index()


def factor_cross_sectional_correlation(
    panel: pd.DataFrame,
    *,
    factors: list[str] | None = None,
    method: str = "spearman",
    min_symbols: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算因子每日横截面相关性的时间均值。

    返回：
    - corr_mean：每对因子日横截面相关性的均值；
    - corr_days：每对因子有多少个交易日满足有效样本数要求。
    """
    p = _ensure_panel(panel)
    factor_list = [str(x) for x in (factors or list(p.columns)) if str(x) in p.columns]
    if not factor_list:
        return pd.DataFrame(), pd.DataFrame()

    # pandas computes the same pairwise daily correlation matrix in compiled
    # groupby paths.  Aggregating that panel removes the Python day × factor²
    # loop, which is material when the gate is reevaluated every week.
    daily = (
        p[factor_list]
        .groupby(level="date", sort=True)
        .corr(method=method, min_periods=int(min_symbols))
        .replace([np.inf, -np.inf], np.nan)
    )
    if daily.empty:
        empty_float = pd.DataFrame(np.nan, index=factor_list, columns=factor_list)
        empty_int = pd.DataFrame(0, index=factor_list, columns=factor_list, dtype=int)
        return empty_float, empty_int
    corr_mean = daily.groupby(level=1, sort=False).mean().reindex(
        index=factor_list, columns=factor_list
    )
    counts = daily.groupby(level=1, sort=False).count().reindex(
        index=factor_list, columns=factor_list, fill_value=0
    ).astype(int)
    for f in factor_list:
        if counts.loc[f, f] > 0:
            corr_mean.loc[f, f] = 1.0
    return corr_mean, counts


def _selection_lookup(selection: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if selection is None or selection.empty or "factor" not in selection.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for rec in selection.to_dict("records"):
        out[str(rec.get("factor"))] = rec
    return out


def _decision_rank(decision: str) -> int:
    return {"PASS": 0, "WATCH": 1, "REJECT": 2}.get(str(decision).upper(), 9)


def _safe_score(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _choose_keep_drop(
    factor_a: str,
    factor_b: str,
    selection_map: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    rec_a = selection_map.get(factor_a, {})
    rec_b = selection_map.get(factor_b, {})
    dec_a = str(rec_a.get("decision", "")).upper()
    dec_b = str(rec_b.get("decision", "")).upper()
    rank_a = _decision_rank(dec_a)
    rank_b = _decision_rank(dec_b)
    if rank_a < rank_b:
        return factor_a, factor_b, "better_selection_decision"
    if rank_b < rank_a:
        return factor_b, factor_a, "better_selection_decision"

    score_a = _safe_score(rec_a.get("factor_score"))
    score_b = _safe_score(rec_b.get("factor_score"))
    if np.isfinite(score_a) and np.isfinite(score_b) and not np.isclose(score_a, score_b):
        if score_a > score_b:
            return factor_a, factor_b, "higher_factor_score"
        return factor_b, factor_a, "higher_factor_score"
    if np.isfinite(score_a) and not np.isfinite(score_b):
        return factor_a, factor_b, "higher_factor_score"
    if np.isfinite(score_b) and not np.isfinite(score_a):
        return factor_b, factor_a, "higher_factor_score"
    return factor_a, factor_b, "stable_factor_order"


def build_factor_redundancy_report(
    corr_mean: pd.DataFrame,
    corr_days: pd.DataFrame,
    *,
    selection: pd.DataFrame | None = None,
    threshold: float = 0.7,
    min_days: int = 20,
) -> pd.DataFrame:
    """找出绝对相关性超过阈值的因子对，并给出建议保留 / 降级对象。"""
    if corr_mean is None or corr_mean.empty:
        return pd.DataFrame(columns=REDUNDANCY_COLUMNS)
    factors = list(corr_mean.index.astype(str))
    selection_map = _selection_lookup(selection)
    rows: list[dict[str, Any]] = []
    for i, factor_a in enumerate(factors):
        for factor_b in factors[i + 1 :]:
            corr = corr_mean.loc[factor_a, factor_b] if factor_b in corr_mean.columns else np.nan
            days = corr_days.loc[factor_a, factor_b] if not corr_days.empty and factor_b in corr_days.columns else 0
            if pd.isna(corr) or int(days) < int(min_days):
                continue
            abs_corr = abs(float(corr))
            if abs_corr < float(threshold):
                continue
            keep, drop, reason = _choose_keep_drop(factor_a, factor_b, selection_map)
            rec_a = selection_map.get(factor_a, {})
            rec_b = selection_map.get(factor_b, {})
            rows.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "correlation": float(corr),
                    "abs_correlation": abs_corr,
                    "n_days": int(days),
                    "decision_a": rec_a.get("decision", ""),
                    "decision_b": rec_b.get("decision", ""),
                    "score_a": rec_a.get("factor_score", np.nan),
                    "score_b": rec_b.get("factor_score", np.nan),
                    "recommended_keep": keep,
                    "recommended_drop": drop,
                    "reason": reason,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=REDUNDANCY_COLUMNS)
    out = out.sort_values(["abs_correlation", "factor_a", "factor_b"], ascending=[False, True, True])
    return out[REDUNDANCY_COLUMNS].reset_index(drop=True)


def prune_redundant_factors(
    factors: list[str],
    redundancy_report: pd.DataFrame,
) -> list[str]:
    """从候选因子池中剔除被高相关因子对建议降级的因子。"""
    current = [str(x) for x in factors]
    if redundancy_report is None or redundancy_report.empty:
        return current
    current_set = set(current)
    drop_set: set[str] = set()
    for rec in redundancy_report.to_dict("records"):
        a = str(rec.get("factor_a", ""))
        b = str(rec.get("factor_b", ""))
        drop = str(rec.get("recommended_drop", ""))
        keep = str(rec.get("recommended_keep", ""))
        if a in current_set and b in current_set and keep in current_set and drop in current_set:
            drop_set.add(drop)
    pruned = [f for f in current if f not in drop_set]
    return pruned or current
