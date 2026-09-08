"""
因子入选与剔除机制。

本模块不重新计算 IC / 分组收益 / 样本外表现，而是复用已有诊断表，
把每个候选因子打成 PASS / WATCH / REJECT，给主融合策略一个可审计的
候选池。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SELECTION_COLUMNS = [
    "factor",
    "decision",
    "selected_for_fusion",
    "reasons",
    "coverage",
    "valid_dates",
    "valid_symbols",
    "factor_score",
    "fusion_weight",
    "mean_ic",
    "ic_ir",
    "positive_rate",
    "top_minus_bottom_ann",
    "monotonicity_score",
    "monitor_status",
    "monitor_severity",
    "validation_ic_mean",
    "validation_positive_rate",
    "validation_excess_ann_return",
    "validation_top_minus_bottom_ann",
    "multi_horizon_status",
    "supportive_horizon_rate",
    "rolling_oos_status",
    "supportive_window_rate",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_status(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _frame_with_factor(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "factor" not in frame.columns:
        return pd.DataFrame(columns=["factor"])
    out = frame.copy()
    out["factor"] = out["factor"].astype(str)
    return out


def build_factor_selection_table(
    *,
    factors: list[str],
    factor_coverage: pd.DataFrame | None = None,
    factor_weight_summary: pd.DataFrame | None = None,
    factor_decay_monitor: pd.DataFrame | None = None,
    multi_horizon_summary: pd.DataFrame | None = None,
    rolling_out_of_sample_summary: pd.DataFrame | None = None,
    min_coverage: float = 0.5,
    min_valid_dates: int = 60,
    min_factor_score: float = 0.15,
    min_mean_ic: float = 0.0,
    min_positive_rate: float = 0.5,
    min_top_minus_bottom_ann: float = 0.0,
    min_monotonicity: float = 0.4,
) -> pd.DataFrame:
    """
    生成因子准入表。

    规则尽量保守：
    - 覆盖率 / 有效日期不足，直接 REJECT；
    - 样本外状态 FAILED / DEGRADED，直接 REJECT；
    - 因子分数、IC、正 IC 占比、Top-Bottom、单调性均达标，PASS；
    - 没有硬伤但指标不够强，WATCH。
    """
    factor_order = [str(x) for x in factors]
    base = pd.DataFrame({"factor": factor_order})

    cov = _frame_with_factor(factor_coverage)
    weight = _frame_with_factor(factor_weight_summary)
    monitor = _frame_with_factor(factor_decay_monitor).rename(
        columns={
            "status": "monitor_status",
            "severity": "monitor_severity",
        }
    )
    multi = _frame_with_factor(multi_horizon_summary).rename(
        columns={"status": "multi_horizon_status"}
    )
    rolling = _frame_with_factor(rolling_out_of_sample_summary).rename(
        columns={"status": "rolling_oos_status"}
    )

    keep_cov = ["factor", "coverage", "valid_dates", "valid_symbols"]
    keep_weight = [
        "factor",
        "factor_score",
        "fusion_weight",
        "mean_ic",
        "ic_ir",
        "positive_rate",
        "top_minus_bottom_ann",
        "monotonicity_score",
    ]
    keep_monitor = [
        "factor",
        "monitor_status",
        "monitor_severity",
        "validation_ic_mean",
        "validation_positive_rate",
        "validation_excess_ann_return",
        "validation_top_minus_bottom_ann",
    ]
    keep_multi = ["factor", "multi_horizon_status", "supportive_horizon_rate"]
    keep_rolling = ["factor", "rolling_oos_status", "supportive_window_rate"]
    for col in keep_cov:
        if col not in cov.columns:
            cov[col] = np.nan
    for col in keep_weight:
        if col not in weight.columns:
            weight[col] = np.nan
    for col in keep_monitor:
        if col not in monitor.columns:
            monitor[col] = np.nan
    for col in keep_multi:
        if col not in multi.columns:
            multi[col] = np.nan
    for col in keep_rolling:
        if col not in rolling.columns:
            rolling[col] = np.nan

    out = (
        base.merge(cov[keep_cov], on="factor", how="left")
        .merge(weight[keep_weight], on="factor", how="left")
        .merge(monitor[keep_monitor], on="factor", how="left")
        .merge(multi[keep_multi], on="factor", how="left")
        .merge(rolling[keep_rolling], on="factor", how="left")
    )

    rows: list[dict[str, Any]] = []
    for rec in out.to_dict("records"):
        reasons: list[str] = []
        coverage = _safe_float(rec.get("coverage"))
        valid_dates = _safe_float(rec.get("valid_dates"), default=0.0)
        score = _safe_float(rec.get("factor_score"))
        mean_ic = _safe_float(rec.get("mean_ic"))
        pos_rate = _safe_float(rec.get("positive_rate"))
        top_bottom = _safe_float(rec.get("top_minus_bottom_ann"))
        mono = _safe_float(rec.get("monotonicity_score"))
        monitor_status = _safe_status(rec.get("monitor_status"))
        multi_status = _safe_status(rec.get("multi_horizon_status"))
        rolling_status = _safe_status(rec.get("rolling_oos_status"))
        enhanced_validation = bool(multi_status or rolling_status)

        hard_reject = False
        if np.isfinite(coverage) and coverage < min_coverage:
            reasons.append("coverage_below_threshold")
            hard_reject = True
        if valid_dates < min_valid_dates:
            reasons.append("valid_dates_below_threshold")
            hard_reject = True
        if not np.isfinite(mean_ic):
            reasons.append("ic_unavailable")
            hard_reject = True
        if monitor_status in {"FAILED", "DEGRADED"}:
            reasons.append("fixed_split_%s" % monitor_status.lower())
            if not enhanced_validation:
                hard_reject = True
        if multi_status == "FAILED":
            reasons.append("multi_horizon_failed")
            hard_reject = True

        pass_checks: list[bool] = []
        if np.isfinite(score):
            ok = score >= min_factor_score
            pass_checks.append(ok)
            if not ok:
                reasons.append("factor_score_below_threshold")
        if np.isfinite(mean_ic):
            ok = mean_ic >= min_mean_ic
            pass_checks.append(ok)
            if not ok:
                reasons.append("mean_ic_below_threshold")
        if np.isfinite(pos_rate):
            ok = pos_rate >= min_positive_rate
            pass_checks.append(ok)
            if not ok:
                reasons.append("positive_rate_below_threshold")
        if np.isfinite(top_bottom):
            ok = top_bottom >= min_top_minus_bottom_ann
            pass_checks.append(ok)
            if not ok:
                reasons.append("top_bottom_below_threshold")
        if np.isfinite(mono):
            ok = mono >= min_monotonicity
            pass_checks.append(ok)
            if not ok:
                reasons.append("monotonicity_below_threshold")
        if multi_status:
            ok = multi_status in {"ROBUST", "SUPPORTIVE"}
            pass_checks.append(ok)
            if not ok:
                reasons.append("multi_horizon_not_supportive")
        if rolling_status:
            ok = rolling_status in {"ROBUST", "WATCH"}
            pass_checks.append(ok)
            if not ok:
                reasons.append("rolling_oos_unstable")

        if hard_reject:
            decision = "REJECT"
        elif pass_checks and all(pass_checks):
            decision = "PASS"
        else:
            decision = "WATCH"

        rec["decision"] = decision
        rec["selected_for_fusion"] = decision == "PASS"
        rec["reasons"] = ";".join(dict.fromkeys(reasons))
        rows.append(rec)

    selected = pd.DataFrame(rows)
    for col in SELECTION_COLUMNS:
        if col not in selected.columns:
            selected[col] = np.nan
    selected["_order"] = selected["factor"].map({name: i for i, name in enumerate(factor_order)})
    selected["_decision_order"] = selected["decision"].map({"PASS": 0, "WATCH": 1, "REJECT": 2}).fillna(9)
    selected = selected.sort_values(["_decision_order", "_order"], ascending=[True, True])
    selected = selected.drop(columns=["_order", "_decision_order"]).reset_index(drop=True)
    return selected[SELECTION_COLUMNS]


def selected_factors_for_fusion(selection: pd.DataFrame, fallback_factors: list[str]) -> list[str]:
    """
    取主融合候选池。

    优先 PASS；如果没有 PASS，则使用 WATCH；如果仍为空，回退到原始因子池，避免主流程中断。
    """
    if selection is None or selection.empty or "decision" not in selection.columns:
        return [str(x) for x in fallback_factors]
    passed = selection.loc[selection["decision"].astype(str) == "PASS", "factor"].astype(str).tolist()
    if passed:
        return passed
    watched = selection.loc[selection["decision"].astype(str) == "WATCH", "factor"].astype(str).tolist()
    if watched:
        return watched
    return [str(x) for x in fallback_factors]
