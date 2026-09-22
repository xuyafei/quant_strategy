"""Point-in-time factor admission and dynamic fusion for walk-forward backtests."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.data_quality import factor_coverage
from analysis.factor_composite import build_factor_composite_scores
from analysis.factor_diagnostics import batch_factor_group_returns
from analysis.factor_redundancy import (
    build_factor_redundancy_report,
    factor_cross_sectional_correlation,
    prune_redundant_factors,
)
from analysis.factor_selection import build_factor_selection_table
from analysis.factor_validation import (
    build_factor_decay_monitor,
    build_multi_horizon_out_of_sample_validation,
    build_out_of_sample_validation,
    build_rolling_out_of_sample_validation,
    summarize_multi_horizon_validation,
    summarize_rolling_out_of_sample_validation,
)
from analysis.ic import daily_ic_spearman, ic_distribution_summary, ic_rolling_stability
from config import Settings
from factors.preprocess import cross_sectional_zscore
from models.factor_weighting import build_factor_weight_summary


WALK_FORWARD_SUMMARY_COLUMNS = [
    "date",
    "signal_status",
    "history_start",
    "history_end",
    "history_days",
    "rolling_windows",
    "pass_factors",
    "selected_factors",
    "style_factors",
    "n_pass",
    "n_selected",
    "n_styles",
    "reason",
]


_GATE_WORKER_CONTEXT: tuple[Any, ...] | None = None


def validate_walk_forward_audit(
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, int]:
    """Fail closed when a saved walk-forward audit is not point-in-time consistent."""
    if summary.empty:
        raise ValueError("walk-forward 调仓摘要为空")
    if summary["date"].duplicated().any():
        raise ValueError("walk-forward 调仓摘要存在重复日期")

    checks = {
        "rebalance_dates": int(len(summary)),
        "active_rebalances": int(summary["signal_status"].eq("ACTIVE").sum()),
        "decision_rows": int(len(decisions)),
        "weight_rows": int(len(weights)),
    }
    if not decisions.empty:
        if decisions.duplicated(["as_of_date", "factor"]).any():
            raise ValueError("walk-forward 因子审计存在重复的调仓日/因子")
        cutoff = pd.to_datetime(decisions["history_end"], errors="coerce")
        as_of = pd.to_datetime(decisions["as_of_date"], errors="coerce")
        if bool((cutoff.notna() & (cutoff >= as_of)).any()):
            raise ValueError("walk-forward 因子准入读取了调仓日或未来数据")
    if not weights.empty:
        cutoff = pd.to_datetime(weights["history_end"], errors="coerce")
        as_of = pd.to_datetime(weights["date"], errors="coerce")
        if bool((cutoff.notna() & (cutoff >= as_of)).any()):
            raise ValueError("walk-forward 因子权重读取了调仓日或未来数据")
        totals = weights.groupby("date")["weight"].sum()
        if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-9):
            raise ValueError("walk-forward 因子权重之和不为 1")
    return checks


def _resample_freq_alias(freq: str) -> str:
    return {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(str(freq), str(freq))


def walk_forward_rebalance_dates(prices: pd.DataFrame, settings: Settings) -> pd.DatetimeIndex:
    """Return the same rebalance calendar as the portfolio backtest."""
    if prices.empty:
        return pd.DatetimeIndex([])
    frame = prices.dropna(how="all").sort_index()
    dates = pd.DatetimeIndex(
        [
            group.index[-1]
            for _, group in frame.groupby(pd.Grouper(freq=_resample_freq_alias(settings.rebalance_freq)))
            if not group.empty
        ]
    )
    if bool(getattr(settings, "force_final_rebalance", False)) and len(frame.index):
        dates = dates.union(pd.DatetimeIndex([frame.index[-1]]))
    return dates.sort_values()


def _panel_on_dates(panel: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    mask = panel.index.get_level_values("date").isin(pd.Index(dates))
    return panel.loc[mask]


def _factor_weight_summary_for_history(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    ic_by_name: dict[str, pd.Series] = {}
    for factor in panel.columns:
        series = panel[factor]
        if series.notna().sum() == 0:
            continue
        try:
            ic_by_name[str(factor)] = daily_ic_spearman(
                series,
                prices,
                forward_days=int(settings.ic_forward_days),
            )
        except Exception:
            continue
    if not ic_by_name:
        return pd.DataFrame()
    distribution = ic_distribution_summary(ic_by_name)
    rolling = ic_rolling_stability(ic_by_name, windows=settings.ic_rolling_windows)
    _, groups = batch_factor_group_returns(
        panel,
        prices,
        factors=list(panel.columns),
        group_count=int(settings.factor_group_count),
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        trading_days_per_year=int(settings.trading_days_per_year),
    )
    windows = tuple(int(x) for x in settings.ic_rolling_windows)
    return build_factor_weight_summary(
        distribution,
        rolling,
        groups,
        factors=list(panel.columns),
        preferred_rolling_window=max(windows) if windows else None,
    )


def evaluate_factor_gate_asof(
    panel_history: pd.DataFrame,
    prices_history: pd.DataFrame,
    settings: Settings,
    *,
    factors: list[str],
    eligible_mask_history: pd.Series | None = None,
    benchmark_prices_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every admission input using only the supplied historical slice."""
    available = [str(x) for x in factors if str(x) in panel_history.columns]
    history = panel_history[available].dropna(axis=1, how="all")
    if history.empty or history.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    active_factors = list(history.columns)
    coverage = factor_coverage(history, eligible_mask=eligible_mask_history)
    weights = _factor_weight_summary_for_history(history, prices_history, settings)
    fixed = build_out_of_sample_validation(
        history,
        prices_history,
        settings,
        factors=active_factors,
        train_ratio=settings.factor_weight_train_ratio,
        benchmark_prices=benchmark_prices_history,
    )
    decay = build_factor_decay_monitor(fixed)
    rolling_detail = build_rolling_out_of_sample_validation(
        history,
        prices_history,
        settings,
        factors=active_factors,
        benchmark_prices=benchmark_prices_history,
    )
    rolling = summarize_rolling_out_of_sample_validation(rolling_detail)
    multi_detail = build_multi_horizon_out_of_sample_validation(
        history,
        prices_history,
        settings,
        factors=active_factors,
        horizons=tuple(settings.factor_validation_horizons),
        include_next_rebalance=bool(settings.factor_validation_include_next_rebalance),
        train_ratio=settings.factor_weight_train_ratio,
        benchmark_prices=benchmark_prices_history,
    )
    multi = summarize_multi_horizon_validation(multi_detail)
    selection = build_factor_selection_table(
        factors=available,
        factor_coverage=coverage,
        factor_weight_summary=weights,
        factor_decay_monitor=decay,
        multi_horizon_summary=multi,
        rolling_out_of_sample_summary=rolling,
    )

    correlation, correlation_days = factor_cross_sectional_correlation(
        history,
        factors=active_factors,
        method="spearman",
        min_symbols=max(5, int(settings.top_k)),
    )
    redundancy = build_factor_redundancy_report(
        correlation,
        correlation_days,
        selection=selection,
        threshold=0.70,
        min_days=max(20, int(settings.rolling_factor_weight_min_days // 2)),
    )
    return selection, rolling, multi, redundancy


def _evaluate_gate_for_date(
    raw_date: pd.Timestamp,
    panel: pd.DataFrame,
    price_frame: pd.DataFrame,
    settings: Settings,
    factors: list[str],
    eligible_mask: pd.Series | None,
    benchmark: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    dt = pd.Timestamp(raw_date)
    history_dates = pd.DatetimeIndex(price_frame.index[price_frame.index < dt])
    if lookback > 0:
        history_dates = history_dates[-lookback:]
    panel_history = _panel_on_dates(panel[factors], history_dates)
    mask_history = None
    if eligible_mask is not None:
        mask_history = eligible_mask.reindex(panel_history.index).fillna(False).astype(bool)
    benchmark_history = benchmark.loc[benchmark.index.isin(history_dates)]
    prices_history = price_frame.loc[price_frame.index.isin(history_dates)]
    try:
        selection, rolling, _multi, redundancy = evaluate_factor_gate_asof(
            panel_history,
            prices_history,
            settings,
            factors=factors,
            eligible_mask_history=mask_history,
            benchmark_prices_history=benchmark_history,
        )
        return selection, rolling, redundancy, ""
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"gate_failed:{type(exc).__name__}"


def _init_gate_worker(*context: Any) -> None:
    global _GATE_WORKER_CONTEXT
    _GATE_WORKER_CONTEXT = context


def _process_gate_worker(
    raw_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if _GATE_WORKER_CONTEXT is None:
        raise RuntimeError("walk-forward gate worker context is not initialized")
    return _evaluate_gate_for_date(raw_date, *_GATE_WORKER_CONTEXT)


def _gate_cache_path(
    cache_root: Path,
    raw_date: pd.Timestamp,
    *,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    factors: list[str],
) -> Path:
    payload = {
        "panel_shape": list(panel.shape),
        "panel_start": str(panel.index.get_level_values("date").min()),
        "panel_end": str(panel.index.get_level_values("date").max()),
        "price_shape": list(prices.shape),
        "price_start": str(prices.index.min()),
        "price_end": str(prices.index.max()),
        "factors": factors,
        "rebalance_freq": settings.rebalance_freq,
        "validation_horizons": list(settings.factor_validation_horizons),
        "include_next_rebalance": settings.factor_validation_include_next_rebalance,
        "train_ratio": settings.factor_weight_train_ratio,
        "rolling_windows": list(settings.ic_rolling_windows),
        "lookback": settings.walk_forward_history_lookback_days,
        "top_k": settings.top_k,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / digest / f"{pd.Timestamp(raw_date):%Y%m%d}.pkl"


def _clean_weights(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(clean.sum())
    if not len(clean):
        return clean
    if total <= 1e-18:
        return pd.Series(1.0 / len(clean), index=clean.index, dtype=float)
    return clean / total


def _constrain_weights(values: pd.Series, settings: Settings) -> pd.Series:
    weights = _clean_weights(values)
    n = len(weights)
    if n == 0:
        return weights
    floor = float(settings.rolling_factor_weight_min_weight)
    cap = float(settings.rolling_factor_weight_max_weight)
    if not np.isfinite(floor) or floor < 0 or floor * n >= 1:
        floor = 0.0
    if not np.isfinite(cap) or cap <= 0 or cap * n < 1:
        cap = 1.0
    if floor > 0:
        weights = pd.Series(floor, index=weights.index) + (1.0 - floor * n) * weights
        weights = weights / float(weights.sum())
    if cap >= 1:
        return weights
    result = weights.to_numpy(dtype=float).copy()
    for _ in range(n + 2):
        over = result > cap + 1e-12
        if not bool(over.any()):
            break
        excess = float((result[over] - cap).sum())
        result[over] = cap
        under = ~over
        room = np.maximum(cap - result[under], 0.0)
        if float(room.sum()) <= 1e-12:
            break
        result[under] += excess * room / float(room.sum())
    return _clean_weights(pd.Series(result, index=weights.index))


def _style_weights(
    style_history: pd.DataFrame,
    prices_history: pd.DataFrame,
    settings: Settings,
    previous: pd.Series | None,
) -> tuple[pd.Series, str]:
    styles = list(style_history.columns)
    reason = "computed"
    summary = _factor_weight_summary_for_history(style_history, prices_history, settings)
    if summary.empty:
        raw = pd.Series(1.0 / len(styles), index=styles, dtype=float)
        reason = "equal_fallback"
    else:
        raw = summary.set_index("factor")["fusion_weight"].reindex(styles).fillna(0.0)
    current = _constrain_weights(raw, settings)
    smoothing = float(settings.rolling_factor_weight_smoothing)
    smoothing = min(max(smoothing if np.isfinite(smoothing) else 1.0, 0.0), 1.0)
    if previous is not None and smoothing < 1.0:
        prior = previous.reindex(styles).fillna(0.0)
        current = _constrain_weights(smoothing * current + (1.0 - smoothing) * prior, settings)
        reason += "_smoothed"
    return current, reason


def build_walk_forward_factor_fusion(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    *,
    factors: list[str],
    eligible_mask: pd.Series | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "thread",
    gate_cache_dir: Path | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build one fused score per rebalance using a gate evaluated strictly before that date.

    Returns ``(fused_scores, factor_decisions, rebalance_summary, style_weight_log)``.
    Warm-up/no-PASS dates contain explicit all-NaN scores so the backtest can move to cash.
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为 MultiIndex(date, symbol)")
    work = panel.copy().sort_index()
    work.index = work.index.set_names(["date", "symbol"])
    factors = list(dict.fromkeys(str(x) for x in factors if str(x) in work.columns))
    if not factors:
        raise ValueError("没有可用于 walk-forward 准入的因子")
    price_frame = prices.sort_index().sort_index(axis=1)
    benchmark = (benchmark_prices if benchmark_prices is not None else prices).sort_index()
    rebalance_dates = walk_forward_rebalance_dates(price_frame, settings)
    min_history = int(settings.walk_forward_min_history_days)
    min_windows = int(settings.walk_forward_min_rolling_windows)
    lookback = int(settings.walk_forward_history_lookback_days)

    score_parts: list[pd.Series] = []
    decision_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    previous_style_weights: pd.Series | None = None
    all_symbols = pd.Index(price_frame.columns.astype(str), name="symbol")

    context = (work, price_frame, settings, factors, eligible_mask, benchmark, lookback)

    def evaluate_at_date(
        raw_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        return _evaluate_gate_for_date(raw_date, *context)

    gate_results: dict[
        pd.Timestamp,
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str],
    ] = {}
    eligible_gate_dates = [
        pd.Timestamp(dt)
        for dt in rebalance_dates
        if int((price_frame.index < pd.Timestamp(dt)).sum()) >= min_history
    ]
    workers = max(1, int(n_jobs))
    pending_dates: list[pd.Timestamp] = []
    cache_paths: dict[pd.Timestamp, Path] = {}
    for dt in eligible_gate_dates:
        if gate_cache_dir is None:
            pending_dates.append(dt)
            continue
        cache_path = _gate_cache_path(
            Path(gate_cache_dir),
            dt,
            panel=work,
            prices=price_frame,
            settings=settings,
            factors=factors,
        )
        cache_paths[dt] = cache_path
        if cache_path.is_file():
            try:
                gate_results[dt] = pd.read_pickle(cache_path)
                continue
            except Exception:
                pass
        pending_dates.append(dt)

    def save_gate_result(dt: pd.Timestamp) -> None:
        cache_path = cache_paths.get(dt)
        result = gate_results[dt]
        if cache_path is None or result[3]:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        pd.to_pickle(result, temporary)
        temporary.replace(cache_path)

    if workers > 1 and len(pending_dates) > 1:
        backend = str(parallel_backend).strip().lower()
        if backend == "process":
            executor_context = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_gate_worker,
                initargs=context,
            )
            submit = lambda executor, dt: executor.submit(_process_gate_worker, dt)
        elif backend == "thread":
            executor_context = ThreadPoolExecutor(max_workers=workers)
            submit = lambda executor, dt: executor.submit(evaluate_at_date, dt)
        else:
            raise ValueError("parallel_backend 须为 thread 或 process")
        with executor_context as executor:
            futures = {submit(executor, dt): dt for dt in pending_dates}
            for future in as_completed(futures):
                dt = futures[future]
                gate_results[dt] = future.result()
                save_gate_result(dt)
                print(f"walk_forward_gate_completed={len(gate_results)}/{len(eligible_gate_dates)} date={dt:%Y-%m-%d}")
    else:
        for dt in pending_dates:
            gate_results[dt] = evaluate_at_date(dt)
            save_gate_result(dt)

    for raw_date in rebalance_dates:
        dt = pd.Timestamp(raw_date)
        history_dates = pd.DatetimeIndex(price_frame.index[price_frame.index < dt])
        if lookback > 0:
            history_dates = history_dates[-lookback:]
        history_days = int(len(history_dates))
        history_start = pd.Timestamp(history_dates[0]) if history_days else pd.NaT
        history_end = pd.Timestamp(history_dates[-1]) if history_days else pd.NaT

        def add_empty(status: str, reason: str, rolling_windows: int = 0) -> None:
            idx = pd.MultiIndex.from_product([[dt], all_symbols], names=["date", "symbol"])
            score_parts.append(pd.Series(np.nan, index=idx, dtype=float))
            warmup = pd.DataFrame(
                {
                    "factor": factors,
                    "decision": status,
                    "selected_for_fusion": False,
                    "selected_after_redundancy": False,
                    "reasons": reason,
                    "as_of_date": dt,
                    "history_start": history_start,
                    "history_end": history_end,
                    "history_days": history_days,
                    "rolling_windows": rolling_windows,
                }
            )
            decision_parts.append(warmup)
            summaries.append(
                {
                    "date": dt,
                    "signal_status": status,
                    "history_start": history_start,
                    "history_end": history_end,
                    "history_days": history_days,
                    "rolling_windows": rolling_windows,
                    "pass_factors": "",
                    "selected_factors": "",
                    "style_factors": "",
                    "n_pass": 0,
                    "n_selected": 0,
                    "n_styles": 0,
                    "reason": reason,
                }
            )

        if history_days < min_history:
            add_empty("WARMUP", "history_days_below_threshold")
            continue

        selection, rolling, redundancy, gate_error = (
            gate_results[dt] if workers > 1 else evaluate_at_date(dt)
        )
        if gate_error:
            add_empty("ERROR", gate_error)
            continue

        rolling_windows = 0
        if not rolling.empty and "n_windows" in rolling.columns:
            rolling_windows = int(pd.to_numeric(rolling["n_windows"], errors="coerce").max())
        if rolling_windows < min_windows:
            add_empty("WARMUP", "rolling_windows_below_threshold", rolling_windows)
            continue

        passed = selection.loc[selection["decision"] == "PASS", "factor"].astype(str).tolist()
        selected = prune_redundant_factors(passed, redundancy) if passed else []
        audit = selection.copy()
        audit["selected_after_redundancy"] = audit["factor"].astype(str).isin(selected)
        audit["as_of_date"] = dt
        audit["history_start"] = history_start
        audit["history_end"] = history_end
        audit["history_days"] = history_days
        audit["rolling_windows"] = rolling_windows
        if not selected:
            decision_parts.append(audit)
            idx = pd.MultiIndex.from_product([[dt], all_symbols], names=["date", "symbol"])
            score_parts.append(pd.Series(np.nan, index=idx, dtype=float))
            summaries.append(
                {
                    "date": dt,
                    "signal_status": "NO_PASS",
                    "history_start": history_start,
                    "history_end": history_end,
                    "history_days": history_days,
                    "rolling_windows": rolling_windows,
                    "pass_factors": ",".join(passed),
                    "selected_factors": "",
                    "style_factors": "",
                    "n_pass": len(passed),
                    "n_selected": 0,
                    "n_styles": 0,
                    "reason": "no_pass_after_redundancy",
                }
            )
            continue

        upto_dates = history_dates.union(pd.DatetimeIndex([dt]))
        panel_upto = _panel_on_dates(work[factors], upto_dates)
        style_panel, components = build_factor_composite_scores(
            panel_upto,
            eligible_factors=selected,
            min_components=1,
        )
        styles = list(style_panel.columns)
        if not styles:
            add_empty("ERROR", "no_style_composite", rolling_windows)
            continue
        weight_dates = history_dates
        weight_lookback = int(settings.rolling_factor_weight_lookback_days)
        if weight_lookback > 0:
            weight_dates = weight_dates[-weight_lookback:]
        style_history = _panel_on_dates(style_panel, weight_dates)
        weight_prices = price_frame.loc[price_frame.index.isin(weight_dates)]
        try:
            style_weights, weight_reason = _style_weights(
                style_history,
                weight_prices,
                settings,
                previous_style_weights,
            )
        except Exception as exc:
            add_empty("ERROR", f"weight_failed:{type(exc).__name__}", rolling_windows)
            continue
        zscore = cross_sectional_zscore(style_panel)
        try:
            current = zscore.xs(dt, level="date")[styles]
            fused = current.mul(style_weights, axis=1).sum(axis=1, min_count=1)
        except KeyError:
            add_empty("ERROR", "missing_rebalance_factor_values", rolling_windows)
            continue
        fused = fused.reindex(all_symbols)
        idx = pd.MultiIndex.from_product([[dt], all_symbols], names=["date", "symbol"])
        score_parts.append(pd.Series(fused.to_numpy(dtype=float), index=idx))
        decision_parts.append(audit)
        previous_style_weights = style_weights

        component_map = components.set_index("composite_factor")["eligible_components"].to_dict()
        for style in styles:
            weight_rows.append(
                {
                    "date": dt,
                    "style_factor": style,
                    "components": component_map.get(style, ""),
                    "weight": float(style_weights.get(style, np.nan)),
                    "history_start": pd.Timestamp(weight_dates[0]) if len(weight_dates) else pd.NaT,
                    "history_end": pd.Timestamp(weight_dates[-1]) if len(weight_dates) else pd.NaT,
                    "history_days": int(len(weight_dates)),
                    "reason": weight_reason,
                }
            )
        summaries.append(
            {
                "date": dt,
                "signal_status": "ACTIVE",
                "history_start": history_start,
                "history_end": history_end,
                "history_days": history_days,
                "rolling_windows": rolling_windows,
                "pass_factors": ",".join(passed),
                "selected_factors": ",".join(selected),
                "style_factors": ",".join(styles),
                "n_pass": len(passed),
                "n_selected": len(selected),
                "n_styles": len(styles),
                "reason": weight_reason,
            }
        )

    fused_scores = pd.concat(score_parts).sort_index() if score_parts else pd.Series(dtype=float)
    fused_scores.name = "FUSED_WALK_FORWARD_SCORE_WEIGHTED"
    if not fused_scores.empty:
        fused_scores.index = fused_scores.index.set_names(["date", "symbol"])
    decisions = pd.concat(decision_parts, ignore_index=True, sort=False) if decision_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries, columns=WALK_FORWARD_SUMMARY_COLUMNS)
    weights = pd.DataFrame(weight_rows)
    return fused_scores, decisions, summary, weights
