"""
主入口（**MVP 主流程**）：
1）先构建多因子原始面板（只算一次）；
2）数据质量与覆盖率报告；
3）IC 与可选落盘；
4）各因子独立回测（`config.portfolio_weighting`：equal / max_sharpe / risk_parity）并打印每期 Top-K 及权重、绩效；
5）多因子融合：IC 滞后滚动列权、训练段静态综合权重、调仓日前滚动综合权重三条路线并行验证。
6）样本外验证与因子失效监控：训练段 vs 验证段对比 IC、多头超额和分组单调性。
7）构造股票池等权基准，计算超额收益、跟踪误差与信息比率。
8）由调仓日志估算换手率与交易成本。
9）由调仓日志计算 HHI / effective_n 等持仓集中度指标。
10）可选保存运行配置、绩效汇总、调仓日志、决策审计日志与图表，形成可复现实验记录。
非 MVP：`live` 信号/模拟盘、`fuse_models` 高阶 method；详见 README「MVP 定稿」。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.benchmark import (
    equal_weight_benchmark_nav,
    excess_nav_frame,
    summarize_excess,
)
from analysis.data_quality import (
    factor_coverage,
    factor_daily_coverage,
    price_coverage,
    rebalance_coverage,
)
from analysis.factor_diagnostics import batch_factor_group_returns, batch_factor_long_excess
from analysis.factor_composite import build_factor_composite_scores
from analysis.factor_redundancy import (
    build_factor_redundancy_report,
    factor_cross_sectional_correlation,
    prune_redundant_factors,
)
from analysis.factor_selection import build_factor_selection_table, selected_factors_for_fusion
from analysis.factor_weight_stability import (
    factor_weight_drift_events,
    factor_weight_portfolio_drift,
    factor_weight_stability_summary,
)
from analysis.style_exposure import (
    batch_style_exposure,
    style_exposure_return_link,
    summarize_style_exposure,
)
from analysis.factor_validation import (
    build_factor_decay_monitor,
    build_out_of_sample_validation,
    build_rolling_out_of_sample_validation,
    save_factor_validation_outputs,
    summarize_rolling_out_of_sample_validation,
)
from analysis.ic import (
    daily_ic_spearman,
    ic_distribution_summary,
    ic_rolling_stability,
    save_ic_diagnostics,
    save_ic_series,
    summarize_ic,
)
from analysis.market_regime import (
    build_market_regime_frame,
    save_market_regime_outputs,
    strategy_regime_performance,
    summarize_regime_days,
    summarize_strategy_regime_robustness,
)
from analysis.performance import summarize
from analysis.plotting import (
    plot_effective_n,
    plot_factor_coverage,
    plot_ic,
    plot_nav,
    plot_turnover,
    plot_weights,
    rebalance_log_to_weights_frame,
)
from analysis.risk_exposure import (
    concentration_frame,
    effective_n_wide,
    summarize_concentration,
)
from analysis.turnover import summarize_turnover, turnover_frame, turnover_wide
from backtest.backtest_multi import run_multi_backtest
from backtest.backtest_single import run_single_backtest
from backtest.backtest_utils import long_to_wide, wide_to_long
from config import Settings, get_settings
from factors.factor_ml import ML_SCORE_NAME, build_ml_score_factor
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel
from factors.preprocess import cross_sectional_zscore, preprocess_factor_panel
from live.cache_io import (
    save_decision_logs,
    save_performance_summary,
    save_rebalance_logs,
    save_risk_exposure_logs,
    save_risk_exposure_summary,
    save_run_cache,
    save_run_config,
    save_data_quality_reports,
    save_factor_diagnostics,
    save_style_exposure_outputs,
    save_turnover_logs,
)
from live.data_feed import fetch_daily_panel, load_prices_from_csv
from live.stock_pool import load_stock_pool, load_stock_pool_frame, normalize_ts_code
from models.factor_weighting import build_factor_weight_summary
from models.fusion import (
    fuse_equal_weight_zscore,
    fuse_ic_weighted_zscore,
    fuse_static_weight_zscore,
)


def _build_fused_zscore_panel(
    sub: pd.DataFrame,
    ic_by_name: dict[str, pd.Series],
    settings: Settings,
) -> tuple[pd.Series, str]:
    """
    融合得分：默认 IC 滞后滚动列权；缺 IC / 配置关闭 / 异常时回退等权 z-score。
    返回 (Series, 模式标签 ic_rolling_weighted | equal_zscore)。
    """
    cols = list(sub.columns)
    use_ic = bool(getattr(settings, "fusion_use_ic_weights", True))
    win = int(getattr(settings, "fusion_ic_rolling_window", 60))
    min_p = int(getattr(settings, "fusion_ic_min_periods", 20))
    ic_sub = {k: ic_by_name[k] for k in cols if k in ic_by_name}
    if not use_ic:
        return fuse_equal_weight_zscore(sub), "equal_zscore"
    if len(ic_sub) != len(cols) or not cols:
        miss = sorted(set(cols) - set(ic_sub.keys()))
        if miss:
            print("【融合】IC 加权需各列均有 IC，缺 %s → 等权 z-score 融合" % miss)
        return fuse_equal_weight_zscore(sub), "equal_zscore"
    try:
        return (
            fuse_ic_weighted_zscore(sub, ic_sub, rolling_window=win, min_periods=min_p),
            "ic_rolling_weighted",
        )
    except Exception as e:
        print("【融合】IC 加权失败，回退等权 z-score:", e)
        return fuse_equal_weight_zscore(sub), "equal_zscore"


def _panel_date_mask(panel: pd.DataFrame, dates: pd.Index) -> np.ndarray:
    vals = panel.index.get_level_values("date")
    return vals.isin(pd.Index(dates))


def _split_factor_weight_train_test(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.Index, pd.Index]:
    panel_dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    dates = pd.Index(prices.index).intersection(panel_dates).sort_values()
    if len(dates) < 4:
        raise ValueError("可用于训练/验证切分的日期太少")

    ratio = float(getattr(settings, "factor_weight_train_ratio", 0.5))
    ratio = min(max(ratio, 0.2), 0.8)
    split_pos = int(len(dates) * ratio)
    split_pos = min(max(split_pos, 2), len(dates) - 1)
    return dates[:split_pos], dates[split_pos:]


def _build_train_factor_weight_summary(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    """
    训练段计算综合权重，验证段只使用固定权重。

    这一步和全样本 `factor_weight_summary` 不同：后者用于诊断审计；这里的
    `factor_weight_train_summary` 是后续 `FUSED_SCORE_WEIGHTED` 真正使用的权重来源。
    """
    train_dates, test_dates = _split_factor_weight_train_test(panel, prices, settings)
    train_end = pd.Timestamp(train_dates[-1])
    test_start = pd.Timestamp(test_dates[0])
    train_panel = panel.loc[_panel_date_mask(panel, train_dates)]
    train_prices = prices.loc[prices.index <= train_end]

    train_ic_by_name: dict[str, pd.Series] = {}
    for fname in panel.columns:
        col = train_panel[fname]
        if col.notna().sum() == 0:
            continue
        try:
            train_ic_by_name[fname] = daily_ic_spearman(
                col,
                train_prices,
                forward_days=settings.ic_forward_days,
            )
        except Exception:
            continue
    if not train_ic_by_name:
        raise ValueError("训练段没有可用 IC，无法计算静态综合权重")

    train_ic_distribution = ic_distribution_summary(train_ic_by_name)
    train_ic_rolling = ic_rolling_stability(
        train_ic_by_name,
        windows=settings.ic_rolling_windows,
    )
    _, train_group_summary = batch_factor_group_returns(
        train_panel,
        train_prices,
        factors=list(panel.columns),
        group_count=settings.factor_group_count,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        trading_days_per_year=settings.trading_days_per_year,
    )
    windows = tuple(int(w) for w in settings.ic_rolling_windows)
    preferred_window = max(windows) if windows else None
    summary = build_factor_weight_summary(
        train_ic_distribution,
        train_ic_rolling,
        train_group_summary,
        factors=list(panel.columns),
        preferred_rolling_window=preferred_window,
    )
    if summary.empty:
        raise ValueError("训练段综合权重表为空")

    weights = {
        str(rec["factor"]): float(rec["fusion_weight"])
        for rec in summary.to_dict("records")
        if pd.notna(rec.get("fusion_weight"))
    }
    meta = {
        "factor_weight_train_ratio": float(getattr(settings, "factor_weight_train_ratio", 0.5)),
        "train_start": pd.Timestamp(train_dates[0]),
        "train_end": train_end,
        "test_start": test_start,
        "test_end": pd.Timestamp(test_dates[-1]),
        "factor_weight_source": "factor_weight_train_summary",
        "fusion_weight_by_factor": weights,
    }
    return summary, weights, meta


def _factor_weight_summary_for_history(
    panel_hist: pd.DataFrame,
    prices_hist: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    ic_by_name: dict[str, pd.Series] = {}
    for fname in panel_hist.columns:
        col = panel_hist[fname]
        if col.notna().sum() == 0:
            continue
        try:
            ic_by_name[fname] = daily_ic_spearman(
                col,
                prices_hist,
                forward_days=settings.ic_forward_days,
            )
        except Exception:
            continue
    if not ic_by_name:
        return pd.DataFrame()

    ic_distribution = ic_distribution_summary(ic_by_name)
    ic_rolling = ic_rolling_stability(ic_by_name, windows=settings.ic_rolling_windows)
    _, group_summary = batch_factor_group_returns(
        panel_hist,
        prices_hist,
        factors=list(panel_hist.columns),
        group_count=settings.factor_group_count,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        trading_days_per_year=settings.trading_days_per_year,
    )
    windows = tuple(int(w) for w in settings.ic_rolling_windows)
    preferred_window = max(windows) if windows else None
    return build_factor_weight_summary(
        ic_distribution,
        ic_rolling,
        group_summary,
        factors=list(panel_hist.columns),
        preferred_rolling_window=preferred_window,
    )


def _clean_factor_weights(
    weights: dict[str, float],
    factors: list[str],
) -> pd.Series:
    raw = pd.Series([float(weights.get(f, 0.0)) for f in factors], index=factors, dtype=float)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 1e-18:
        return pd.Series(1.0 / len(factors), index=factors, dtype=float)
    return raw / total


def _constrain_factor_weights(
    weights: pd.Series,
    *,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    factors = list(weights.index)
    n = len(factors)
    if n == 0:
        return pd.Series(dtype=float)

    w = _clean_factor_weights(weights.to_dict(), factors)
    floor = float(min_weight)
    if not np.isfinite(floor) or floor < 0.0 or floor * n >= 1.0:
        floor = 0.0
    cap = float(max_weight)
    if not np.isfinite(cap) or cap <= 0.0 or cap * n < 1.0:
        cap = 1.0
    cap = min(max(cap, floor), 1.0)

    if floor > 0.0:
        w = pd.Series(floor, index=factors, dtype=float) + (1.0 - floor * n) * w
        w = w / float(w.sum())

    if cap >= 1.0:
        return w

    arr = w.to_numpy(dtype=float).copy()
    for _ in range(n + 2):
        over = arr > cap + 1e-12
        if not bool(over.any()):
            break
        excess = float((arr[over] - cap).sum())
        arr[over] = cap
        under = ~over
        room = np.maximum(cap - arr[under], 0.0)
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        arr[under] += excess * room / room_sum
    out = pd.Series(arr, index=factors, dtype=float).clip(lower=0.0)
    total = float(out.sum())
    if total <= 1e-18:
        return pd.Series(1.0 / n, index=factors, dtype=float)
    return out / total


def _last_rebalance_dates(prices: pd.DataFrame, settings: Settings) -> pd.DatetimeIndex:
    rf = _resample_freq_alias(settings.rebalance_freq)
    if prices.empty:
        return pd.DatetimeIndex([])
    price_frame = prices.dropna(how="all").sort_index()
    dates = pd.DatetimeIndex(
        [group.index[-1] for _, group in price_frame.groupby(pd.Grouper(freq=rf)) if not group.empty]
    )
    if bool(getattr(settings, "force_final_rebalance", False)) and len(prices.index) > 0:
        dates = dates.union(pd.DatetimeIndex([prices.index[-1]]))
    return dates.sort_values()


def _build_rolling_score_weighted_fusion(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    """
    每个调仓日前用历史窗口重新计算综合因子权重，生成滚动融合得分。

    权重只使用调仓日之前的历史数据；样本不足或计算失败时使用上一期权重，
    若没有上一期则回退等权。输出的日志用于审计每期因子权重如何变化。
    """
    factors = list(panel.columns)
    if not factors:
        raise ValueError("滚动融合至少需要一列因子")

    price_dates = pd.DatetimeIndex(prices.index).sort_values()
    rebal_dates = _last_rebalance_dates(prices, settings)
    if len(rebal_dates) == 0:
        raise ValueError("无调仓日，无法构造滚动综合权重")

    lookback_days = int(getattr(settings, "rolling_factor_weight_lookback_days", 120))
    min_days = int(getattr(settings, "rolling_factor_weight_min_days", 60))
    min_w = float(getattr(settings, "rolling_factor_weight_min_weight", 0.0))
    max_w = float(getattr(settings, "rolling_factor_weight_max_weight", 1.0))
    smoothing = float(getattr(settings, "rolling_factor_weight_smoothing", 1.0))
    if not np.isfinite(smoothing):
        smoothing = 1.0
    smoothing = min(max(smoothing, 0.0), 1.0)

    z = cross_sectional_zscore(panel)
    pieces: list[pd.Series] = []
    rows: list[dict[str, object]] = []
    prev_weights: pd.Series | None = None

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        history_dates = price_dates[price_dates < dt]
        if lookback_days > 0:
            history_dates = history_dates[-lookback_days:]
        history_days = int(len(history_dates))
        hist_start = pd.Timestamp(history_dates[0]) if history_days else pd.NaT
        hist_end = pd.Timestamp(history_dates[-1]) if history_days else pd.NaT

        factor_scores = pd.Series(np.nan, index=factors, dtype=float)
        raw_weights = pd.Series(1.0 / len(factors), index=factors, dtype=float)
        reason = "computed"
        summary = pd.DataFrame()

        if history_days < min_days:
            reason = "insufficient_history_previous" if prev_weights is not None else "insufficient_history_equal"
            raw_weights = prev_weights.copy() if prev_weights is not None else raw_weights
        else:
            try:
                panel_hist = panel.loc[_panel_date_mask(panel, history_dates)]
                prices_hist = prices.loc[prices.index.isin(history_dates)]
                summary = _factor_weight_summary_for_history(panel_hist, prices_hist, settings)
                if summary.empty:
                    reason = "empty_summary_previous" if prev_weights is not None else "empty_summary_equal"
                    raw_weights = prev_weights.copy() if prev_weights is not None else raw_weights
                else:
                    factor_scores = pd.Series(
                        {
                            str(rec["factor"]): float(rec.get("factor_score", np.nan))
                            for rec in summary.to_dict("records")
                        },
                        dtype=float,
                    ).reindex(factors)
                    raw_weights = _clean_factor_weights(
                        {
                            str(rec["factor"]): float(rec.get("fusion_weight", 0.0))
                            for rec in summary.to_dict("records")
                        },
                        factors,
                    )
            except Exception:
                reason = "calc_failed_previous" if prev_weights is not None else "calc_failed_equal"
                raw_weights = prev_weights.copy() if prev_weights is not None else raw_weights

        constrained = _constrain_factor_weights(raw_weights, min_weight=min_w, max_weight=max_w)
        if prev_weights is not None and smoothing < 1.0:
            blended = smoothing * constrained + (1.0 - smoothing) * prev_weights.reindex(factors).fillna(0.0)
            final_weights = _constrain_factor_weights(blended, min_weight=min_w, max_weight=max_w)
            if reason == "computed":
                reason = "computed_smoothed"
        else:
            final_weights = constrained

        try:
            z_dt = z.xs(dt, level="date")
        except KeyError:
            prev_weights = final_weights
            continue
        fused_dt = z_dt.mul(final_weights, axis=1).sum(axis=1)
        idx = pd.MultiIndex.from_product([[dt], fused_dt.index], names=["date", "symbol"])
        pieces.append(pd.Series(fused_dt.to_numpy(dtype=float), index=idx))

        for factor in factors:
            rows.append(
                {
                    "date": dt,
                    "factor": factor,
                    "factor_score": float(factor_scores.get(factor, np.nan)),
                    "raw_weight": float(raw_weights.get(factor, np.nan)),
                    "constrained_weight": float(constrained.get(factor, np.nan)),
                    "final_weight": float(final_weights.get(factor, np.nan)),
                    "history_start": hist_start,
                    "history_end": hist_end,
                    "history_days": history_days,
                    "reason": reason,
                    "min_weight": min_w,
                    "max_weight": max_w,
                    "smoothing": smoothing,
                }
            )
        prev_weights = final_weights

    if not pieces:
        raise ValueError("滚动综合权重没有生成任何调仓日得分")

    fused = pd.concat(pieces).sort_index().astype(float)
    fused.name = "fused_zscore_rolling_score_weighted"
    fused.index = fused.index.set_names(["date", "symbol"])
    log = pd.DataFrame(rows)
    meta = {
        "fusion_mode": "rolling_score_weighted",
        "rolling_factor_weight_lookback_days": lookback_days,
        "rolling_factor_weight_min_days": min_days,
        "rolling_factor_weight_min_weight": min_w,
        "rolling_factor_weight_max_weight": max_w,
        "rolling_factor_weight_smoothing": smoothing,
        "rolling_weight_rebalances": int(log["date"].nunique()) if not log.empty else 0,
    }
    return fused, log, meta


# 无 CSV 时用于 Tushare 拉取（可按需增删；需与积分权限匹配）
_DEFAULT_TS_SYMBOLS = [
    "600519.SH",
    "601318.SH",
    "600036.SH",
    "601166.SH",
    "600030.SH",
    "601398.SH",
    "601288.SH",
    "000858.SZ",
]


def _demo_price_wide() -> pd.DataFrame:
    """合成收盘价宽表（工作日 × 多标的），便于无 CSV / 无 Token 时跑通流水线。"""
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2023-01-01", periods=280)
    syms = ["600519.SH", "000001.SZ", "601318.SH", "600036.SH", "601166.SH"]
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, size=(len(days), len(syms))), axis=0)
    return pd.DataFrame(px, index=days, columns=syms)


def _industry_map_from_stock_pool(settings: Settings) -> dict[str, str]:
    pool_path = settings.stock_pool_path
    if pool_path is None or not Path(pool_path).is_file():
        return {}
    try:
        pool = load_stock_pool_frame(pool_path, code_col=settings.stock_pool_code_col)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for rec in pool.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol"))
        if not symbol:
            continue
        industry = str(rec.get("sub_industry") or "").strip()
        if not industry:
            industry = str(rec.get("theme") or "").strip()
        if industry:
            out[symbol] = industry
    return out


def _attach_industry_to_long_df(long_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if long_df.empty or "ts_code" not in long_df.columns:
        return long_df
    industry_col = str(getattr(settings, "industry_col", "industry") or "industry")
    if industry_col in long_df.columns and long_df[industry_col].notna().any():
        return long_df
    industry_map = _industry_map_from_stock_pool(settings)
    if not industry_map:
        return long_df
    out = long_df.copy()
    out["_norm_ts_code"] = out["ts_code"].map(normalize_ts_code)
    out[industry_col] = out["_norm_ts_code"].map(industry_map).fillna("")
    out = out.drop(columns=["_norm_ts_code"])
    n_known = int((out[industry_col].astype(str).str.strip() != "").sum())
    if n_known:
        print("已从股票池补充行业字段 %r: %d 行" % (industry_col, n_known))
    return out


def _industry_series_from_long_df(
    long_df: pd.DataFrame,
    panel_index: pd.MultiIndex,
    settings: Settings,
) -> pd.Series | None:
    industry_col = str(getattr(settings, "industry_col", "industry") or "industry")
    need = {"trade_date", "ts_code", industry_col}
    if long_df.empty or not need.issubset(long_df.columns):
        return None
    data = long_df[list(need)].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["ts_code"] = data["ts_code"].map(normalize_ts_code)
    data[industry_col] = data[industry_col].fillna("").astype(str).str.strip()
    data = data[data[industry_col] != ""].drop_duplicates(["trade_date", "ts_code"], keep="last")
    if data.empty:
        return None
    industry_wide = data.pivot(index="trade_date", columns="ts_code", values=industry_col).sort_index()
    panel_dates = pd.Index(panel_index.get_level_values("date")).unique().sort_values()
    industry_wide = industry_wide.reindex(panel_dates).ffill()
    try:
        industry_long = industry_wide.stack(future_stack=True)
    except TypeError:
        industry_long = industry_wide.stack(dropna=False)
    industry_long.index = industry_long.index.set_names(["date", "symbol"])
    return industry_long.reindex(panel_index)


def _research_price_column(long_df: pd.DataFrame, settings: Settings) -> str:
    """选择研究回测价格列；优先复权价，缺失时回退原始价格。"""
    research_col = str(getattr(settings, "research_price_col", "") or "").strip()
    fallback_col = str(getattr(settings, "price_col", "close") or "close").strip()
    if research_col and research_col in long_df.columns and long_df[research_col].notna().any():
        return research_col
    return fallback_col


def _price_wides_for_research_and_execution(
    long_df: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """返回研究价格宽表、交易价格宽表和各自列名。"""
    research_col = _research_price_column(long_df, settings)
    execution_col = str(getattr(settings, "execution_price_col", "") or settings.price_col)
    if execution_col not in long_df.columns:
        execution_col = settings.price_col
    research_prices = long_to_wide(long_df, research_col)
    execution_prices = long_to_wide(long_df, execution_col)
    return research_prices, execution_prices, research_col, execution_col


def _load_market_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取本地/远程行情数据，返回 `(long_df, research_prices, execution_prices)`。

    优先级：
    1. `data/prices_demo.csv`
    2. `settings.tushare_price_cache_path`
    3. `settings.stock_pool_path` + Tushare 拉取并写缓存
    4. 默认示例股票 + Tushare 拉取
    5. 合成数据兜底
    """
    demo_path = settings.data_dir / "prices_demo.csv"
    if demo_path.is_file():
        print("加载本地数据:", demo_path)
        long_df = load_prices_from_csv(demo_path)
        long_df = _attach_industry_to_long_df(long_df, settings)
        research_prices, execution_prices, research_col, execution_col = _price_wides_for_research_and_execution(
            long_df,
            settings,
        )
        print("价格口径: research=%s, execution=%s" % (research_col, execution_col))
        return long_df, research_prices, execution_prices

    cache_path = settings.tushare_price_cache_path
    if cache_path is not None and Path(cache_path).is_file():
        print("加载 Tushare 本地行情缓存:", cache_path)
        long_df = load_prices_from_csv(cache_path)
        long_df = _attach_industry_to_long_df(long_df, settings)
        research_prices, execution_prices, research_col, execution_col = _price_wides_for_research_and_execution(
            long_df,
            settings,
        )
        print(
            "本地缓存已对齐: %d 只股票, %d 个交易日；价格口径 research=%s, execution=%s"
            % (research_prices.shape[1], research_prices.shape[0], research_col, execution_col)
        )
        return long_df, research_prices, execution_prices

    symbols = list(_DEFAULT_TS_SYMBOLS)
    pool_path = settings.stock_pool_path
    if pool_path is not None and Path(pool_path).is_file():
        symbols = load_stock_pool(pool_path, code_col=settings.stock_pool_code_col)
        print("加载股票池: %s，%d 只股票" % (pool_path, len(symbols)))
    else:
        print("未找到股票池文件 %s，使用默认示例股票池" % pool_path)

    try:
        print(
            "尝试从 Tushare 拉取（区间 %s ~ %s，股票数=%d）"
            % (settings.backtest_start, settings.backtest_end, len(symbols))
        )
        long_df = fetch_daily_panel(
            symbols,
            settings.backtest_start,
            settings.backtest_end,
            include_adj_factor=True,
            adjustment_mode=getattr(settings, "adjustment_mode", "qfq"),
        )
        long_df = _attach_industry_to_long_df(long_df, settings)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            long_df.to_csv(cache_path, index=False, date_format="%Y-%m-%d")
            print("Tushare 行情已缓存:", cache_path)
        research_prices, execution_prices, research_col, execution_col = _price_wides_for_research_and_execution(
            long_df,
            settings,
        )
        print(
            "Tushare 已对齐: %d 只股票, %d 个交易日；价格口径 research=%s, execution=%s"
            % (research_prices.shape[1], research_prices.shape[0], research_col, execution_col)
        )
        return long_df, research_prices, execution_prices
    except Exception as e:
        print("Tushare 不可用，回退合成数据:", e)
        prices = _demo_price_wide()
        long_df = wide_to_long(prices, settings.price_col)
        long_df = _attach_industry_to_long_df(long_df, settings)
        return long_df, prices, prices


def _resample_freq_alias(freq: str) -> str:
    return {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(freq, freq)


def _print_backtest_block(title: str, nav: pd.Series, meta: dict, stats: dict) -> None:
    print(title)
    print(
        "  回测设置: top_k=%s, 再平衡=%s, 手续费率=%s, 持仓权重=%s"
        % (
            meta.get("top_k"),
            meta.get("rebalance_freq"),
            meta.get("commission_rate"),
            meta.get("portfolio_weighting", "equal"),
        )
    )
    print("  单票权重上限: %s" % meta.get("max_position_weight", ""))
    print("  单次换手上限: %s" % meta.get("max_rebalance_turnover", ""))
    print("  调仓次数: %s" % meta.get("n_rebalances"))
    log = meta.get("rebalance_log") or []
    if log:
        print("  —— 各再平衡日 Top-%s 选股与权重 ——" % meta.get("top_k", "?"))
        for rec in log:
            dt = rec.get("date")
            dts = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            picks = rec.get("picks") or []
            wt = rec.get("weights") or []
            wl = rec.get("weighting", "")
            parts = []
            for i, p in enumerate(picks):
                wv = wt[i] if i < len(wt) else float("nan")
                parts.append("%s(%.3f)" % (p, wv))
            print("    %s  [%s]  %s" % (dts, wl, ", ".join(parts)))
    print("  —— 绩效指标 ——")
    print("  年化收益:     %.4f  (约 %.2f%%)" % (stats["ann_return"], stats["ann_return"] * 100))
    print("  年化波动率:   %.4f  (约 %.2f%%)" % (stats["ann_vol"], stats["ann_vol"] * 100))
    print("  夏普比率:     %.4f  (无风险利率按 0)" % stats["sharpe"])
    print("  最大回撤:     %.4f  (约 %.2f%%)" % (stats["max_drawdown"], stats["max_drawdown"] * 100))
    print("  期末净值:     %.6f  (期初归一 1.0)" % nav.iloc[-1])
    print()


def main() -> None:
    settings = get_settings()
    print(
        "配置: portfolio_weighting=%s（equal=等权；max_sharpe=夏普；risk_parity=风险平价）"
        % settings.portfolio_weighting
    )
    long_df, research_prices, execution_prices = _load_market_data(settings)
    prices = research_prices

    print("\n构建因子面板（多列原始因子，只计算一次）…")
    try:
        panel = build_four_factor_panel(research_prices, long_df, settings)
    except Exception as e:
        print("因子面板构建失败:", e)
        return

    print(
        "面板形状: %d 行 × %d 列 (列=%s)"
        % (panel.shape[0], panel.shape[1], list(panel.columns))
    )

    ml_score_log = pd.DataFrame()
    factor_order = [f for f in DEFAULT_FACTOR_ORDER if f in panel.columns]
    if getattr(settings, "enable_ml_score", True):
        try:
            ml_score, ml_score_log = build_ml_score_factor(
                panel[factor_order].dropna(axis=1, how="all"),
                prices,
                settings,
                feature_cols=factor_order,
            )
            if ml_score.notna().sum() > 0:
                panel[ML_SCORE_NAME] = ml_score.reindex(panel.index)
                factor_order.append(ML_SCORE_NAME)
                backend = (
                    str(ml_score_log["model_backend"].iloc[-1])
                    if not ml_score_log.empty and "model_backend" in ml_score_log.columns
                    else str(getattr(settings, "ml_score_model", ""))
                )
                print(
                    "机器学习打分因子已生成: %s  有效样本=%d  refit=%d  backend=%s"
                    % (
                        ML_SCORE_NAME,
                        int(panel[ML_SCORE_NAME].notna().sum()),
                        int(len(ml_score_log)),
                        backend,
                    )
                )
            else:
                print("机器学习打分因子跳过: 有效预测为空")
        except Exception as e:
            print("机器学习打分因子生成失败（不影响基础因子流程）:", e)
    else:
        print("机器学习打分因子已关闭: enable_ml_score=False")

    print("本次参与诊断/回测的因子顺序: %s" % factor_order)
    industry_ser = _industry_series_from_long_df(long_df, panel.index, settings)
    by_industry = bool(getattr(settings, "factor_standardize_by_industry", False)) and industry_ser is not None
    panel_zscore = preprocess_factor_panel(
        panel,
        industry=industry_ser,
        industry_col=str(getattr(settings, "industry_col", "industry") or "industry"),
        by_industry=by_industry,
        min_industry_count=int(getattr(settings, "factor_industry_min_count", 3)),
    )
    research_panel = panel_zscore
    standardize_label = (
        "行业内 winsorize + z-score（小行业回退全截面）"
        if by_industry
        else "横截面 winsorize + z-score"
    )
    print(
        "标准化面板: %d 行 × %d 列（%s）"
        % (panel_zscore.shape[0], panel_zscore.shape[1], standardize_label)
    )

    data_quality_reports: dict[str, pd.DataFrame] = {}
    try:
        rebalance_dates = _last_rebalance_dates(prices, settings)
        data_quality_reports = {
            "price_coverage": price_coverage(prices),
            "factor_coverage": factor_coverage(panel),
            "factor_daily_coverage": factor_daily_coverage(panel),
            "rebalance_coverage": rebalance_coverage(
                panel,
                prices,
                rebalance_dates,
                factors=factor_order,
            ),
        }
        fc = data_quality_reports["factor_coverage"]
        print("\n========== 数据质量与覆盖率 ==========\n")
        for rec in fc.to_dict("records"):
            print(
                "【覆盖】%s  valid=%d/%d  coverage=%.2f%%"
                % (
                    rec["factor"],
                    rec["valid_cells"],
                    rec["total_cells"],
                    rec["coverage"] * 100.0,
                )
            )
        print()
    except Exception as e:
        print("数据质量报告生成失败（不影响主流程）:", e)

    if settings.persist_run_outputs:
        try:
            paths = save_run_cache(settings, long_df, execution_prices, panel, panel_zscore=panel_zscore)
            print(
                "数据已落盘: %s"
                % ", ".join("%s=%s" % (k, v) for k, v in paths.items())
            )
            if not ml_score_log.empty:
                ml_log_dir = settings.output_dir / "factor_diagnostics"
                ml_log_dir.mkdir(parents=True, exist_ok=True)
                ml_log_path = ml_log_dir / "ml_score_training_log.csv"
                ml_score_log.to_csv(ml_log_path, index=False)
                print("机器学习打分因子训练日志已保存:", ml_log_path.resolve())
        except Exception as e:
            print("落盘失败（不影响回测）:", e)
        if data_quality_reports:
            try:
                dq_paths = save_data_quality_reports(settings, data_quality_reports)
                print(
                    "数据质量报告已保存: %d 份 → 目录 %s"
                    % (len(dq_paths), (settings.output_dir / "data_quality").resolve())
                )
                plot_factor_coverage(
                    data_quality_reports["factor_coverage"],
                    title="因子有效覆盖率",
                    save_path=settings.output_dir / "data_quality" / "factor_coverage.png",
                )
                print(
                    "因子覆盖率图已保存:",
                    (settings.output_dir / "data_quality" / "factor_coverage.png").resolve(),
                )
            except Exception as e:
                print("数据质量报告落盘失败（不影响主流程）:", e)

    ic_by_name: dict[str, pd.Series] = {}
    backtest_meta_by_name: dict[str, dict] = {}
    performance_by_name: dict[str, dict] = {}
    print(
        "\n========== IC（截面 Spearman：因子@t vs 前瞻 %d 日收盘收益）==========\n"
        % settings.ic_forward_days
    )
    for fname in factor_order:
        if fname not in research_panel.columns:
            print("【IC】%s 跳过: 面板中无该列\n" % fname)
            continue
        col = research_panel[fname]
        if col.notna().sum() == 0:
            print("【IC】%s 跳过: 整列无有效值\n" % fname)
            continue
        try:
            ic_ser = daily_ic_spearman(
                col,
                prices,
                forward_days=settings.ic_forward_days,
            )
            ic_by_name[fname] = ic_ser
            st = summarize_ic(ic_ser)
            print("【IC】%s" % fname)
            print(
                "  mean_IC=%.4f  std_IC=%.4f  IC_IR=%.4f  胜率=%.2f%%  有效日=%d"
                % (
                    st["mean_ic"],
                    st["std_ic"],
                    st["ic_ir"],
                    st["hit_rate"] * 100.0,
                    st["n_days"],
                )
            )
            print()
        except Exception as e:
            print("【IC】%s 跳过: %s\n" % (fname, e))
    try:
        sub_ic = research_panel[factor_order].dropna(axis=1, how="all")
        if sub_ic.shape[1] > 0:
            fused_ic, fusion_mode = _build_fused_zscore_panel(sub_ic, ic_by_name, settings)
            ic_f = daily_ic_spearman(
                fused_ic,
                prices,
                forward_days=settings.ic_forward_days,
            )
            ic_by_name["FUSED_ZSCORE"] = ic_f
            stf = summarize_ic(ic_f)
            print("【IC】FUSED_ZSCORE（列: %s，融合=%s）" % (list(sub_ic.columns), fusion_mode))
            print(
                "  mean_IC=%.4f  std_IC=%.4f  IC_IR=%.4f  胜率=%.2f%%  有效日=%d"
                % (
                    stf["mean_ic"],
                    stf["std_ic"],
                    stf["ic_ir"],
                    stf["hit_rate"] * 100.0,
                    stf["n_days"],
                )
            )
            print()
    except Exception as e:
        print("【IC】FUSED_ZSCORE 跳过: %s\n" % e)

    ic_distribution = pd.DataFrame()
    ic_rolling = pd.DataFrame()
    if ic_by_name:
        try:
            ic_distribution = ic_distribution_summary(ic_by_name)
            ic_rolling = ic_rolling_stability(
                ic_by_name,
                windows=settings.ic_rolling_windows,
            )
            print("========== IC 分布与稳定性 ==========\n")
            for rec in ic_distribution.to_dict("records"):
                print(
                    "【IC分布】%s  median=%.4f  p25=%.4f  p75=%.4f  正IC=%.2f%%  负IC=%.2f%%"
                    % (
                        rec["factor"],
                        rec["median"],
                        rec["p25"],
                        rec["p75"],
                        rec["positive_rate"] * 100.0,
                        rec["negative_rate"] * 100.0,
                    )
                )
            if not ic_rolling.empty:
                for rec in ic_rolling.to_dict("records"):
                    print(
                        "【IC滚动】%s  win=%d  last_mean=%.4f  mean>0比例=%.2f%%"
                        % (
                            rec["factor"],
                            rec["window"],
                            rec["rolling_mean_last"],
                            rec["rolling_mean_positive_rate"] * 100.0,
                        )
                    )
            print()
        except Exception as e:
            print("IC 分布与稳定性诊断失败（不影响回测）:", e)
            print()

    if settings.persist_run_outputs and ic_by_name:
        try:
            ic_paths = save_ic_series(settings, ic_by_name)
            print("IC 序列已落盘:", ", ".join("%s=%s" % (k, v) for k, v in ic_paths.items()))
            if not ic_distribution.empty or not ic_rolling.empty:
                ic_diag_paths = save_ic_diagnostics(settings, ic_distribution, ic_rolling)
                print(
                    "IC 诊断已保存:",
                    ", ".join("%s=%s" % (k, v.resolve()) for k, v in ic_diag_paths.items()),
                )
            print()
        except Exception as e:
            print("IC 落盘失败（不影响回测）:", e)
            print()

    long_excess_summary = pd.DataFrame()
    group_return_detail = pd.DataFrame()
    group_return_summary = pd.DataFrame()
    factor_weight_summary = pd.DataFrame()
    factor_weight_train_summary = pd.DataFrame()
    rolling_factor_weight_log = pd.DataFrame()
    factor_weight_stability = pd.DataFrame()
    factor_weight_drift = pd.DataFrame()
    factor_weight_portfolio_drift_frame = pd.DataFrame()
    factor_selection_summary = pd.DataFrame()
    factor_composite_components = pd.DataFrame()
    factor_composite_scores = pd.DataFrame()
    factor_correlation_matrix = pd.DataFrame()
    factor_correlation_days = pd.DataFrame()
    factor_redundancy_report = pd.DataFrame()
    out_of_sample_validation = pd.DataFrame()
    factor_decay_monitor = pd.DataFrame()
    rolling_out_of_sample_validation = pd.DataFrame()
    rolling_out_of_sample_summary = pd.DataFrame()
    score_weighted_weights_by_factor: dict[str, float] = {}
    score_weighted_meta: dict[str, object] = {}
    fused_rolling_score_weighted = pd.Series(dtype=float)
    rolling_score_weighted_meta: dict[str, object] = {}
    fusion_factor_order = list(factor_order)
    print("========== 因子 Top-K 多头超额诊断 ==========\n")
    try:
        long_excess_summary, _ = batch_factor_long_excess(
            research_panel,
            prices,
            factors=factor_order,
            top_k=settings.top_k,
            rebalance_freq=settings.rebalance_freq,
            price_col=settings.price_col,
            periods=settings.trading_days_per_year,
        )
        if long_excess_summary.empty:
            print("【多头超额】跳过: 无有效因子列\n")
        else:
            for rec in long_excess_summary.to_dict("records"):
                print(
                    "【多头超额】%s  ann=%.4f  excess_ann=%.4f  TE=%.4f  IR=%.4f  rebalances=%d"
                    % (
                        rec["factor"],
                        rec["ann_return"],
                        rec["excess_ann_return"],
                        rec["tracking_error"],
                        rec["information_ratio"],
                        rec["n_rebalances"],
                    )
                )
            print()
    except Exception as e:
        print("【多头超额】跳过: %s\n" % e)

    print("========== 因子分组收益与单调性 ==========\n")
    try:
        group_return_detail, group_return_summary = batch_factor_group_returns(
            research_panel,
            prices,
            factors=factor_order,
            group_count=settings.factor_group_count,
            rebalance_freq=settings.rebalance_freq,
            price_col=settings.price_col,
            trading_days_per_year=settings.trading_days_per_year,
        )
        if group_return_summary.empty:
            print("【分组收益】跳过: 无有效分组结果\n")
        else:
            factor_level = (
                group_return_summary.sort_values(["factor", "group"])
                .groupby("factor", as_index=False)
                .tail(1)
            )
            for rec in factor_level.to_dict("records"):
                print(
                    "【分组收益】%s  top-bottom=%.4f  ann=%.4f  hit=%.2f%%  monotonicity=%.2f"
                    % (
                        rec["factor"],
                        rec["top_minus_bottom_mean"],
                        rec["top_minus_bottom_ann"],
                        rec["top_minus_bottom_hit_rate"] * 100.0,
                        rec["monotonicity_score"],
                    )
                )
            print()
    except Exception as e:
        print("【分组收益】跳过: %s\n" % e)

    if not ic_distribution.empty:
        print("========== 多因子权重建议（全样本诊断；训练/滚动支路另行使用）==========\n")
        try:
            windows = tuple(int(w) for w in settings.ic_rolling_windows)
            preferred_window = max(windows) if windows else None
            factor_weight_summary = build_factor_weight_summary(
                ic_distribution,
                ic_rolling,
                group_return_summary,
                factors=factor_order,
                preferred_rolling_window=preferred_window,
            )
            if factor_weight_summary.empty:
                print("【因子权重】跳过: 无有效评分输入\n")
            else:
                for rec in factor_weight_summary.to_dict("records"):
                    print(
                        "【因子权重】%s  score=%.4f  weight=%.2f%%  meanIC=%.4f  mono=%.2f"
                        % (
                            rec["factor"],
                            rec["factor_score"],
                            rec["fusion_weight"] * 100.0,
                            rec["mean_ic"],
                            rec["monotonicity_score"],
                        )
                    )
                print()
        except Exception as e:
            print("【因子权重】跳过: %s\n" % e)

    print("========== 样本外验证与因子失效监控 ==========\n")
    try:
        out_of_sample_validation = build_out_of_sample_validation(
            research_panel[factor_order].dropna(axis=1, how="all"),
            prices,
            settings,
            factors=factor_order,
            train_ratio=settings.factor_weight_train_ratio,
        )
        factor_decay_monitor = build_factor_decay_monitor(out_of_sample_validation)
        rolling_out_of_sample_validation = build_rolling_out_of_sample_validation(
            research_panel[factor_order].dropna(axis=1, how="all"),
            prices,
            settings,
            factors=factor_order,
        )
        rolling_out_of_sample_summary = summarize_rolling_out_of_sample_validation(
            rolling_out_of_sample_validation
        )
        if out_of_sample_validation.empty:
            print("【样本外】跳过: 无有效验证结果\n")
        else:
            train_start = out_of_sample_validation["train_start"].iloc[0]
            train_end = out_of_sample_validation["train_end"].iloc[0]
            val_start = out_of_sample_validation["validation_start"].iloc[0]
            val_end = out_of_sample_validation["validation_end"].iloc[0]
            print("【样本外】train=%s ~ %s；validation=%s ~ %s" % (train_start, train_end, val_start, val_end))
            for rec in factor_decay_monitor.to_dict("records"):
                print(
                    "【失效监控】%s  status=%s  valIC=%.4f  valExcess=%.4f  valTopBottom=%.4f  reasons=%s"
                    % (
                        rec["factor"],
                        rec["status"],
                        rec["validation_ic_mean"],
                        rec["validation_excess_ann_return"],
                        rec["validation_top_minus_bottom_ann"],
                        rec["reasons"] or "-",
                    )
                )
            print()
        if not rolling_out_of_sample_summary.empty:
            n_windows = int(rolling_out_of_sample_validation["window_id"].nunique())
            print(
                "【滚动样本外】windows=%d；train_days=%d；validation_days=%d；step_days=%d"
                % (
                    n_windows,
                    int(settings.rolling_oos_train_days),
                    int(settings.rolling_oos_validation_days),
                    int(settings.rolling_oos_step_days),
                )
            )
            for rec in rolling_out_of_sample_summary.head(8).to_dict("records"):
                print(
                    "【滚动样本外】%s  status=%s  stable=%.2f%%  valIC=%.4f  valExcess=%.4f  topBottom=%.4f"
                    % (
                        rec["factor"],
                        rec["status"],
                        float(rec["stable_window_rate"]) * 100.0,
                        float(rec["avg_validation_ic_mean"]),
                        float(rec["avg_validation_excess_ann_return"]),
                        float(rec["avg_validation_top_minus_bottom_ann"]),
                    )
                )
            print()
    except Exception as e:
        print("【样本外】跳过: %s\n" % e)

    print("========== 因子入选与剔除 ==========\n")
    try:
        coverage_frame = data_quality_reports.get("factor_coverage", pd.DataFrame())
        factor_selection_summary = build_factor_selection_table(
            factors=factor_order,
            factor_coverage=coverage_frame,
            factor_weight_summary=factor_weight_summary,
            factor_decay_monitor=factor_decay_monitor,
        )
        fusion_factor_order = selected_factors_for_fusion(factor_selection_summary, factor_order)
        if factor_selection_summary.empty:
            print("【因子准入】跳过: 无有效准入结果，主融合使用原始因子池\n")
        else:
            for rec in factor_selection_summary.to_dict("records"):
                print(
                    "【因子准入】%s  decision=%s  selected=%s  coverage=%.2f%%  score=%.4f  reasons=%s"
                    % (
                        rec["factor"],
                        rec["decision"],
                        rec["selected_for_fusion"],
                        float(rec["coverage"]) * 100.0 if pd.notna(rec["coverage"]) else float("nan"),
                        float(rec["factor_score"]) if pd.notna(rec["factor_score"]) else float("nan"),
                        rec["reasons"] or "-",
                    )
                )
            print("【因子准入】主融合候选池: %s\n" % fusion_factor_order)
    except Exception as e:
        fusion_factor_order = list(factor_order)
        print("【因子准入】跳过: %s；主融合使用原始因子池\n" % e)

    print("========== 因子相关性与冗余分析 ==========\n")
    try:
        factor_correlation_matrix, factor_correlation_days = factor_cross_sectional_correlation(
            research_panel[factor_order].dropna(axis=1, how="all"),
            factors=factor_order,
            method="spearman",
            min_symbols=max(5, int(settings.top_k)),
        )
        factor_redundancy_report = build_factor_redundancy_report(
            factor_correlation_matrix,
            factor_correlation_days,
            selection=factor_selection_summary,
            threshold=0.70,
            min_days=max(20, int(settings.rolling_factor_weight_min_days // 2)),
        )
        before_redundancy = list(fusion_factor_order)
        fusion_factor_order = prune_redundant_factors(fusion_factor_order, factor_redundancy_report)
        if factor_redundancy_report.empty:
            print("【因子冗余】未发现 abs(corr) >= 0.70 且有效天数充足的高相关因子对\n")
        else:
            for rec in factor_redundancy_report.head(10).to_dict("records"):
                print(
                    "【因子冗余】%s vs %s  corr=%.4f  days=%d  keep=%s  drop=%s  reason=%s"
                    % (
                        rec["factor_a"],
                        rec["factor_b"],
                        rec["correlation"],
                        rec["n_days"],
                        rec["recommended_keep"],
                        rec["recommended_drop"],
                        rec["reason"],
                    )
                )
            print("【因子冗余】准入候选池: %s" % before_redundancy)
            print("【因子冗余】去冗余后候选池: %s\n" % fusion_factor_order)
    except Exception as e:
        print("【因子冗余】跳过: %s；主融合沿用准入候选池\n" % e)

    print("========== 因子分层与复合因子体系 ==========\n")
    try:
        factor_composite_scores, factor_composite_components = build_factor_composite_scores(
            research_panel[factor_order].dropna(axis=1, how="all"),
            eligible_factors=fusion_factor_order,
            min_components=1,
        )
        if factor_composite_scores.empty:
            print("【复合因子】跳过: 去冗余候选池无法形成有效风格层，主融合沿用原始候选因子\n")
        else:
            research_panel = pd.concat([research_panel, factor_composite_scores], axis=1)
            raw_fusion_factors = list(fusion_factor_order)
            fusion_factor_order = list(factor_composite_scores.columns)
            for cname in fusion_factor_order:
                try:
                    ic_comp = daily_ic_spearman(
                        factor_composite_scores[cname],
                        prices,
                        forward_days=settings.ic_forward_days,
                    )
                    if ic_comp.dropna().shape[0] > 0:
                        ic_by_name[cname] = ic_comp
                except Exception as ic_exc:
                    print("【复合因子】%s IC 计算跳过: %s" % (cname, ic_exc))
            for rec in factor_composite_components.to_dict("records"):
                print(
                    "【复合因子】%s  components=%s  coverage=%.2f%%"
                    % (
                        rec["composite_factor"],
                        rec["eligible_components"] or "-",
                        float(rec["coverage"]) * 100.0,
                    )
                )
            print("【复合因子】原始去冗余候选池: %s" % raw_fusion_factors)
            print("【复合因子】主融合改用风格层候选池: %s\n" % fusion_factor_order)
    except Exception as e:
        print("【复合因子】跳过: %s；主融合沿用去冗余原始因子池\n" % e)

    try:
        sub_weight = research_panel[fusion_factor_order].dropna(axis=1, how="all")
        if sub_weight.shape[1] > 0:
            (
                factor_weight_train_summary,
                score_weighted_weights_by_factor,
                score_weighted_meta,
            ) = _build_train_factor_weight_summary(sub_weight, prices, settings)
            print("========== 训练段综合权重（用于验证段 FUSED_SCORE_WEIGHTED）==========\n")
            print(
                "【训练/验证】train=%s ~ %s；test=%s ~ %s；候选因子=%s"
                % (
                    pd.Timestamp(score_weighted_meta["train_start"]).strftime("%Y-%m-%d"),
                    pd.Timestamp(score_weighted_meta["train_end"]).strftime("%Y-%m-%d"),
                    pd.Timestamp(score_weighted_meta["test_start"]).strftime("%Y-%m-%d"),
                    pd.Timestamp(score_weighted_meta["test_end"]).strftime("%Y-%m-%d"),
                    list(sub_weight.columns),
                )
            )
            for rec in factor_weight_train_summary.to_dict("records"):
                print(
                    "【训练权重】%s  score=%.4f  weight=%.2f%%"
                    % (
                        rec["factor"],
                        rec["factor_score"],
                        rec["fusion_weight"] * 100.0,
                    )
                )
            print()
    except Exception as e:
        print("【训练段综合权重】跳过: %s\n" % e)

    try:
        sub_rolling = research_panel[fusion_factor_order].dropna(axis=1, how="all")
        if sub_rolling.shape[1] > 0:
            (
                fused_rolling_score_weighted,
                rolling_factor_weight_log,
                rolling_score_weighted_meta,
            ) = _build_rolling_score_weighted_fusion(sub_rolling, prices, settings)
            print("========== 滚动综合权重（用于 FUSED_ROLLING_SCORE_WEIGHTED）==========\n")
            if not rolling_factor_weight_log.empty:
                last_dt = pd.Timestamp(rolling_factor_weight_log["date"].max())
                latest = rolling_factor_weight_log[rolling_factor_weight_log["date"] == last_dt]
                print(
                    "【滚动权重】共 %d 个调仓日；最近一期=%s；候选因子=%s"
                    % (
                        int(rolling_score_weighted_meta.get("rolling_weight_rebalances", 0)),
                        last_dt.strftime("%Y-%m-%d"),
                        list(sub_rolling.columns),
                    )
                )
                for rec in latest.to_dict("records"):
                    print(
                        "  %s  final_weight=%.2f%%  reason=%s"
                        % (
                            rec["factor"],
                            rec["final_weight"] * 100.0,
                            rec["reason"],
                        )
                    )
            print()
    except Exception as e:
        print("【滚动综合权重】跳过: %s\n" % e)

    print("========== 因子权重稳定性与漂移监控 ==========\n")
    try:
        factor_weight_stability = factor_weight_stability_summary(rolling_factor_weight_log)
        factor_weight_drift = factor_weight_drift_events(rolling_factor_weight_log)
        factor_weight_portfolio_drift_frame = factor_weight_portfolio_drift(rolling_factor_weight_log)
        if factor_weight_stability.empty:
            print("【权重漂移】跳过: 无滚动权重日志\n")
        else:
            watch = factor_weight_stability[factor_weight_stability["status"] != "PASS"]
            latest_port = (
                factor_weight_portfolio_drift_frame.sort_values("date").tail(1)
                if not factor_weight_portfolio_drift_frame.empty
                else pd.DataFrame()
            )
            for rec in factor_weight_stability.to_dict("records"):
                print(
                    "【权重稳定】%s  latest=%.2f%%  avg_change=%.2f%%  max_change=%.2f%%  status=%s"
                    % (
                        rec["factor"],
                        float(rec["latest_weight"]) * 100.0,
                        float(rec["avg_abs_change"]) * 100.0,
                        float(rec["max_abs_change"]) * 100.0,
                        rec["status"],
                    )
                )
            if not latest_port.empty:
                rec = latest_port.iloc[0].to_dict()
                print(
                    "【组合权重漂移】最近一期=%s  dominant=%s %.2f%%  effective_factor_n=%.2f  weight_turnover=%.2f%%"
                    % (
                        pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"),
                        rec["dominant_factor"],
                        float(rec["dominant_weight"]) * 100.0,
                        float(rec["effective_factor_n"]),
                        float(rec["weight_turnover"]) * 100.0,
                    )
                )
            if not factor_weight_drift.empty:
                latest_events = factor_weight_drift.sort_values("date").tail(5)
                print("【权重漂移事件】最近事件:")
                for rec in latest_events.to_dict("records"):
                    print(
                        "  %s  %s  %s  change=%.2f%%  severity=%s"
                        % (
                            pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"),
                            rec["factor"],
                            rec["event_type"],
                            float(rec["weight_change"]) * 100.0,
                            rec["severity"],
                        )
                    )
            elif watch.empty:
                print("【权重漂移事件】未发现超过阈值的权重跳变")
            print()
    except Exception as e:
        print("【权重漂移】跳过: %s\n" % e)

    if settings.persist_run_outputs and (
        not long_excess_summary.empty
        or not group_return_detail.empty
        or not group_return_summary.empty
        or not factor_weight_summary.empty
        or not factor_weight_train_summary.empty
        or not rolling_factor_weight_log.empty
        or not factor_weight_stability.empty
        or not factor_weight_drift.empty
        or not factor_weight_portfolio_drift_frame.empty
        or not factor_selection_summary.empty
        or not factor_composite_components.empty
        or not factor_composite_scores.empty
        or not factor_correlation_matrix.empty
        or not factor_redundancy_report.empty
        or not out_of_sample_validation.empty
        or not factor_decay_monitor.empty
        or not rolling_out_of_sample_validation.empty
        or not rolling_out_of_sample_summary.empty
    ):
        try:
            diag_paths = save_factor_diagnostics(
                settings,
                long_excess_summary,
                group_return_detail=group_return_detail,
                group_return_summary=group_return_summary,
                factor_weight_summary=factor_weight_summary,
                factor_weight_train_summary=factor_weight_train_summary,
                rolling_factor_weight_log=rolling_factor_weight_log,
                factor_selection_summary=factor_selection_summary,
                factor_composite_components=factor_composite_components,
                factor_composite_scores=factor_composite_scores,
                factor_correlation_matrix=factor_correlation_matrix,
                factor_correlation_days=factor_correlation_days,
                factor_redundancy_report=factor_redundancy_report,
                factor_weight_stability_summary=factor_weight_stability,
                factor_weight_drift_events=factor_weight_drift,
                factor_weight_portfolio_drift=factor_weight_portfolio_drift_frame,
            )
            print(
                "因子诊断已保存: %s"
                % ", ".join("%s=%s" % (k, v.resolve()) for k, v in diag_paths.items())
            )
            if (
                not out_of_sample_validation.empty
                or not factor_decay_monitor.empty
                or not rolling_out_of_sample_validation.empty
                or not rolling_out_of_sample_summary.empty
            ):
                validation_paths = save_factor_validation_outputs(
                    settings,
                    out_of_sample_validation,
                    factor_decay_monitor,
                    rolling_validation=rolling_out_of_sample_validation,
                    rolling_summary=rolling_out_of_sample_summary,
                )
                print(
                    "样本外验证已保存: %s"
                    % ", ".join("%s=%s" % (k, v.resolve()) for k, v in validation_paths.items())
                )
            print()
        except Exception as e:
            print("因子诊断落盘失败（不影响回测）:", e)
            print()

    nav_curves: dict[str, pd.Series] = {}

    print("\n========== 一、单因子回测（各列独立，横截面 Top-K + 配置持仓权重）==========\n")
    for fname in factor_order:
        if fname not in research_panel.columns:
            print("【因子】%s 跳过: 面板中无该列\n" % fname)
            continue
        col = research_panel[fname]
        if col.notna().sum() == 0:
            print("【因子】%s 跳过: 整列无有效值\n" % fname)
            continue
        try:
            nav, meta = run_single_backtest(
                fname,
                prices=prices,
                settings=settings,
                factor_values=col,
                long_prices=long_df,
            )
            stats = summarize(nav, periods=settings.trading_days_per_year)
            nav_curves[fname] = nav
            backtest_meta_by_name[fname] = meta
            performance_by_name[fname] = stats
            _print_backtest_block("【因子】%s" % fname, nav, meta, stats)
        except Exception as e:
            print("【因子】%s 跳过: %s\n" % (fname, e))

    print("========== 二、多因子融合（IC rolling + 静态综合权重 + 滚动综合权重）==========\n")
    try:
        sub = research_panel[fusion_factor_order].dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print("【融合】跳过: 无有效因子列\n")
            return
        fused, fusion_mode = _build_fused_zscore_panel(sub, ic_by_name, settings)
        nav_f, meta_f = run_multi_backtest(
            fused=fused,
            prices=prices,
            settings=settings,
            factor_name="FUSED_ZSCORE",
            long_prices=long_df,
        )
        stats_f = summarize(nav_f, periods=settings.trading_days_per_year)
        nav_curves["FUSED_ZSCORE"] = nav_f
        backtest_meta_by_name["FUSED_ZSCORE"] = meta_f
        performance_by_name["FUSED_ZSCORE"] = stats_f
        _print_backtest_block(
            "【融合】FUSED_ZSCORE（参与列: %s；融合=%s）" % (list(sub.columns), fusion_mode),
            nav_f,
            meta_f,
            stats_f,
        )
        if score_weighted_weights_by_factor and score_weighted_meta:
            fused_sw_all = fuse_static_weight_zscore(sub, score_weighted_weights_by_factor)
            test_start = pd.Timestamp(score_weighted_meta["test_start"])
            test_end = pd.Timestamp(score_weighted_meta["test_end"])
            fused_sw = fused_sw_all.loc[
                fused_sw_all.index.get_level_values("date") >= test_start
            ]
            prices_test = prices.loc[(prices.index >= test_start) & (prices.index <= test_end)]
            nav_sw, meta_sw = run_multi_backtest(
                fused=fused_sw,
                prices=prices_test,
                settings=settings,
                factor_name="FUSED_SCORE_WEIGHTED",
                long_prices=long_df,
            )
            meta_sw.update(score_weighted_meta)
            meta_sw["fusion_mode"] = "static_score_weighted_train_test"
            stats_sw = summarize(nav_sw, periods=settings.trading_days_per_year)
            nav_curves["FUSED_SCORE_WEIGHTED"] = nav_sw
            backtest_meta_by_name["FUSED_SCORE_WEIGHTED"] = meta_sw
            performance_by_name["FUSED_SCORE_WEIGHTED"] = stats_sw
            _print_backtest_block(
                "【融合】FUSED_SCORE_WEIGHTED（训练段权重；验证段回测）",
                nav_sw,
                meta_sw,
                stats_sw,
            )
        else:
            print("【融合】FUSED_SCORE_WEIGHTED 跳过: 无训练段综合权重\n")
        if not fused_rolling_score_weighted.empty:
            nav_rw, meta_rw = run_multi_backtest(
                fused=fused_rolling_score_weighted,
                prices=prices,
                settings=settings,
                factor_name="FUSED_ROLLING_SCORE_WEIGHTED",
                long_prices=long_df,
            )
            meta_rw.update(rolling_score_weighted_meta)
            stats_rw = summarize(nav_rw, periods=settings.trading_days_per_year)
            nav_curves["FUSED_ROLLING_SCORE_WEIGHTED"] = nav_rw
            backtest_meta_by_name["FUSED_ROLLING_SCORE_WEIGHTED"] = meta_rw
            performance_by_name["FUSED_ROLLING_SCORE_WEIGHTED"] = stats_rw
            _print_backtest_block(
                "【融合】FUSED_ROLLING_SCORE_WEIGHTED（调仓日前滚动综合权重）",
                nav_rw,
                meta_rw,
                stats_rw,
            )
        else:
            print("【融合】FUSED_ROLLING_SCORE_WEIGHTED 跳过: 无滚动综合权重得分\n")
    except Exception as e:
        print("【融合】跳过: %s\n" % e)

    style_exposure = pd.DataFrame()
    style_exposure_summary = pd.DataFrame()
    style_exposure_return = pd.DataFrame()
    if backtest_meta_by_name and not factor_composite_scores.empty:
        try:
            fused_strategy_names = [
                name for name in backtest_meta_by_name.keys() if str(name).startswith("FUSED_")
            ]
            style_exposure = batch_style_exposure(
                backtest_meta_by_name,
                factor_composite_scores,
                strategies=fused_strategy_names,
            )
            style_exposure_summary = summarize_style_exposure(style_exposure)
            style_exposure_return = style_exposure_return_link(style_exposure, nav_curves)
            if not style_exposure_summary.empty:
                print("========== 三、风格层暴露与下一期收益关联 ==========\n")
                for rec in style_exposure_summary.sort_values(
                    ["strategy", "avg_abs_exposure"],
                    ascending=[True, False],
                ).to_dict("records"):
                    print(
                        "【风格暴露】%s/%s  avg=%.4f  latest=%.4f  positive_rate=%.2f%%  coverage=%.2f%%"
                        % (
                            rec["strategy"],
                            rec["style"],
                            rec["avg_exposure"],
                            rec["latest_exposure"],
                            rec["positive_rate"] * 100.0,
                            rec["avg_score_coverage"] * 100.0,
                        )
                    )
                if not style_exposure_return.empty:
                    print()
                    for rec in style_exposure_return.sort_values(
                        ["strategy", "exposure_next_return_corr"],
                        ascending=[True, False],
                    ).to_dict("records"):
                        corr = rec["exposure_next_return_corr"]
                        print(
                            "【风格收益关联】%s/%s  corr=%s  pos_next_ret=%.4f  nonpos_next_ret=%s"
                            % (
                                rec["strategy"],
                                rec["style"],
                                "nan" if pd.isna(corr) else "%.4f" % corr,
                                rec["avg_next_return_when_positive"]
                                if not pd.isna(rec["avg_next_return_when_positive"])
                                else float("nan"),
                                "nan"
                                if pd.isna(rec["avg_next_return_when_nonpositive"])
                                else "%.4f" % rec["avg_next_return_when_nonpositive"],
                            )
                        )
                print()
            if settings.persist_run_outputs and not style_exposure.empty:
                paths = save_style_exposure_outputs(
                    settings,
                    style_exposure,
                    style_exposure_summary,
                    style_exposure_return,
                )
                print(
                    "风格暴露分析已保存: %s\n"
                    % ", ".join("%s=%s" % (k, v.resolve()) for k, v in paths.items())
                )
        except Exception as e:
            print("【风格暴露】跳过: %s\n" % e)

    benchmark_nav: pd.Series | None = None
    market_regime_frame = pd.DataFrame()
    market_regime_days = pd.DataFrame()
    market_regime_detail = pd.DataFrame()
    market_regime_summary = pd.DataFrame()
    if nav_curves:
        try:
            nav_index = pd.DataFrame(nav_curves).index
            benchmark_nav = equal_weight_benchmark_nav(
                prices,
                dates=nav_index,
                price_col=settings.price_col,
            )
            if not benchmark_nav.empty:
                benchmark_stats = summarize(
                    benchmark_nav,
                    periods=settings.trading_days_per_year,
                )
                performance_by_name[benchmark_nav.name] = benchmark_stats
                for name, nav in nav_curves.items():
                    performance_by_name[name].update(
                        summarize_excess(
                            nav,
                            benchmark_nav,
                            periods=settings.trading_days_per_year,
                        )
                    )
                print("========== 四、基准与超额收益 ==========\n")
                print("【基准】%s（股票池每日等权）" % benchmark_nav.name)
                print(
                    "  年化收益: %.4f  年化波动: %.4f  夏普: %.4f  最大回撤: %.4f"
                    % (
                        benchmark_stats["ann_return"],
                        benchmark_stats["ann_vol"],
                        benchmark_stats["sharpe"],
                        benchmark_stats["max_drawdown"],
                    )
                )
                for name in nav_curves:
                    st = performance_by_name[name]
                    print(
                        "【超额】%s  excess_ann_return=%.4f  tracking_error=%.4f  IR=%.4f"
                        % (
                            name,
                            st["excess_ann_return"],
                            st["tracking_error"],
                            st["information_ratio"],
                        )
                    )
                print()
                market_regime_frame = build_market_regime_frame(
                    benchmark_nav,
                    lookback_days=settings.market_regime_lookback_days,
                    bull_return_threshold=settings.market_regime_bull_return_threshold,
                    bear_return_threshold=settings.market_regime_bear_return_threshold,
                    bear_drawdown_threshold=settings.market_regime_bear_drawdown_threshold,
                )
                market_regime_days = summarize_regime_days(market_regime_frame)
                market_regime_detail = strategy_regime_performance(
                    nav_curves,
                    benchmark_nav,
                    market_regime_frame,
                    periods=settings.trading_days_per_year,
                )
                market_regime_summary = summarize_strategy_regime_robustness(
                    market_regime_detail
                )
                if not market_regime_days.empty:
                    print("========== 四A、牛熊市分段表现 ==========\n")
                    print("【市场状态覆盖】")
                    for rec in market_regime_days.to_dict("records"):
                        print(
                            "  %s: %d 天，占比 %.2f%%"
                            % (
                                rec["regime"],
                                int(rec["n_days"]),
                                float(rec["day_ratio"]) * 100.0,
                            )
                        )
                    if not market_regime_summary.empty:
                        print("【策略分段摘要】")
                        for rec in market_regime_summary.head(8).to_dict("records"):
                            print(
                                "  %s status=%s bull_excess=%s bear_excess=%s sideways_excess=%s"
                                % (
                                    rec["strategy"],
                                    rec["status"],
                                    "nan"
                                    if pd.isna(rec["bull_excess_ann_return"])
                                    else "%.4f" % float(rec["bull_excess_ann_return"]),
                                    "nan"
                                    if pd.isna(rec["bear_excess_ann_return"])
                                    else "%.4f" % float(rec["bear_excess_ann_return"]),
                                    "nan"
                                    if pd.isna(rec["sideways_excess_ann_return"])
                                    else "%.4f" % float(rec["sideways_excess_ann_return"]),
                                )
                            )
                    print()
                if settings.persist_run_outputs and not market_regime_frame.empty:
                    paths = save_market_regime_outputs(
                        settings.output_dir / "market_regime",
                        market_regime_frame,
                        market_regime_days,
                        market_regime_detail,
                        market_regime_summary,
                    )
                    print(
                        "牛熊市分段报告已保存: %s\n"
                        % ", ".join("%s=%s" % (k, v.resolve()) for k, v in paths.items())
                    )
            else:
                print("【基准】跳过: 股票池等权基准为空\n")
        except Exception as e:
            print("【基准】跳过: %s\n" % e)

    turnover_by_name: dict[str, pd.DataFrame] = {}
    if backtest_meta_by_name:
        print("========== 五、换手率与交易成本 ==========\n")
        for name, meta in backtest_meta_by_name.items():
            log = meta.get("rebalance_log") or []
            tf = turnover_frame(log, commission_rate=settings.commission_rate)
            turnover_by_name[name] = tf
            st = summarize_turnover(log, commission_rate=settings.commission_rate)
            if name in performance_by_name:
                performance_by_name[name].update(st)
            print(
                "【换手】%s  avg=%.4f  max=%.4f  total=%.4f  estimated_cost=%.6f"
                % (
                    name,
                    st["avg_turnover"],
                    st["max_turnover"],
                    st["total_turnover"],
                    st["estimated_total_cost"],
                )
            )
        print()

    concentration_by_name: dict[str, pd.DataFrame] = {}
    concentration_summary_by_name: dict[str, dict] = {}
    if backtest_meta_by_name:
        print("========== 六、风险暴露与持仓集中度 ==========\n")
        for name, meta in backtest_meta_by_name.items():
            log = meta.get("rebalance_log") or []
            cf = concentration_frame(log)
            concentration_by_name[name] = cf
            st = summarize_concentration(log)
            concentration_summary_by_name[name] = st
            if name in performance_by_name:
                performance_by_name[name].update(st)
            print(
                "【集中度】%s  avg_effective_n=%.2f  min_effective_n=%.2f  avg_top1=%.2f%%  max_hhi=%.4f"
                % (
                    name,
                    st["avg_effective_n"],
                    st["min_effective_n"],
                    st["avg_top1_weight"] * 100.0,
                    st["max_hhi"],
                )
            )
        print()

    if settings.persist_run_outputs and ic_by_name:
        outd = settings.output_dir
        outd.mkdir(parents=True, exist_ok=True)
        try:
            ic_frame = pd.DataFrame(ic_by_name)
            plot_ic(
                ic_frame,
                title="日 IC 对比（Spearman，前瞻 %d 交易日）" % settings.ic_forward_days,
                save_path=outd / "ic_compare.png",
                rolling_window=None,
            )
            for name, ser in ic_by_name.items():
                safe = name.replace("/", "_")
                plot_ic(
                    ser,
                    title="%s 日 IC（前瞻 %d 日）" % (name, settings.ic_forward_days),
                    save_path=outd / ("ic_timeseries_%s.png" % safe),
                    rolling_window=20,
                )
            print("IC 图已保存:", outd / "ic_compare.png", "及各因子 ic_timeseries_*.png")
        except Exception as e:
            print("IC 作图失败（不影响主流程）:", e)

    if settings.persist_run_outputs and backtest_meta_by_name:
        outd = settings.output_dir
        outd.mkdir(parents=True, exist_ok=True)
        try:
            cfg_path = save_run_config(settings)
            perf_path = save_performance_summary(settings, performance_by_name)
            log_paths = save_rebalance_logs(settings, backtest_meta_by_name)
            decision_paths = save_decision_logs(settings, backtest_meta_by_name)
            turnover_paths = save_turnover_logs(settings, turnover_by_name)
            risk_paths = save_risk_exposure_logs(settings, concentration_by_name)
            risk_summary_path = save_risk_exposure_summary(settings, concentration_summary_by_name)
            print("运行配置已保存:", cfg_path.resolve())
            print("绩效汇总已保存:", perf_path.resolve())
            if log_paths:
                print("调仓日志已保存: %d 份 → 目录 %s" % (len(log_paths), (outd / "rebalance_logs").resolve()))
            if decision_paths:
                print("决策审计日志已保存: %d 份 → 目录 %s" % (len(decision_paths), (outd / "decision_logs").resolve()))
            if turnover_paths:
                print("换手日志已保存: %d 份 → 目录 %s" % (len(turnover_paths), (outd / "turnover_logs").resolve()))
            if risk_paths:
                print(
                    "集中度日志已保存: %d 份 → 目录 %s"
                    % (len(risk_paths), (outd / "risk_exposure" / "concentration_logs").resolve())
                )
                print("集中度汇总已保存:", risk_summary_path.resolve())
        except Exception as e:
            print("实验记录落盘失败（不影响主流程）:", e)
        try:
            n_w = 0
            for name, meta in backtest_meta_by_name.items():
                log = meta.get("rebalance_log") or []
                if not log:
                    continue
                wf = rebalance_log_to_weights_frame(log)
                if wf.empty:
                    continue
                safe = name.replace("/", "_")
                plot_weights(
                    wf,
                    title="%s 再平衡权重（堆叠）" % name,
                    save_path=outd / ("weights_%s.png" % safe),
                    kind="area",
                )
                n_w += 1
            if n_w:
                print("权重堆叠图已保存: %d 张 → 目录 %s" % (n_w, outd.resolve()))
        except Exception as e:
            print("权重作图失败（不影响主流程）:", e)
        try:
            tw = turnover_wide(turnover_by_name)
            if not tw.empty:
                plot_turnover(
                    tw,
                    title="各策略逐期换手率",
                    save_path=outd / "turnover_compare.png",
                )
                print("换手率对比图已保存:", (outd / "turnover_compare.png").resolve())
        except Exception as e:
            print("换手率作图失败（不影响主流程）:", e)
        try:
            en = effective_n_wide(concentration_by_name)
            if not en.empty:
                plot_effective_n(
                    en,
                    title="各策略有效持仓数（1 / HHI）",
                    save_path=outd / "risk_exposure" / "effective_n_compare.png",
                )
                print(
                    "有效持仓数对比图已保存:",
                    (outd / "risk_exposure" / "effective_n_compare.png").resolve(),
                )
        except Exception as e:
            print("集中度作图失败（不影响主流程）:", e)

    if nav_curves:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = settings.output_dir / "nav_compare.png"
        try:
            plot_df = pd.DataFrame(nav_curves)
            if benchmark_nav is not None and not benchmark_nav.empty:
                plot_df[benchmark_nav.name] = benchmark_nav
            plot_nav(
                plot_df,
                title="单因子 / 融合 / 基准（归一化净值，起点≈1）",
                save_path=chart_path,
                normalize=True,
            )
            print("净值对比图已保存:", chart_path.resolve())
        except Exception as e:
            print("作图失败:", e)
        if benchmark_nav is not None and not benchmark_nav.empty:
            excess_chart_path = settings.output_dir / "excess_nav_compare.png"
            try:
                xnav = excess_nav_frame(nav_curves, benchmark_nav)
                if not xnav.empty:
                    plot_nav(
                        xnav,
                        title="相对股票池等权基准的超额净值（起点≈1）",
                        save_path=excess_chart_path,
                        normalize=True,
                    )
                    print("超额净值图已保存:", excess_chart_path.resolve())
            except Exception as e:
                print("超额净值作图失败:", e)


if __name__ == "__main__":
    main()
