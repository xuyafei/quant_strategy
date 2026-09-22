"""
单因子回测。契约：输入因子名或预计算 PanelLong，输出 NavSeries + 元信息。

月末调仓、Top-K 多头、收盘价成交、单边手续费；持仓权重见 config.portfolio_weighting（equal / max_sharpe / risk_parity）。
若 config.max_position_weight / max_industry_weight / target_volatility / min_positions
/ max_rebalance_turnover 可行，会在目标权重生成后限制单票、行业、组合波动、
最低分散度与单次换手。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.backtest_utils import prices_to_wide_close, to_returns, wide_to_long
from config import Settings, get_settings
from factors import FACTOR_REGISTRY
from models.optimizer import maximize_sharpe, risk_parity


def _resample_freq_alias(freq: str) -> str:
    """Pandas 2.2+ 推荐 ME/YE；旧别名 M/Q/A 在此映射。"""
    return {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(freq, freq)


def _rebalance_dates(prices_wide: pd.DataFrame, settings: Any) -> pd.DatetimeIndex:
    """返回调仓日；可选把样本最后一个交易日也纳入调仓日。"""
    rf = _resample_freq_alias(settings.rebalance_freq)
    if prices_wide.empty:
        return pd.DatetimeIndex([])
    price_frame = prices_wide.dropna(how="all").sort_index()
    dates = pd.DatetimeIndex(
        [group.index[-1] for _, group in price_frame.groupby(pd.Grouper(freq=rf)) if not group.empty]
    )
    if bool(getattr(settings, "force_final_rebalance", False)) and len(prices_wide.index) > 0:
        dates = dates.union(pd.DatetimeIndex([prices_wide.index[-1]]))
    return dates.sort_values()


def _portfolio_value(cash: float, shares: pd.Series, px: pd.Series) -> float:
    stock_val = 0.0
    for s in shares.index:
        p = px.get(s)
        if p is not None and np.isfinite(float(p)):
            stock_val += float(shares[s]) * float(p)
    return cash + stock_val


def _rebalance_topk_equal_weight(
    cash: float,
    shares: pd.Series,
    px: pd.Series,
    picks: List[str],
    commission_rate: float,
) -> Tuple[float, pd.Series]:
    """先卖后买；买端若现金不足则按比例缩放。"""
    shares = shares.copy()
    if not picks:
        return cash, shares

    w = 1.0 / len(picks)
    nav0 = _portfolio_value(cash, shares, px)
    tgt_dollar = {s: (nav0 * w if s in picks else 0.0) for s in shares.index}

    cur_dollar = pd.Series(0.0, index=shares.index)
    for s in shares.index:
        p = px.get(s)
        if p is not None and np.isfinite(float(p)):
            cur_dollar[s] = float(shares[s]) * float(p)

    for s in shares.index:
        tgt = tgt_dollar[s]
        cur = float(cur_dollar[s])
        if tgt >= cur - 1e-9:
            continue
        sell_dollar = cur - tgt
        p = px[s]
        if not np.isfinite(float(p)) or sell_dollar <= 0:
            continue
        fee = commission_rate * sell_dollar
        cash += sell_dollar - fee
        shares[s] -= sell_dollar / float(p)
        cur_dollar[s] = float(shares[s]) * float(p) if np.isfinite(float(p)) else 0.0

    buy_list: List[Tuple[str, float]] = []
    for s in picks:
        p = px.get(s)
        if p is None or not np.isfinite(float(p)):
            continue
        tgt = nav0 * w
        cur = float(shares[s]) * float(p)
        need = tgt - cur
        if need > 1e-9:
            buy_list.append((s, need))

    total_with_fee = sum(n * (1.0 + commission_rate) for _, n in buy_list)
    scale = 1.0
    if total_with_fee > cash + 1e-9 and total_with_fee > 0:
        scale = cash / total_with_fee

    for s, need in buy_list:
        p = px[s]
        buy_dollar = need * scale
        if buy_dollar <= 1e-12:
            continue
        fee = commission_rate * buy_dollar
        pay = buy_dollar + fee
        if pay > cash + 1e-9:
            buy_dollar = cash / (1.0 + commission_rate)
            fee = commission_rate * buy_dollar
            pay = buy_dollar + fee
        cash -= pay
        shares[s] += buy_dollar / float(p)

    return cash, shares


def _rebalance_to_target_weights(
    cash: float,
    shares: pd.Series,
    px: pd.Series,
    picks: List[str],
    pick_weights: List[float],
    commission_rate: float,
    valuation_px: pd.Series | None = None,
) -> Tuple[float, pd.Series]:
    """与等权调仓同一撮合逻辑；pick_weights 为目标股票仓位，和小于 1 时剩余保留现金。"""
    shares = shares.copy()
    if not picks or not pick_weights:
        return cash, shares
    if len(picks) != len(pick_weights):
        raise ValueError("picks 与 pick_weights 长度须一致")
    ssum = float(sum(pick_weights))
    if ssum <= 1e-12:
        return cash, shares
    if ssum > 1.0 + 1e-12:
        pw = [float(w) / ssum for w in pick_weights]
    else:
        pw = [max(float(w), 0.0) for w in pick_weights]
    wmap = dict(zip(picks, pw))
    nav0 = _portfolio_value(cash, shares, valuation_px if valuation_px is not None else px)
    tgt_dollar = {s: (nav0 * float(wmap[s]) if s in wmap else 0.0) for s in shares.index}

    cur_dollar = pd.Series(0.0, index=shares.index)
    for s in shares.index:
        p = px.get(s)
        if p is not None and np.isfinite(float(p)):
            cur_dollar[s] = float(shares[s]) * float(p)

    for s in shares.index:
        tgt = tgt_dollar[s]
        cur = float(cur_dollar[s])
        if tgt >= cur - 1e-9:
            continue
        sell_dollar = cur - tgt
        p = px[s]
        if not np.isfinite(float(p)) or sell_dollar <= 0:
            continue
        fee = commission_rate * sell_dollar
        cash += sell_dollar - fee
        shares[s] -= sell_dollar / float(p)
        cur_dollar[s] = float(shares[s]) * float(p) if np.isfinite(float(p)) else 0.0

    buy_list: List[Tuple[str, float]] = []
    for s in picks:
        p = px.get(s)
        if p is None or not np.isfinite(float(p)):
            continue
        tgt = nav0 * float(wmap[s])
        cur = float(shares[s]) * float(p)
        need = tgt - cur
        if need > 1e-9:
            buy_list.append((s, need))

    total_with_fee = sum(n * (1.0 + commission_rate) for _, n in buy_list)
    scale = 1.0
    if total_with_fee > cash + 1e-9 and total_with_fee > 0:
        scale = cash / total_with_fee

    for s, need in buy_list:
        p = px[s]
        buy_dollar = need * scale
        if buy_dollar <= 1e-12:
            continue
        fee = commission_rate * buy_dollar
        pay = buy_dollar + fee
        if pay > cash + 1e-9:
            buy_dollar = cash / (1.0 + commission_rate)
            fee = commission_rate * buy_dollar
            pay = buy_dollar + fee
        cash -= pay
        shares[s] += buy_dollar / float(p)

    return cash, shares


def _apply_max_position_cap(
    weights: List[float],
    max_weight: float,
) -> Tuple[List[float], bool]:
    """
    对权重施加单票上限，并把被裁掉的权重迭代分配给未触顶标的。

    返回 (新权重, 是否发生裁剪)。当 max_weight 不可行（例如 2 只股票上限 40%，总和无法到 100%）
    时返回归一后的原权重并标记未裁剪。
    """
    if not weights:
        return [], False
    arr = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        arr = np.full(len(weights), 1.0 / len(weights), dtype=float)
    else:
        arr = arr / total

    cap = float(max_weight)
    if not np.isfinite(cap) or cap <= 0.0 or cap >= 1.0:
        return arr.tolist(), False
    if cap * len(arr) < 1.0 - 1e-12:
        return arr.tolist(), False

    capped = np.minimum(arr, cap)
    changed = bool(np.any(arr > cap + 1e-12))
    for _ in range(len(arr) + 2):
        gap = 1.0 - float(capped.sum())
        if gap <= 1e-12:
            break
        room = np.maximum(cap - capped, 0.0)
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        add = np.minimum(room, gap * room / room_sum)
        capped += add

    s = float(capped.sum())
    if s <= 1e-12:
        return arr.tolist(), False
    capped = capped / s
    return capped.tolist(), changed or bool(np.any(capped > cap + 1e-9))


def _cap_label(label: str) -> str:
    if label.endswith("_capped"):
        return label
    return "%s_capped" % label


def _target_weight_map(picks: List[str], weights: List[float]) -> Dict[str, float]:
    if not picks or not weights:
        return {}
    arr = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return {}
    out: Dict[str, float] = {}
    for sym, w in zip(picks, arr / total):
        if float(w) <= 1e-12:
            continue
        out[str(sym)] = out.get(str(sym), 0.0) + float(w)
    return out


def _target_weight_lists(
    target: Dict[str, float],
    preferred_order: List[str],
) -> Tuple[List[str], List[float]]:
    ordered: List[str] = []
    seen: set[str] = set()
    for sym in preferred_order:
        ss = str(sym)
        if ss in seen or target.get(ss, 0.0) <= 1e-12:
            continue
        ordered.append(ss)
        seen.add(ss)
    for sym in sorted(target):
        if sym in seen or target.get(sym, 0.0) <= 1e-12:
            continue
        ordered.append(sym)
        seen.add(sym)
    weights = [float(target[s]) for s in ordered]
    return ordered, weights


def _apply_rebalance_turnover_cap(
    prev_target: Dict[str, float],
    target: Dict[str, float],
    max_turnover: float,
) -> Tuple[Dict[str, float], bool, float, float]:
    """
    限制单次目标权重变化：target' = prev + scale * (target - prev)。

    返回 (新目标权重, 是否节流, 原始目标换手, scale)。首次建仓不节流。
    """
    clean_target = {str(k): float(v) for k, v in target.items() if float(v) > 1e-12}
    if not clean_target:
        return {}, False, 0.0, 1.0

    clean_prev = {str(k): float(v) for k, v in prev_target.items() if float(v) > 1e-12}
    symbols = sorted(set(clean_prev) | set(clean_target))
    turnover = float(
        sum(abs(clean_target.get(sym, 0.0) - clean_prev.get(sym, 0.0)) for sym in symbols)
    )

    cap = float(max_turnover)
    if not clean_prev or not np.isfinite(cap) or cap <= 0.0 or turnover <= cap + 1e-12:
        return clean_target, False, turnover, 1.0

    scale = max(0.0, min(1.0, cap / turnover))
    capped: Dict[str, float] = {}
    for sym in symbols:
        w = clean_prev.get(sym, 0.0) + scale * (clean_target.get(sym, 0.0) - clean_prev.get(sym, 0.0))
        if w > 1e-12:
            capped[sym] = float(w)
    total = float(sum(capped.values()))
    return capped, True, turnover, scale


def _apply_min_positions_rule(
    target: Dict[str, float],
    settings: Settings,
) -> tuple[Dict[str, float], dict[str, Any]]:
    min_pos = int(getattr(settings, "min_positions", 0) or 0)
    exposure = float(getattr(settings, "min_positions_exposure", 1.0) or 1.0)
    enabled = min_pos > 0
    clean = {str(k): float(v) for k, v in target.items() if float(v) > 1e-12}
    n_pos = int(len(clean))
    exposure = max(0.0, min(1.0, exposure)) if np.isfinite(exposure) else 1.0
    meta: dict[str, Any] = {
        "min_positions_enabled": bool(enabled),
        "min_positions": min_pos,
        "min_positions_actual": n_pos,
        "min_positions_exposure": exposure,
        "min_positions_applied": False,
        "cash_target_weight": max(0.0, 1.0 - float(sum(clean.values()))),
    }
    if not enabled or not clean or n_pos >= min_pos:
        return clean, meta

    current_sum = float(sum(clean.values()))
    if current_sum <= 1e-12:
        return {}, meta
    target_sum = min(current_sum, exposure)
    scale = max(0.0, min(1.0, target_sum / current_sum))
    scaled = {sym: w * scale for sym, w in clean.items() if w * scale > 1e-12}
    meta["min_positions_applied"] = True
    meta["cash_target_weight"] = max(0.0, 1.0 - float(sum(scaled.values())))
    return scaled, meta


def _apply_gross_exposure_multiplier(
    target: Dict[str, float],
    multiplier: float,
) -> tuple[Dict[str, float], dict[str, Any]]:
    """Scale the whole risky sleeve while leaving the remainder in cash."""
    value = float(multiplier)
    if not np.isfinite(value):
        value = 1.0
    value = max(0.0, min(1.0, value))
    scaled = {
        str(sym): float(weight) * value
        for sym, weight in target.items()
        if float(weight) * value > 1e-12
    }
    return scaled, {
        "gross_exposure_multiplier": value,
        "gross_exposure_scaled": bool(value < 1.0 - 1e-12 and bool(target)),
        "gross_target_weight": float(sum(scaled.values())),
        "cash_target_weight_after_gross": max(0.0, 1.0 - float(sum(scaled.values()))),
    }


def _rebalance_override_for_date(
    overrides: Any,
    dt: pd.Timestamp,
) -> dict[str, Any]:
    """Return an exact-date, precomputed risk override without forward filling."""
    if overrides is None:
        return {}
    if isinstance(overrides, pd.DataFrame):
        frame = overrides
        if "date" in frame.columns:
            frame = frame.set_index("date")
        index = pd.to_datetime(frame.index)
        matches = np.flatnonzero(index == pd.Timestamp(dt))
        if len(matches) == 0:
            return {}
        row = frame.iloc[int(matches[-1])]
        return {str(k): v for k, v in row.items() if pd.notna(v)}
    if isinstance(overrides, Mapping):
        for key in (pd.Timestamp(dt), pd.Timestamp(dt).strftime("%Y-%m-%d")):
            value = overrides.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        return {}
    raise TypeError("rebalance_overrides 须为 DataFrame 或按日期索引的 Mapping")


def _turnover_cap_label(label: str) -> str:
    if label.endswith("_turnover_capped"):
        return label
    return "%s_turnover_capped" % label


def _estimate_mu_cov_for_picks(
    prices_wide: pd.DataFrame,
    picks: List[str],
    end_dt: pd.Timestamp,
    *,
    window: int,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    用 end_dt 当日及之前收盘价，取 [end_dt 往前 window 个交易日] 的日简单收益样本，
    估计 mu（日均收益向量）、cov（样本协方差）；列顺序与 picks 一致。
    """
    if not picks:
        return None
    try:
        sub = prices_wide.loc[:end_dt, picks]
    except (KeyError, TypeError):
        return None
    if sub.shape[1] < len(picks):
        return None
    sub = sub.dropna(axis=0, how="any")
    if len(sub) < 2:
        return None
    tail = sub.iloc[-min(len(sub), window + 1) :]
    rets = tail.pct_change().iloc[1:].dropna(how="any")
    if len(rets) < min_obs:
        return None
    mu = rets.mean().reindex(picks).to_numpy(dtype=float)
    cov = rets[picks].cov().reindex(index=picks, columns=picks).to_numpy(dtype=float)
    if not np.all(np.isfinite(mu)) or mu.shape[0] != len(picks):
        return None
    if cov.shape != (len(picks), len(picks)) or not np.all(np.isfinite(cov)):
        return None
    return mu, cov


def _apply_volatility_target(
    target: Dict[str, float],
    prices_wide: pd.DataFrame,
    dt: pd.Timestamp,
    settings: Settings,
) -> tuple[Dict[str, float], dict[str, Any]]:
    target_vol = float(getattr(settings, "target_volatility", 0.0) or 0.0)
    enabled = bool(np.isfinite(target_vol) and target_vol > 0.0)
    meta: dict[str, Any] = {
        "volatility_target_enabled": enabled,
        "target_volatility": target_vol,
        "portfolio_estimated_volatility": float("nan"),
        "volatility_target_scale": 1.0,
        "cash_target_weight": max(0.0, 1.0 - float(sum(target.values()))),
        "volatility_target_applied": False,
        "volatility_target_missing_data": False,
    }
    if not target:
        return {}, meta
    if not enabled:
        return dict(target), meta

    symbols = list(target)
    est = _estimate_mu_cov_for_picks(
        prices_wide,
        symbols,
        dt,
        window=int(getattr(settings, "volatility_target_lookback_days", 60) or 60),
        min_obs=int(getattr(settings, "volatility_target_min_obs", 20) or 20),
    )
    if est is None:
        meta["volatility_target_missing_data"] = True
        return dict(target), meta
    _, cov = est
    weights = np.asarray([float(target[s]) for s in symbols], dtype=float)
    port_var = float(weights @ cov @ weights)
    if not np.isfinite(port_var) or port_var < 0.0:
        meta["volatility_target_missing_data"] = True
        return dict(target), meta

    ann_vol = float(np.sqrt(max(port_var, 0.0)) * np.sqrt(float(settings.trading_days_per_year)))
    meta["portfolio_estimated_volatility"] = ann_vol
    if ann_vol <= target_vol + 1e-12 or ann_vol <= 1e-12:
        meta["cash_target_weight"] = max(0.0, 1.0 - float(sum(target.values())))
        return dict(target), meta

    scale = max(0.0, min(1.0, target_vol / ann_vol))
    scaled = {sym: float(w) * scale for sym, w in target.items() if float(w) * scale > 1e-12}
    meta["volatility_target_scale"] = float(scale)
    meta["cash_target_weight"] = max(0.0, 1.0 - float(sum(scaled.values())))
    meta["volatility_target_applied"] = True
    return scaled, meta


def _metric_wide_from_long(
    data: Any,
    value_col: str,
) -> pd.DataFrame | None:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    need = {"trade_date", "ts_code", value_col}
    if not need.issubset(data.columns):
        return None
    try:
        out = data.pivot(index="trade_date", columns="ts_code", values=value_col)
    except Exception:
        return None
    out.index = pd.to_datetime(out.index)
    return out.sort_index().sort_index(axis=1)


def _liquidity_wide_frames(
    data: Any,
    prices_wide: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    volume = _metric_wide_from_long(data, "volume")
    amount = _metric_wide_from_long(data, "amount")
    if amount is None:
        amount = _metric_wide_from_long(data, "turnover")
    if amount is None and volume is not None:
        aligned_close = prices_wide.reindex(index=volume.index, columns=volume.columns)
        amount = volume.astype(float) * aligned_close.astype(float)
    return volume, amount


def _bool_wide_from_long(
    data: Any,
    candidates: tuple[str, ...],
) -> pd.DataFrame | None:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    for col in candidates:
        frame = _metric_wide_from_long(data, col)
        if frame is not None:
            return frame.astype(bool)
    return None


def _trade_status_wide_frames(
    data: Any,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    suspended = _bool_wide_from_long(data, ("is_suspended", "suspended"))
    limit_up = _bool_wide_from_long(data, ("is_limit_up", "limit_up"))
    limit_down = _bool_wide_from_long(data, ("is_limit_down", "limit_down"))
    return suspended, limit_up, limit_down


def _industry_wide_from_long(
    data: Any,
    industry_col: str,
) -> pd.DataFrame | None:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    need = {"trade_date", "ts_code", industry_col}
    if not need.issubset(data.columns):
        return None
    try:
        out = data.pivot(index="trade_date", columns="ts_code", values=industry_col)
    except Exception:
        return None
    out.index = pd.to_datetime(out.index)
    return out.sort_index().sort_index(axis=1)


def _industry_map_for_symbols(
    symbols: List[str],
    dt: pd.Timestamp,
    settings: Settings,
    industry_wide: pd.DataFrame | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    max_weight = float(getattr(settings, "max_industry_weight", 0.0) or 0.0)
    enabled = max_weight > 0.0 and max_weight < 1.0
    tech_weight = float(getattr(settings, "max_tech_growth_weight", 0.0) or 0.0)
    tech_enabled = tech_weight > 0.0 and tech_weight < 1.0
    needs_industry = enabled or tech_enabled
    meta = {
        "industry_cap_enabled": bool(enabled),
        "max_industry_weight": max_weight,
        "industry_missing_data": bool(needs_industry and industry_wide is None),
    }
    out: dict[str, str] = {}
    if not needs_industry:
        return out, meta
    for sym in symbols:
        ss = str(sym)
        industry = "UNKNOWN"
        if industry_wide is not None and ss in industry_wide.columns:
            try:
                hist = industry_wide.loc[industry_wide.index <= dt, ss].dropna()
            except KeyError:
                hist = pd.Series(dtype=object)
            if not hist.empty:
                val = hist.iloc[-1]
                if pd.notna(val) and str(val).strip():
                    industry = str(val).strip()
        out[ss] = industry
    return out, meta


def _apply_industry_weight_cap(
    target: Dict[str, float],
    industry_by_symbol: dict[str, str],
    max_industry_weight: float,
    max_position_weight: float = 0.0,
) -> tuple[Dict[str, float], bool, dict[str, float]]:
    if not target:
        return {}, False, {}
    clean = {str(k): max(float(v), 0.0) for k, v in target.items() if float(v) > 1e-12}
    total = float(sum(clean.values()))
    if total <= 1e-12:
        return {}, False, {}
    target_total = min(total, 1.0)
    weights = {k: v / total * target_total for k, v in clean.items()}
    cap = float(max_industry_weight)
    if not np.isfinite(cap) or cap <= 0.0 or cap >= 1.0:
        exposure = _industry_exposure(weights, industry_by_symbol)
        return weights, False, exposure

    industries = {industry_by_symbol.get(sym, "UNKNOWN") for sym in weights}
    position_cap = float(max_position_weight)
    if not np.isfinite(position_cap) or position_cap <= 0.0 or position_cap >= 1.0:
        position_cap = 1.0

    # 先裁行业和单票，再只向同时具有“行业余量 + 单票余量”的标的回填。
    # 若约束组合不可行，剩余部分保留现金，绝不通过最终归一化重新突破上限。
    capped = {sym: min(weight, position_cap) for sym, weight in weights.items()}
    exposure = _industry_exposure(capped, industry_by_symbol)
    for ind, exp in exposure.items():
        if exp <= cap + 1e-12:
            continue
        scale = cap / exp
        for sym in capped:
            if industry_by_symbol.get(sym, "UNKNOWN") == ind:
                capped[sym] *= scale

    exposure = _industry_exposure(capped, industry_by_symbol)
    gap = max(target_total - float(sum(capped.values())), 0.0)
    room_by_symbol = {sym: max(position_cap - weight, 0.0) for sym, weight in capped.items()}
    symbols_by_industry: dict[str, list[str]] = {ind: [] for ind in industries}
    for sym in capped:
        symbols_by_industry[industry_by_symbol.get(sym, "UNKNOWN")].append(sym)

    room_by_industry: dict[str, float] = {}
    for ind, symbols_in_industry in symbols_by_industry.items():
        industry_room = max(cap - exposure.get(ind, 0.0), 0.0)
        position_room = float(sum(room_by_symbol[sym] for sym in symbols_in_industry))
        room_by_industry[ind] = min(industry_room, position_room)

    total_room = float(sum(room_by_industry.values()))
    distributable = min(gap, total_room)
    if distributable > 1e-12 and total_room > 1e-12:
        for ind, industry_room in room_by_industry.items():
            if industry_room <= 1e-12:
                continue
            industry_add = distributable * industry_room / total_room
            symbols_in_industry = symbols_by_industry[ind]
            symbol_room_total = float(sum(room_by_symbol[sym] for sym in symbols_in_industry))
            if symbol_room_total <= 1e-12:
                continue
            for sym in symbols_in_industry:
                capped[sym] += industry_add * room_by_symbol[sym] / symbol_room_total

    capped = {k: v for k, v in capped.items() if v > 1e-12}
    changed = any(abs(capped.get(sym, 0.0) - weight) > 1e-12 for sym, weight in weights.items())
    exposure = _industry_exposure(capped, industry_by_symbol)
    return capped, changed, exposure


def _industry_exposure(
    target: Dict[str, float],
    industry_by_symbol: dict[str, str],
) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for sym, weight in target.items():
        ind = industry_by_symbol.get(str(sym), "UNKNOWN")
        exposure[ind] = exposure.get(ind, 0.0) + float(weight)
    return exposure


def _apply_aggregate_exposure_cap(
    target: Dict[str, float],
    in_group_by_symbol: dict[str, bool],
    max_group_weight: float,
    *,
    max_position_weight: float = 0.0,
    industry_by_symbol: dict[str, str] | None = None,
    max_industry_weight: float = 0.0,
) -> tuple[Dict[str, float], bool, float]:
    """Cap one cross-industry risk bucket and redistribute only into feasible non-group room."""
    if not target:
        return {}, False, 0.0
    weights = {
        str(sym): max(float(weight), 0.0)
        for sym, weight in target.items()
        if np.isfinite(float(weight)) and float(weight) > 1e-12
    }
    cap = float(max_group_weight)
    exposure = float(sum(w for sym, w in weights.items() if in_group_by_symbol.get(sym, False)))
    if not np.isfinite(cap) or cap <= 0.0 or cap >= 1.0 or exposure <= cap + 1e-12:
        return weights, False, exposure

    before_total = float(sum(weights.values()))
    scale = cap / exposure
    for sym in list(weights):
        if in_group_by_symbol.get(sym, False):
            weights[sym] *= scale

    position_cap = float(max_position_weight)
    if not np.isfinite(position_cap) or position_cap <= 0.0 or position_cap >= 1.0:
        position_cap = 1.0
    industry_cap = float(max_industry_weight)
    industry_enabled = bool(np.isfinite(industry_cap) and 0.0 < industry_cap < 1.0)
    industries = industry_by_symbol or {}

    # 将裁掉的风险预算回填给已入选的非科技股票；若单票/行业约束不可行则保留现金。
    gap = max(before_total - float(sum(weights.values())), 0.0)
    for _ in range(len(weights) + 2):
        if gap <= 1e-12:
            break
        industry_exposure = _industry_exposure(weights, industries)
        room: dict[str, float] = {}
        for sym, weight in weights.items():
            if in_group_by_symbol.get(sym, False):
                continue
            available = max(position_cap - weight, 0.0)
            if industry_enabled:
                ind = industries.get(sym, "UNKNOWN")
                available = min(
                    available,
                    max(industry_cap - industry_exposure.get(ind, 0.0), 0.0),
                )
            if available > 1e-12:
                room[sym] = available
        total_room = float(sum(room.values()))
        if total_room <= 1e-12:
            break
        distributable = min(gap, total_room)
        base_total = float(sum(weights[sym] for sym in room))
        for sym, available in room.items():
            basis = weights[sym] / base_total if base_total > 1e-12 else available / total_room
            weights[sym] += min(distributable * basis, available)
        new_gap = max(before_total - float(sum(weights.values())), 0.0)
        if new_gap >= gap - 1e-12:
            break
        gap = new_gap

    exposure = float(sum(w for sym, w in weights.items() if in_group_by_symbol.get(sym, False)))
    return {sym: w for sym, w in weights.items() if w > 1e-12}, True, exposure


def _status_value(frame: pd.DataFrame | None, dt: pd.Timestamp, sym: str) -> bool:
    if frame is None or sym not in frame.columns:
        return False
    try:
        val = frame.loc[dt, sym]
    except KeyError:
        return False
    if pd.isna(val):
        return False
    return bool(val)


def _trade_status_for_symbols(
    symbols: List[str],
    dt: pd.Timestamp,
    settings: Settings,
    suspended_wide: pd.DataFrame | None,
    limit_up_wide: pd.DataFrame | None,
    limit_down_wide: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    enabled = bool(getattr(settings, "enable_trade_status_filter", False))
    meta = {
        "trade_status_filter_enabled": bool(enabled),
        "trade_status_missing_data": bool(
            enabled and suspended_wide is None and limit_up_wide is None and limit_down_wide is None
        ),
    }
    status: dict[str, dict[str, bool]] = {}
    for sym in symbols:
        ss = str(sym)
        status[ss] = {
            "is_suspended": _status_value(suspended_wide, dt, ss) if enabled else False,
            "is_limit_up": _status_value(limit_up_wide, dt, ss) if enabled else False,
            "is_limit_down": _status_value(limit_down_wide, dt, ss) if enabled else False,
        }
    return status, meta


def _trade_block_reason(
    prev_weight: float,
    target_weight: float,
    status: dict[str, bool],
    enabled: bool,
) -> str:
    if not enabled:
        return ""
    eps = 1e-9
    if bool(status.get("is_suspended", False)) and abs(target_weight - prev_weight) > eps:
        return "blocked_by_suspension"
    if bool(status.get("is_limit_up", False)) and target_weight > prev_weight + eps:
        return "blocked_by_limit_up"
    if bool(status.get("is_limit_down", False)) and target_weight < prev_weight - eps:
        return "blocked_by_limit_down"
    return ""


def _apply_trade_status_constraints(
    prev_target: Dict[str, float],
    target: Dict[str, float],
    status_by_symbol: dict[str, dict[str, bool]],
    enabled: bool,
) -> tuple[Dict[str, float], dict[str, str], bool]:
    clean_target = {str(k): float(v) for k, v in target.items() if float(v) > 1e-12}
    clean_prev = {str(k): float(v) for k, v in prev_target.items() if float(v) > 1e-12}
    if not enabled or not clean_target:
        return clean_target, {}, False

    symbols = sorted(set(clean_prev) | set(clean_target))
    fixed: dict[str, float] = {}
    flexible: dict[str, float] = {}
    blocked: dict[str, str] = {}
    for sym in symbols:
        prev_w = float(clean_prev.get(sym, 0.0))
        tgt_w = float(clean_target.get(sym, 0.0))
        reason = _trade_block_reason(prev_w, tgt_w, status_by_symbol.get(sym, {}), enabled)
        if reason:
            blocked[sym] = reason
            if prev_w > 1e-12:
                fixed[sym] = prev_w
        elif tgt_w > 1e-12:
            flexible[sym] = tgt_w

    fixed_sum = float(sum(fixed.values()))
    target_sum = min(1.0, max(float(sum(clean_target.values())), fixed_sum))
    if fixed_sum >= target_sum - 1e-12:
        return ({k: v for k, v in fixed.items() if v > 1e-12}, blocked, bool(blocked))

    flex_sum = float(sum(flexible.values()))
    out = dict(fixed)
    if flex_sum > 1e-12:
        remain = max(target_sum - fixed_sum, 0.0)
        for sym, w in flexible.items():
            out[sym] = float(w) / flex_sum * remain
    return ({k: v for k, v in out.items() if v > 1e-12}, blocked, bool(blocked))


def _liquidity_filter_symbols(
    symbols: List[str],
    dt: pd.Timestamp,
    settings: Settings,
    volume_wide: pd.DataFrame | None,
    amount_wide: pd.DataFrame | None,
) -> tuple[List[str], dict[str, Any]]:
    lookback = int(getattr(settings, "liquidity_lookback_days", 20) or 20)
    min_vol = float(getattr(settings, "min_avg_volume", 0.0) or 0.0)
    min_amt = float(getattr(settings, "min_avg_amount", 0.0) or 0.0)
    enabled = min_vol > 0.0 or min_amt > 0.0
    meta = {
        "liquidity_filter_enabled": bool(enabled),
        "liquidity_lookback_days": lookback,
        "min_avg_volume": min_vol,
        "min_avg_amount": min_amt,
        "liquidity_missing_data": False,
    }
    if not enabled:
        return list(symbols), meta
    if (min_vol > 0.0 and volume_wide is None) or (min_amt > 0.0 and amount_wide is None):
        meta["liquidity_missing_data"] = True
        return [], meta

    passed: list[str] = []
    for sym in symbols:
        ok = True
        if min_vol > 0.0 and volume_wide is not None:
            if sym not in volume_wide.columns:
                ok = False
            else:
                v = volume_wide.loc[volume_wide.index <= dt, sym].tail(max(1, lookback))
                avg_v = float(v.dropna().mean()) if not v.dropna().empty else float("nan")
                ok = ok and np.isfinite(avg_v) and avg_v >= min_vol
        if min_amt > 0.0 and amount_wide is not None:
            if sym not in amount_wide.columns:
                ok = False
            else:
                a = amount_wide.loc[amount_wide.index <= dt, sym].tail(max(1, lookback))
                avg_a = float(a.dropna().mean()) if not a.dropna().empty else float("nan")
                ok = ok and np.isfinite(avg_a) and avg_a >= min_amt
        if ok:
            passed.append(sym)
    return passed, meta


def _decision_action(previous_weight: float, final_weight: float) -> str:
    eps = 1e-9
    if previous_weight <= eps and final_weight > eps:
        return "buy"
    if previous_weight > eps and final_weight <= eps:
        return "sell"
    if final_weight > previous_weight + eps:
        return "increase"
    if final_weight < previous_weight - eps:
        return "decrease"
    if final_weight > eps:
        return "hold"
    return "skip"


def _decision_reason(
    *,
    in_candidate: bool,
    passed_liquidity: bool,
    selected: bool,
    previous_weight: float,
    raw_target_weight: float,
    final_target_weight: float,
    weighting: str,
    turnover_capped: bool,
    liquidity_enabled: bool,
    trade_block_reason: str,
    industry_cap_adjusted: bool,
    tech_growth_cap_adjusted: bool,
    volatility_target_scaled: bool,
    min_positions_scaled: bool,
) -> str:
    eps = 1e-9
    if in_candidate and liquidity_enabled and not passed_liquidity:
        return "filtered_by_liquidity"
    if selected:
        reasons = ["selected_topk"]
        if "fallback" in weighting:
            reasons.append("optimizer_fallback_equal_weight")
        if "_capped" in weighting.replace("_turnover_capped", ""):
            reasons.append("position_cap_applied")
        if turnover_capped and abs(final_target_weight - raw_target_weight) > eps:
            reasons.append("turnover_cap_adjusted")
        if industry_cap_adjusted and abs(final_target_weight - raw_target_weight) > eps:
            reasons.append("industry_cap_adjusted")
        if tech_growth_cap_adjusted and abs(final_target_weight - raw_target_weight) > eps:
            reasons.append("tech_growth_cap_adjusted")
        if volatility_target_scaled and abs(final_target_weight - raw_target_weight) > eps:
            reasons.append("volatility_target_scaled")
        if min_positions_scaled and abs(final_target_weight - raw_target_weight) > eps:
            reasons.append("min_positions_scaled")
        if trade_block_reason:
            reasons.append(trade_block_reason)
        return "|".join(reasons)
    if trade_block_reason:
        return trade_block_reason
    if raw_target_weight <= eps and final_target_weight > eps and previous_weight > eps and turnover_capped:
        return "not_selected_but_retained_by_turnover_cap"
    if in_candidate:
        return "not_selected_outside_topk"
    if previous_weight > eps and final_target_weight <= eps:
        return "previous_holding_exited_not_in_valid_candidates"
    if previous_weight > eps:
        return "previous_holding_retained_by_turnover_cap"
    return "not_in_valid_candidates"


def _build_decision_records(
    *,
    dt: pd.Timestamp,
    scores: pd.Series,
    candidate_symbols: List[str],
    eligible_symbols: List[str],
    selected_picks: List[str],
    raw_target_map: Dict[str, float],
    final_target_map: Dict[str, float],
    previous_target_map: Dict[str, float],
    weighting: str,
    turnover_capped: bool,
    liquidity_meta: dict[str, Any],
    industry_target_map: Dict[str, float] | None = None,
    tech_growth_target_map: Dict[str, float] | None = None,
    tech_growth_meta: dict[str, Any] | None = None,
    volatility_target_map: Dict[str, float] | None = None,
    volatility_target_meta: dict[str, Any] | None = None,
    min_positions_target_map: Dict[str, float] | None = None,
    min_positions_meta: dict[str, Any] | None = None,
    trade_status_by_symbol: dict[str, dict[str, bool]] | None = None,
    trade_block_reasons: dict[str, str] | None = None,
    trade_status_meta: dict[str, Any] | None = None,
    industry_by_symbol: dict[str, str] | None = None,
    industry_cap_meta: dict[str, Any] | None = None,
    industry_cap_applied: bool = False,
) -> List[Dict[str, Any]]:
    candidate_rank = {str(sym): i + 1 for i, sym in enumerate(candidate_symbols)}
    eligible_set = {str(sym) for sym in eligible_symbols}
    selected_rank = {str(sym): i + 1 for i, sym in enumerate(selected_picks)}
    all_symbols = (
        list(candidate_symbols)
        + [s for s in previous_target_map if s not in candidate_rank]
        + [s for s in final_target_map if s not in candidate_rank and s not in previous_target_map]
    )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    liquidity_enabled = bool(liquidity_meta.get("liquidity_filter_enabled", False))
    trade_status_by_symbol = trade_status_by_symbol or {}
    trade_block_reasons = trade_block_reasons or {}
    trade_status_meta = trade_status_meta or {
        "trade_status_filter_enabled": False,
        "trade_status_missing_data": False,
    }
    industry_target_map = industry_target_map or {}
    tech_growth_target_map = tech_growth_target_map or {}
    tech_growth_meta = tech_growth_meta or {
        "tech_growth_cap_enabled": False,
        "max_tech_growth_weight": 0.0,
        "tech_growth_exposure": 0.0,
        "tech_growth_cap_applied": False,
    }
    volatility_target_map = volatility_target_map or {}
    volatility_target_meta = volatility_target_meta or {
        "volatility_target_enabled": False,
        "target_volatility": 0.0,
        "portfolio_estimated_volatility": float("nan"),
        "volatility_target_scale": 1.0,
        "cash_target_weight": 0.0,
        "volatility_target_applied": False,
        "volatility_target_missing_data": False,
    }
    min_positions_target_map = min_positions_target_map or {}
    min_positions_meta = min_positions_meta or {
        "min_positions_enabled": False,
        "min_positions": 0,
        "min_positions_actual": 0,
        "min_positions_exposure": 1.0,
        "min_positions_applied": False,
        "cash_target_weight": float(volatility_target_meta.get("cash_target_weight", 0.0) or 0.0),
    }
    industry_by_symbol = industry_by_symbol or {}
    industry_cap_meta = industry_cap_meta or {
        "industry_cap_enabled": False,
        "max_industry_weight": 0.0,
        "industry_missing_data": False,
    }
    for sym in all_symbols:
        ss = str(sym)
        if ss in seen:
            continue
        seen.add(ss)
        prev_w = float(previous_target_map.get(ss, 0.0) or 0.0)
        raw_w = float(raw_target_map.get(ss, 0.0) or 0.0)
        industry_w = float(industry_target_map.get(ss, raw_w) or 0.0)
        tech_w = float(tech_growth_target_map.get(ss, industry_w) or 0.0)
        vol_w = float(volatility_target_map.get(ss, tech_w) or 0.0)
        minpos_w = float(min_positions_target_map.get(ss, vol_w) or 0.0)
        final_w = float(final_target_map.get(ss, 0.0) or 0.0)
        in_candidate = ss in candidate_rank
        passed_liquidity = ss in eligible_set if in_candidate else False
        selected = ss in selected_rank
        score = scores.get(ss, np.nan)
        status = trade_status_by_symbol.get(ss, {})
        trade_block_reason = trade_block_reasons.get(ss, "")
        rows.append(
            {
                "date": dt,
                "symbol": ss,
                "factor_score": float(score) if np.isfinite(float(score)) else float("nan"),
                "factor_rank": candidate_rank.get(ss, ""),
                "passed_liquidity_filter": bool(passed_liquidity),
                "selected_by_signal": bool(selected),
                "selected_rank": selected_rank.get(ss, ""),
                "previous_weight": prev_w,
                "raw_target_weight": raw_w,
                "final_target_weight": final_w,
                "weighting": weighting,
                "turnover_capped": bool(turnover_capped),
                "is_suspended": bool(status.get("is_suspended", False)),
                "is_limit_up": bool(status.get("is_limit_up", False)),
                "is_limit_down": bool(status.get("is_limit_down", False)),
                "trade_blocked": bool(trade_block_reason),
                "trade_block_reason": trade_block_reason,
                "industry": industry_by_symbol.get(ss, ""),
                "is_tech_growth": bool(tech_growth_meta.get("tech_growth_cap_enabled", False))
                and bool(tech_growth_meta.get("tech_growth_members", {}).get(ss, False)),
                "industry_cap_applied": bool(industry_cap_applied),
                "action": _decision_action(prev_w, final_w),
                "decision_reason": _decision_reason(
                    in_candidate=in_candidate,
                    passed_liquidity=passed_liquidity,
                    selected=selected,
                    previous_weight=prev_w,
                    raw_target_weight=raw_w,
                    final_target_weight=final_w,
                    weighting=weighting,
                    turnover_capped=turnover_capped,
                    liquidity_enabled=liquidity_enabled,
                    trade_block_reason=trade_block_reason,
                    industry_cap_adjusted=bool(industry_cap_applied and abs(industry_w - raw_w) > 1e-9),
                    tech_growth_cap_adjusted=bool(
                        tech_growth_meta.get("tech_growth_cap_applied", False)
                        and abs(tech_w - industry_w) > 1e-9
                    ),
                    volatility_target_scaled=bool(
                        volatility_target_meta.get("volatility_target_applied", False)
                        and abs(vol_w - tech_w) > 1e-9
                    ),
                    min_positions_scaled=bool(
                        min_positions_meta.get("min_positions_applied", False)
                        and abs(minpos_w - vol_w) > 1e-9
                    ),
                ),
                "n_candidates_before_liquidity": int(len(candidate_symbols)),
                "n_candidates_after_liquidity": int(len(eligible_symbols)),
                **liquidity_meta,
                **trade_status_meta,
                **industry_cap_meta,
                **{k: v for k, v in tech_growth_meta.items() if k != "tech_growth_members"},
                **volatility_target_meta,
                **min_positions_meta,
            }
        )
    return rows


def _weights_for_rebalance(
    prices_wide: pd.DataFrame,
    picks: List[str],
    dt: pd.Timestamp,
    settings: Settings,
) -> Tuple[List[float], str]:
    """
    返回与 picks 对齐的权重列表（和为 1）及模式标签：
    equal / max_sharpe / max_sharpe_fallback / risk_parity / risk_parity_fallback。
    """
    n = len(picks)
    if n == 0:
        return [], "equal"
    eq = [1.0 / n] * n
    max_pos = float(getattr(settings, "max_position_weight", 0.0) or 0.0)
    mode = str(getattr(settings, "portfolio_weighting", "equal") or "equal").strip().lower()
    if n < 2 or mode == "equal":
        w, capped = _apply_max_position_cap(eq, max_pos)
        return w, _cap_label("equal") if capped else "equal"

    win = int(getattr(settings, "optimizer_return_window", 60))
    min_obs = int(getattr(settings, "optimizer_min_obs", 15))

    if mode == "max_sharpe":
        est = _estimate_mu_cov_for_picks(prices_wide, picks, dt, window=win, min_obs=min_obs)
        if est is None:
            return eq, "max_sharpe_fallback"
        mu, cov = est
        try:
            w = maximize_sharpe(mu, cov, risk_free=0.0)
        except Exception:
            return eq, "max_sharpe_fallback"
        w = np.maximum(np.asarray(w, dtype=float), 0.0)
        s = float(w.sum())
        if s <= 1e-12:
            return eq, "max_sharpe_fallback"
        w = (w / s).tolist()
        if len(w) != n:
            return eq, "max_sharpe_fallback"
        w, capped = _apply_max_position_cap(w, max_pos)
        return w, _cap_label("max_sharpe") if capped else "max_sharpe"

    if mode == "risk_parity":
        est = _estimate_mu_cov_for_picks(prices_wide, picks, dt, window=win, min_obs=min_obs)
        if est is None:
            return eq, "risk_parity_fallback"
        _, cov = est
        try:
            w = np.asarray(risk_parity(cov), dtype=float)
        except Exception:
            return eq, "risk_parity_fallback"
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 1e-12:
            return eq, "risk_parity_fallback"
        w = (w / s).tolist()
        if len(w) != n:
            return eq, "risk_parity_fallback"
        w, capped = _apply_max_position_cap(w, max_pos)
        return w, _cap_label("risk_parity") if capped else "risk_parity"

    w, capped = _apply_max_position_cap(eq, max_pos)
    return w, _cap_label("equal") if capped else "equal"


def run_single_backtest(
    factor_name: str,
    *,
    factor_values: Optional[pd.Series] = None,
    prices: Optional[pd.DataFrame] = None,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    lookback: Optional[int] = None,
    **kwargs: Any,
) -> tuple[pd.Series, Dict[str, Any]]:
    """
    :param factor_name: FACTOR_REGISTRY 中的键；若已传 factor_values 则仅用于 meta 记录
    :param factor_values: MultiIndex(date, symbol) 因子
    :param prices: 收盘价宽表（日期索引 × ts_code 列）或契约长表
    :param top_k: 多头只数，默认取 settings.top_k
    :param lookback: 动量窗口（仅当自动计算 MOMENTUM 时）
    :return: (nav, meta)；nav 索引为 date，值为净值
    """
    if prices is None:
        raise ValueError("run_single_backtest 需要 prices")

    settings = settings or get_settings()
    k = int(top_k if top_k is not None else settings.top_k)
    if k < 1:
        raise ValueError("top_k 须 >= 1")

    prices_wide = prices_to_wide_close(
        prices,
        date_col="trade_date",
        symbol_col="ts_code",
        close_col=settings.price_col,
    )
    prices_wide = prices_wide.sort_index().sort_index(axis=1)
    # 停牌或临时缺报价不能让持仓市值凭空归零；估值使用最近可得价格，交易仍使用
    # 当日原始价格，因此缺报价标的不会在陈旧价格上被买卖。
    valuation_prices_wide = prices_wide.ffill()
    liquidity_data = kwargs.get("liquidity_data")
    if liquidity_data is None:
        liquidity_data = kwargs.get("long_prices")
    volume_wide, amount_wide = _liquidity_wide_frames(liquidity_data, prices_wide)
    trade_status_data = kwargs.get("trade_status_data")
    if trade_status_data is None:
        trade_status_data = kwargs.get("long_prices")
    suspended_wide, limit_up_wide, limit_down_wide = _trade_status_wide_frames(trade_status_data)
    industry_data = kwargs.get("industry_data")
    if industry_data is None:
        industry_data = kwargs.get("long_prices")
    industry_wide = _industry_wide_from_long(
        industry_data,
        str(getattr(settings, "industry_col", "industry") or "industry"),
    )
    empty_signal_policy = str(kwargs.get("empty_signal_policy", "hold") or "hold").strip().lower()
    if empty_signal_policy not in {"hold", "cash"}:
        raise ValueError("empty_signal_policy 须为 hold 或 cash")
    rebalance_overrides = kwargs.get("rebalance_overrides")

    if factor_values is None:
        if factor_name not in FACTOR_REGISTRY:
            raise KeyError(f"未知因子: {factor_name}，可注册于 factors.FACTOR_REGISTRY")
        fn = FACTOR_REGISTRY[factor_name]
        if factor_name in ("MOMENTUM", "MOMENTUM_60D"):
            default_lb = (
                getattr(settings, "momentum_long_lookback", 60)
                if factor_name == "MOMENTUM_60D"
                else settings.momentum_lookback
            )
            lb = int(lookback if lookback is not None else default_lb)
            factor_values = fn(prices_wide, lookback=lb)
        elif factor_name == "REVERSAL_5D":
            lb = int(lookback if lookback is not None else getattr(settings, "reversal_lookback", 5))
            factor_values = fn(prices_wide, lookback=lb)
        elif factor_name == "VOLATILITY":
            ret = to_returns(prices_wide)
            vw = int(kwargs.get("vol_window") or settings.vol_window)
            factor_values = fn(
                ret,
                window=vw,
                annualize=True,
                trading_days=settings.trading_days_per_year,
            )
        elif factor_name == "VOLUME_RATIO_20D":
            long_px = kwargs.get("long_prices")
            if long_px is None or "volume" not in getattr(long_px, "columns", []):
                raise ValueError("VOLUME_RATIO_20D 需要 long_prices 且含 volume 列")
            vol_wide = long_px.pivot(index="trade_date", columns="ts_code", values="volume")
            factor_values = fn(
                vol_wide,
                window=int(kwargs.get("volume_ratio_window") or getattr(settings, "volume_ratio_window", 20)),
            )
        elif factor_name in ("PE", "ROE"):
            long_px = kwargs.get("long_prices")
            if long_px is None:
                long_px = wide_to_long(prices_wide, settings.price_col)
            fina = kwargs.get("finance_df")
            if fina is None or getattr(fina, "empty", True):
                from live.data_feed import fetch_fina_indicator_panel

                fina = fetch_fina_indicator_panel(
                    list(prices_wide.columns),
                    settings.backtest_start,
                    settings.backtest_end,
                    history_years=settings.fina_history_years,
                    token=kwargs.get("token"),
                )
            if fina.empty:
                raise ValueError(
                    "财务数据为空，无法计算 PE/ROE（检查 Tushare 积分/权限或 finance_df）"
                )
            factor_values = fn(fina, long_px)
        else:
            raise ValueError(f"因子 {factor_name} 需预计算 factor_values 或未支持自动计算")

    factor_values = factor_values.copy()
    factor_values.index = factor_values.index.set_names(["date", "symbol"])

    rebalance_dates = _rebalance_dates(prices_wide, settings)

    symbols = list(prices_wide.columns)
    shares = pd.Series(0.0, index=symbols)
    cash = 1.0
    nav_records: List[Tuple[pd.Timestamp, float]] = []
    n_rebalances = 0
    rebalance_log: List[Dict[str, Any]] = []
    decision_log: List[Dict[str, Any]] = []
    prev_target_weights: Dict[str, float] = {}

    for dt in prices_wide.index:
        px = prices_wide.loc[dt]
        valuation_px = valuation_prices_wide.loc[dt]
        nav = _portfolio_value(cash, shares, valuation_px)
        nav_records.append((dt, nav))

        if dt not in rebalance_dates:
            continue

        try:
            sc = factor_values.xs(dt, level=0)
        except KeyError:
            sc = pd.Series(dtype=float)
        sc = sc.dropna()
        if sc.empty:
            if empty_signal_policy == "cash" and prev_target_weights:
                cash, shares = _rebalance_to_target_weights(
                    cash,
                    shares,
                    px,
                    [],
                    [],
                    settings.commission_rate,
                    valuation_px=valuation_px,
                )
                rebalance_log.append(
                    {
                        "date": dt,
                        "picks": [],
                        "selected_picks": [],
                        "weights": [],
                        "weighting": "empty_signal_cash",
                        "target_turnover": float(sum(abs(x) for x in prev_target_weights.values())),
                        "turnover_capped": False,
                        "turnover_scale": 1.0,
                        "n_candidates_before_liquidity": 0,
                        "n_candidates_after_liquidity": 0,
                        "n_trade_blocked": 0,
                        "industry_cap_applied": False,
                        "max_industry_exposure": 0.0,
                        "n_industries": 0,
                        "cash_target_weight": 1.0,
                        "empty_signal_policy": empty_signal_policy,
                    }
                )
                prev_target_weights = {}
                n_rebalances += 1
            continue
        sc = sc.sort_values(ascending=False)
        risk_override = _rebalance_override_for_date(rebalance_overrides, pd.Timestamp(dt))
        override_tech_cap = float(
            risk_override.get(
                "max_tech_growth_weight",
                getattr(settings, "max_tech_growth_weight", 0.0) or 0.0,
            )
        )
        gross_exposure_multiplier = float(
            risk_override.get("gross_exposure_multiplier", 1.0)
        )
        effective_settings = replace(settings, max_tech_growth_weight=override_tech_cap)
        risk_override_meta = {
            "risk_state": str(risk_override.get("risk_state", "STATIC")),
            "risk_override_applied": bool(risk_override),
        }
        candidate_symbols: list[str] = []
        for sym in sc.index:
            if sym not in px.index:
                continue
            if not np.isfinite(float(px[sym])):
                continue
            candidate_symbols.append(str(sym))
        eligible_symbols, liquidity_meta = _liquidity_filter_symbols(
            candidate_symbols,
            pd.Timestamp(dt),
            settings,
            volume_wide,
            amount_wide,
        )
        status_symbols = list(dict.fromkeys(candidate_symbols + list(prev_target_weights.keys())))
        industry_by_symbol, industry_cap_meta = _industry_map_for_symbols(
            status_symbols,
            pd.Timestamp(dt),
            effective_settings,
            industry_wide,
        )
        tech_growth_industries = {
            str(value).strip()
            for value in getattr(effective_settings, "tech_growth_industries", ())
            if str(value).strip()
        }
        tech_growth_members = {
            sym: industry_by_symbol.get(sym, "") in tech_growth_industries
            for sym in status_symbols
        }
        max_tech_growth_weight = override_tech_cap
        tech_growth_meta: dict[str, Any] = {
            "tech_growth_cap_enabled": bool(0.0 < max_tech_growth_weight < 1.0),
            "max_tech_growth_weight": max_tech_growth_weight,
            "tech_growth_exposure": 0.0,
            "tech_growth_cap_applied": False,
            "tech_growth_members": tech_growth_members,
        }
        trade_status_by_symbol, trade_status_meta = _trade_status_for_symbols(
            status_symbols,
            pd.Timestamp(dt),
            effective_settings,
            suspended_wide,
            limit_up_wide,
            limit_down_wide,
        )
        eligible_set = set(eligible_symbols)
        picks = []
        for sym in sc.index:
            ss = str(sym)
            if ss not in eligible_set:
                continue
            picks.append(ss)
            if len(picks) >= k:
                break
        if not picks:
            if candidate_symbols:
                rebalance_log.append(
                    {
                        "date": dt,
                        "picks": [],
                        "selected_picks": [],
                        "weights": [],
                        "weighting": "no_trade_liquidity",
                        "target_turnover": 0.0,
                        "turnover_capped": False,
                        "turnover_scale": 1.0,
                        "n_candidates_before_liquidity": int(len(candidate_symbols)),
                        "n_candidates_after_liquidity": int(len(eligible_symbols)),
                        **liquidity_meta,
                        **trade_status_meta,
                        **industry_cap_meta,
                        **{k: v for k, v in tech_growth_meta.items() if k != "tech_growth_members"},
                    }
                )
                decision_log.extend(
                    _build_decision_records(
                        dt=pd.Timestamp(dt),
                        scores=sc,
                        candidate_symbols=candidate_symbols,
                        eligible_symbols=eligible_symbols,
                        selected_picks=[],
                        raw_target_map={},
                        final_target_map=prev_target_weights,
                        previous_target_map=prev_target_weights,
                        weighting="no_trade_liquidity",
                        turnover_capped=False,
                        liquidity_meta=liquidity_meta,
                        trade_status_by_symbol=trade_status_by_symbol,
                        trade_block_reasons={},
                        trade_status_meta=trade_status_meta,
                        industry_by_symbol=industry_by_symbol,
                        industry_cap_meta=industry_cap_meta,
                        industry_cap_applied=False,
                        tech_growth_meta=tech_growth_meta,
                    )
                )
            continue

        selected_picks = list(picks)
        pick_weights, w_label = _weights_for_rebalance(
            prices_wide, selected_picks, dt, effective_settings
        )
        raw_target_map = _target_weight_map(selected_picks, pick_weights)
        industry_target_map, industry_cap_applied, industry_exposure = _apply_industry_weight_cap(
            raw_target_map,
            industry_by_symbol,
            float(getattr(effective_settings, "max_industry_weight", 0.0) or 0.0),
            float(getattr(effective_settings, "max_position_weight", 0.0) or 0.0),
        )
        tech_growth_target_map, tech_growth_cap_applied, tech_growth_exposure = (
            _apply_aggregate_exposure_cap(
                industry_target_map,
                tech_growth_members,
                max_tech_growth_weight,
                max_position_weight=float(
                    getattr(effective_settings, "max_position_weight", 0.0) or 0.0
                ),
                industry_by_symbol=industry_by_symbol,
                max_industry_weight=float(
                    getattr(effective_settings, "max_industry_weight", 0.0) or 0.0
                ),
            )
        )
        industry_exposure = _industry_exposure(tech_growth_target_map, industry_by_symbol)
        tech_growth_meta.update(
            {
                "tech_growth_exposure": float(tech_growth_exposure),
                "tech_growth_cap_applied": bool(tech_growth_cap_applied),
            }
        )
        volatility_target_map, volatility_target_meta = _apply_volatility_target(
            tech_growth_target_map,
            prices_wide,
            pd.Timestamp(dt),
            effective_settings,
        )
        min_positions_target_map, min_positions_meta = _apply_min_positions_rule(
            volatility_target_map,
            effective_settings,
        )
        gross_target_map, gross_exposure_meta = _apply_gross_exposure_multiplier(
            min_positions_target_map,
            gross_exposure_multiplier,
        )
        target_map, turnover_capped, target_turnover, turnover_scale = _apply_rebalance_turnover_cap(
            prev_target_weights,
            gross_target_map,
            float(getattr(effective_settings, "max_rebalance_turnover", 0.0) or 0.0),
        )
        if turnover_capped:
            w_label = _turnover_cap_label(w_label)
        target_map, trade_block_reasons, trade_blocked = _apply_trade_status_constraints(
            prev_target_weights,
            target_map,
            trade_status_by_symbol,
            bool(getattr(settings, "enable_trade_status_filter", False)),
        )
        target_picks, target_weights = _target_weight_lists(
            target_map,
            selected_picks + list(prev_target_weights.keys()),
        )
        if not target_picks:
            if selected_picks:
                rebalance_log.append(
                    {
                        "date": dt,
                        "picks": [],
                        "selected_picks": list(selected_picks),
                        "weights": [],
                        "weighting": "no_trade_status",
                        "target_turnover": float(target_turnover),
                        "turnover_capped": bool(turnover_capped),
                        "turnover_scale": float(turnover_scale),
                        "n_candidates_before_liquidity": int(len(candidate_symbols)),
                        "n_candidates_after_liquidity": int(len(eligible_symbols)),
                        "n_trade_blocked": int(len(trade_block_reasons)),
                        "industry_cap_applied": bool(industry_cap_applied),
                        "max_industry_exposure": float(max(industry_exposure.values())) if industry_exposure else 0.0,
                        "n_industries": int(len(industry_exposure)),
                        **liquidity_meta,
                        **trade_status_meta,
                        **industry_cap_meta,
                        **{k: v for k, v in tech_growth_meta.items() if k != "tech_growth_members"},
                        **volatility_target_meta,
                        **min_positions_meta,
                        **gross_exposure_meta,
                        **risk_override_meta,
                        "cash_target_weight": 1.0,
                    }
                )
                decision_log.extend(
                    _build_decision_records(
                        dt=pd.Timestamp(dt),
                        scores=sc,
                        candidate_symbols=candidate_symbols,
                        eligible_symbols=eligible_symbols,
                        selected_picks=selected_picks,
                        raw_target_map=raw_target_map,
                        industry_target_map=industry_target_map,
                        tech_growth_target_map=tech_growth_target_map,
                        tech_growth_meta=tech_growth_meta,
                        volatility_target_map=volatility_target_map,
                        volatility_target_meta=volatility_target_meta,
                        min_positions_target_map=min_positions_target_map,
                        min_positions_meta=min_positions_meta,
                        final_target_map={},
                        previous_target_map=prev_target_weights,
                        weighting="no_trade_status",
                        turnover_capped=turnover_capped,
                        liquidity_meta=liquidity_meta,
                        trade_status_by_symbol=trade_status_by_symbol,
                        trade_block_reasons=trade_block_reasons,
                        trade_status_meta=trade_status_meta,
                        industry_by_symbol=industry_by_symbol,
                        industry_cap_meta=industry_cap_meta,
                        industry_cap_applied=industry_cap_applied,
                    )
                )
            continue

        cash, shares = _rebalance_to_target_weights(
            cash,
            shares,
            px,
            target_picks,
            target_weights,
            settings.commission_rate,
            valuation_px=valuation_px,
        )

        rebalance_log.append(
            {
                "date": dt,
                "picks": list(target_picks),
                "selected_picks": list(selected_picks),
                "weights": [float(x) for x in target_weights],
                "weighting": w_label,
                "target_turnover": float(target_turnover),
                "turnover_capped": bool(turnover_capped),
                "turnover_scale": float(turnover_scale),
                "n_candidates_before_liquidity": int(len(candidate_symbols)),
                "n_candidates_after_liquidity": int(len(eligible_symbols)),
                "n_trade_blocked": int(len(trade_block_reasons)),
                "industry_cap_applied": bool(industry_cap_applied),
                "max_industry_exposure": float(max(industry_exposure.values())) if industry_exposure else 0.0,
                "n_industries": int(len(industry_exposure)),
                **liquidity_meta,
                **trade_status_meta,
                **industry_cap_meta,
                **{k: v for k, v in tech_growth_meta.items() if k != "tech_growth_members"},
                **volatility_target_meta,
                **min_positions_meta,
                **gross_exposure_meta,
                **risk_override_meta,
                "cash_target_weight": max(0.0, 1.0 - float(sum(target_weights))),
            }
        )
        final_target_map = dict(zip(target_picks, [float(x) for x in target_weights]))
        decision_log.extend(
            _build_decision_records(
                dt=pd.Timestamp(dt),
                scores=sc,
                candidate_symbols=candidate_symbols,
                eligible_symbols=eligible_symbols,
                selected_picks=selected_picks,
                raw_target_map=raw_target_map,
                industry_target_map=industry_target_map,
                tech_growth_target_map=tech_growth_target_map,
                tech_growth_meta=tech_growth_meta,
                volatility_target_map=volatility_target_map,
                volatility_target_meta=volatility_target_meta,
                min_positions_target_map=min_positions_target_map,
                min_positions_meta=min_positions_meta,
                final_target_map=final_target_map,
                previous_target_map=prev_target_weights,
                weighting=w_label,
                turnover_capped=turnover_capped,
                liquidity_meta=liquidity_meta,
                trade_status_by_symbol=trade_status_by_symbol,
                trade_block_reasons=trade_block_reasons,
                trade_status_meta=trade_status_meta,
                industry_by_symbol=industry_by_symbol,
                industry_cap_meta=industry_cap_meta,
                industry_cap_applied=industry_cap_applied,
            )
        )
        prev_target_weights = final_target_map
        n_rebalances += 1

    nav = pd.Series(
        [v for _, v in nav_records],
        index=pd.DatetimeIndex([d for d, _ in nav_records], name="date"),
        name="nav",
        dtype=float,
    )
    meta: Dict[str, Any] = {
        "factor_name": factor_name,
        "top_k": k,
        "rebalance_freq": settings.rebalance_freq,
        "commission_rate": settings.commission_rate,
        "n_rebalances": n_rebalances,
        "portfolio_weighting": getattr(settings, "portfolio_weighting", "equal"),
        "max_position_weight": getattr(settings, "max_position_weight", 0.0),
        "max_rebalance_turnover": getattr(settings, "max_rebalance_turnover", 0.0),
        "enable_trade_status_filter": getattr(settings, "enable_trade_status_filter", False),
        "max_industry_weight": getattr(settings, "max_industry_weight", 0.0),
        "max_tech_growth_weight": getattr(settings, "max_tech_growth_weight", 0.0),
        "tech_growth_industries": list(getattr(settings, "tech_growth_industries", ())),
        "target_volatility": getattr(settings, "target_volatility", 0.0),
        "min_positions": getattr(settings, "min_positions", 0),
        "empty_signal_policy": empty_signal_policy,
        "rebalance_log": rebalance_log,
        "decision_log": decision_log,
    }
    return nav, meta
