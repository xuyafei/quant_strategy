"""
截面 IC：因子在交易日 t 的取值 vs 前瞻收盘收益（避免与「仅用 ≤t 信息」的因子重叠同一根价格 bar）。

工程约定（与机构常见口径一致，可改 `config.ic_forward_days`）：
- 前瞻收益 r：close(t+h)/close(t) - 1，h 默认 1（亦可 5、20 等）；若策略为月度调仓，可另选
  「持有至下次调仓日收益」以与回测对齐，本模块当前实现固定 h 日。
- 信息集：因子列在 (t, symbol) 处仅使用 ≤t 已可得数据（见各因子实现）；IC 用 t 日因子对齐
  从 t 到 t+h 的收益，收益起点为 t 收盘、终点为 t+h 收盘，不包含 t 之后才能知道的因子修订。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import Settings


def forward_returns_long(
    prices_wide: pd.DataFrame,
    *,
    forward_days: int = 1,
) -> pd.Series:
    """
    宽表收盘价 -> 长表前瞻简单收益：r_{t,s} = close_{t+h}/close_{t,s} - 1。
    """
    if forward_days < 1:
        raise ValueError("forward_days 须 >= 1")
    px = prices_wide.sort_index().sort_index(axis=1).astype(float)
    ret = px.shift(-forward_days) / px - 1.0
    s = ret.stack()
    s.index.names = ["date", "symbol"]
    return s


def daily_ic_spearman(
    factor: pd.Series,
    prices_wide: pd.DataFrame,
    *,
    forward_days: int = 1,
    min_names: int = 3,
) -> pd.Series:
    """
    每个交易日截面上，因子与未来收益的 Spearman 相关系数（秩相关，对极端值更稳健）。

    :param factor: MultiIndex (date, symbol)，与 prices 列名（标的）一致。
    :param min_names: 当日有效 (因子, 收益) 对少于该数则当日 IC 记为 NaN。
    :return: 日频 Series，索引为 date，名称为 ic。
    """
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
        raise TypeError("factor 须为二级 MultiIndex (date, symbol)")
    fac = factor.astype(float).copy()
    fac.index = fac.index.set_names(["date", "symbol"])
    fwd = forward_returns_long(prices_wide, forward_days=forward_days)
    df = pd.concat([fac.rename("f"), fwd.rename("r")], axis=1, join="inner")
    all_dates = pd.DatetimeIndex(df.index.get_level_values(0).unique()).sort_values()
    clean = df[["f", "r"]].dropna(how="any")
    if clean.empty:
        return pd.Series(np.nan, index=all_dates, name="ic", dtype=float)

    # Spearman is Pearson correlation after within-date average ranking.  Doing the
    # rank, centering and reduction in vectorized groupby operations avoids a
    # Python/scipy call for every trading day while preserving tie semantics.
    grouped = clean.groupby(level=0, sort=False)
    ranked = grouped[["f", "r"]].rank(method="average")
    centered = ranked - ranked.groupby(level=0, sort=False).transform("mean")
    numerator = (centered["f"] * centered["r"]).groupby(level=0, sort=False).sum()
    denominator = np.sqrt(
        centered["f"].pow(2).groupby(level=0, sort=False).sum()
        * centered["r"].pow(2).groupby(level=0, sort=False).sum()
    )
    correlation = numerator / denominator.replace(0.0, np.nan)
    counts = grouped.size()
    correlation = correlation.where(counts >= int(min_names))
    return correlation.reindex(all_dates).astype(float).rename("ic")


def summarize_ic(ic: pd.Series) -> Dict[str, Any]:
    """均值、标准差、IC_IR（均值/标准差）、胜率、有效日数。"""
    v = ic.dropna().astype(float)
    if v.empty:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "ic_ir": np.nan,
            "hit_rate": np.nan,
            "n_days": 0,
        }
    m = float(v.mean())
    s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    ir = (m / s) if s > 0 else np.nan
    hr = float((v > 0).mean())
    return {
        "mean_ic": m,
        "std_ic": s,
        "ic_ir": ir,
        "hit_rate": hr,
        "n_days": int(len(v)),
    }


def ic_distribution_summary(ic_by_name: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    汇总各因子的 IC 分布。

    相比 `summarize_ic` 的均值视角，本函数补充分位数、正负占比和极端值，
    用于判断 IC 是否由少数日期支撑，或是否长期正负摇摆。
    """
    rows: list[dict[str, Any]] = []
    for name, ser in ic_by_name.items():
        v = ser.dropna().astype(float)
        base: dict[str, Any] = {"factor": str(name)}
        if v.empty:
            base.update(
                {
                    "n_days": 0,
                    "mean_ic": np.nan,
                    "std_ic": np.nan,
                    "ic_ir": np.nan,
                    "positive_rate": np.nan,
                    "negative_rate": np.nan,
                    "zero_rate": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "min_ic": np.nan,
                    "max_ic": np.nan,
                    "abs_mean_ic": np.nan,
                }
            )
            rows.append(base)
            continue
        st = summarize_ic(v)
        qs = v.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        base.update(
            {
                "n_days": st["n_days"],
                "mean_ic": st["mean_ic"],
                "std_ic": st["std_ic"],
                "ic_ir": st["ic_ir"],
                "positive_rate": float((v > 0).mean()),
                "negative_rate": float((v < 0).mean()),
                "zero_rate": float((v == 0).mean()),
                "p05": float(qs.loc[0.05]),
                "p25": float(qs.loc[0.25]),
                "median": float(qs.loc[0.50]),
                "p75": float(qs.loc[0.75]),
                "p95": float(qs.loc[0.95]),
                "min_ic": float(v.min()),
                "max_ic": float(v.max()),
                "abs_mean_ic": float(v.abs().mean()),
            }
        )
        rows.append(base)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("factor").reset_index(drop=True)
    return out


def ic_rolling_stability(
    ic_by_name: Dict[str, pd.Series],
    *,
    windows: tuple[int, ...] | list[int] = (20, 60),
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    汇总各因子的滚动 IC 稳定性。

    对每个窗口输出 rolling mean 的均值、末值、正值占比，以及 rolling std 的均值。
    这些字段用于观察因子近期是否衰减、是否长期保持同向。
    """
    rows: list[dict[str, Any]] = []
    clean_windows = [int(w) for w in windows if int(w) > 0]
    for name, ser in ic_by_name.items():
        v = ser.dropna().astype(float).sort_index()
        for win in clean_windows:
            mp = int(min_periods) if min_periods is not None else max(2, min(win, max(len(v), 1)))
            row: dict[str, Any] = {
                "factor": str(name),
                "window": win,
                "min_periods": mp,
                "n_days": int(len(v)),
            }
            if v.empty or len(v) < mp:
                row.update(
                    {
                        "rolling_mean_last": np.nan,
                        "rolling_mean_avg": np.nan,
                        "rolling_mean_std": np.nan,
                        "rolling_mean_positive_rate": np.nan,
                        "rolling_std_avg": np.nan,
                        "rolling_ir_last": np.nan,
                    }
                )
                rows.append(row)
                continue
            rmean = v.rolling(win, min_periods=mp).mean().dropna()
            rstd = v.rolling(win, min_periods=mp).std().dropna()
            last_mean = float(rmean.iloc[-1]) if not rmean.empty else np.nan
            last_std = float(rstd.iloc[-1]) if not rstd.empty else np.nan
            row.update(
                {
                    "rolling_mean_last": last_mean,
                    "rolling_mean_avg": float(rmean.mean()) if not rmean.empty else np.nan,
                    "rolling_mean_std": float(rmean.std(ddof=1)) if len(rmean) > 1 else 0.0,
                    "rolling_mean_positive_rate": float((rmean > 0).mean()) if not rmean.empty else np.nan,
                    "rolling_std_avg": float(rstd.mean()) if not rstd.empty else np.nan,
                    "rolling_ir_last": float(last_mean / last_std) if np.isfinite(last_std) and last_std > 0 else np.nan,
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["factor", "window"]).reset_index(drop=True)
    return out


def save_ic_diagnostics(
    settings: Settings,
    distribution_summary: pd.DataFrame,
    rolling_stability: pd.DataFrame,
) -> Dict[str, Path]:
    """将 IC 分布与滚动稳定性诊断写入 output/ic_diagnostics/。"""
    base = settings.output_dir / "ic_diagnostics"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    p_dist = base / "ic_distribution_summary.csv"
    distribution_summary.to_csv(p_dist, index=False)
    out["ic_distribution_summary"] = p_dist

    p_roll = base / "ic_rolling_stability.csv"
    rolling_stability.to_csv(p_roll, index=False)
    out["ic_rolling_stability"] = p_roll

    return out


def save_ic_series(settings: Settings, ic_by_name: Dict[str, pd.Series]) -> Dict[str, Path]:
    """将各因子/融合的日 IC 写入 output/cache/ic_<name>.csv。"""
    base = settings.output_dir / "cache"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, ser in ic_by_name.items():
        safe = name.replace("/", "_")
        p = base / ("ic_%s.csv" % safe)
        ser.to_csv(p, date_format="%Y-%m-%d", header=True)
        out[name] = p
    return out
