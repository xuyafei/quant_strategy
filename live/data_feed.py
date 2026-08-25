"""
行情数据接入：Tushare / AkShare 与本地 CSV。输出列名需符合 docs/INTERFACE_AND_CONTRACTS.md §2.1。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from config import get_tushare_token
from storage.price_adjustment import add_adjusted_close


_REQUIRED_OHLCV = {"open", "high", "low", "close"}


def get_data_tushare(
    symbol: str,
    start: str,
    end: str,
    *,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    拉取单日频 OHLCV。列: trade_date, ts_code, open, high, low, close, volume
    start/end 建议 YYYYMMDD 或 YYYY-MM-DD（将规范为 YYYYMMDD）。
    """
    try:
        import tushare as ts
    except ImportError as e:
        raise ImportError("需要安装 tushare: pip install tushare") from e

    tok = (token or "").strip() or get_tushare_token()
    pro = ts.pro_api(tok)

    def _norm(d: str) -> str:
        return d.replace("-", "")[:8]

    df = pro.daily(ts_code=symbol, start_date=_norm(start), end_date=_norm(end))
    if df.empty:
        return df
    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "volume" not in df.columns:
        raise ValueError("Tushare daily 返回缺少 volume/vol")
    for c in _REQUIRED_OHLCV:
        if c not in df.columns:
            raise ValueError(f"Tushare daily 返回缺少 {c!r}")
    return df[
        ["trade_date", "ts_code", "open", "high", "low", "close", "volume"]
    ].reset_index(drop=True)


def fetch_fina_indicator_panel(
    symbols: List[str],
    start: str,
    end: str,
    *,
    history_years: int = 2,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    批量拉取上市公司财务指标（fina_indicator），用于 PE/ROE 等因子。
    向前多取 history_years 年数据，便于 merge_asof 在样本期初也有可用财报。
    须含列：ts_code, ann_date, eps, roe（Tushare 默认字段名）。
    """
    try:
        import tushare as ts
    except ImportError as e:
        raise ImportError("需要安装 tushare: pip install tushare") from e

    tok = (token or "").strip() or get_tushare_token()
    pro = ts.pro_api(tok)

    def _norm(d: str) -> str:
        return d.replace("-", "")[:8]

    end_s = _norm(end)
    start_dt = pd.to_datetime(_norm(start), format="%Y%m%d") - pd.DateOffset(
        years=int(history_years)
    )
    start_s = start_dt.strftime("%Y%m%d")

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = pro.fina_indicator(ts_code=sym, start_date=start_s, end_date=end_s)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "ann_date" in out.columns:
        out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    return out.sort_values(["ts_code", "ann_date"]).reset_index(drop=True)


def load_fina_indicator_from_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    从本地 CSV 加载 Tushare fina_indicator 缓存。

    统一 `ann_date` 为 datetime，并按 `ts_code/ann_date` 排序。这个缓存用于把财务数据
    和行情缓存解耦：回测可以复用同一份财务快照，避免每次主流程都现场请求 Tushare。
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "ann_date" in df.columns:
        df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    if "ts_code" in df.columns and "ann_date" in df.columns:
        df = df.sort_values(["ts_code", "ann_date"]).reset_index(drop=True)
    return df


def fetch_daily_panel(
    symbols: List[str],
    start: str,
    end: str,
    *,
    token: Optional[str] = None,
    include_adj_factor: bool = False,
    adjustment_mode: str = "qfq",
) -> pd.DataFrame:
    """
    批量拉取日行情，合并为一张长表（含 trade_date, ts_code, OHLCV）。
    跳过拉取结果为空的标的。
    """
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = get_data_tushare(sym, start, end, token=token)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError("所有标的均无日线数据，请检查 ts_code、日期区间与积分权限")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if include_adj_factor:
        adj = fetch_adj_factor_panel(symbols, start, end, token=token)
        out = merge_adj_factor(out, adj, adjustment_mode=adjustment_mode)
    return out


def get_adj_factor_tushare(
    symbol: str,
    start: str,
    end: str,
    *,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """拉取单只股票的 Tushare 复权因子。"""
    try:
        import tushare as ts
    except ImportError as e:
        raise ImportError("需要安装 tushare: pip install tushare") from e

    tok = (token or "").strip() or get_tushare_token()
    pro = ts.pro_api(tok)

    def _norm(d: str) -> str:
        return d.replace("-", "")[:8]

    df = pro.adj_factor(ts_code=symbol, start_date=_norm(start), end_date=_norm(end))
    if df is None or df.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "adj_factor"])
    required = {"trade_date", "ts_code", "adj_factor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Tushare adj_factor 返回缺少列: %s" % ", ".join(sorted(missing)))
    out = df[["trade_date", "ts_code", "adj_factor"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    return out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def fetch_adj_factor_panel(
    symbols: List[str],
    start: str,
    end: str,
    *,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """批量拉取复权因子长表。"""
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = get_adj_factor_tushare(sym, start, end, token=token)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "adj_factor"])
    return pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def merge_adj_factor(
    prices: pd.DataFrame,
    adj_factor: pd.DataFrame,
    *,
    adjustment_mode: str = "qfq",
) -> pd.DataFrame:
    """把复权因子合并到日线行情，并生成研究用 `adj_close`。"""
    if prices.empty:
        return prices.copy()
    if adj_factor.empty:
        out = prices.copy()
        if "adj_factor" not in out.columns:
            out["adj_factor"] = pd.NA
        if "adj_close" not in out.columns:
            out["adj_close"] = pd.NA
        return out
    left = prices.copy()
    right = adj_factor[["trade_date", "ts_code", "adj_factor"]].copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"], errors="coerce")
    right["trade_date"] = pd.to_datetime(right["trade_date"], errors="coerce")
    left["ts_code"] = left["ts_code"].astype(str).str.strip()
    right["ts_code"] = right["ts_code"].astype(str).str.strip()
    merged = left.merge(right, on=["trade_date", "ts_code"], how="left")
    return add_adjusted_close(merged, mode=adjustment_mode)


def load_prices_from_csv(
    path: Union[str, Path],
    *,
    parse_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    从单个 CSV 加载行情；统一 trade_date 为 datetime，vol -> volume。
    必需列: trade_date, ts_code, open, high, low, close, volume 或 vol
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    parse_dates = parse_dates or ["trade_date"]
    df = pd.read_csv(path, parse_dates=parse_dates)
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    need = {"trade_date", "ts_code", "volume"} | _REQUIRED_OHLCV
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df
