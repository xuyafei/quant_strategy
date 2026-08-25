"""SQLite 数据仓库读写与缓存导出。"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from backtest.backtest_utils import long_to_wide
from storage.database import connect_database, get_table_columns, initialize_database
from storage.price_adjustment import add_adjusted_close


def _date_text(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def _datetime_text(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")


def _clean_symbol_list(symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return []
    return [str(s).strip() for s in symbols if str(s).strip()]


def _upsert_frame(
    database: str | Path,
    table: str,
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
) -> int:
    """按主键 upsert DataFrame，返回写入行数。"""
    if frame.empty:
        return 0
    initialize_database(database)
    table_columns = get_table_columns(database, table)
    columns = [col for col in table_columns if col in frame.columns and col != "updated_at"]
    if not columns:
        return 0

    data = frame[columns].copy()
    data = data.where(pd.notna(data), None)
    records = list(data.itertuples(index=False, name=None))
    if not records:
        return 0

    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join(columns)
    key_sql = ", ".join(key_columns)
    update_columns = [col for col in columns if col not in set(key_columns)]
    update_sql = ", ".join("%s = excluded.%s" % (col, col) for col in update_columns)
    if "updated_at" in table_columns:
        update_sql = (update_sql + ", " if update_sql else "") + "updated_at = CURRENT_TIMESTAMP"
    sql = (
        "INSERT INTO %s (%s) VALUES (%s) "
        "ON CONFLICT(%s) DO UPDATE SET %s"
        % (table, col_sql, placeholders, key_sql, update_sql)
    )
    with closing(connect_database(database)) as conn:
        conn.executemany(sql, records)
        conn.commit()
    return len(records)


def upsert_prices_daily(
    database: str | Path,
    prices: pd.DataFrame,
    *,
    source: str = "",
) -> int:
    """写入日线行情到 `prices_daily`。"""
    if prices.empty:
        return 0
    df = prices.copy()
    if "date" in df.columns and "trade_date" not in df.columns:
        df = df.rename(columns={"date": "trade_date"})
    if "symbol" in df.columns and "ts_code" not in df.columns:
        df = df.rename(columns={"symbol": "ts_code"})
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    required = {"trade_date", "ts_code", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("prices_daily 缺少列: %s" % ", ".join(sorted(missing)))
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    if "volume" not in df.columns:
        df["volume"] = pd.NA
    if "amount" not in df.columns:
        df["amount"] = pd.NA
    if "adj_factor" in df.columns and (
        "adj_close" not in df.columns or df["adj_close"].isna().all()
    ):
        df = add_adjusted_close(df, mode="qfq")
    df["trade_date"] = _date_text(df["trade_date"])
    df["source"] = source
    df = df.dropna(subset=["trade_date"])
    df = df[df["ts_code"] != ""]
    df = df.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return _upsert_frame(
        database,
        "prices_daily",
        df,
        key_columns=("trade_date", "ts_code"),
    )


def load_prices_daily(
    database: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """从 `prices_daily` 读取日线行情长表。"""
    initialize_database(database)
    where: list[str] = []
    params: list[object] = []
    if start:
        where.append("trade_date >= ?")
        params.append(pd.to_datetime(start).strftime("%Y-%m-%d"))
    if end:
        where.append("trade_date <= ?")
        params.append(pd.to_datetime(end).strftime("%Y-%m-%d"))
    clean_symbols = _clean_symbol_list(symbols)
    if clean_symbols:
        where.append("ts_code IN (%s)" % ", ".join(["?"] * len(clean_symbols)))
        params.extend(clean_symbols)
    sql = (
        "SELECT trade_date, ts_code, open, high, low, close, "
        "adj_factor, adj_close, volume, amount FROM prices_daily"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts_code, trade_date"
    with closing(connect_database(database)) as conn:
        out = pd.read_sql_query(sql, conn, params=params)
    if not out.empty:
        out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def export_price_cache(
    database: str | Path,
    output_dir: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
    price_col: str = "close",
) -> dict[str, Path]:
    """从 SQLite 导出行情缓存；默认保留真实交易用 `prices_wide_close.csv`。"""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_long = load_prices_daily(database, start=start, end=end, symbols=symbols)
    paths: dict[str, Path] = {}
    long_path = out_dir / "prices_long.csv"
    prices_long.to_csv(long_path, index=False, date_format="%Y-%m-%d")
    paths["prices_long"] = long_path
    if not prices_long.empty:
        wide = long_to_wide(prices_long, price_col)
    else:
        wide = pd.DataFrame()
    wide_path = out_dir / "prices_wide_close.csv"
    wide.to_csv(wide_path, date_format="%Y-%m-%d")
    paths["prices_wide_close"] = wide_path
    if "adj_close" in prices_long.columns and prices_long["adj_close"].notna().any():
        wide_adj = long_to_wide(prices_long, "adj_close")
        wide_adj_path = out_dir / "prices_wide_adj_close.csv"
        wide_adj.to_csv(wide_adj_path, date_format="%Y-%m-%d")
        paths["prices_wide_adj_close"] = wide_adj_path
    return paths


def upsert_fina_indicator(
    database: str | Path,
    fina: pd.DataFrame,
    *,
    source: str = "",
) -> int:
    """写入 Tushare fina_indicator 财务指标。"""
    if fina.empty:
        return 0
    df = fina.copy()
    if "symbol" in df.columns and "ts_code" not in df.columns:
        df = df.rename(columns={"symbol": "ts_code"})
    required = {"ts_code", "ann_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("fina_indicator 缺少列: %s" % ", ".join(sorted(missing)))
    if "end_date" not in df.columns:
        df["end_date"] = df["ann_date"]
    df["ann_date"] = _date_text(df["ann_date"])
    df["end_date"] = _date_text(df["end_date"])
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    if "fcff" not in df.columns:
        for candidate in ("fcff_ps", "free_cashflow_ps"):
            if candidate in df.columns:
                df["fcff"] = df[candidate]
                break
    df["source"] = source
    df = df.dropna(subset=["ann_date", "end_date"])
    df = df[df["ts_code"] != ""]
    df = df.drop_duplicates(["ts_code", "ann_date", "end_date"], keep="last")
    return _upsert_frame(
        database,
        "fina_indicator",
        df,
        key_columns=("ts_code", "ann_date", "end_date"),
    )


def load_fina_indicator(
    database: str | Path,
    *,
    start_ann_date: str | None = None,
    end_ann_date: str | None = None,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """从 `fina_indicator` 读取财务指标。"""
    initialize_database(database)
    where: list[str] = []
    params: list[object] = []
    if start_ann_date:
        where.append("ann_date >= ?")
        params.append(pd.to_datetime(start_ann_date).strftime("%Y-%m-%d"))
    if end_ann_date:
        where.append("ann_date <= ?")
        params.append(pd.to_datetime(end_ann_date).strftime("%Y-%m-%d"))
    clean_symbols = _clean_symbol_list(symbols)
    if clean_symbols:
        where.append("ts_code IN (%s)" % ", ".join(["?"] * len(clean_symbols)))
        params.extend(clean_symbols)
    sql = "SELECT * FROM fina_indicator"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts_code, ann_date, end_date"
    with closing(connect_database(database)) as conn:
        out = pd.read_sql_query(sql, conn, params=params)
    for col in ("ann_date", "end_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _factor_panel_to_long(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    rename = {}
    if "date" in frame.columns and "trade_date" not in frame.columns:
        rename["date"] = "trade_date"
    if "symbol" in frame.columns and "ts_code" not in frame.columns:
        rename["symbol"] = "ts_code"
    if rename:
        frame = frame.rename(columns=rename)
    required = {"trade_date", "ts_code"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("factor panel 缺少列: %s" % ", ".join(sorted(missing)))
    id_cols = {"trade_date", "ts_code", "source", "factor_version"}
    factor_cols = [col for col in frame.columns if col not in id_cols]
    if not factor_cols:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "factor_name", "factor_value", "factor_version", "source"]
        )
    long = frame.melt(
        id_vars=[col for col in ("trade_date", "ts_code", "factor_version", "source") if col in frame.columns],
        value_vars=factor_cols,
        var_name="factor_name",
        value_name="factor_value",
    )
    return long


def upsert_factor_panel_daily(
    database: str | Path,
    panel: pd.DataFrame,
    *,
    source: str = "",
    factor_version: str = "v1",
) -> int:
    """写入日频因子面板；输入可为 MultiIndex 面板或 reset_index 后宽表。"""
    if panel.empty:
        return 0
    long = _factor_panel_to_long(panel)
    if long.empty:
        return 0
    long["trade_date"] = _date_text(long["trade_date"])
    long["ts_code"] = long["ts_code"].astype(str).str.strip()
    long["factor_name"] = long["factor_name"].astype(str)
    long["factor_value"] = pd.to_numeric(long["factor_value"], errors="coerce")
    if "factor_version" not in long.columns:
        long["factor_version"] = factor_version
    else:
        long["factor_version"] = long["factor_version"].fillna(factor_version).astype(str)
    if "source" not in long.columns:
        long["source"] = source
    else:
        long["source"] = long["source"].fillna(source).astype(str)
    long = long.dropna(subset=["trade_date"])
    long = long[long["ts_code"] != ""]
    long = long[long["factor_name"] != ""]
    long = long.drop_duplicates(
        ["trade_date", "ts_code", "factor_name", "factor_version"],
        keep="last",
    )
    return _upsert_frame(
        database,
        "factor_panel_daily",
        long,
        key_columns=("trade_date", "ts_code", "factor_name", "factor_version"),
    )


def load_factor_panel_daily(
    database: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
    factor_names: Iterable[str] | None = None,
    factor_version: str = "v1",
) -> pd.DataFrame:
    """从 `factor_panel_daily` 读取宽表因子面板，索引为 `(date, symbol)`。"""
    initialize_database(database)
    where = ["factor_version = ?"]
    params: list[object] = [factor_version]
    if start:
        where.append("trade_date >= ?")
        params.append(pd.to_datetime(start).strftime("%Y-%m-%d"))
    if end:
        where.append("trade_date <= ?")
        params.append(pd.to_datetime(end).strftime("%Y-%m-%d"))
    clean_symbols = _clean_symbol_list(symbols)
    if clean_symbols:
        where.append("ts_code IN (%s)" % ", ".join(["?"] * len(clean_symbols)))
        params.extend(clean_symbols)
    clean_factors = _clean_symbol_list(factor_names)
    if clean_factors:
        where.append("factor_name IN (%s)" % ", ".join(["?"] * len(clean_factors)))
        params.extend(clean_factors)
    sql = (
        "SELECT trade_date, ts_code, factor_name, factor_value FROM factor_panel_daily "
        "WHERE %s ORDER BY trade_date, ts_code, factor_name" % " AND ".join(where)
    )
    with closing(connect_database(database)) as conn:
        long = pd.read_sql_query(sql, conn, params=params)
    if long.empty:
        return pd.DataFrame()
    long["trade_date"] = pd.to_datetime(long["trade_date"])
    out = long.pivot_table(
        index=["trade_date", "ts_code"],
        columns="factor_name",
        values="factor_value",
        aggfunc="last",
    ).sort_index()
    out.index = out.index.set_names(["date", "symbol"])
    out.columns.name = None
    return out


def export_factor_panel_cache(
    database: str | Path,
    output_dir: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
    factor_version: str = "v1",
) -> Path:
    """从 SQLite 导出 `factor_panel.csv`。"""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_factor_panel_daily(
        database,
        start=start,
        end=end,
        symbols=symbols,
        factor_version=factor_version,
    )
    path = out_dir / "factor_panel.csv"
    if panel.empty:
        pd.DataFrame(columns=["date", "symbol"]).to_csv(path, index=False)
    else:
        panel.reset_index().to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
