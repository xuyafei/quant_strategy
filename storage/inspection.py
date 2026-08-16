"""SQLite 数据质量与巡检日报。"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from storage.database import CORE_TABLES, connect_database, initialize_database


STATUS_RANK = {"PASS": 0, "WATCH": 1, "NA": 2, "BLOCK": 3}


def _clean_symbols(symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return []
    return sorted({str(s).strip() for s in symbols if str(s).strip()})


def _worst_status(statuses: Iterable[str]) -> str:
    values = [str(s) for s in statuses if str(s)]
    if not values:
        return "NA"
    return max(values, key=lambda s: STATUS_RANK.get(s, 2))


def _scalar(database: str | Path, sql: str, params: tuple[object, ...] = ()) -> object:
    with closing(connect_database(database)) as conn:
        row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _table_exists(database: str | Path, table: str) -> bool:
    return bool(
        _scalar(
            database,
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
    )


def _row_count(database: str | Path, table: str) -> int:
    if not _table_exists(database, table):
        return 0
    return int(_scalar(database, "SELECT COUNT(*) FROM %s" % table) or 0)


def _distinct_count(database: str | Path, table: str, column: str) -> int:
    if not _table_exists(database, table):
        return 0
    return int(_scalar(database, "SELECT COUNT(DISTINCT %s) FROM %s" % (column, table)) or 0)


def _min_max(database: str | Path, table: str, column: str) -> tuple[str, str]:
    if not _table_exists(database, table):
        return "", ""
    with closing(connect_database(database)) as conn:
        row = conn.execute("SELECT MIN(%s), MAX(%s) FROM %s" % (column, column, table)).fetchone()
    if not row:
        return "", ""
    return str(row[0] or ""), str(row[1] or "")


def build_table_summary(database: str | Path) -> pd.DataFrame:
    """核心表行数与基础日期范围摘要。"""
    initialize_database(database)
    date_columns = {
        "prices_daily": "trade_date",
        "fina_indicator": "ann_date",
        "factor_panel_daily": "trade_date",
        "announcement_events": "ann_date",
        "news_sentiment": "publish_time",
        "universe_snapshot": "snapshot_date",
    }
    symbol_columns = {
        "prices_daily": "ts_code",
        "fina_indicator": "ts_code",
        "factor_panel_daily": "ts_code",
        "announcement_events": "ts_code",
        "news_sentiment": "ts_code",
        "universe_snapshot": "ts_code",
    }
    rows: list[dict[str, object]] = []
    for table in CORE_TABLES:
        exists = _table_exists(database, table)
        row_count = _row_count(database, table)
        date_col = date_columns.get(table)
        symbol_col = symbol_columns.get(table)
        first_date, last_date = _min_max(database, table, date_col) if date_col else ("", "")
        status = "PASS" if exists and row_count > 0 else ("WATCH" if exists else "BLOCK")
        if table == "storage_metadata" and exists:
            status = "PASS"
        rows.append(
            {
                "table": table,
                "exists": bool(exists),
                "row_count": row_count,
                "distinct_symbols": _distinct_count(database, table, symbol_col) if symbol_col else "",
                "first_date": first_date,
                "last_date": last_date,
                "status": status,
                "message": "ok" if status == "PASS" else "empty_table" if exists else "missing_table",
            }
        )
    return pd.DataFrame(rows)


def build_price_health(
    database: str | Path,
    *,
    expected_symbols: Iterable[str] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    max_stale_days: int = 5,
) -> pd.DataFrame:
    """检查行情表覆盖、最新日期和股票池缺失。"""
    initialize_database(database)
    symbols = _clean_symbols(expected_symbols)
    if not _table_exists(database, "prices_daily"):
        return pd.DataFrame(
            [
                {
                    "check": "prices_daily",
                    "status": "BLOCK",
                    "message": "missing_table",
                }
            ]
        )

    row_count = _row_count(database, "prices_daily")
    distinct_symbols = _distinct_count(database, "prices_daily", "ts_code")
    first_date, last_date = _min_max(database, "prices_daily", "trade_date")
    db_symbols: set[str] = set()
    with closing(connect_database(database)) as conn:
        db_symbols = {str(row[0]) for row in conn.execute("SELECT DISTINCT ts_code FROM prices_daily")}
    missing_symbols = sorted(set(symbols) - db_symbols) if symbols else []

    stale_days = ""
    if as_of_date is not None and last_date:
        stale_days = int((pd.Timestamp(as_of_date).normalize() - pd.Timestamp(last_date).normalize()).days)

    status = "PASS"
    messages: list[str] = []
    if row_count == 0:
        status = "BLOCK"
        messages.append("empty_prices")
    if missing_symbols:
        status = _worst_status([status, "WATCH"])
        messages.append("missing_symbols=%d" % len(missing_symbols))
    if stale_days != "" and int(stale_days) > int(max_stale_days):
        status = _worst_status([status, "WATCH"])
        messages.append("stale_days=%s" % stale_days)
    if not messages:
        messages.append("ok")
    return pd.DataFrame(
        [
            {
                "check": "prices_daily",
                "status": status,
                "row_count": row_count,
                "distinct_symbols": distinct_symbols,
                "expected_symbols": len(symbols) if symbols else "",
                "missing_symbols": len(missing_symbols),
                "missing_symbol_list": ",".join(missing_symbols[:20]),
                "first_trade_date": first_date,
                "last_trade_date": last_date,
                "as_of_date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d") if as_of_date is not None else "",
                "stale_days": stale_days,
                "message": "; ".join(messages),
            }
        ]
    )


def build_fina_health(
    database: str | Path,
    *,
    expected_symbols: Iterable[str] | None = None,
    required_columns: Iterable[str] = ("roe", "eps", "fcff_ps", "ocf_to_profit"),
) -> pd.DataFrame:
    """检查财务表覆盖和关键字段非空率。"""
    initialize_database(database)
    symbols = _clean_symbols(expected_symbols)
    if not _table_exists(database, "fina_indicator"):
        return pd.DataFrame(
            [{"check": "fina_indicator", "status": "BLOCK", "message": "missing_table"}]
        )

    row_count = _row_count(database, "fina_indicator")
    db_symbols: set[str] = set()
    with closing(connect_database(database)) as conn:
        db_symbols = {str(row[0]) for row in conn.execute("SELECT DISTINCT ts_code FROM fina_indicator")}
    missing_symbols = sorted(set(symbols) - db_symbols) if symbols else []
    _, last_ann_date = _min_max(database, "fina_indicator", "ann_date")

    rows: list[dict[str, object]] = []
    status = "PASS"
    if row_count == 0:
        status = "WATCH"
    if missing_symbols:
        status = _worst_status([status, "WATCH"])
    rows.append(
        {
            "check": "fina_indicator",
            "field": "__table__",
            "status": status,
            "row_count": row_count,
            "distinct_symbols": len(db_symbols),
            "expected_symbols": len(symbols) if symbols else "",
            "missing_symbols": len(missing_symbols),
            "missing_symbol_list": ",".join(missing_symbols[:20]),
            "last_ann_date": last_ann_date,
            "coverage": "",
            "message": "ok" if status == "PASS" else "coverage_warning",
        }
    )

    with closing(connect_database(database)) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(fina_indicator)").fetchall()}
        for col in required_columns:
            if col not in columns:
                rows.append(
                    {
                        "check": "fina_indicator_field",
                        "field": col,
                        "status": "WATCH",
                        "row_count": row_count,
                        "valid_rows": 0,
                        "coverage": 0.0,
                        "message": "missing_column",
                    }
                )
                continue
            valid = int(
                conn.execute("SELECT COUNT(*) FROM fina_indicator WHERE %s IS NOT NULL" % col).fetchone()[0]
            )
            coverage = float(valid / row_count) if row_count else 0.0
            field_status = "PASS" if coverage > 0 else "WATCH"
            rows.append(
                {
                    "check": "fina_indicator_field",
                    "field": col,
                    "status": field_status,
                    "row_count": row_count,
                    "valid_rows": valid,
                    "coverage": coverage,
                    "message": "ok" if field_status == "PASS" else "no_valid_value",
                }
            )
    return pd.DataFrame(rows)


def build_factor_health(
    database: str | Path,
    *,
    expected_symbols: Iterable[str] | None = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """检查因子表中每个因子的日期、股票和非空覆盖。"""
    initialize_database(database)
    symbols = _clean_symbols(expected_symbols)
    if not _table_exists(database, "factor_panel_daily"):
        return pd.DataFrame(
            [{"factor": "__table__", "status": "BLOCK", "message": "missing_table"}]
        )
    total_rows = _row_count(database, "factor_panel_daily")
    if total_rows == 0:
        return pd.DataFrame(
            [{"factor": "__table__", "status": "WATCH", "row_count": 0, "message": "empty_table"}]
        )
    sql = """
        SELECT
            factor_name,
            COUNT(*) AS total_cells,
            SUM(CASE WHEN factor_value IS NOT NULL THEN 1 ELSE 0 END) AS valid_cells,
            COUNT(DISTINCT trade_date) AS valid_dates,
            COUNT(DISTINCT ts_code) AS valid_symbols,
            MIN(trade_date) AS first_date,
            MAX(trade_date) AS last_date
        FROM factor_panel_daily
        GROUP BY factor_name
        ORDER BY factor_name
    """
    with closing(connect_database(database)) as conn:
        out = pd.read_sql_query(sql, conn)
    if out.empty:
        return out
    out = out.rename(columns={"factor_name": "factor"})
    out["coverage"] = out["valid_cells"] / out["total_cells"].replace(0, pd.NA)
    out["expected_symbols"] = len(symbols) if symbols else ""
    out["status"] = out["coverage"].map(lambda x: "PASS" if float(x or 0) >= min_coverage else "WATCH")
    out["message"] = out["status"].map(lambda s: "ok" if s == "PASS" else "low_factor_coverage")
    return out


def build_cache_file_health(cache_dir: str | Path) -> pd.DataFrame:
    """检查兼容缓存文件是否存在、是否非空。"""
    base = Path(cache_dir).expanduser()
    files = {
        "prices_long": base / "prices_long.csv",
        "prices_wide_close": base / "prices_wide_close.csv",
        "factor_panel": base / "factor_panel.csv",
    }
    rows: list[dict[str, object]] = []
    for name, path in files.items():
        exists = path.is_file()
        rows_count: int | str = ""
        columns: int | str = ""
        status = "PASS" if exists else "WATCH"
        message = "ok" if exists else "missing_file"
        if exists:
            try:
                df = pd.read_csv(path)
                rows_count = len(df)
                columns = len(df.columns)
                if rows_count == 0:
                    status = "WATCH"
                    message = "empty_file"
            except Exception as exc:
                status = "WATCH"
                message = "read_error=%s" % exc
        rows.append(
            {
                "cache_name": name,
                "path": str(path),
                "exists": bool(exists),
                "rows": rows_count,
                "columns": columns,
                "status": status,
                "message": message,
            }
        )
    return pd.DataFrame(rows)


def summarize_database_quality(parts: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """汇总各巡检表的最差状态。"""
    statuses: list[str] = []
    for df in parts.values():
        if not df.empty and "status" in df.columns:
            statuses.extend([str(x) for x in df["status"].dropna().tolist()])
    overall = _worst_status(statuses)
    return {
        "overall_status": overall,
        "pass_count": statuses.count("PASS"),
        "watch_count": statuses.count("WATCH"),
        "block_count": statuses.count("BLOCK"),
        "na_count": statuses.count("NA"),
        "check_count": len(statuses),
    }


def build_database_quality_report(
    database: str | Path,
    *,
    expected_symbols: Iterable[str] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    cache_dir: str | Path | None = None,
    max_price_stale_days: int = 5,
    factor_min_coverage: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """构建数据库巡检日报的所有明细表。"""
    initialize_database(database)
    symbols = _clean_symbols(expected_symbols)
    parts = {
        "table_summary": build_table_summary(database),
        "price_health": build_price_health(
            database,
            expected_symbols=symbols,
            as_of_date=as_of_date,
            max_stale_days=max_price_stale_days,
        ),
        "fina_health": build_fina_health(database, expected_symbols=symbols),
        "factor_health": build_factor_health(
            database,
            expected_symbols=symbols,
            min_coverage=factor_min_coverage,
        ),
    }
    if cache_dir is not None:
        parts["cache_file_health"] = build_cache_file_health(cache_dir)
    summary = summarize_database_quality(parts)
    parts["summary"] = pd.DataFrame([summary])
    return parts


def database_quality_report_markdown(parts: Mapping[str, pd.DataFrame]) -> str:
    """把数据库巡检结果转换成 Markdown。"""
    summary = summarize_database_quality(parts)
    lines = [
        "# 数据库巡检日报",
        "",
        "## 总控结论",
        "",
        "- overall_status: `%s`" % summary["overall_status"],
        "- checks: %s, PASS=%s, WATCH=%s, BLOCK=%s, NA=%s"
        % (
            summary["check_count"],
            summary["pass_count"],
            summary["watch_count"],
            summary["block_count"],
            summary["na_count"],
        ),
    ]
    for name, df in parts.items():
        if name == "summary":
            continue
        lines.extend(["", "## %s" % name, ""])
        if df.empty:
            lines.append("无数据。")
            continue
        preview = df.head(20).copy()
        lines.append(preview.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def save_database_quality_report(
    database: str | Path,
    output_dir: str | Path,
    *,
    expected_symbols: Iterable[str] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    cache_dir: str | Path | None = None,
    max_price_stale_days: int = 5,
    factor_min_coverage: float = 0.5,
) -> dict[str, Path]:
    """保存数据库巡检 CSV 与 Markdown 报告。"""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = build_database_quality_report(
        database,
        expected_symbols=expected_symbols,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        max_price_stale_days=max_price_stale_days,
        factor_min_coverage=factor_min_coverage,
    )
    paths: dict[str, Path] = {}
    for name, df in parts.items():
        path = out_dir / ("%s.csv" % name)
        df.to_csv(path, index=False)
        paths[name] = path
    md_path = out_dir / "database_quality_report.md"
    md_path.write_text(database_quality_report_markdown(parts), encoding="utf-8")
    paths["markdown"] = md_path
    return paths
