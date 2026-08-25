"""SQLite 数据库初始化与表结构定义。"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable

from config import Settings


SCHEMA_VERSION = "3"

CORE_TABLES = (
    "prices_daily",
    "fina_indicator",
    "factor_panel_daily",
    "announcement_events",
    "news_sentiment",
    "universe_snapshot",
    "storage_metadata",
)


DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS prices_daily (
        trade_date TEXT NOT NULL,
        ts_code TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_factor REAL,
        adj_close REAL,
        volume REAL,
        amount REAL,
        source TEXT DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, ts_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fina_indicator (
        ts_code TEXT NOT NULL,
        ann_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        roe REAL,
        grossprofit_margin REAL,
        netprofit_margin REAL,
        debt_to_assets REAL,
        or_yoy REAL,
        netprofit_yoy REAL,
        ocfps REAL,
        cfps REAL,
        fcff REAL,
        fcff_ps REAL,
        fcfe_ps REAL,
        free_cashflow_ps REAL,
        ocf_to_profit REAL,
        ocf_to_opincome REAL,
        salescash_to_or REAL,
        ocf_to_or REAL,
        q_ocf_to_sales REAL,
        netprofit_cash_cover REAL,
        cashflow_to_profit REAL,
        eps REAL,
        pe_ttm REAL,
        source TEXT DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ts_code, ann_date, end_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_panel_daily (
        trade_date TEXT NOT NULL,
        ts_code TEXT NOT NULL,
        factor_name TEXT NOT NULL,
        factor_value REAL,
        factor_version TEXT NOT NULL DEFAULT 'v1',
        source TEXT DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, ts_code, factor_name, factor_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS announcement_events (
        event_key TEXT PRIMARY KEY,
        ts_code TEXT NOT NULL,
        ann_date TEXT NOT NULL,
        title TEXT NOT NULL,
        event_type TEXT DEFAULT '',
        event_score REAL,
        source TEXT DEFAULT '',
        url TEXT DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS news_sentiment (
        item_key TEXT PRIMARY KEY,
        ts_code TEXT NOT NULL,
        publish_time TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT DEFAULT '',
        source TEXT DEFAULT '',
        url TEXT DEFAULT '',
        sentiment_score REAL,
        risk_level TEXT DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_snapshot (
        snapshot_date TEXT NOT NULL,
        universe_name TEXT NOT NULL DEFAULT 'default',
        ts_code TEXT NOT NULL,
        name TEXT DEFAULT '',
        industry TEXT DEFAULT '',
        theme TEXT DEFAULT '',
        active INTEGER NOT NULL DEFAULT 1,
        exclude_reason TEXT DEFAULT '',
        source TEXT DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (snapshot_date, universe_name, ts_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS storage_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
)

TABLE_COLUMN_MIGRATIONS = {
    "prices_daily": {
        "adj_factor": "REAL",
        "adj_close": "REAL",
    },
    "fina_indicator": {
        "cfps": "REAL",
        "fcff_ps": "REAL",
        "fcfe_ps": "REAL",
        "free_cashflow_ps": "REAL",
        "ocf_to_profit": "REAL",
        "ocf_to_opincome": "REAL",
        "salescash_to_or": "REAL",
        "ocf_to_or": "REAL",
        "q_ocf_to_sales": "REAL",
        "netprofit_cash_cover": "REAL",
        "cashflow_to_profit": "REAL",
    }
}


INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol_date ON prices_daily (ts_code, trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_fina_indicator_symbol_ann ON fina_indicator (ts_code, ann_date)",
    "CREATE INDEX IF NOT EXISTS idx_factor_panel_name_date ON factor_panel_daily (factor_name, trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_factor_panel_symbol_date ON factor_panel_daily (ts_code, trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_announcement_events_symbol_date ON announcement_events (ts_code, ann_date)",
    "CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_time ON news_sentiment (ts_code, publish_time)",
    "CREATE INDEX IF NOT EXISTS idx_universe_snapshot_date_name ON universe_snapshot (snapshot_date, universe_name)",
)


def default_database_path(settings: Settings) -> Path:
    """返回工程默认 SQLite 文件路径。"""
    return settings.database_path or settings.data_dir / "quant_strategy.db"


def connect_database(path: str | Path) -> sqlite3.Connection:
    """连接 SQLite，并打开外键约束。"""
    conn = sqlite3.connect(str(Path(path).expanduser()))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database(path: str | Path) -> Path:
    """初始化 SQLite 数据库表结构；可重复执行。"""
    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect_database(db_path)) as conn:
        for statement in DDL_STATEMENTS:
            conn.execute(statement)
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
        for table, columns in TABLE_COLUMN_MIGRATIONS.items():
            existing = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(%s)" % table).fetchall()
            }
            for column, column_type in columns.items():
                if column not in existing:
                    conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, column_type))
        conn.execute(
            """
            INSERT INTO storage_metadata(key, value, updated_at)
            VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (SCHEMA_VERSION,),
        )
        conn.commit()
    return db_path


def list_database_tables(path: str | Path) -> list[str]:
    """列出数据库中的表名。"""
    with closing(connect_database(path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [str(row[0]) for row in rows]


def get_table_columns(path: str | Path, table: str) -> list[str]:
    """读取指定表的列名。"""
    with closing(connect_database(path)) as conn:
        rows = conn.execute("PRAGMA table_info(%s)" % table).fetchall()
    return [str(row[1]) for row in rows]


def missing_core_tables(path: str | Path, expected: Iterable[str] = CORE_TABLES) -> list[str]:
    """返回缺失的核心表。"""
    existing = set(list_database_tables(path))
    return [name for name in expected if name not in existing]
