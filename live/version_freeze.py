"""Live version freeze manifest helpers.

This module records the strategy, stock pool, key settings and source file
hashes that should be treated as fixed during a small-capital live trial.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config import Settings
from live.cache_io import settings_to_dict
from live.stock_pool import load_stock_pool_frame


DEFAULT_LIVE_STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"

DEFAULT_FROZEN_SOURCE_FILES: tuple[str, ...] = (
    "config.py",
    "main.py",
    "backtest/backtest_multi.py",
    "models/fusion.py",
    "models/factor_weighting.py",
    "models/optimizer.py",
    "factors/panel_builder.py",
    "factors/preprocess.py",
    "live/order_builder.py",
    "live/order_precheck.py",
    "live/risk_gate.py",
    "live/risk_limits.py",
    "live/stress_test.py",
    "live/drawdown_control.py",
    "live/capacity_impact.py",
    "live/manual_confirmation.py",
    "live/execution_feedback.py",
)


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_output(root: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def collect_git_state(root: Path) -> dict[str, Any]:
    """Collect commit and dirty state without requiring Git to be available."""
    status = _git_output(root, ["status", "--short"])
    return {
        "commit": _git_output(root, ["rev-parse", "HEAD"]),
        "branch": _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": bool(status),
        "dirty_files": [line for line in status.splitlines() if line.strip()],
    }


def collect_source_hashes(root: Path, files: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Hash the source files that define the frozen strategy and live controls."""
    rows: list[dict[str, Any]] = []
    for rel in files or DEFAULT_FROZEN_SOURCE_FILES:
        path = root / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )
    return rows


def collect_stock_pool_snapshot(settings: Settings) -> dict[str, Any]:
    """Summarize and hash the configured stock pool file."""
    path = settings.stock_pool_path
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.is_file() else "",
        "total_count": 0,
        "active_count": 0,
        "sample_symbols": [],
        "sample_names": [],
        "error": "",
    }
    if not path.exists():
        snapshot["error"] = "stock_pool_missing"
        return snapshot
    try:
        pool = load_stock_pool_frame(path, code_col=settings.stock_pool_code_col)
    except Exception as exc:
        snapshot["error"] = str(exc)
        return snapshot

    active = pool[pool["enabled"]].copy() if "enabled" in pool.columns else pool.copy()
    snapshot["total_count"] = int(len(pool))
    snapshot["active_count"] = int(len(active))
    snapshot["sample_symbols"] = active["symbol"].dropna().astype(str).head(10).tolist()
    if "name" in active.columns:
        snapshot["sample_names"] = active["name"].fillna("").astype(str).head(10).tolist()
    return snapshot


def build_freeze_manifest(
    settings: Settings,
    *,
    strategy: str = DEFAULT_LIVE_STRATEGY,
    as_of_date: Any | None = None,
    run_time: str = "09:35",
    capital: float | None = None,
    operator: str = "",
    notes: str = "",
    source_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable live freeze manifest."""
    if as_of_date is None:
        as_of = pd.Timestamp.today().normalize()
    else:
        as_of = pd.Timestamp(as_of_date)

    return {
        "manifest_type": "live_version_freeze",
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "operator": operator,
        "notes": notes,
        "live_policy": {
            "strategy": strategy,
            "capital": float(settings.paper_initial_cash if capital is None else capital),
            "run_time": run_time,
            "rebalance_freq": settings.rebalance_freq,
            "manual_confirmation_required": True,
            "auto_submit_orders": False,
            "allow_intraday_discretionary_trade": False,
        },
        "strategy_settings": {
            "top_k": settings.top_k,
            "portfolio_weighting": settings.portfolio_weighting,
            "research_price_col": settings.research_price_col,
            "execution_price_col": settings.execution_price_col,
            "adjustment_mode": settings.adjustment_mode,
            "factor_standardize_by_industry": settings.factor_standardize_by_industry,
            "fusion_use_ic_weights": settings.fusion_use_ic_weights,
            "rolling_factor_weight_lookback_days": settings.rolling_factor_weight_lookback_days,
            "rolling_factor_weight_min_days": settings.rolling_factor_weight_min_days,
        },
        "risk_settings": {
            "commission_rate": settings.commission_rate,
            "max_position_weight": settings.max_position_weight,
            "max_rebalance_turnover": settings.max_rebalance_turnover,
            "enable_trade_status_filter": settings.enable_trade_status_filter,
            "max_industry_weight": settings.max_industry_weight,
            "target_volatility": settings.target_volatility,
            "min_positions": settings.min_positions,
            "min_positions_exposure": settings.min_positions_exposure,
            "order_lot_size": settings.order_lot_size,
            "min_order_amount": settings.min_order_amount,
            "order_cash_buffer": settings.order_cash_buffer,
        },
        "data_settings": {
            "backtest_start": settings.backtest_start,
            "backtest_end": settings.backtest_end,
            "stock_pool_code_col": settings.stock_pool_code_col,
            "tushare_price_cache_path": str(settings.tushare_price_cache_path or ""),
            "fina_indicator_cache_path": str(settings.fina_indicator_cache_path or ""),
            "announcement_event_path": str(settings.announcement_event_path or ""),
            "database_path": str(settings.database_path or ""),
        },
        "broker_settings": {
            "broker_mode": settings.broker_mode,
            "broker_provider": settings.broker_provider,
            "broker_account_id": settings.broker_account_id,
        },
        "stock_pool": collect_stock_pool_snapshot(settings),
        "git": collect_git_state(settings.project_root),
        "source_hashes": collect_source_hashes(settings.project_root, source_files),
        "settings_snapshot": settings_to_dict(settings),
    }


def _flatten(prefix: str, value: Any, rows: list[dict[str, str]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten("%s.%s" % (prefix, key) if prefix else str(key), item, rows)
    elif isinstance(value, list):
        rows.append({"key": prefix, "value": json.dumps(value, ensure_ascii=False)})
    else:
        rows.append({"key": prefix, "value": "" if value is None else str(value)})


def manifest_to_rows(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    """Flatten a manifest into key/value rows for CSV review."""
    rows: list[dict[str, str]] = []
    _flatten("", manifest, rows)
    return rows


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for value in row:
            text = "" if value is None else str(value)
            values.append(text.replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_freeze_report(manifest: Mapping[str, Any]) -> str:
    """Render a human-readable Markdown freeze report."""
    policy = manifest.get("live_policy", {})
    strategy = policy.get("strategy", "")
    stock_pool = manifest.get("stock_pool", {})
    git = manifest.get("git", {})
    risk = manifest.get("risk_settings", {})
    strategy_settings = manifest.get("strategy_settings", {})
    source_hashes = manifest.get("source_hashes", [])
    missing_sources = [row.get("path", "") for row in source_hashes if not row.get("exists")]

    source_rows = [
        [row.get("path", ""), "yes" if row.get("exists") else "no", str(row.get("sha256", ""))[:12]]
        for row in source_hashes
    ]
    lines = [
        "# 实盘前版本冻结清单",
        "",
        "## 冻结范围",
        "",
        "- 策略：`%s`" % strategy,
        "- 冻结日期：%s" % manifest.get("as_of_date", ""),
        "- 运行时间：%s" % policy.get("run_time", ""),
        "- 调仓频率：`%s`" % policy.get("rebalance_freq", ""),
        "- 小资金规模：%.2f" % float(policy.get("capital", 0.0) or 0.0),
        "- 人工确认：%s" % ("必须" if policy.get("manual_confirmation_required") else "不要求"),
        "- 自动下单：%s" % ("开启" if policy.get("auto_submit_orders") else "关闭"),
        "",
        "## 股票池",
        "",
        "- 文件：`%s`" % stock_pool.get("path", ""),
        "- 文件存在：%s" % ("是" if stock_pool.get("exists") else "否"),
        "- SHA256：`%s`" % stock_pool.get("sha256", ""),
        "- 总股票数：%s" % stock_pool.get("total_count", 0),
        "- 启用股票数：%s" % stock_pool.get("active_count", 0),
        "- 样例代码：%s" % ", ".join(stock_pool.get("sample_symbols", []) or []),
        "- 样例名称：%s" % ", ".join([x for x in (stock_pool.get("sample_names", []) or []) if x]),
        "",
        "## 策略参数",
        "",
        _markdown_table(
            [
                ["Top-K", strategy_settings.get("top_k")],
                ["组合配权", strategy_settings.get("portfolio_weighting")],
                ["研究价格", strategy_settings.get("research_price_col")],
                ["交易价格", strategy_settings.get("execution_price_col")],
                ["复权模式", strategy_settings.get("adjustment_mode")],
                ["行业内标准化", strategy_settings.get("factor_standardize_by_industry")],
                ["滚动因子权重窗口", strategy_settings.get("rolling_factor_weight_lookback_days")],
            ],
            ["项目", "冻结值"],
        ),
        "",
        "## 风控参数",
        "",
        _markdown_table(
            [
                ["单票权重上限", risk.get("max_position_weight")],
                ["单次换手上限", risk.get("max_rebalance_turnover")],
                ["行业权重上限", risk.get("max_industry_weight")],
                ["波动率目标", risk.get("target_volatility")],
                ["最小持仓数量", risk.get("min_positions")],
                ["订单手数", risk.get("order_lot_size")],
                ["最小订单金额", risk.get("min_order_amount")],
                ["现金缓冲", risk.get("order_cash_buffer")],
            ],
            ["项目", "冻结值"],
        ),
        "",
        "## Git 与源码",
        "",
        "- 分支：`%s`" % git.get("branch", ""),
        "- Commit：`%s`" % git.get("commit", ""),
        "- 工作区是否有未提交改动：%s" % ("是" if git.get("is_dirty") else "否"),
        "- 缺失源码文件：%s" % (", ".join(missing_sources) if missing_sources else "无"),
        "",
        _markdown_table(source_rows, ["文件", "存在", "SHA256 前 12 位"]),
        "",
        "## 处理原则",
        "",
        "这份清单不是收益承诺，也不是自动下单授权。它只说明：从这个日期开始，"
        "小资金实盘观察期使用哪一个策略版本、哪一个股票池、哪一套数据口径和哪一组风控参数。",
        "",
        "如果后续调整策略、股票池、调仓频率、价格口径或关键风控参数，应重新生成一份新的冻结清单。",
    ]
    return "\n".join(lines) + "\n"


def save_freeze_outputs(
    settings: Settings,
    manifest: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Save JSON, CSV and Markdown freeze outputs."""
    base = output_dir or (settings.output_dir / "live_freeze" / str(manifest.get("as_of_date", "")))
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "freeze_manifest.json"
    csv_path = base / "freeze_manifest.csv"
    md_path = base / "freeze_report.md"

    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerows(manifest_to_rows(manifest))
    md_path.write_text(format_freeze_report(manifest), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def load_freeze_manifest(path: Path | str | None) -> dict[str, Any]:
    """Load a freeze manifest JSON; missing path returns an empty dict."""
    if path is None:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError("未找到实盘冻结清单: %s" % p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("实盘冻结清单格式错误: %s" % p)
    data.setdefault("manifest_path", str(p))
    return data
