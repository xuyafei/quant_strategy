"""QMT 只读快照校验、客户端核对与连续运行验收。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CHECK_COLUMNS = ["check", "status", "actual", "expected", "detail"]


def _row(check: str, status: str, actual: Any, expected: Any, detail: str = "") -> dict[str, Any]:
    return {"check": check, "status": status, "actual": actual, "expected": expected, "detail": detail}


def _read_first(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("文件没有数据行: %s" % path)
    return frame.iloc[0]


def validate_qmt_snapshot(
    snapshot_dir: Path | str,
    *,
    ui_account_path: Path | str | None = None,
    ui_positions_path: Path | str | None = None,
    amount_tolerance: float = 1.0,
) -> pd.DataFrame:
    """验证单日快照；可用人工从客户端抄录的标准 CSV 做双边核对。"""
    root = Path(snapshot_dir)
    required = ["account.csv", "positions.csv", "orders.csv", "trades.csv", "manifest.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return pd.DataFrame([
            _row("required_files", "BLOCK", ",".join(missing), "all files", "缺少只读快照文件")
        ], columns=CHECK_COLUMNS)

    rows: list[dict[str, Any]] = []
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows.append(_row("read_only", "PASS" if manifest.get("read_only") is True else "BLOCK",
                     manifest.get("read_only"), True, "必须保持只读"))
    account_status = str(manifest.get("account_status", "UNKNOWN"))
    rows.append(_row("account_status", "PASS" if account_status in {"OK", "CLOSED"} else "WATCH",
                     account_status, "OK or CLOSED", "旧版 SDK 无状态查询时可能为 UNKNOWN"))
    warnings = manifest.get("query_warnings", []) or []
    rows.append(_row("query_warnings", "PASS" if not warnings else "WATCH",
                     len(warnings), 0, "; ".join(map(str, warnings))))

    account = _read_first(root / "account.csv")
    cash = float(account["cash"])
    market_value = float(account["market_value"])
    total_asset = float(account["total_asset"])
    nonnegative = min(cash, market_value, total_asset) >= 0
    rows.append(_row("account_nonnegative", "PASS" if nonnegative else "BLOCK",
                     min(cash, market_value, total_asset), ">= 0"))
    asset_gap = abs(total_asset - cash - market_value)
    rows.append(_row("asset_equation", "PASS" if asset_gap <= amount_tolerance else "WATCH",
                     asset_gap, "<= %.2f" % amount_tolerance,
                     "普通股票账户通常满足总资产≈可用资金+持仓市值；冻结资金等口径可能造成差异"))

    positions = pd.read_csv(root / "positions.csv")
    if positions.empty:
        rows.append(_row("position_rows", "PASS", 0, ">= 0", "空仓允许为空表"))
    else:
        required_pos = {"symbol", "shares", "available_shares", "market_value"}
        missing_pos = sorted(required_pos - set(positions.columns))
        rows.append(_row("position_schema", "PASS" if not missing_pos else "BLOCK",
                         ",".join(missing_pos), "no missing columns"))
        if not missing_pos:
            invalid = positions[
                positions["symbol"].astype(str).str.strip().eq("")
                | (pd.to_numeric(positions["shares"], errors="coerce") < 0)
                | (pd.to_numeric(positions["available_shares"], errors="coerce") < 0)
                | (pd.to_numeric(positions["available_shares"], errors="coerce")
                   > pd.to_numeric(positions["shares"], errors="coerce"))
            ]
            rows.append(_row("position_values", "PASS" if invalid.empty else "BLOCK",
                             len(invalid), 0, "可用数量必须在 0 到总持仓之间"))
            duplicate_count = int(positions["symbol"].astype(str).duplicated().sum())
            rows.append(_row("position_duplicates", "PASS" if duplicate_count == 0 else "BLOCK",
                             duplicate_count, 0))
            position_mv = float(pd.to_numeric(positions["market_value"], errors="coerce").fillna(0).sum())
            mv_gap = abs(position_mv - market_value)
            rows.append(_row("position_market_value", "PASS" if mv_gap <= amount_tolerance else "WATCH",
                             mv_gap, "<= %.2f" % amount_tolerance,
                             "基金、逆回购或其他资产可能不在股票持仓表中"))

    if ui_account_path:
        ui_account = _read_first(Path(ui_account_path))
        for field in ("cash", "market_value", "total_asset"):
            gap = abs(float(account[field]) - float(ui_account[field]))
            rows.append(_row("ui_account_%s" % field, "PASS" if gap <= amount_tolerance else "BLOCK",
                             gap, "<= %.2f" % amount_tolerance, "与 MiniQMT 客户端人工抄录值核对"))
    else:
        rows.append(_row("ui_account_reconciliation", "NA", "not provided", "canonical account CSV"))

    if ui_positions_path:
        ui_positions = pd.read_csv(ui_positions_path)
        left = positions[["symbol", "shares", "available_shares"]].copy() if not positions.empty else pd.DataFrame(
            columns=["symbol", "shares", "available_shares"]
        )
        right = ui_positions[["symbol", "shares", "available_shares"]].copy()
        merged = left.merge(right, on="symbol", how="outer", suffixes=("_api", "_ui")).fillna(0)
        share_gap = (
            (pd.to_numeric(merged["shares_api"]) - pd.to_numeric(merged["shares_ui"])).abs().sum()
            + (pd.to_numeric(merged["available_shares_api"])
               - pd.to_numeric(merged["available_shares_ui"])).abs().sum()
        )
        rows.append(_row("ui_positions_reconciliation", "PASS" if share_gap == 0 else "BLOCK",
                         int(share_gap), 0, "总持仓和可用持仓均须一致"))
    else:
        rows.append(_row("ui_positions_reconciliation", "NA", "not provided", "canonical positions CSV"))
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def audit_qmt_snapshot_history(
    account_root: Path | str,
    *,
    min_days: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """检查多个交易日快照是否连续通过结构校验。"""
    root = Path(account_root)
    manifests = sorted(root.glob("*/manifest.json"))
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        checks = validate_qmt_snapshot(manifest_path.parent)
        statuses = set(checks["status"].astype(str))
        rows.append({
            "trade_date": manifest_path.parent.name,
            "status": "BLOCK" if "BLOCK" in statuses else ("WATCH" if "WATCH" in statuses else "PASS"),
            "block_count": int((checks["status"] == "BLOCK").sum()),
            "watch_count": int((checks["status"] == "WATCH").sum()),
            "snapshot_dir": str(manifest_path.parent),
        })
    detail = pd.DataFrame(rows, columns=["trade_date", "status", "block_count", "watch_count", "snapshot_dir"])
    distinct_days = detail["trade_date"].nunique() if not detail.empty else 0
    block_days = int((detail["status"] == "BLOCK").sum()) if not detail.empty else 0
    summary = {
        "status": "PASS" if distinct_days >= min_days and block_days == 0 else "PENDING",
        "observed_days": int(distinct_days),
        "required_days": int(min_days),
        "block_days": block_days,
        "note": "PASS 只代表快照结构连续通过；至少一个交易日仍须提供客户端人工核对证据",
    }
    return detail, summary


def save_acceptance_report(
    checks: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str = "MiniQMT 只读接入验收报告",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = checks["status"].value_counts().to_dict() if not checks.empty else {}
    lines = ["# %s" % title, "", "状态统计：`%s`" % json.dumps(counts, ensure_ascii=False), "", checks.to_markdown(index=False), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
