#!/usr/bin/env python3
"""生成组合统一风险限额检查报告。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.risk_limits import (
    check_risk_limits,
    default_risk_limits,
    load_risk_limits,
    summarize_risk_limit_checks,
)


DEFAULT_STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按统一风险限额表检查目标组合")
    parser.add_argument("--trade-date", required=True, help="检查日期")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名，默认 FUSED_ROLLING_SCORE_WEIGHTED")
    parser.add_argument("--target-weights", type=Path, default=None, help="目标权重 CSV；默认读取 output/rebalance_logs/<strategy>.csv")
    parser.add_argument("--current-weights", type=Path, default=None, help="可选当前权重 CSV，含 symbol 与 weight/current_weight")
    parser.add_argument("--industry", type=Path, default=None, help="可选行业映射 CSV，含 symbol/ts_code 与 industry/行业")
    parser.add_argument("--risk-gate", type=Path, default=None, help="可选统一风险门禁 CSV")
    parser.add_argument("--order-checks", type=Path, default=None, help="可选订单预检查 CSV")
    parser.add_argument("--limits", type=Path, default=None, help="可选自定义风险限额表 CSV")
    parser.add_argument("--write-default-limits", action="store_true", help="同时输出默认限额模板 CSV")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录，默认 output/portfolio_risk_limits/<strategy>")
    return parser


def _load_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError("未找到文件: %s" % path)
    return pd.read_csv(path)


def _load_target_weights(path: Path, *, trade_date: Any) -> tuple[pd.Timestamp, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError("未找到目标权重文件: %s" % path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("目标权重文件为空: %s" % path)
    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code"
    weight_col = "weight" if "weight" in frame.columns else "target_weight"
    if symbol_col not in frame.columns or weight_col not in frame.columns:
        raise ValueError("目标权重文件须包含 symbol/ts_code 与 weight/target_weight 列")

    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[out["date"] <= pd.Timestamp(trade_date)]
        if out.empty:
            raise ValueError("目标权重文件中没有不晚于 %s 的记录" % pd.Timestamp(trade_date).strftime("%Y-%m-%d"))
        latest_date = pd.Timestamp(out["date"].max())
        out = out[out["date"] == latest_date].copy()
    else:
        latest_date = pd.Timestamp(trade_date)
    # `selected` 是信号层标志，不是最终目标持仓标志。换手节流保留的旧仓位会是
    # selected=False 但 weight>0，风险限额必须把这些仓位计算在内。
    out = out.rename(columns={symbol_col: "symbol", weight_col: "weight"})
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out = out[out["weight"] > 0.0]
    if out.empty:
        raise ValueError("目标权重文件没有有效正权重")
    out = out.groupby("symbol", as_index=False)["weight"].sum().sort_values("symbol")
    return latest_date, out


def _default_rebalance_log(strategy: str) -> Path:
    settings = get_settings()
    return settings.output_dir / "rebalance_logs" / ("%s.csv" % strategy.replace("/", "_"))


def _markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str], max_rows: int = 30) -> str:
    cols = [c for c in columns if c in frame.columns]
    if frame.empty or not cols:
        return "无\n"
    if len(headers) != len(cols):
        headers = cols
    display = frame.loc[:, cols].head(max_rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in display.to_dict("records"):
        values: list[str] = []
        for col in cols:
            value = rec.get(col, "")
            if isinstance(value, float):
                values.append("%.4f" % value)
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.extend(["", "仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame))])
    return "\n".join(lines) + "\n"


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    target_date: pd.Timestamp,
    checks: pd.DataFrame,
) -> None:
    status, detail = summarize_risk_limit_checks(checks)
    status_counts = checks["status"].value_counts().to_dict() if not checks.empty else {}
    risky = checks[checks["status"].isin(["BLOCK", "WATCH"])].copy() if not checks.empty else checks
    lines = [
        "# 组合统一风险限额检查",
        "",
        "## 运行范围",
        "",
        "- 策略：`%s`" % args.strategy,
        "- 检查日期：%s" % args.trade_date,
        "- 目标权重日期：%s" % target_date.strftime("%Y-%m-%d"),
        "- 目标权重文件：%s" % (args.target_weights or _default_rebalance_log(args.strategy)),
        "- 当前权重文件：%s" % (args.current_weights or "未提供"),
        "- 行业映射文件：%s" % (args.industry or "未提供"),
        "- 风险门禁文件：%s" % (args.risk_gate or "未提供"),
        "- 订单预检查文件：%s" % (args.order_checks or "未提供"),
        "",
        "## 总结",
        "",
        "- 总状态：`%s`" % status,
        "- 状态分布：%s" % status_counts,
        "- 明细：%s" % detail,
        "",
        "## BLOCK / WATCH 项",
        "",
        _markdown_table(
            risky,
            ["limit_id", "category", "metric", "status", "observed_value", "warning_threshold", "block_threshold", "details", "action"],
            ["限额", "类别", "指标", "状态", "当前值", "提醒阈值", "阻断阈值", "说明", "处理动作"],
            max_rows=30,
        ),
        "## 全部限额明细",
        "",
        _markdown_table(
            checks,
            ["limit_id", "category", "metric", "status", "observed_value", "warning_threshold", "block_threshold", "direction", "details"],
            ["限额", "类别", "指标", "状态", "当前值", "提醒阈值", "阻断阈值", "方向", "说明"],
            max_rows=50,
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    output_dir = args.output_dir or (settings.output_dir / "portfolio_risk_limits" / args.strategy.replace("/", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = args.target_weights or _default_rebalance_log(args.strategy)
    target_date, target_weights = _load_target_weights(target_path, trade_date=args.trade_date)
    limits = load_risk_limits(str(args.limits) if args.limits else None)
    checks = check_risk_limits(
        limits,
        target_weights,
        trade_date=args.trade_date,
        current_weights=_load_frame(args.current_weights),
        industry=_load_frame(args.industry),
        risk_gate=_load_frame(args.risk_gate),
        order_checks=_load_frame(args.order_checks),
    )

    tag = str(args.trade_date).replace("-", "")
    checks_path = output_dir / ("portfolio_risk_limit_checks_%s.csv" % tag)
    summary_path = output_dir / ("portfolio_risk_limit_summary_%s.md" % tag)
    checks.to_csv(checks_path, index=False)
    _write_summary(summary_path, args=args, target_date=target_date, checks=checks)
    if args.write_default_limits:
        default_risk_limits().to_csv(output_dir / "risk_limits_template.csv", index=False)

    status, detail = summarize_risk_limit_checks(checks)
    print("risk_limit_checks=%s" % checks_path)
    print("risk_limit_summary=%s" % summary_path)
    print("status=%s detail=%s" % (status, detail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
