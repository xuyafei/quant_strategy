"""
每日纸面交易命令行辅助逻辑。

从已有回测输出读取最近一期目标权重和最新价格，再调用 paper_runner。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings, get_settings
from live.account_state import load_account_state
from live.capacity_impact import (
    evaluate_capacity_impact,
    load_capacity_rules,
    load_liquidity_history,
    summarize_capacity_impact,
)
from live.drawdown_control import (
    apply_drawdown_control_to_weights,
    build_current_account_snapshot,
    evaluate_drawdown_control,
    load_account_snapshots,
    load_drawdown_rules,
    summarize_drawdown_control,
)
from live.factor_health_report import (
    build_factor_health_report,
    summarize_factor_health_report,
)
from live.paper_guard import (
    DailyPaperGuardError,
    format_guard_issues,
    raise_on_guard_errors,
    validate_daily_inputs,
    validate_daily_result,
)
from live.paper_run_control import (
    DailyPaperRunControlError,
    load_trading_calendar_from_prices,
    validate_daily_run_control,
)
from live.manual_confirmation import (
    FACTOR_HEALTH_SEVERITY,
    load_factor_decay_monitor,
    save_manual_confirmation,
    summarize_factor_health,
)
from live.paper_runner import run_daily_paper_trade
from live.paper_report import save_daily_paper_report
from live.risk_blacklist import (
    active_risk_blacklist,
    default_risk_blacklist_path,
    load_risk_blacklist,
    summarize_risk_blacklist_for_report,
)
from live.risk_gate import load_risk_gate, summarize_risk_gate_for_report
from live.risk_limits import (
    check_risk_limits,
    load_risk_limits,
    summarize_risk_limit_checks,
)
from live.risk_control_report import (
    build_risk_control_report,
    summarize_risk_control_report,
)
from live.stress_test import (
    load_stress_scenarios,
    run_portfolio_stress_tests,
    summarize_stress_tests,
)
from live.style_exposure_monitor import (
    latest_style_exposure_for_strategy,
    load_style_exposure,
    summarize_style_exposure_for_report,
)
from live.version_freeze import load_freeze_manifest


DEFAULT_STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_latest_target_weights(
    path: Path,
    *,
    trade_date: Any = None,
) -> tuple[pd.Timestamp, pd.Series]:
    """从 rebalance_logs/<strategy>.csv 读取不晚于 trade_date 的最近一期目标权重。"""
    if not path.exists():
        raise FileNotFoundError("未找到调仓日志: %s；请先运行 main.py 生成 rebalance_logs" % path)
    frame = pd.read_csv(path)
    required = {"date", "symbol", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("调仓日志缺少必要列: %s" % ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("调仓日志为空: %s" % path)

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if trade_date is not None:
        dt = pd.Timestamp(trade_date)
        frame = frame[frame["date"] <= dt]
        if frame.empty:
            raise ValueError("调仓日志中没有不晚于 %s 的目标权重" % dt.strftime("%Y-%m-%d"))

    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date].copy()
    if "selected" in latest.columns:
        latest = latest[_to_bool_series(latest["selected"])]
    latest["weight"] = pd.to_numeric(latest["weight"], errors="coerce").fillna(0.0)
    latest = latest[latest["weight"] > 0.0]
    if latest.empty:
        raise ValueError("最近一期调仓日志没有有效目标权重: %s" % latest_date.strftime("%Y-%m-%d"))

    weights = pd.Series(
        latest["weight"].astype(float).to_numpy(),
        index=latest["symbol"].astype(str),
        dtype=float,
    )
    weights = weights.groupby(level=0).sum().sort_index()
    return latest_date, weights


def load_latest_prices(
    path: Path,
    *,
    trade_date: Any = None,
) -> tuple[pd.Timestamp, pd.Series]:
    """从 cache/prices_wide_close.csv 读取最新价格，或读取不晚于 trade_date 的最近价格。"""
    if not path.exists():
        raise FileNotFoundError("未找到价格缓存: %s；请先运行 main.py 生成 prices_wide_close.csv" % path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("价格缓存为空: %s" % path)

    date_col = "date" if "date" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_col: "date"}).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if trade_date is not None:
        dt = pd.Timestamp(trade_date)
        frame = frame[frame["date"] <= dt]
        if frame.empty:
            raise ValueError("价格缓存中没有不晚于 %s 的价格" % dt.strftime("%Y-%m-%d"))

    latest = frame.sort_values("date").iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    prices = latest.drop(labels=["date"]).astype(float)
    prices.index = prices.index.astype(str)
    prices = prices[prices.notna() & (prices > 0.0)].sort_index()
    if prices.empty:
        raise ValueError("最近价格行没有有效价格: %s" % latest_date.strftime("%Y-%m-%d"))
    return latest_date, prices


def load_trade_status(
    path: Path | None,
    *,
    trade_date: Any = None,
) -> pd.DataFrame | None:
    """读取可选交易状态 CSV；若含 date 列，则每个 symbol 取不晚于 trade_date 的最近状态。"""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError("未找到交易状态文件: %s" % path)
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code"
    if symbol_col not in frame.columns:
        raise ValueError("交易状态文件须包含 symbol 或 ts_code 列")

    out = frame.copy()
    if symbol_col != "symbol":
        out = out.rename(columns={symbol_col: "symbol"})
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        if trade_date is not None:
            out = out[out["date"] <= pd.Timestamp(trade_date)]
        if out.empty:
            return out
        out = out.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    return out


def load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    """读取可选 CSV；不存在或未提供时返回 None。"""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError("未找到文件: %s" % path)
    return pd.read_csv(path)


def _current_weights_from_positions(
    positions: pd.DataFrame | None,
    latest_prices: pd.Series,
    *,
    cash: float,
) -> pd.Series:
    if positions is None or positions.empty:
        return pd.Series(dtype=float)
    if "symbol" not in positions.columns or "shares" not in positions.columns:
        return pd.Series(dtype=float)
    frame = positions.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    frame["price"] = frame["symbol"].map(latest_prices.astype(float))
    frame["value"] = frame["shares"] * pd.to_numeric(frame["price"], errors="coerce")
    frame = frame[frame["value"].notna() & (frame["value"] > 0.0)]
    total_asset = float(cash) + float(frame["value"].sum())
    if total_asset <= 0.0 or frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        (frame["value"] / total_asset).to_numpy(),
        index=frame["symbol"].astype(str),
        dtype=float,
    ).groupby(level=0).sum().sort_index()


def _save_risk_limit_checks(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    checks: pd.DataFrame,
) -> Path:
    safe_strategy = str(strategy).replace("/", "_")
    base = settings.output_dir / "portfolio_risk_limits" / safe_strategy
    base.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(trade_date).strftime("%Y%m%d")
    path = base / ("daily_risk_limit_checks_%s.csv" % tag)
    checks.to_csv(path, index=False)
    return path


def _save_stress_tests(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    stress_tests: pd.DataFrame,
) -> Path:
    safe_strategy = str(strategy).replace("/", "_")
    base = settings.output_dir / "stress_tests" / safe_strategy
    base.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(trade_date).strftime("%Y%m%d")
    path = base / ("daily_stress_tests_%s.csv" % tag)
    stress_tests.to_csv(path, index=False)
    return path


def _save_drawdown_control(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    control: pd.DataFrame,
) -> Path:
    safe_strategy = str(strategy).replace("/", "_")
    base = settings.output_dir / "drawdown_control" / safe_strategy
    base.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(trade_date).strftime("%Y%m%d")
    path = base / ("daily_drawdown_control_%s.csv" % tag)
    control.to_csv(path, index=False)
    return path


def _save_capacity_impact(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    safe_strategy = str(strategy).replace("/", "_")
    base = settings.output_dir / "capacity_impact" / safe_strategy
    base.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(trade_date).strftime("%Y%m%d")
    detail_path = base / ("daily_capacity_impact_detail_%s.csv" % tag)
    summary_path = base / ("daily_capacity_impact_summary_%s.csv" % tag)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {"detail": detail_path, "summary": summary_path}


def _save_risk_control_report(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    report: pd.DataFrame,
) -> Path:
    safe_strategy = str(strategy).replace("/", "_")
    base = settings.output_dir / "risk_control_reports" / safe_strategy
    base.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(trade_date).strftime("%Y%m%d")
    path = base / ("daily_risk_control_report_%s.csv" % tag)
    report.to_csv(path, index=False)
    return path


def run_daily_paper_from_outputs(
    settings: Settings,
    *,
    strategy: str = DEFAULT_STRATEGY,
    trade_date: Any = None,
    rebalance_log_path: Path | None = None,
    prices_path: Path | None = None,
    trade_status_path: Path | None = None,
    persist_outputs: bool = True,
    generate_report: bool = True,
    run_guard: bool = True,
    max_price_age_days: int = 7,
    run_control: bool = True,
    allow_non_trading_day: bool = False,
    allow_rerun: bool = False,
    execution_mode: str = "paper_trading",
    generate_manual_confirmation: bool = True,
    factor_decay_monitor_path: Path | None = None,
    style_exposure_path: Path | None = None,
    risk_gate_path: Path | None = None,
    risk_blacklist_path: Path | None = None,
    risk_limits_path: Path | None = None,
    stress_scenarios_path: Path | None = None,
    drawdown_rules_path: Path | None = None,
    capacity_rules_path: Path | None = None,
    liquidity_history_path: Path | None = None,
    freeze_manifest_path: Path | None = None,
    capacity_lookback_days: int | None = None,
    impact_coefficient_bps: float = 100.0,
    industry_path: Path | None = None,
) -> dict[str, Any]:
    """从 output/ 下已有文件读取输入并执行单日纸面交易。"""
    rebalance_path = (
        rebalance_log_path
        if rebalance_log_path is not None
        else settings.output_dir / "rebalance_logs" / ("%s.csv" % strategy.replace("/", "_"))
    )
    price_cache_path = (
        prices_path
        if prices_path is not None
        else settings.output_dir / "cache" / "prices_wide_close.csv"
    )
    liquidity_path = (
        liquidity_history_path
        if liquidity_history_path is not None
        else settings.output_dir / "cache" / "prices_long.csv"
    )
    freeze_manifest = load_freeze_manifest(freeze_manifest_path)

    requested_date = pd.Timestamp(trade_date) if trade_date is not None else None
    price_date, latest_prices = load_latest_prices(price_cache_path, trade_date=requested_date)
    run_date = requested_date if requested_date is not None else price_date
    trading_calendar = load_trading_calendar_from_prices(price_cache_path)
    if run_control:
        validate_daily_run_control(
            settings,
            strategy=strategy,
            trade_date=run_date,
            trading_calendar=trading_calendar,
            persist_outputs=persist_outputs,
            allow_non_trading_day=allow_non_trading_day,
            allow_rerun=allow_rerun,
        )
    target_date, target_weights = load_latest_target_weights(rebalance_path, trade_date=run_date)
    trade_status = load_trade_status(trade_status_path, trade_date=run_date)
    blacklist_path = risk_blacklist_path if risk_blacklist_path is not None else default_risk_blacklist_path(settings)
    risk_blacklist = active_risk_blacklist(load_risk_blacklist(blacklist_path), trade_date=run_date)
    risk_gate = load_risk_gate(risk_gate_path, trade_date=run_date)
    industry = load_optional_csv(industry_path)
    starting_cash, starting_positions = load_account_state(
        settings,
        strategy=strategy,
        default_cash=settings.paper_initial_cash,
    )
    drawdown_control = evaluate_drawdown_control(
        load_drawdown_rules(str(drawdown_rules_path) if drawdown_rules_path is not None else None),
        load_account_snapshots(settings, strategy=strategy),
        build_current_account_snapshot(
            cash=starting_cash,
            positions=starting_positions,
            latest_prices=latest_prices,
        ),
        target_weights,
        trade_date=run_date,
    )
    target_weights_after_drawdown = apply_drawdown_control_to_weights(target_weights, drawdown_control)
    guard_issues = (
        validate_daily_inputs(
            target_weights=target_weights_after_drawdown,
            latest_prices=latest_prices,
            run_date=run_date,
            target_date=target_date,
            price_date=price_date,
            max_price_age_days=max_price_age_days,
            allow_empty_target=target_weights_after_drawdown.empty
            and float(drawdown_control["target_weight_scale"].iloc[0]) <= 1e-12,
        )
        if run_guard
        else []
    )
    raise_on_guard_errors(guard_issues)

    result = run_daily_paper_trade(
        settings,
        strategy=strategy,
        target_weights=target_weights_after_drawdown,
        latest_prices=latest_prices,
        trade_date=run_date,
        trade_status=trade_status,
        risk_blacklist=risk_blacklist,
        persist_outputs=persist_outputs,
        execution_mode=execution_mode,
    )
    result["input_paths"] = {
        "rebalance_log": rebalance_path,
        "prices": price_cache_path,
        "trade_status": trade_status_path,
        "risk_gate": risk_gate_path if risk_gate_path is not None and risk_gate_path.exists() else None,
        "risk_blacklist": blacklist_path if blacklist_path.exists() else None,
        "drawdown_rules": drawdown_rules_path if drawdown_rules_path is not None and drawdown_rules_path.exists() else None,
        "capacity_rules": capacity_rules_path if capacity_rules_path is not None and capacity_rules_path.exists() else None,
        "liquidity_history": liquidity_path if liquidity_path.exists() else None,
        "freeze_manifest": freeze_manifest_path if freeze_manifest_path is not None and freeze_manifest_path.exists() else None,
    }
    result["target_date"] = target_date
    result["price_date"] = price_date
    result["target_weights_before_drawdown"] = target_weights
    result["target_weights_after_drawdown"] = target_weights_after_drawdown
    result["drawdown_control"] = drawdown_control
    result["trading_calendar_latest_date"] = trading_calendar[-1]
    if run_guard:
        guard_issues.extend(validate_daily_result(result))
        raise_on_guard_errors(guard_issues)
    result["guard_issues"] = guard_issues
    factor_monitor = load_factor_decay_monitor(settings, factor_decay_monitor_path)
    result["factor_decay_monitor"] = factor_monitor
    result["freeze_manifest"] = freeze_manifest
    style_exposure_all = load_style_exposure(settings, style_exposure_path)
    result["style_exposure"] = latest_style_exposure_for_strategy(
        style_exposure_all,
        strategy=strategy,
        trade_date=run_date,
    )
    result["factor_health_report"] = build_factor_health_report(settings, strategy=strategy)
    result["risk_gate"] = risk_gate
    result["risk_blacklist"] = risk_blacklist
    capacity_detail, capacity_summary = evaluate_capacity_impact(
        result.get("orders"),
        load_liquidity_history(liquidity_path),
        trade_date=run_date,
        rules=load_capacity_rules(str(capacity_rules_path) if capacity_rules_path is not None else None),
        lookback_days=capacity_lookback_days
        if capacity_lookback_days is not None
        else int(getattr(settings, "liquidity_lookback_days", 20) or 20),
        impact_coefficient_bps=impact_coefficient_bps,
    )
    result["capacity_impact"] = capacity_detail
    result["capacity_impact_summary"] = capacity_summary
    current_weights = _current_weights_from_positions(
        result.get("starting_positions"),
        latest_prices,
        cash=float(result.get("starting_cash", 0.0)),
    )
    result["risk_limit_checks"] = check_risk_limits(
        load_risk_limits(str(risk_limits_path) if risk_limits_path is not None else None),
        target_weights_after_drawdown,
        trade_date=run_date,
        current_weights=current_weights,
        industry=industry,
        risk_gate=risk_gate,
        order_checks=result.get("order_checks"),
    )
    snapshot = result.get("account_snapshot", {})
    result["stress_tests"] = run_portfolio_stress_tests(
        load_stress_scenarios(str(stress_scenarios_path) if stress_scenarios_path is not None else None),
        target_weights_after_drawdown,
        trade_date=run_date,
        total_asset=float(snapshot.get("total_asset", 0.0)),
        industry=industry,
    )
    result["risk_control_report"] = build_risk_control_report(
        trade_date=run_date,
        guard_issues=result.get("guard_issues"),
        risk_gate=result.get("risk_gate"),
        risk_blacklist=result.get("risk_blacklist"),
        drawdown_control=result.get("drawdown_control"),
        capacity_impact_summary=result.get("capacity_impact_summary"),
        order_checks=result.get("order_checks"),
        risk_limit_checks=result.get("risk_limit_checks"),
        stress_tests=result.get("stress_tests"),
    )
    if persist_outputs:
        result.setdefault("paths", {})["drawdown_control"] = _save_drawdown_control(
            settings,
            strategy=strategy,
            trade_date=run_date,
            control=result["drawdown_control"],
        )
        result.setdefault("paths", {})["capacity_impact"] = _save_capacity_impact(
            settings,
            strategy=strategy,
            trade_date=run_date,
            detail=result["capacity_impact"],
            summary=result["capacity_impact_summary"],
        )
        result.setdefault("paths", {})["risk_limit_checks"] = _save_risk_limit_checks(
            settings,
            strategy=strategy,
            trade_date=run_date,
            checks=result["risk_limit_checks"],
        )
        result.setdefault("paths", {})["stress_tests"] = _save_stress_tests(
            settings,
            strategy=strategy,
            trade_date=run_date,
            stress_tests=result["stress_tests"],
        )
        result.setdefault("paths", {})["risk_control_report"] = _save_risk_control_report(
            settings,
            strategy=strategy,
            trade_date=run_date,
            report=result["risk_control_report"],
        )
    if persist_outputs and generate_report:
        report_path = save_daily_paper_report(settings, result)
        result.setdefault("paths", {})["paper_report"] = report_path
    if persist_outputs and generate_manual_confirmation:
        confirm_paths = save_manual_confirmation(
            settings,
            result,
            factor_monitor=factor_monitor,
        )
        result.setdefault("paths", {})["manual_confirmation"] = confirm_paths
    return result


def format_daily_paper_summary(result: dict[str, Any]) -> str:
    """生成命令行摘要。"""
    orders = result["orders"]
    checks = result["order_checks"]
    trades = result["paper_trades"]
    snapshot = result["account_snapshot"]
    paths = result.get("paths", {})
    guard_issues = result.get("guard_issues", [])
    factor_monitor = result.get("factor_decay_monitor")
    style_exposure = result.get("style_exposure")
    factor_health_report = result.get("factor_health_report")
    risk_gate = result.get("risk_gate")
    risk_blacklist = result.get("risk_blacklist")
    risk_limit_checks = result.get("risk_limit_checks")
    stress_tests = result.get("stress_tests")
    drawdown_control = result.get("drawdown_control")
    capacity_impact_summary = result.get("capacity_impact_summary")
    risk_control_report = result.get("risk_control_report")
    freeze_manifest = result.get("freeze_manifest") or {}

    n_orders = int(len(orders))
    n_pass = int((checks["check_status"] == "PASS").sum()) if not checks.empty else 0
    n_block = int((checks["check_status"] == "BLOCK").sum()) if not checks.empty else 0
    n_filled = int((trades["fill_status"] == "FILLED").sum()) if not trades.empty else 0
    n_skipped = int((trades["fill_status"] == "SKIPPED").sum()) if not trades.empty else 0

    lines = [
        "每日纸面交易完成",
        "strategy=%s" % result["strategy"],
        "execution_mode=%s" % result.get("execution_mode", "paper_trading"),
        "trade_date=%s target_date=%s price_date=%s"
        % (
            pd.Timestamp(result["trade_date"]).strftime("%Y-%m-%d"),
            pd.Timestamp(result["target_date"]).strftime("%Y-%m-%d"),
            pd.Timestamp(result["price_date"]).strftime("%Y-%m-%d"),
        ),
        "orders=%d pass=%d block=%d filled=%d skipped=%d"
        % (n_orders, n_pass, n_block, n_filled, n_skipped),
        "cash=%.2f market_value=%.2f total_asset=%.2f n_positions=%d"
        % (
            float(snapshot.get("cash", 0.0)),
            float(snapshot.get("market_value", 0.0)),
            float(snapshot.get("total_asset", 0.0)),
            int(float(snapshot.get("n_positions", 0.0))),
        ),
    ]
    if paths:
        lines.append("outputs:")
        for key, value in paths.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append("  %s.%s=%s" % (key, sub_key, sub_value))
            else:
                lines.append("  %s=%s" % (key, value))
    factor_status, factor_reasons = summarize_factor_health(factor_monitor)
    if freeze_manifest:
        policy = freeze_manifest.get("live_policy", {}) or {}
        git = freeze_manifest.get("git", {}) or {}
        lines.append(
            "freeze_manifest=%s strategy=%s git_dirty=%s"
            % (
                freeze_manifest.get("as_of_date", ""),
                policy.get("strategy", ""),
                bool(git.get("is_dirty", False)),
            )
        )
    if factor_monitor is not None and not factor_monitor.empty:
        monitor = factor_monitor.copy()
        if "status" in monitor.columns:
            status_series = monitor["status"].astype(str).str.upper()
            risky_count = int(status_series.map(FACTOR_HEALTH_SEVERITY).fillna(0).ge(FACTOR_HEALTH_SEVERITY["WATCH"]).sum())
        else:
            risky_count = 0
        lines.append("factor_health=%s risky_factors=%d reason=%s" % (factor_status, risky_count, factor_reasons))
    else:
        lines.append("factor_health=%s reason=%s" % (factor_status, factor_reasons))
    style_status, style_reason = summarize_style_exposure_for_report(style_exposure)
    lines.append("style_exposure=%s detail=%s" % (style_status, style_reason))
    health_status, health_reason = summarize_factor_health_report(factor_health_report)
    lines.append("enhanced_factor_health=%s detail=%s" % (health_status, health_reason))
    gate_status, gate_reason = summarize_risk_gate_for_report(risk_gate)
    lines.append("risk_gate=%s detail=%s" % (gate_status, gate_reason))
    blacklist_status, blacklist_reason = summarize_risk_blacklist_for_report(risk_blacklist)
    lines.append("risk_blacklist=%s detail=%s" % (blacklist_status, blacklist_reason))
    risk_limit_status, risk_limit_reason = summarize_risk_limit_checks(risk_limit_checks)
    lines.append("risk_limits=%s detail=%s" % (risk_limit_status, risk_limit_reason))
    stress_status, stress_reason = summarize_stress_tests(stress_tests)
    lines.append("stress_tests=%s detail=%s" % (stress_status, stress_reason))
    drawdown_status, drawdown_reason = summarize_drawdown_control(drawdown_control)
    lines.append("drawdown_control=%s detail=%s" % (drawdown_status, drawdown_reason))
    capacity_status, capacity_reason = summarize_capacity_impact(capacity_impact_summary)
    lines.append("capacity_impact=%s detail=%s" % (capacity_status, capacity_reason))
    risk_control_status, risk_control_reason = summarize_risk_control_report(risk_control_report)
    lines.append("risk_control_report=%s detail=%s" % (risk_control_status, risk_control_reason))
    if guard_issues:
        lines.append("guard:")
        lines.append(format_guard_issues(guard_issues))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行每日纸面交易流程")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名，默认 FUSED_ROLLING_SCORE_WEIGHTED")
    parser.add_argument("--trade-date", default=None, help="运行日期；默认使用价格缓存最新日期")
    parser.add_argument("--rebalance-log", type=Path, default=None, help="调仓日志 CSV；默认 output/rebalance_logs/<strategy>.csv")
    parser.add_argument("--prices", type=Path, default=None, help="价格宽表 CSV；默认 output/cache/prices_wide_close.csv")
    parser.add_argument("--trade-status", type=Path, default=None, help="可选交易状态 CSV，含 symbol/ts_code 与 is_suspended/is_limit_up/is_limit_down")
    parser.add_argument("--no-persist", action="store_true", help="只运行不写订单、成交和账户状态文件")
    parser.add_argument("--no-report", action="store_true", help="不生成 Markdown 纸面交易日报")
    parser.add_argument("--no-manual-confirm", action="store_true", help="不生成小资金人工确认实盘单")
    parser.add_argument(
        "--factor-decay-monitor",
        type=Path,
        default=None,
        help="因子失效监控 CSV；默认 output/factor_validation/factor_decay_monitor.csv",
    )
    parser.add_argument(
        "--style-exposure",
        type=Path,
        default=None,
        help="风格暴露 CSV；默认 output/factor_diagnostics/style_exposure.csv",
    )
    parser.add_argument(
        "--risk-blacklist",
        type=Path,
        default=None,
        help="风险黑名单 CSV/XLSX；默认 data/risk_blacklist.csv，文件不存在则视为无黑名单",
    )
    parser.add_argument(
        "--risk-gate",
        type=Path,
        default=None,
        help="统一风险门禁 CSV；由 scripts/build_unified_risk_gate.py 生成，用于日报展示 PASS/WATCH/BLOCK 总览",
    )
    parser.add_argument(
        "--risk-limits",
        type=Path,
        default=None,
        help="组合风险限额表 CSV；默认使用 live.risk_limits.default_risk_limits()",
    )
    parser.add_argument(
        "--stress-scenarios",
        type=Path,
        default=None,
        help="组合压力测试情景表 CSV；默认使用 live.stress_test.default_stress_scenarios()",
    )
    parser.add_argument(
        "--drawdown-rules",
        type=Path,
        default=None,
        help="账户回撤止损与降仓规则 CSV；默认使用 live.drawdown_control.default_drawdown_rules()",
    )
    parser.add_argument(
        "--capacity-rules",
        type=Path,
        default=None,
        help="容量与冲击成本规则 CSV；默认使用 live.capacity_impact.default_capacity_rules()",
    )
    parser.add_argument(
        "--liquidity-history",
        type=Path,
        default=None,
        help="流动性历史 CSV，含 date/trade_date、symbol/ts_code、amount/turnover；默认 output/cache/prices_long.csv",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=None,
        help="实盘前版本冻结清单 JSON；由 scripts/build_live_version_freeze.py 生成，用于人工确认单审计",
    )
    parser.add_argument("--capacity-lookback-days", type=int, default=None, help="容量估算平均成交额窗口；默认 Settings.liquidity_lookback_days")
    parser.add_argument("--impact-coefficient-bps", type=float, default=100.0, help="冲击成本估算系数，默认 100 bps * sqrt(参与率)")
    parser.add_argument(
        "--industry",
        type=Path,
        default=None,
        help="行业映射 CSV，含 symbol/ts_code/股票代码 与 industry/分类/行业；用于每日组合风险限额检查",
    )
    parser.add_argument("--no-guard", action="store_true", help="跳过日终输入和结果异常检查")
    parser.add_argument("--max-price-age-days", type=int, default=7, help="价格日期距运行日期超过该天数时给出 warning")
    parser.add_argument("--no-run-control", action="store_true", help="跳过交易日日历和重复运行保护")
    parser.add_argument("--allow-non-trading-day", action="store_true", help="允许在非交易日强制运行")
    parser.add_argument("--allow-rerun", action="store_true", help="允许覆盖同一交易日已有纸面账户快照")
    parser.add_argument(
        "--execution-mode",
        default="paper_trading",
        choices=["paper_trading", "simulated_broker"],
        help="执行模式：paper_trading 使用旧纸面成交；simulated_broker 使用统一模拟券商适配器",
    )
    return parser


def run_daily_paper_from_args(settings: Settings, args: argparse.Namespace) -> int:
    """执行已解析的日终纸面交易 CLI 参数。"""
    try:
        result = run_daily_paper_from_outputs(
            settings,
            strategy=args.strategy,
            trade_date=args.trade_date,
            rebalance_log_path=args.rebalance_log,
            prices_path=args.prices,
            trade_status_path=args.trade_status,
            persist_outputs=not args.no_persist,
            generate_report=not args.no_report,
            run_guard=not args.no_guard,
            max_price_age_days=args.max_price_age_days,
            run_control=not args.no_run_control,
            allow_non_trading_day=args.allow_non_trading_day,
            allow_rerun=args.allow_rerun,
            execution_mode=args.execution_mode,
            generate_manual_confirmation=not args.no_manual_confirm,
            factor_decay_monitor_path=args.factor_decay_monitor,
            style_exposure_path=args.style_exposure,
            risk_gate_path=args.risk_gate,
            risk_blacklist_path=args.risk_blacklist,
            risk_limits_path=args.risk_limits,
            stress_scenarios_path=args.stress_scenarios,
            drawdown_rules_path=args.drawdown_rules,
            capacity_rules_path=args.capacity_rules,
            liquidity_history_path=args.liquidity_history,
            freeze_manifest_path=args.freeze_manifest,
            capacity_lookback_days=args.capacity_lookback_days,
            impact_coefficient_bps=args.impact_coefficient_bps,
            industry_path=args.industry,
        )
    except (DailyPaperGuardError, DailyPaperRunControlError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_daily_paper_summary(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_daily_paper_from_args(get_settings(), args)


if __name__ == "__main__":
    raise SystemExit(main())
