"""
小资金实盘每日 SOP。

这个模块把已经完成的准实盘产物组织成一张当天操作清单。它不生成信号、
不修改账户、不自动下单，只给出每天应该按什么顺序运行和检查。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


SOP_COLUMNS = [
    "date",
    "strategy",
    "phase",
    "step_no",
    "step",
    "owner",
    "timing",
    "command",
    "required_output",
    "gate_status",
    "on_pass",
    "on_watch",
    "on_block",
    "notes",
]


@dataclass(frozen=True)
class SOPStep:
    date: str
    strategy: str
    phase: str
    step_no: int
    step: str
    owner: str
    timing: str
    command: str
    required_output: str
    gate_status: str
    on_pass: str
    on_watch: str
    on_block: str
    notes: str


def live_sop_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "live_sop" / safe


def _date_to_str(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _tag(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _default_path(settings: Settings, strategy: str, trade_date: Any, kind: str) -> str:
    safe = str(strategy).replace("/", "_")
    date_s = _date_to_str(trade_date)
    tag = _tag(trade_date)
    mapping = {
        "freeze": settings.output_dir / "live_freeze" / date_s / "freeze_manifest.json",
        "run_monitor": settings.output_dir / "live_run_monitor" / safe / ("%s_run_monitor.csv" % date_s),
        "risk_control": settings.output_dir
        / "risk_control_reports"
        / safe
        / ("daily_risk_control_report_%s.csv" % tag),
        "manual_confirm": settings.output_dir / "live_orders" / safe / ("%s_manual_confirm.csv" % date_s),
        "paper_report": settings.output_dir / "paper_reports" / safe / ("%s.md" % date_s),
        "execution_feedback": settings.output_dir
        / "execution_feedback"
        / safe
        / ("%s_execution_feedback.csv" % date_s),
        "next_day_review": settings.output_dir
        / "execution_feedback"
        / safe
        / ("%s_next_day_review.csv" % date_s),
        "attribution": settings.output_dir
        / "performance_attribution"
        / safe
        / ("%s_performance_attribution_summary.csv" % date_s),
        "deviation": settings.output_dir / "live_deviation" / safe / ("%s_deviation_summary.csv" % date_s),
        "checklist": settings.output_dir
        / "semi_auto_checklist"
        / safe
        / ("%s_semi_auto_decision.csv" % date_s),
    }
    return str(mapping[kind])


def _step(
    *,
    date_s: str,
    strategy: str,
    phase: str,
    step_no: int,
    step: str,
    owner: str,
    timing: str,
    command: str,
    required_output: str,
    gate_status: str,
    on_pass: str,
    on_watch: str,
    on_block: str,
    notes: str = "",
) -> SOPStep:
    return SOPStep(
        date=date_s,
        strategy=strategy,
        phase=phase,
        step_no=step_no,
        step=step,
        owner=owner,
        timing=timing,
        command=command,
        required_output=required_output,
        gate_status=gate_status,
        on_pass=on_pass,
        on_watch=on_watch,
        on_block=on_block,
        notes=notes,
    )


def build_live_daily_sop(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    freeze_manifest_path: Path | str | None = None,
    broker_positions_path: Path | str | None = None,
    include_broker_reconcile: bool = False,
) -> pd.DataFrame:
    """生成当天小资金实盘 SOP 清单。"""
    date_s = _date_to_str(trade_date)
    freeze_path = str(freeze_manifest_path or _default_path(settings, strategy, trade_date, "freeze"))
    broker_positions = str(broker_positions_path or "")

    steps: list[SOPStep] = [
        _step(
            date_s=date_s,
            strategy=strategy,
            phase="pre_market",
            step_no=1,
            step="更新基础数据与缓存",
            owner="system",
            timing="开盘前或前一晚",
            command="python scripts/update_database_cache.py",
            required_output="output/cache/prices_long.csv; output/cache/prices_wide_close.csv; output/cache/factor_panel.csv",
            gate_status="WATCH",
            on_pass="继续生成研究结果和目标持仓。",
            on_watch="检查数据库巡检结果，确认缺口是否影响今天股票池。",
            on_block="停止当天自动流程，先补齐行情、财务或因子缓存。",
            notes="若当天只做演练，可用已有缓存；真实执行前应确认数据日期新鲜。",
        ),
        _step(
            date_s=date_s,
            strategy=strategy,
            phase="pre_market",
            step_no=2,
            step="生成主策略回测与目标权重",
            owner="system",
            timing="开盘前或前一晚",
            command="python main.py",
            required_output="output/rebalance_logs/%s.csv; output/performance_summary.csv" % strategy,
            gate_status="BLOCK",
            on_pass="读取最近一期目标权重，进入日终或调仓日执行准备。",
            on_watch="若只是观察指标异常，记录后进入人工复核。",
            on_block="没有目标权重就不能继续生成订单。",
            notes="研究回测用复权价格，订单金额和纸面交易继续用原始 close。",
        ),
        _step(
            date_s=date_s,
            strategy=strategy,
            phase="pre_market",
            step_no=3,
            step="冻结实盘前版本",
            owner="operator",
            timing="策略或股票池变更后",
            command="python scripts/build_live_version_freeze.py --as-of-date %s --strategy %s" % (date_s, strategy),
            required_output=freeze_path,
            gate_status="BLOCK",
            on_pass="确认策略版本、股票池、调仓频率、价格口径和风控参数未漂移。",
            on_watch="若只是备注变化，写入人工确认单后继续。",
            on_block="冻结清单缺失时不进入人工下单。",
            notes="策略版本不是每天都要改；改了就必须重新冻结。",
        ),
        _step(
            date_s=date_s,
            strategy=strategy,
            phase="pre_order",
            step_no=4,
            step="运行当日纸面交易链路",
            owner="system",
            timing="调仓日前一晚或调仓日上午",
            command="python scripts/run_daily_paper.py --strategy %s --trade-date %s --execution-mode simulated_broker"
            % (strategy, date_s),
            required_output=_default_path(settings, strategy, trade_date, "paper_report"),
            gate_status="BLOCK",
            on_pass="继续查看订单、预检查、风险总控和人工确认单。",
            on_watch="先看日报里的 WATCH 项，再决定是否降低仓位或跳过部分订单。",
            on_block="当天不下单，修复阻断项后重跑。",
            notes="纸面交易是用同一套订单和风控流程演练，不代表真实成交。",
        ),
        _step(
            date_s=date_s,
            strategy=strategy,
            phase="pre_order",
            step_no=5,
            step="检查运行监控与风险总控",
            owner="operator",
            timing="人工确认前",
            command="python scripts/build_live_run_monitor.py --strategy %s --trade-date %s" % (strategy, date_s),
            required_output="%s; %s"
            % (
                _default_path(settings, strategy, trade_date, "run_monitor"),
                _default_path(settings, strategy, trade_date, "risk_control"),
            ),
            gate_status="BLOCK",
            on_pass="全部关键检查通过后进入半自动执行清单。",
            on_watch="逐项看风险来源，是数据缺口、容量问题、回撤控制还是门禁命中。",
            on_block="任何硬阻断都不进入真实下单。",
            notes="风险总控日报由日终纸面交易生成，运行监控可单独补跑。",
        ),
        _step(
            date_s=date_s,
            strategy=strategy,
            phase="pre_order",
            step_no=6,
            step="生成半自动实盘执行清单",
            owner="operator",
            timing="人工下单前最后一步",
            command="python scripts/build_semi_auto_checklist.py --strategy %s --trade-date %s" % (strategy, date_s),
            required_output=_default_path(settings, strategy, trade_date, "checklist"),
            gate_status="BLOCK",
            on_pass="只有 READY_FOR_MANUAL_ORDER 才进入人工下单。",
            on_watch="MANUAL_REVIEW 需要写清楚继续或放弃的人工理由。",
            on_block="DO_NOT_TRADE 时停止当天交易。",
            notes="这一步是实盘前的刹车片，不能跳过。",
        ),
    ]
    if include_broker_reconcile:
        steps.append(
            _step(
                date_s=date_s,
                strategy=strategy,
                phase="pre_order",
                step_no=7,
                step="纸面账户与真实账户只读对账",
                owner="operator",
                timing="人工下单前",
                command="python scripts/reconcile_paper_broker.py --strategy %s --trade-date %s --broker-positions %s"
                % (strategy, date_s, broker_positions or "<broker_positions.csv>"),
                required_output="output/broker_reconciliation/%s/%s_reconciliation.md"
                % (str(strategy).replace("/", "_"), date_s),
                gate_status="WATCH",
                on_pass="确认真实账户和纸面账户没有明显漂移。",
                on_watch="先解释现金或持仓差异，再决定是否人工修正。",
                on_block="若差异不可解释，停止下单。",
                notes="没有真实券商只读权限时，这一步先保留为空。",
            )
        )
    start_post_no = 8 if include_broker_reconcile else 7
    steps.extend(
        [
            _step(
                date_s=date_s,
                strategy=strategy,
                phase="manual_order",
                step_no=start_post_no,
                step="人工确认并在券商端执行",
                owner="operator",
                timing="调仓日上午",
                command="手工打开 %s 并回填 manual_action / executed_qty / executed_price"
                % _default_path(settings, strategy, trade_date, "manual_confirm"),
                required_output=_default_path(settings, strategy, trade_date, "manual_confirm"),
                gate_status="BLOCK",
                on_pass="按确认单执行通过的订单，并保留截图或成交记录。",
                on_watch="对 WATCH 订单降仓、跳过或拆单，并写明原因。",
                on_block="确认单不存在或风险阻断时不下单。",
                notes="小资金阶段坚持人工确认，系统只给建议。",
            ),
            _step(
                date_s=date_s,
                strategy=strategy,
                phase="post_market",
                step_no=start_post_no + 1,
                step="真实成交回填与执行偏差分析",
                owner="operator",
                timing="成交后或收盘后",
                command="python scripts/build_execution_feedback.py --strategy %s --trade-date %s" % (strategy, date_s),
                required_output=_default_path(settings, strategy, trade_date, "execution_feedback"),
                gate_status="WATCH",
                on_pass="记录实际成交与建议订单基本一致。",
                on_watch="分析未成交、部分成交或滑点偏大的原因。",
                on_block="若出现非确认订单或无法解释成交，暂停后续人工交易。",
                notes="这一步让实盘执行从感觉变成可量化偏差。",
            ),
            _step(
                date_s=date_s,
                strategy=strategy,
                phase="post_market",
                step_no=start_post_no + 2,
                step="表现归因与持仓偏差分析",
                owner="system",
                timing="收盘后",
                command="python scripts/build_live_performance_attribution.py --strategy %s --trade-date %s\npython scripts/build_live_deviation_analysis.py --strategy %s --trade-date %s"
                % (strategy, date_s, strategy, date_s),
                required_output="%s; %s"
                % (
                    _default_path(settings, strategy, trade_date, "attribution"),
                    _default_path(settings, strategy, trade_date, "deviation"),
                ),
                gate_status="WATCH",
                on_pass="确认收益来源和目标跟踪偏差可解释。",
                on_watch="若偏差变大，次日先修正持仓再考虑新交易。",
                on_block="若账户状态、价格或成交记录无法对齐，停止自动滚动。",
                notes="归因回答赚亏来自哪里，偏差分析回答组合是否还跟着策略走。",
            ),
            _step(
                date_s=date_s,
                strategy=strategy,
                phase="next_day_review",
                step_no=start_post_no + 3,
                step="次日复盘并更新运行监控",
                owner="operator",
                timing="下一个交易日",
                command="python scripts/build_live_run_monitor.py --strategy %s --trade-date %s" % (strategy, date_s),
                required_output=_default_path(settings, strategy, trade_date, "next_day_review"),
                gate_status="WATCH",
                on_pass="继续观察下一次调仓或纸面交易。",
                on_watch="把异常写入黑名单、参数复核或数据修复清单。",
                on_block="若次日复盘无法完成，不扩大实盘规模。",
                notes="小资金阶段最重要的不是交易次数，而是每次交易都能复盘。",
            ),
        ]
    )
    return pd.DataFrame([x.__dict__ for x in steps], columns=SOP_COLUMNS)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 100) -> str:
    if frame.empty:
        return "无\n"
    rows = frame.head(max_rows)
    lines = [
        "| " + " | ".join(rows.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(rows.columns)) + " |",
    ]
    for rec in rows.to_dict("records"):
        values: list[str] = []
        for col in rows.columns:
            value = rec.get(col, "")
            text = str(value).replace("\n", "<br/>")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_live_sop_report(sop: pd.DataFrame) -> str:
    """生成每日 SOP Markdown 报告。"""
    if sop.empty:
        return "# 小资金实盘每日 SOP\n\n无步骤。\n"
    rec0 = sop.iloc[0].to_dict()
    strategy = str(rec0.get("strategy", ""))
    date_s = str(rec0.get("date", ""))
    phases = sop.groupby("phase", sort=False)["step"].count().to_dict()
    lines = [
        "# 小资金实盘每日 SOP - %s - %s" % (strategy, date_s),
        "",
        "这份 SOP 用来把研究回测、纸面交易、风险总控、人工确认、成交回填、归因和偏差分析串成一天的操作顺序。它不替你下单，也不改变策略，只把每天该做什么、看什么、什么情况停手写清楚。",
        "",
        "## 阶段",
        "",
    ]
    for phase, count in phases.items():
        lines.append("- `%s`：%d 步" % (phase, count))
    lines.extend(
        [
            "",
            "## 总原则",
            "",
            "- 有 `BLOCK`，当天不进入真实下单。",
            "- 有 `WATCH`，可以人工复核，但必须写明继续、降仓、跳过或延后的理由。",
            "- 只有半自动执行清单给出 `READY_FOR_MANUAL_ORDER`，才进入小资金人工下单。",
            "- 真实成交后必须回填，次日必须复盘；否则策略看起来在运行，实际上账户已经脱离系统。",
            "",
            "## 操作清单",
            "",
            _markdown_table(sop),
            "",
            "## 输出位置",
            "",
            "默认输出到 `output/live_sop/<strategy>/`。CSV 用于机器读取，Markdown 用于人工照着执行和复盘。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_live_daily_sop(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    sop: pd.DataFrame,
) -> dict[str, Path]:
    base = live_sop_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    date_s = _date_to_str(trade_date)
    csv_path = base / ("%s_daily_sop.csv" % date_s)
    md_path = base / ("%s_daily_sop.md" % date_s)
    sop.to_csv(csv_path, index=False)
    md_path.write_text(build_live_sop_report(sop), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}
