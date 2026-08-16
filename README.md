# Quant Strategy（MVP）

模块化量化研究项目：**数据存储工程化 → 数据 → 因子面板 → 因子清洗与行业内标准化 → 数据质量 → IC → 因子诊断（Top-K 多头超额 + 分组收益单调性）→ 多因子权重建议 → 样本外验证与因子失效监控 → 融合回测（IC 滚动列权 + 训练段静态综合权重 + 调仓日前滚动综合权重）→ 可交易性 / 流动性过滤 → 回测（Top-K + 等权 / 夏普 / 风险平价）→ 单票 / 行业 / 波动率 / 最小持仓约束 → 决策审计日志 → 基准与超额收益 → 换手与成本 → 风险暴露与集中度 → 绩效与作图 → 实验记录落盘 → 回撤止损与降仓 → 目标权重转订单计划 → 容量与冲击成本 → 订单预检查 → 风险预警 / 黑名单阻断 → 统一风险限额表 → 组合压力测试 → 风险总控日报 → 纸面交易 → 账户状态持久化 → 每日纸面交易运行器 → 日终纸面交易脚本 → 纸面交易日报 → 增强因子健康日报 → 运行失败 / 异常检查 → 交易日日历 / 重复运行保护 → 每日调度入口 → 统一券商接口协议 / 模拟券商适配器 → 日终纸面交易接入统一券商接口 → 真实券商只读 Adapter 骨架**。

**文档与代码**：以 `main.py` 与 `config.Settings` 为准；更新行为后请同步修改 `docs/ENGINEERING_OVERVIEW.md`、`docs/FLOW_AND_MODULES.md` 及本 README 相关段落（仓库无自动文档校验）。

### MVP 定稿（范围）

**本仓库 MVP 已交付**，指下面闭环可稳定跑通、用于研究与对内演示；不要求实盘下单。

| 在 MVP 内 | 不在 MVP 内（后续扩展） |
|------------|-------------------------|
| 行情接入（CSV / Tushare / 合成兜底）、多因子面板（量价 + 估值 + 质量 + 成长 + 现金流 + 公告事件）、横截面/行业内标准化、数据质量 / 覆盖率报告、IC 与可选 CSV/图、因子 Top-K 多头超额诊断、分组收益与单调性分析、多因子权重建议表、样本外验证与因子失效监控 | `live/signal_system.generate_signals`、真实券商 API |
| 月末再平衡、Top-K、可交易性 / 流动性过滤、停牌 / 涨跌停交易约束、`portfolio_weighting`：`equal` / `max_sharpe` / `risk_parity`，`max_position_weight` 单票权重上限，`max_industry_weight` 行业权重上限，`target_volatility` 波动率目标与现金仓位，`min_positions` 最小持仓数量，`max_rebalance_turnover` 单次换手上限 | `fuse_models` 除 `mean_zscore` / `mean` 外的 `method`（如 `dynamic`、`xgboost`） |
| 单因子回测 + **IC 驱动或等权** z-score 融合回测 + **训练段静态综合权重**验证回测 + **调仓日前滚动综合权重**回测、`meta["rebalance_log"]`、`meta["decision_log"]` | `main` 未接 `run_multi_backtest(factors, weights)` 原始因子线性加权入口（代码已有，非主流程） |
| 绩效 `summarize`、股票池等权基准、超额收益 / 跟踪误差 / 信息比率、换手率与预估成本、HHI / effective_n 持仓集中度、净值/IC/权重/换手/集中度/覆盖率图、`performance_summary.csv`、`run_config.json`、调仓/决策审计/换手/集中度日志 CSV、`persist_run_outputs` 落盘、`storage.database` SQLite 表结构初始化、`live.order_builder` 目标权重转订单计划、`live.drawdown_control` 账户级回撤止损与降仓、`live.capacity_impact` 容量与冲击成本估算、`live.order_precheck` 订单预检查、`live.risk_blacklist` 风险预警与黑名单阻断、`live.risk_gate` 统一风险门禁、`live.risk_limits` 统一风险限额表、`live.stress_test` 组合压力测试、`live.risk_control_report` 风险总控日报、`live.announcement_source` 真实公告数据源标准化、`live.event_risk_filter` 公告事件风险候选、`live.negative_sentiment_filter` 负面舆情风险候选、`live.paper_trading` 虚拟账户模拟成交、`live.account_state` 纸面账户状态持久化、`live.paper_runner` 单日纸面交易运行器、`scripts/run_daily_paper.py` 日终纸面交易脚本、`live.paper_report` 纸面交易日报、`live.factor_health_report` 增强因子健康总览、`live.manual_confirmation` 小资金人工确认实盘单、`live.execution_feedback` 真实成交回填与执行偏差分析、`live.paper_guard` 运行失败 / 异常检查、`live.paper_run_control` 交易日日历 / 重复运行保护、`scripts/run_scheduled_daily_paper.py` 每日调度入口、`live.broker` 统一券商接口协议与模拟券商适配器、真实券商只读 Adapter 骨架、日终纸面交易可通过 `--execution-mode simulated_broker` 走统一券商接口 | 真实券商交易 API、实时风控与订单路由 |

## 文档

- **项目介绍（MVP 工程）**：[docs/MVP_PROJECT_ARTICLE.md](docs/MVP_PROJECT_ARTICLE.md) — Quant Strategy 的定位、模块关系、默认全流程与主流程表、数据/因子/IC/融合/回测与后续扩展方向
- **主流程与各模块**：[docs/FLOW_AND_MODULES.md](docs/FLOW_AND_MODULES.md)（含 Mermaid 流程图）
- **工程总览（技术细节）**：[docs/ENGINEERING_OVERVIEW.md](docs/ENGINEERING_OVERVIEW.md)
- **接口与数据契约**：[docs/INTERFACE_AND_CONTRACTS.md](docs/INTERFACE_AND_CONTRACTS.md)
- **代码结构**：[docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)

原创长文与小红书草稿默认只在本地保留，并通过 `.gitignore` 排除，避免随公开代码仓库发布。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Token：优先环境变量 `TUSHARE_TOKEN`；未设置时使用 `config.py` 内 `_TUSHARE_TOKEN_LOCAL`（**勿将含真实密钥的 config 提交远程**）。真实股票池默认读取 `data/stock_pool.xlsx`，也可用环境变量 `QUANT_STOCK_POOL_PATH` 指向本机 Excel/CSV；行情拉取后默认缓存到 `data/prices_tushare_cache.csv`，也可用 `QUANT_TUSHARE_PRICE_CACHE` 改路径。SQLite 数据库默认路径为 `data/quant_strategy.db`，也可用 `QUANT_DATABASE_PATH` 改路径；数据库文件默认不提交远程。

## 目录结构

```
data/           # 原始/演示数据（如 prices_demo.csv、stock_pool.xlsx、本地行情缓存）
output/         # 运行生成：nav_compare.png、excess_nav_compare.png、turnover_compare.png、performance_summary.csv、cache/、data_quality/、factor_diagnostics/、market_regime/、risk_exposure/ 等
factors/        # 因子与 panel_builder；factor_events 支持公告总分和公告类型分层因子
backtest/       # backtest_single、backtest_multi、utils
models/         # fusion、factor_weighting、optimizer
analysis/       # performance、benchmark、turnover、risk_exposure、data_quality、factor_diagnostics、factor_validation、market_regime、plotting、ic
live/           # data_feed、stock_pool、cache_io、order_builder、drawdown_control、capacity_impact、order_precheck、risk_blacklist、risk_gate、risk_limits、stress_test、risk_control_report、announcement_source、news_source、event_risk_filter、negative_sentiment_filter、paper_trading、broker、account_state、paper_runner、paper_report、factor_health_report、manual_confirmation、execution_feedback、paper_guard、paper_run_control、daily_paper_cli；signal 非 MVP 占位
storage/        # SQLite 表结构与后续数据库读写层
scripts/        # init_database.py、update_database_cache.py、build_database_quality_report.py、build_live_universe.py、fetch_tushare_announcements.py、fetch_akshare_stock_news.py、build_event_risk_filter.py、build_announcement_event_type_analysis.py、build_announcement_event_type_backtest.py、build_announcement_event_type_risk_filter_backtest.py、build_negative_sentiment_filter.py、build_drawdown_control.py、build_capacity_impact.py、build_portfolio_risk_limits.py、build_portfolio_stress_tests.py、run_daily_paper.py、build_execution_feedback.py、run_scheduled_daily_paper.py 等日常运行入口
config.py
main.py
```

## 运行

```bash
python main.py
```

### SQLite 数据库初始化

第一版数据库层只负责建表，不改变当前 `main.py` 的 CSV 缓存流程：

```bash
python scripts/init_database.py
```

默认生成：

```text
data/quant_strategy.db
```

当前核心表：

```text
prices_daily          日线行情
fina_indicator        财务指标
factor_panel_daily    日频因子长表
announcement_events   公告事件
news_sentiment        新闻舆情
universe_snapshot     股票池快照
```

### 数据库缓存更新

第 96 篇之后，SQLite 可以开始参与主流程前的数据准备。脚本会读取本地行情 / 财务 / 因子 CSV，按主键 upsert 到 SQLite，再导出当前 `main.py` 和日终纸面交易仍然兼容的缓存文件：

```bash
python scripts/update_database_cache.py
```

常用参数：

```bash
python scripts/update_database_cache.py \
  --prices-csv data/prices_tushare_cache.csv \
  --fina-csv data/fina_indicator_cache.csv \
  --factor-panel-csv output/cache/factor_panel.csv \
  --start 2025-01-01 \
  --end 2026-08-15
```

导出结果：

```text
output/cache/prices_long.csv
output/cache/prices_wide_close.csv
output/cache/factor_panel.csv
```

这一步仍然不强制改造 `main.py`：数据库作为数据底座，导出的缓存作为兼容层。

### 数据库巡检日报

第 97 篇之后，可以对 SQLite 数据库和导出缓存做日常巡检：

```bash
python scripts/build_database_quality_report.py
```

常用参数：

```bash
python scripts/build_database_quality_report.py \
  --database data/quant_strategy.db \
  --stock-pool data/stock_pool.xlsx \
  --as-of-date 2026-08-16
```

输出目录：

```text
output/database_quality/
```

核心输出：

```text
table_summary.csv
price_health.csv
fina_health.csv
factor_health.csv
cache_file_health.csv
summary.csv
database_quality_report.md
```

巡检日报只做检查，不改变回测、调仓或订单。它回答的是：数据库里的数据是否完整、新鲜、可用，导出的缓存是否足够支撑当天运行。

### 当前 `main.py` 实际顺序（与代码一致）

0. **数据存储工程化**：`storage.database` 定义本地 SQLite 表结构，`scripts/init_database.py` 可初始化 `prices_daily`、`fina_indicator`、`factor_panel_daily`、`announcement_events`、`news_sentiment`、`universe_snapshot`。`storage.warehouse` 提供行情、财务和因子面板 upsert、读取和缓存导出；`scripts/update_database_cache.py` 可把本地 CSV 缓存增量写入 SQLite，再导出 `output/cache/prices_long.csv`、`prices_wide_close.csv` 和 `factor_panel.csv`；`storage.inspection` 与 `scripts/build_database_quality_report.py` 生成数据库巡检日报，检查表结构、行情新鲜度、财务覆盖、因子覆盖和缓存文件状态。
1. **数据**：`data/prices_demo.csv` 优先；否则读取 Tushare 行情缓存；若缓存不存在，则从 `Settings.stock_pool_path` 指定的 Excel/CSV 股票池读取标的并拉取 Tushare 日线，同时写入 `Settings.tushare_price_cache_path`；若股票池不存在才使用 `main._DEFAULT_TS_SYMBOLS` 示例股票池；失败则合成宽表。得到 `prices`（宽表）与 `long_df`。
2. **基础因子面板**：`factors.panel_builder.build_four_factor_panel`（历史命名保留；当前默认十五列：`MOMENTUM`、`MOMENTUM_60D`、`REVERSAL_5D`、`VOLATILITY`、`VOLUME_RATIO_20D`、`PE`、`ROE`、`GROSS_MARGIN`、`NET_MARGIN`、`LOW_DEBT_TO_ASSETS`、`REVENUE_GROWTH`、`PROFIT_GROWTH`、`FREE_CASH_FLOW_YIELD`、`CASH_PROFIT_QUALITY`、`ANNOUNCEMENT_EVENT_SCORE`）。公告事件因子默认读取 `data/announcement_events.csv`，文件不存在时该列为空；`factors.factor_events.calc_announcement_event_type_scores` 可把同一公告表拆成回购、减持、问询处罚、分红、合同项目等类型分层因子，用于单独诊断或后续接入策略。
3. **机器学习打分因子**：若 `enable_ml_score=True`，`factors.factor_ml.build_ml_score_factor` 用已有因子面板滚动训练梯度提升类模型，预测未来收益并追加 `ML_SCORE`；该列只作为候选因子进入后续 IC、分组收益、样本外验证和回测。
4. **因子清洗与行业内标准化**：`factors.preprocess.preprocess_factor_panel` 默认读取 `industry_col` 行业字段，在同一交易日同一行业内做 winsorize + z-score；缺行业或行业样本少于 `factor_industry_min_count` 时回退全股票池横截面 z-score。原始 `factor_panel.csv` 保留审计，IC/诊断/回测使用标准化后的研究面板。
5. **数据质量**：`analysis.data_quality` 输出价格覆盖、因子覆盖、调仓日覆盖报告；若 `persist_run_outputs`，保存到 `output/data_quality/`。
6. **落盘**：若 `persist_run_outputs`，`live.cache_io.save_run_cache` → `output/cache/`（`prices_long.csv`、`prices_wide_close.csv`、`factor_panel.csv`、`factor_panel_zscore.csv`、`run_meta.txt`）；若生成 `ML_SCORE`，另写 `output/factor_diagnostics/ml_score_training_log.csv`。
7. **IC 与稳定性诊断**：`analysis.ic` 对各因子列及 **与融合同构的** FUSED 得分算日截面 Spearman；同时输出 IC 分布分位数、正负占比和滚动稳定性；若 `persist_run_outputs`，另存 `ic_*.csv` 与 `output/ic_diagnostics/*.csv`。
8. **因子诊断**：`analysis.factor_diagnostics.batch_factor_long_excess` 对每个因子构造 Top-K 等权多头腿，并相对股票池等权基准输出 `excess_ann_return`、`tracking_error`、`information_ratio`；`batch_factor_group_returns` 按 `Settings.factor_group_count` 分组计算持有期收益、Top-Bottom、胜率与 `monotonicity_score`。
9. **多因子权重建议**：`models.factor_weighting.build_factor_weight_summary` 综合 IC 分布、rolling IC、Top-Bottom 与单调性，输出 `factor_score` 与 `fusion_weight`。全样本表用于诊断审计；训练段表用于 `FUSED_SCORE_WEIGHTED`；滚动权重日志用于 `FUSED_ROLLING_SCORE_WEIGHTED`；`analysis.factor_weight_stability` 进一步监控滚动权重的稳定性、漂移事件和组合层主导因子。
10. **样本外验证与因子失效监控**：`analysis.factor_validation` 按 `factor_weight_train_ratio` 切成训练段和验证段，分别计算 IC、多头超额、Top-Bottom 与单调性，并生成 `OK/WATCH/DEGRADED/FAILED` 状态表；同时可按滚动窗口输出 `rolling_out_of_sample_validation.csv` 与 `rolling_out_of_sample_summary.csv`，观察因子跨时间窗口是否稳定。
10A. **多股票池验证**：`analysis.multi_universe_validation` 与 `scripts/build_multi_universe_validation.py` 可读取多个已完成回测的 output 目录，汇总策略绩效和因子多头超额，输出跨股票池稳健性表。
11. **单因子回测**：对每列 `run_single_backtest(fname, factor_values=col, ...)`（**预计算因子**，不调注册表重算）。
12. **融合回测**：第一条是 **IC 滞后滚动列权（默认）或等权** z-score → `FUSED_ZSCORE`；第二条是训练段 `fusion_weight` 固定后应用到验证段 → `FUSED_SCORE_WEIGHTED`；第三条是每个调仓日前只用历史窗口重新计算权重 → `FUSED_ROLLING_SCORE_WEIGHTED`。三条都通过 `run_multi_backtest(fused=...)` 进入同一套 Top-K 回测。
13. **可交易性过滤**：若配置了 `min_avg_volume` 或 `min_avg_amount`，回测会在 Top-K 前按过去 `liquidity_lookback_days` 的平均成交量 / 成交额过滤候选股票；过滤前后候选数写入 `rebalance_log`。
14. **行业权重上限**：若 `max_industry_weight` 在 `(0, 1)`，回测会读取 `industry_col` 指定的行业字段，在目标权重生成后限制单个行业暴露，并把 `industry_cap_applied`、`max_industry_exposure`、`n_industries` 写入调仓日志。
15. **波动率目标**：若 `target_volatility > 0`，回测会用历史协方差估算目标组合年化波动；当估算波动超过目标时，只降低股票仓位、不加杠杆，剩余权重作为现金，并记录 `volatility_target_scale`、`cash_target_weight`。
16. **最小持仓数量**：若 `min_positions > 0` 且有效目标持仓数不足，回测会把股票总仓位缩到 `min_positions_exposure`，剩余作为现金，并把 `min_positions_applied` 写入日志。
17. **停牌 / 涨跌停约束**：若 `enable_trade_status_filter=True`，回测会读取 `is_suspended`、`is_limit_up`、`is_limit_down`，限制停牌买卖、涨停买入 / 加仓、跌停卖出 / 减仓，并把阻断原因写入 `decision_log`。
18. **决策审计日志**：回测同步生成 `meta["decision_log"]`，逐股票记录因子分数、排序、是否通过流动性过滤、是否入选、所属行业、交易状态、上期权重、原始目标权重、最终目标权重、动作和原因标签。
19. **风格层暴露与收益关联**：`analysis.style_exposure` 从融合策略调仓日志、复合风格分数和净值曲线计算逐期风格暴露、暴露汇总与暴露-下一期收益关联，解释策略实际偏向量价、质量等哪类风格。
20. **基准与超额收益**：`analysis.benchmark.equal_weight_benchmark_nav` 构造股票池等权基准；每条策略补 `benchmark_ann_return`、`excess_ann_return`、`tracking_error`、`information_ratio`。
21. **换手与成本**：`analysis.turnover` 从 `meta["rebalance_log"]` 估算逐期 `turnover`、`estimated_cost`，并汇总 `avg_turnover`、`total_turnover`、`estimated_total_cost`。
22. **风险暴露与集中度**：`analysis.risk_exposure` 从同一份调仓日志计算 `hhi`、`effective_n`、`top1_weight`、`top3_weight` 等，判断策略是否过度集中。
23. **实验记录与作图**：若 `persist_run_outputs`，保存 `output/cache/run_config.json`、`output/performance_summary.csv`、`output/ic_diagnostics/*.csv`、`output/factor_diagnostics/long_excess_summary.csv`、`output/factor_diagnostics/group_return_detail.csv`、`output/factor_diagnostics/group_return_summary.csv`、`output/factor_diagnostics/factor_weight_summary.csv`、`output/factor_diagnostics/factor_weight_train_summary.csv`、`output/factor_diagnostics/rolling_factor_weight_log.csv`、`output/factor_diagnostics/factor_weight_stability_summary.csv`、`output/factor_diagnostics/factor_weight_drift_events.csv`、`output/factor_diagnostics/factor_weight_portfolio_drift.csv`、`output/factor_diagnostics/style_exposure*.csv`、`output/factor_diagnostics/ml_score_training_log.csv`、`output/factor_validation/out_of_sample_validation.csv`、`output/factor_validation/factor_decay_monitor.csv`、`output/factor_validation/rolling_out_of_sample_validation.csv`、`output/factor_validation/rolling_out_of_sample_summary.csv`、`output/market_regime/*.csv`、`output/rebalance_logs/*.csv`、`output/decision_logs/*.csv`、`output/turnover_logs/*.csv`、`output/risk_exposure/*.csv`、`output/data_quality/*.csv`、`ic_compare.png`、`ic_timeseries_*.png`、`weights_*.png`、`turnover_compare.png`、`risk_exposure/effective_n_compare.png`；`plot_nav` → `output/nav_compare.png`，超额净值 → `output/excess_nav_compare.png`。多股票池验证脚本另写 `output/multi_universe_validation/*.csv`。
23. **订单计划**：`live.order_builder` 可把最近一期目标权重、当前持仓、现金 / 总资产和最新价格转换成 `BUY/SELL`、目标股数、调整股数、预估金额与交易原因；`live.cache_io.save_order_plans` 可保存到 `output/order_plans/*.csv`。该层不连接券商、不模拟成交。
24. **订单预检查**：`live.order_precheck` 对订单计划做现金、可卖数量、买入手数、最小订单金额、停牌 / 涨停买入 / 跌停卖出、风险黑名单检查；`live.cache_io.save_order_checks` 可保存到 `output/order_checks/*.csv`。该层只输出 `PASS/BLOCK` 和原因，不修改订单。
25. **纸面交易**：`live.paper_trading` 只执行通过预检查的订单，按手续费更新虚拟现金和持仓，记录 `FILLED/SKIPPED`、现金变化、持仓变化与原因；`live.cache_io.save_paper_trades` 可保存到 `output/paper_trades/*.csv`。
26. **纸面账户状态**：`live.account_state` 保存 / 读取纸面账户现金、持仓和每日快照，输出到 `output/paper_account/<strategy>/account.csv`、`positions.csv`、`snapshots.csv`。
27. **每日纸面交易运行器**：`live.paper_runner.run_daily_paper_trade` 读取上一日纸面账户状态，串联订单生成、预检查、成交执行、持仓更新、账户快照和落盘，形成可每天调用一次的纸面交易入口。默认沿用 `paper_trading`，也可用 `execution_mode="simulated_broker"` 通过统一券商接口执行。
28. **日终纸面交易脚本**：`scripts/run_daily_paper.py` 从 `output/rebalance_logs/<strategy>.csv` 读取最近一期目标权重，从 `output/cache/prices_wide_close.csv` 读取最新价格，调用每日纸面交易运行器并打印摘要。
29. **纸面交易日报与增强因子健康总览**：`live.paper_report` 将单日纸面运行结果整理为 Markdown，默认写入 `output/paper_reports/<strategy>/<date>.md`，便于复盘每天买卖、阻断、成交、持仓、账户变化、因子失效监控、目标组合风格暴露、统一风险门禁、风险黑名单和增强因子健康总览；`live.factor_health_report` 会读取 `factor_decay_monitor.csv`、`rolling_out_of_sample_summary.csv`、`factor_selection_summary.csv`、`factor_redundancy_report.csv`、`factor_weight_stability_summary.csv`、`factor_weight_drift_events.csv`、`strategy_regime_summary.csv`，压缩成日终可读的健康摘要。
30. **小资金人工确认实盘单**：`live.manual_confirmation` 基于同一份订单计划和预检查结果生成 `output/live_orders/<strategy>/<date>_manual_confirm.csv/.md`，预留人工执行回填字段；该层只给建议，不自动下单。
31. **真实成交回填与执行偏差分析**：`live.execution_feedback` 读取人工确认单中回填的 `executed_qty/executed_price`，比较建议订单和真实执行，输出 `output/execution_feedback/<strategy>/` 下的逐笔偏差、汇总和 Markdown 报告。
32. **运行失败 / 异常检查**：`live.paper_guard` 在日终纸面交易前后检查目标权重、价格日期、价格有效性、账户现金、持仓、订单检查和成交日志；ERROR 级问题直接阻断运行，WARNING 级问题进入命令摘要与日报。
33. **交易日日历 / 重复运行保护**：`live.paper_run_control` 从价格缓存提取交易日日历，默认阻断非交易日运行；若同一策略同一日期已有纸面账户快照，默认阻断重复写入，避免无意覆盖账户状态。
34. **每日调度入口**：`scripts/run_scheduled_daily_paper.py` 包装日终纸面交易命令，适合交给 cron / launchd / 服务器调度器调用，并把 stdout、stderr、参数和退出码写入 `output/scheduler_logs/<date>.log`。
35. **实盘目标池确认**：`scripts/build_live_universe.py` 从人工股票池和价格缓存生成 `stock_pool_filter_report_<date>.csv` 与 `active_universe_<date>.csv`，记录哪些股票通过、哪些被剔除以及剔除原因。后续纸面交易和券商接口应优先读取确认后的 active universe。
36. **统一券商接口协议与通道 Factory**：`live.broker` 定义 `BrokerAdapter`、`BrokerAccount`、`BrokerPosition`、`BrokerOrder`，并提供 `SimulatedBroker`；`live.broker_factory` 根据 `broker_mode/broker_provider` 创建模拟、只读或后续真实 Adapter。策略与订单层只依赖 `get_account/get_positions/get_orders/submit_order/cancel_order` 等统一方法；未来 QMT / PTrade / 掘金只需实现同一协议并注册到 Factory。
37. **日终纸面交易接入统一券商接口**：`scripts/run_daily_paper.py --execution-mode simulated_broker` 可把日终订单计划交给 `SimulatedBroker` 执行，并保留原有 `paper_trades`、账户状态和 Markdown 日报输出，同时额外返回统一券商订单回报 `broker_orders`。
38. **真实券商只读 Adapter 骨架**：`live.broker.RealBrokerReadOnlyAdapter` 和 `RealBrokerConfig` 定义真实券商接入的只读入口，可查询账户、持仓和订单快照；`submit_order/cancel_order` 会抛出 `BrokerReadOnlyError`，防止尚未验证前误下单。
39. **纸面账户 / 真实账户只读对账**：`live.broker_reconcile` 对比纸面账户和只读券商账户的现金、总资产和逐股票持仓差异；`scripts/reconcile_paper_broker.py` 可读取券商导出的账户 / 持仓 CSV，输出 `output/broker_reconciliation/<strategy>/` 下的 CSV 与 Markdown 对账报告。
40. **风险预警与黑名单机制**：`live.risk_blacklist` 可读取 `data/risk_blacklist.csv` 或 `--risk-blacklist` 指定的 CSV/XLSX，把人工风险、公告观察、舆情观察等标记标准化为有效黑名单；`live.order_precheck` 命中后默认直接 `BLOCK`，`live.paper_report` 和命令摘要会展示风险等级、原因和来源。
41. **公告事件风险过滤**：`live.event_risk_filter` 从 `announcement_events.csv` 中识别问询、处罚、立案、诉讼、退市风险等负面公告，输出风险候选；`scripts/build_event_risk_filter.py` 可生成 `event_risk_candidates_<date>.csv`，并可选导出 `risk_blacklist_<date>.csv` 供日终纸面交易使用。
42. **真实公告数据源与负面舆情过滤**：`live.announcement_source` 可把 Tushare 公告接口返回值标准化成 `announcement_events.csv`；`live.negative_sentiment_filter` 可把外部新闻 / 舆情 CSV/XLSX 转成 `BLACKLIST/WATCH` 风险候选，并可选导出黑名单文件接入订单预检查。
43. **统一风险门禁**：`live.risk_gate` 合并人工黑名单、公告风险候选和负面舆情候选，按 `BLOCK > WATCH > PASS` 输出统一门禁；`scripts/build_unified_risk_gate.py` 可导出日终纸面交易可读取的 `risk_blacklist_<date>.csv`。
44. **统一风险限额表**：`live.risk_limits` 把单票权重、Top3 集中度、effective_n、最低持仓数、现金缓冲、行业权重、单次换手、风险门禁命中和订单阻断统一成 `PASS/WATCH/BLOCK/NA` 限额检查；`scripts/build_portfolio_risk_limits.py` 可从目标权重、当前权重、行业映射、风险门禁和订单预检查文件生成 `portfolio_risk_limit_checks_<date>.csv` 与 Markdown 摘要。
45. **组合压力测试**：`live.stress_test` 对目标组合施加市场下跌、第一大持仓下跌、前三大持仓下跌、第一大行业下跌等情景，输出 `PASS/WATCH/BLOCK/NA` 压力测试结果；`scripts/build_portfolio_stress_tests.py` 可单独生成 `portfolio_stress_tests_<date>.csv` 与 Markdown 摘要，日终纸面交易会默认生成 `daily_stress_tests_<date>.csv` 并写入日报。
46. **回撤止损与降仓控制**：`live.drawdown_control` 读取纸面账户历史快照和当前持仓估值，按历史峰值计算账户回撤；默认 5% 回撤降到 70% 目标仓位、10% 回撤降到 50%、15% 回撤转现金。日终纸面交易会在订单生成前缩放目标权重，输出 `output/drawdown_control/<strategy>/daily_drawdown_control_<date>.csv` 并写入日报；`scripts/build_drawdown_control.py` 可单独生成检查表。
47. **容量与冲击成本**：`live.capacity_impact` 读取订单计划和过去 N 日成交额，估算单笔订单参与率、冲击成本 bps、冲击成本金额和当前订单距离容量阈值还有多少空间；默认单笔参与率超过 5% 进入 `WATCH`、超过 10% 进入 `BLOCK`。日终纸面交易会默认读取 `output/cache/prices_long.csv` 作为流动性历史，输出 `output/capacity_impact/<strategy>/daily_capacity_impact_*.csv` 并写入日报；`scripts/build_capacity_impact.py` 可单独生成检查表。
48. **风险总控日报**：`live.risk_control_report` 汇总运行检查、统一风险门禁、风险黑名单、回撤止损与降仓、容量与冲击成本、订单预检查、组合风险限额和组合压力测试，按 `BLOCK > WATCH > NA > PASS` 给出当天总控状态；日终纸面交易会输出 `output/risk_control_reports/<strategy>/daily_risk_control_report_<date>.csv` 并写入 Markdown 日报。
49. **数据存储工程化**：`storage.database` 定义本地 SQLite 表结构，默认初始化 `data/quant_strategy.db`，核心表包括 `prices_daily`、`fina_indicator`、`factor_panel_daily`、`announcement_events`、`news_sentiment`、`universe_snapshot`。`storage.warehouse` 支持行情、财务和因子面板增量写库与导出缓存；`storage.inspection` 支持数据库巡检日报，数据库用于长期基础数据，`output/` 继续保存每次实验、日报和图表。

### 日终纸面交易

先运行 `python main.py` 生成目标权重和价格缓存，再运行：

```bash
python scripts/run_daily_paper.py
```

常用参数：

```bash
python scripts/run_daily_paper.py --strategy FUSED_ROLLING_SCORE_WEIGHTED
python scripts/run_daily_paper.py --trade-date 2024-01-26
python scripts/run_daily_paper.py --trade-status data/trade_status.csv
python scripts/run_daily_paper.py --no-persist
python scripts/run_daily_paper.py --no-report
python scripts/run_daily_paper.py --no-manual-confirm
python scripts/run_daily_paper.py --factor-decay-monitor output/factor_validation/factor_decay_monitor.csv
python scripts/run_daily_paper.py --risk-blacklist data/risk_blacklist.csv
python scripts/run_daily_paper.py --industry data/stock_pool_ftse_china_a50_20260710.csv
python scripts/run_daily_paper.py --risk-limits config/risk_limits.csv
python scripts/run_daily_paper.py --stress-scenarios config/stress_scenarios.csv
python scripts/run_daily_paper.py --drawdown-rules config/drawdown_rules.csv
python scripts/run_daily_paper.py --capacity-rules config/capacity_rules.csv
python scripts/run_daily_paper.py --liquidity-history output/cache/prices_long.csv
python scripts/run_daily_paper.py --no-guard
python scripts/run_daily_paper.py --max-price-age-days 3
python scripts/run_daily_paper.py --allow-non-trading-day
python scripts/run_daily_paper.py --allow-rerun
python scripts/run_daily_paper.py --execution-mode simulated_broker
```

### 公告事件风险过滤

先从真实公告源生成统一公告事件表。Tushare Token 只从运行环境读取，建议用环境变量 `TUSHARE_TOKEN`，不要写进代码或文档：

```bash
python scripts/fetch_tushare_announcements.py --symbols 000001.SZ,600519.SH --start 2026-01-01 --end 2026-07-10 --output data/announcement_events.csv
```

公告事件表接入后，可先生成风险候选：

```bash
python scripts/build_event_risk_filter.py --events data/announcement_events.csv --as-of-date 2026-07-10
```

如需把 `BLACKLIST` 风险事件导出成黑名单文件：

```bash
python scripts/build_event_risk_filter.py --events data/announcement_events.csv --as-of-date 2026-07-10 --write-blacklist
```

输出默认写入 `output/event_risk_filter/`。生成的 `risk_blacklist_<date>.csv` 可以通过 `scripts/run_daily_paper.py --risk-blacklist ...` 接入订单预检查。

### 公告类型分层诊断

公告事件表接入后，可把回购、增持、减持、问询处罚、分红、合同项目等类型拆开诊断：

```bash
python scripts/build_announcement_event_type_analysis.py \
  --universe A50=data/prices_ftse_china_a50_real_20250101_20260710.csv\|data/stock_pool_ftse_china_a50_20260710.csv\|data/announcement_events.csv
```

脚本输出默认写入 `output/announcement_event_type_analysis/`，包括类型事件数、覆盖率、IC、多头超额、分组收益和 `type_factor_decision_table.csv`。这一步用于判断哪些公告类型适合作为收益候选，哪些更适合作为风险过滤输入，不会自动改变主策略权重。

公告类型诊断之后，可进一步比较“不用公告 / 公告总分 / 公告类型收益因子 / 公告类型收益+风险混合”的组合回测：

```bash
python scripts/build_announcement_event_type_backtest.py \
  --universe A50=data/prices_ftse_china_a50_real_20250101_20260710.csv\|data/stock_pool_ftse_china_a50_20260710.csv\|data/announcement_events.csv\|data/fina_indicator_ftse_china_a50_20250101_20260710.csv
```

脚本默认只运行 `ROLLING` 主策略口径，输出 `performance_summary.csv`、`incremental_effect.csv`、`rolling_incremental_return.png`、`nav_compare.png`、`rebalance_log_rolling.csv` 和各场景因子集合。`--include-equal` 可额外运行等权融合口径。

如果要把风险类公告从 alpha 打分中拆出来，改成调仓前候选股过滤：

```bash
python scripts/build_announcement_event_type_risk_filter_backtest.py \
  --prices data/prices_ftse_china_a50_real_20250101_20260710.csv \
  --stock-pool data/stock_pool_ftse_china_a50_20260710.csv \
  --events data/announcement_events.csv \
  --fina data/fina_indicator_ftse_china_a50_20250101_20260710.csv
```

脚本输出默认写入 `output/announcement_event_type_risk_filter_backtest/`，包括过滤前后 `performance_summary.csv`、`risk_filter_incremental_effect.csv`、`risk_filter_log.csv`、`rebalance_log_rolling.csv` 和 `nav_compare.png`。这一步用于验证负面公告作为风险门禁时，是否真的改变调仓候选和最终组合。

### 负面舆情风险过滤

新闻、舆情或人工整理的消息表可先统一成风险候选：

```bash
python scripts/build_negative_sentiment_filter.py --sentiment data/news_sentiment.csv --as-of-date 2026-07-10
```

如需导出给订单预检查使用的黑名单：

```bash
python scripts/build_negative_sentiment_filter.py --sentiment data/news_sentiment.csv --as-of-date 2026-07-10 --write-blacklist
```

舆情表至少需要股票代码和发布时间，标题 / 正文 / 来源 / 新闻链接 / 情绪分可选。若没有情绪分，工程会用负面关键词生成保守分数；若同一表中部分行缺少情绪分，缺失行会回退到关键词打分。该模块当前定位为新闻 / 舆情入口和风险过滤候选，不直接替代策略因子。新闻来源可以是 AkShare 近期个股新闻、手工 CSV、Tushare 权限接口或商业数据源；稳定缓存后再做历史回测和 alpha 验证。

AkShare 近期个股新闻可先拉成统一 `news_sentiment` 表：

```bash
python scripts/fetch_akshare_stock_news.py \
  --stock-pool data/stock_pool_ftse_china_a50_20260710.csv \
  --output data/news_sentiment_akshare.csv \
  --merge-existing
```

`factors.factor_news` 已定义 `NEWS_SENTIMENT_DECAY`、`NEWS_NEGATIVE_RISK_SCORE`、`NEWS_NEGATIVE_COUNT_7D`、`NEWS_HEAT_7D` 四个 MVP 新闻 / 舆情日频因子。AkShare 当前更像近期新闻接口，不是长历史新闻库，因此这一步优先服务每日缓存、风险过滤和后续逐步积累样本。

若要做近期新闻链路的烟雾级验证，可运行：

```bash
python scripts/build_news_sentiment_smoke_backtest.py \
  --sentiment output/news_sentiment_akshare_a50_sample.csv \
  --start 2026-07-01 \
  --end 2026-07-27 \
  --output-dir output/news_sentiment_smoke_backtest_a50_sample
```

该脚本只用于验证“真实新闻 → 日频因子 → 风险过滤对比”链路，不代表新闻因子已经完成长区间有效性验证。

若要在同一 A50 短窗口里比较“基础策略 / 加公告 / 加新闻 / 公告+新闻”四组表现，可运行：

```bash
python scripts/build_a50_event_news_weekly_smoke_backtest.py \
  --warmup-start 2026-04-01 \
  --start 2026-07-01 \
  --end 2026-07-26
```

若要把负面舆情作为风险门禁，而不是作为 alpha 因子加分，可运行：

```bash
python scripts/build_negative_sentiment_filter_backtest.py \
  --warmup-start 2026-04-01 \
  --start 2026-07-01 \
  --end 2026-07-26
```

该脚本会比较不过滤、只过滤 `BLACKLIST`、同时过滤 `BLACKLIST/WATCH` 三种口径，并输出负面候选、调仓日风险命中日志、净值对比和绩效汇总。

公告风险、负面舆情和人工黑名单可以进一步合并成统一门禁：

```bash
python scripts/build_unified_risk_gate.py \
  --trade-date 2026-07-24 \
  --stock-pool data/stock_pool_ftse_china_a50_20260710.csv \
  --events data/announcement_events_a50_20260701_20260726.csv \
  --sentiment data/news_sentiment_a50_20260701_20260726.csv \
  --output-dir output/unified_risk_gate_a50_20260724 \
  --write-blacklist \
  --include-watch-in-blacklist
```

输出包括统一门禁、风险明细、摘要和可选 `risk_blacklist_<date>.csv`。后续日终纸面交易可通过 `--risk-blacklist` 接入这份统一风险结果。

### 容量与冲击成本

订单计划生成后，可以单独估算订单参与率与冲击成本：

```bash
python scripts/build_capacity_impact.py \
  --trade-date 2026-07-24 \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --orders output/order_plans/FUSED_ROLLING_SCORE_WEIGHTED.csv \
  --liquidity-history output/cache/prices_long.csv \
  --write-default-rules
```

脚本默认按订单金额占过去 N 日平均成交额的比例计算参与率，并用简化平方根模型估算冲击成本。默认单笔参与率超过 5% 进入 `WATCH`，超过 10% 进入 `BLOCK`；缺少流动性数据时输出 `NA`，不能当作通过。

### 统一风险限额检查

目标权重进入纸面交易或人工确认前，可以先跑组合层统一风险限额检查：

```bash
python scripts/build_portfolio_risk_limits.py \
  --trade-date 2026-07-24 \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --industry data/stock_pool_ftse_china_a50_20260710.csv \
  --risk-gate output/unified_risk_gate_a50_20260724/risk_gate_20260724.csv \
  --order-checks output/order_checks/FUSED_ROLLING_SCORE_WEIGHTED.csv \
  --write-default-limits
```

脚本默认读取 `output/rebalance_logs/<strategy>.csv` 里不晚于 `--trade-date` 的最近一期目标权重，输出 `portfolio_risk_limit_checks_<date>.csv` 和 `portfolio_risk_limit_summary_<date>.md`。默认限额覆盖单票权重、Top3 集中度、effective_n、最低持仓数、现金缓冲、行业权重、单次换手、风险门禁命中和订单阻断；`WATCH` 进入人工复核，`BLOCK` 不应直接自动执行，`NA` 表示缺少输入，不能当作通过。

### 组合压力测试

目标权重进入纸面交易或人工确认前，也可以单独跑压力测试：

```bash
python scripts/build_portfolio_stress_tests.py \
  --trade-date 2026-07-24 \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --industry data/stock_pool_ftse_china_a50_20260710.csv \
  --total-asset 1000000
```

脚本默认读取 `output/rebalance_logs/<strategy>.csv` 里不晚于 `--trade-date` 的最近一期目标权重，输出 `portfolio_stress_tests_<date>.csv` 和 `portfolio_stress_test_summary_<date>.md`。默认情景覆盖市场下跌 3% / 5% / 8%、第一大持仓下跌 10%、前三大持仓下跌 8%、第一大行业下跌 8%；`WATCH` 表示坏情况已经接近账户承受阈值，`BLOCK` 表示不建议继续自动加仓，`NA` 多数意味着缺少行业映射等必要输入。

### 日终脚本综合输出

脚本默认读取 `output/rebalance_logs/<strategy>.csv` 与 `output/cache/prices_wide_close.csv`，输出订单计划、回撤止损与降仓检查、容量与冲击成本估算、订单预检查、组合风险限额检查、组合压力测试、风险总控日报、纸面成交、纸面账户状态、Markdown 日报和小资金人工确认单。`--no-persist` 可用于只检查流程和摘要，不写账户文件；`--no-report` 可只写 CSV 与账户状态，不生成日报；`--no-manual-confirm` 可关闭人工确认单；`--factor-decay-monitor` 可指定因子失效监控 CSV，并写入命令摘要、Markdown 日报和人工确认单；`--style-exposure` 可指定 `style_exposure.csv`，默认读取 `output/factor_diagnostics/style_exposure.csv` 并把最近一期目标组合风格暴露写入命令摘要和 Markdown 日报；`--risk-gate` 可指定统一风险门禁 CSV，用于命令摘要和 Markdown 日报展示 `PASS/WATCH/BLOCK`；`--risk-blacklist` 可指定风险黑名单，默认读取 `data/risk_blacklist.csv`，文件不存在则视为无黑名单，并进入订单预检查；`--risk-limits` 可指定自定义组合风险限额表，默认使用工程内置限额；`--stress-scenarios` 可指定自定义压力测试情景表，默认使用工程内置情景；`--drawdown-rules` 可指定自定义回撤止损与降仓规则，默认使用工程内置三档规则；`--capacity-rules` 可指定容量与冲击成本阈值表；`--liquidity-history` 可指定日频成交额历史，默认读取 `output/cache/prices_long.csv`；`--capacity-lookback-days` 控制平均成交额窗口；`--impact-coefficient-bps` 控制冲击成本代理模型斜率；`--industry` 可传行业映射，让日报中的行业最大权重限额和第一大行业压力测试从 `NA` 变成可判断状态；增强因子健康总览默认自动读取 `output/factor_validation/`、`output/factor_diagnostics/` 与 `output/market_regime/` 下的最新诊断 CSV，不重新计算研究指标；`--no-guard` 可临时关闭运行检查；`--max-price-age-days` 控制价格日期超过多少自然日后给出 stale warning；`--allow-non-trading-day` 允许在非交易日强制运行；`--allow-rerun` 允许覆盖同一交易日已有纸面账户快照；`--execution-mode simulated_broker` 可让日终流程通过统一模拟券商执行订单。

### 纸面账户 / 券商只读对账

当真实券商或量化终端能导出账户和持仓 CSV 后，可先做只读对账，不下单：

```bash
python scripts/reconcile_paper_broker.py \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --trade-date 2026-06-22 \
  --broker-account data/broker_account.csv \
  --broker-positions data/broker_positions.csv
```

账户 CSV 至少包含 `cash/market_value/total_asset`；持仓 CSV 至少包含 `symbol/shares`，可选 `available_shares`。输出包括账户差异、持仓差异和 Markdown 对账报告。

### 真实成交回填

人工在券商终端执行后，把真实成交数量和价格回填到人工确认单，再运行：

```bash
python scripts/build_execution_feedback.py \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --trade-date 2026-06-22
```

也可显式指定确认单：

```bash
python scripts/build_execution_feedback.py \
  --manual-confirm output/live_orders/FUSED_ROLLING_SCORE_WEIGHTED/2026-06-22_manual_confirm.csv
```

脚本输出到 `output/execution_feedback/<strategy>/`，包括逐笔执行偏差、汇总表和 Markdown 报告。该步骤只分析真实执行结果，不修改纸面账户、不连接券商。

### 每日调度入口

调度入口只负责“运行一次并记录日志”，不在 Python 内部常驻循环。可把它交给 cron、launchd 或服务器调度器：

```bash
python scripts/run_scheduled_daily_paper.py --strategy FUSED_ROLLING_SCORE_WEIGHTED
python scripts/run_scheduled_daily_paper.py --log-date 2024-01-26 --strategy TEST --no-persist
```

未识别参数会透传给 `run_daily_paper.py`，调度日志写到 `output/scheduler_logs/<date>.log`。脚本退出码与日终纸面交易一致，便于系统调度器判断成功或失败。

### 实盘目标池确认

在接券商接口前，先把人工研究池过滤成当日可用的 active universe：

```bash
python scripts/build_live_universe.py \
  --stock-pool data/stock_pool.xlsx \
  --prices output/cache/prices_wide_close.csv \
  --trade-date 2026-06-23
```

输出默认写入 `output/live_universe/`：

- `stock_pool_filter_report_<date>.csv`：完整过滤报告，含 `active` 与 `exclude_reason`。
- `active_universe_<date>.csv`：当天确认后的实盘目标池，只保留通过过滤的股票。

多股票池可用 `--output-subdir live_universe/<pool_name>` 分目录保存，避免同一日期互相覆盖。

### 回测与配置要点

- **再平衡**：默认 `config.rebalance_freq = "ME"`（月末）；**Top-K** 默认 `top_k=5`；因子截面**降序**取前 K。
- **IC 稳定性**：`config.ic_rolling_windows` 默认 `(20, 60)`；诊断层会统计 IC 分位数、正负占比、滚动均值和滚动正值比例。
- **因子分组**：`config.factor_group_count` 默认 `5`；诊断层按因子从低到高分组，`G1` 为低分组，`G5` 为高分组，观察 Top-Bottom 与单调性。
- **机器学习打分因子**：`enable_ml_score=True` 时，`ML_SCORE` 会用已有因子特征滚动训练并预测未来 `ml_score_forward_days` 日收益；`ml_score_model` 可设为 `lightgbm`、`catboost`、`xgboost`、`hist_gradient_boosting` 或 `auto`，缺少可选依赖时会回退到 sklearn 实现。它只是候选因子，仍需经过 IC、分组收益、样本外验证和回测。
- **公告事件因子**：`announcement_event_path` 默认指向 `data/announcement_events.csv`，也可用 `QUANT_ANNOUNCEMENT_EVENT_PATH` 指定。`factors.factor_events` 支持中文列名和显式 `event_score`；若没有分数字段，会用公告标题关键词生成粗略正负分，并按 `announcement_event_effective_days` 向后衰减成 `ANNOUNCEMENT_EVENT_SCORE`。`calc_announcement_event_type_scores` 可进一步按回购、增持、减持、问询处罚、业绩预告、分红、质押、诉讼、合同项目等类型生成分层事件因子。`scripts/fetch_tushare_announcements.py` 可从真实公告源生成同一格式文件，Token 只从运行环境读取。
- **行业内标准化**：`factor_standardize_by_industry=True` 默认开启；`main.py` 会从股票池 `子行业` / `分类` 或行情长表 `industry_col` 读取行业，在同一交易日同一行业内做 winsorize + z-score。若行业缺失或行业样本少于 `factor_industry_min_count`，该部分回退全股票池横截面 z-score。可用 `QUANT_FACTOR_STANDARDIZE_BY_INDUSTRY=0` 做关闭对照。
- **多因子权重建议**：全样本 `factor_weight_summary.csv` 用于观察权重是否合理；训练段 `factor_weight_train_summary.csv` 生成 `FUSED_SCORE_WEIGHTED`；滚动日志 `rolling_factor_weight_log.csv` 记录每个调仓日前实际使用的因子权重，并生成 `FUSED_ROLLING_SCORE_WEIGHTED`；`factor_weight_stability_summary.csv`、`factor_weight_drift_events.csv`、`factor_weight_portfolio_drift.csv` 用于观察滚动权重是否稳定、是否有跳变、是否被单一因子主导。
- **样本外验证与因子失效监控**：`analysis.factor_validation` 复用 IC、多头超额和分组收益口径，按 `factor_weight_train_ratio` 切分训练段 / 验证段，保存 `output/factor_validation/out_of_sample_validation.csv` 与 `factor_decay_monitor.csv`；滚动样本外验证按 `rolling_oos_train_days` / `rolling_oos_validation_days` / `rolling_oos_step_days` 多窗口复查因子稳定性，保存 `rolling_out_of_sample_validation.csv` 与 `rolling_out_of_sample_summary.csv`。
- **多股票池验证**：`scripts/build_multi_universe_validation.py` 读取多个已完成回测输出目录，生成 `strategy_universe_performance.csv`、`strategy_universe_robustness.csv`、`factor_universe_performance.csv`、`factor_universe_robustness.csv`，用于判断策略和因子是否只在某一个股票池里有效。
- **参数敏感性分析**：`scripts/build_parameter_sensitivity.py` 复用已有 `output/cache` 中的价格和标准化因子面板，对 `top_k`、调仓频率、配权方式、单票上限、换手上限、波动率目标等参数做一维扰动，生成 `parameter_sensitivity_detail.csv` 与 `parameter_sensitivity_summary.csv`，用于判断策略是否过度依赖某个精确参数。
- **牛熊市分段表现**：`analysis.market_regime` 用股票池等权基准的滚动收益和回撤把交易日标记为 `BULL/BEAR/SIDEWAYS`，并输出 `output/market_regime/*.csv`，用于观察每条策略在牛市、熊市和震荡市里的收益、超额、回撤和稳健性。
- **滚动因子权重保护**：`rolling_factor_weight_lookback_days` 控制历史窗口，`rolling_factor_weight_min_days` 控制最少历史样本，`rolling_factor_weight_min_weight` / `rolling_factor_weight_max_weight` 控制因子权重上下限，`rolling_factor_weight_smoothing` 控制新旧权重平滑。
- **可交易性过滤**：`min_avg_volume` / `min_avg_amount` 默认为 `0`，表示关闭；设为正数后，回测会在 Top-K 前用过去 `liquidity_lookback_days` 的平均成交量 / 成交额过滤候选股票。调仓日志会记录 `n_candidates_before_liquidity`、`n_candidates_after_liquidity`、`liquidity_filter_enabled` 等字段。
- **停牌 / 涨跌停约束**：`enable_trade_status_filter` 默认 `False`；开启后，回测读取 `long_prices` / `trade_status_data` 中的 `is_suspended`、`is_limit_up`、`is_limit_down`。停牌不能买卖，涨停不能买入 / 加仓，跌停不能卖出 / 减仓；`decision_logs` 会记录 `trade_blocked` 与 `trade_block_reason`。
- **持仓权重**：`config.portfolio_weighting`：`"equal"`、**`"max_sharpe"`（当前默认）** 或 **`"risk_parity"`**；后两者在再平衡日对 Top-K 用历史日收益估协方差（夏普另需 μ），分别调用 `models.optimizer.maximize_sharpe` / `risk_parity`，样本不足等失败则等权。
- **单票权重上限**：`config.max_position_weight` 默认 `0.4`；当优化权重可行且超过上限时，会裁剪并重新分配剩余权重，`rebalance_log[].weighting` 记录为 `max_sharpe_capped` / `risk_parity_capped` 等。
- **行业权重上限**：`config.max_industry_weight` 默认 `0`，表示关闭；设为 `(0, 1)` 后，回测会从 `long_prices` / `industry_data` 中读取 `industry_col`（默认 `industry`），限制单个行业目标权重，避免组合因为 Top-K 或优化器把资金集中到同一行业。
- **波动率目标**：`config.target_volatility` 默认 `0`，表示关闭；设为正数后，回测用 `volatility_target_lookback_days` 的历史收益协方差估算组合年化波动，若超过目标则按比例降低股票仓位，剩余记为现金。该 MVP 只降风险，不主动加杠杆。
- **最小持仓数量**：`config.min_positions` 默认 `0`，表示关闭；开启后若有效目标持仓数少于阈值，会把股票总仓位缩到 `min_positions_exposure`，剩余记为现金，避免可交易标的不足时硬满仓。
- **订单生成、容量冲击、预检查、纸面交易、账户状态、日报、人工确认单、真实成交回填、异常检查、运行控制与调度入口**：`config.order_lot_size` 默认 `100`，用于 A 股一手约束；`config.min_order_amount` 默认 `0`，可过滤金额太小的碎片订单；`config.order_cash_buffer` 默认 `0`，用于买入后现金缓冲检查；`config.paper_initial_cash` 默认 `1_000_000`，用于虚拟账户初始化。`live.capacity_impact` 会在订单计划生成后估算单笔订单参与率和冲击成本，缺少流动性数据时记为 `NA`，不能当作通过。`live.risk_control_report` 会把运行检查、风险门禁、黑名单、回撤、容量、订单预检查、风险限额和压力测试汇总成总控日报，优先级为 `BLOCK > WATCH > NA > PASS`。`live.paper_runner.run_daily_paper_trade` 可把这些步骤串成单日纸面交易流程，`scripts/run_daily_paper.py` 可从已有回测输出里自动读取输入并运行，`--execution-mode simulated_broker` 可通过统一模拟券商执行订单，`live.paper_report` 会生成 Markdown 日报并展示因子失效监控、增强因子健康总览、目标组合风格暴露、回撤止损与降仓、容量与冲击成本、组合风险限额、组合压力测试和风险总控日报，`live.factor_health_report` 会把因子准入、滚动样本外、权重漂移、冗余和牛熊市分段压缩成可读摘要，`live.style_exposure_monitor` 会从 `style_exposure.csv` 读取当前策略最近一期风格暴露，`live.manual_confirmation` 会生成小资金人工确认单，`live.execution_feedback` 会读取人工回填后的确认单并生成执行偏差报告，`live.paper_guard` 会在运行前后拦截 ERROR 级异常并记录 WARNING 级风险提示，`live.paper_run_control` 会默认阻断非交易日运行和重复覆盖同日快照，`scripts/run_scheduled_daily_paper.py` 可作为系统调度器的单次运行入口。`config.broker_mode` 默认 `simulated`，`live.broker_factory.create_broker_adapter` 可按 `broker_mode/broker_provider` 创建模拟或只读 Adapter；真实券商建议先用 `real_readonly` 验证资金、持仓和订单读取。当前不真实下单。
- **单次换手上限**：`config.max_rebalance_turnover` 默认 `1.0`；首次建仓不节流，之后若目标权重变化超过上限，会按比例向新目标移动，`rebalance_log` 记录 `target_turnover`、`turnover_capped`、`turnover_scale`。
- **调仓记录**：`meta["rebalance_log"]`；`main` 会打印每期标的与权重，并在 `persist_run_outputs=True` 时保存到 `output/rebalance_logs/*.csv`，其中包含流动性过滤前后候选数、行业上限是否触发、最大行业暴露、目标波动缩放比例、最小持仓检查和现金目标仓位等。
- **决策审计记录**：`meta["decision_log"]`；在 `persist_run_outputs=True` 时保存到 `output/decision_logs/*.csv`，逐股票解释 `buy` / `sell` / `increase` / `decrease` / `hold` / `skip` 及原因，并记录所属行业与行业上限调整标记。
- **换手记录**：`analysis.turnover` 以调仓目标权重变化估算成交占比，并在 `persist_run_outputs=True` 时保存到 `output/turnover_logs/*.csv`。
- **集中度记录**：`analysis.risk_exposure` 以 HHI 与 effective_n 衡量持仓是否集中，并在 `persist_run_outputs=True` 时保存到 `output/risk_exposure/`。

### 依赖

见 `requirements.txt`（含 **pandas、openpyxl、numpy、scipy、matplotlib、scikit-learn、tushare** 等）。

### 非 MVP（占位或扩展）

- `live/signal_system.generate_signals`
- `models.fusion.fuse_models`：仅 `mean_zscore` / `mean` 可用，其它 `method` 会报错

### 测试

```bash
python3 -m unittest tests.test_optimizer tests.test_backtest_multi tests.test_backtest_single tests.test_plotting tests.test_fusion tests.test_cache_io tests.test_benchmark tests.test_turnover tests.test_data_quality tests.test_risk_exposure tests.test_storage_database tests.test_storage_warehouse tests.test_storage_inspection tests.test_factors tests.test_factor_events tests.test_announcement_source tests.test_negative_sentiment_filter tests.test_factor_ml tests.test_factor_preprocess tests.test_factor_diagnostics tests.test_factor_validation tests.test_multi_universe_validation tests.test_parameter_sensitivity tests.test_market_regime tests.test_factor_weight_stability tests.test_style_exposure tests.test_style_exposure_monitor tests.test_ic tests.test_factor_weighting tests.test_stock_pool tests.test_order_builder tests.test_order_precheck tests.test_capacity_impact tests.test_risk_blacklist tests.test_event_risk_filter tests.test_paper_trading tests.test_broker tests.test_broker_reconcile tests.test_account_state tests.test_paper_runner tests.test_risk_limits tests.test_stress_test tests.test_risk_control_report tests.test_daily_paper_cli tests.test_paper_report tests.test_manual_confirmation tests.test_execution_feedback tests.test_paper_guard tests.test_paper_run_control tests.test_paper_scheduler -v
```
