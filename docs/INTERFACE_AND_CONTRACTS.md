# 接口与数据契约（Interface & Data Contracts）

本文档约定各模块的**输入/输出形态**与**对齐规则**，不规定具体算法。实现时可替换内部逻辑，但**对外契约**应尽量稳定，便于回测与实盘共用同一套数据结构。

---

## 1. 时间与标识

| 字段 | 类型 | 说明 |
|------|------|------|
| `trade_date` / `date` | `datetime64[ns]` 或 `YYYY-MM-DD` 字符串（读入后统一为日频 Timestamp，**交易日**，非自然日需与行情对齐） | 截面排序、再平衡、标签对齐的基准轴 |
| `symbol` / `ts_code` | `str` | 如 `600519.SH`、`000001.SZ`；**全项目统一一种命名**（推荐 Tushare 风格 `ts_code`） |

**约定**：长表（long）面板使用二级索引 `(date, symbol)`，且 `date` 升序、`symbol` 字典序，无重复键。

---

## 2. 磁盘数据（`data/`）

### 2.0 SQLite 数据库：`data/quant_strategy.db` 或环境变量 `QUANT_DATABASE_PATH`

第一版数据库层只定义长期复用的基础数据表，不替代 `output/` 下的单次运行结果。默认路径为 `data/quant_strategy.db`，可通过 `QUANT_DATABASE_PATH` 覆盖。数据库文件属于本地数据资产，不提交到 GitHub。

初始化入口：

```bash
python scripts/init_database.py
```

增量写库与缓存导出入口：

```bash
python scripts/update_database_cache.py
```

核心表契约：

| 表名 | 主键 | 用途 |
|------|------|------|
| `prices_daily` | `(trade_date, ts_code)` | 日线行情；当前包含 `open/high/low/close/adj_factor/adj_close/volume/amount/source/updated_at` |
| `fina_indicator` | `(ts_code, ann_date, end_date)` | 财务指标；财务因子按 `ann_date` 对齐，避免未来函数 |
| `factor_panel_daily` | `(trade_date, ts_code, factor_name, factor_version)` | 日频因子长表；新增因子只增加行，不频繁改宽表字段 |
| `announcement_events` | `event_key` | 公告事件；用于公告事件因子、公告类型分层和公告风险过滤 |
| `news_sentiment` | `item_key` | 新闻舆情；用于新闻日频因子与负面舆情风险过滤 |
| `universe_snapshot` | `(snapshot_date, universe_name, ts_code)` | 股票池快照；记录候选池、启用状态、行业主题与剔除原因 |
| `storage_metadata` | `key` | 数据库元信息；当前记录 `schema_version` |

读写接口：

| 函数 | 输入 | 输出 |
|------|------|------|
| `storage.warehouse.upsert_prices_daily` | 行情长表，含 `trade_date/ts_code/open/high/low/close`，可选 `adj_factor/adj_close/volume/amount` | 写入 `prices_daily`，同一 `(trade_date, ts_code)` 重复时更新；若有 `adj_factor` 但无 `adj_close`，按前复权口径生成研究价格 |
| `storage.warehouse.load_prices_daily` | 数据库路径、日期区间、可选股票列表 | 行情长表 |
| `storage.warehouse.export_price_cache` | 数据库路径、输出目录、日期区间、可选股票列表 | `prices_long.csv`、交易口径 `prices_wide_close.csv`；若有复权价，额外导出研究口径 `prices_wide_adj_close.csv` |
| `storage.warehouse.upsert_fina_indicator` | 财务指标表，含 `ts_code/ann_date`，可选 `end_date` 与各类财务字段 | 写入 `fina_indicator`，同一 `(ts_code, ann_date, end_date)` 重复时更新 |
| `storage.warehouse.upsert_factor_panel_daily` | MultiIndex 因子面板或含 `date/symbol` 的宽表 | 写入 `factor_panel_daily` 长表 |
| `storage.warehouse.export_factor_panel_cache` | 数据库路径、输出目录、日期区间、可选股票列表 | `factor_panel.csv` |
| `storage.inspection.build_database_quality_report` | 数据库路径、可选股票池、巡检日期、缓存目录 | 表级、行情、财务、因子、缓存文件巡检 DataFrame |
| `storage.inspection.save_database_quality_report` | 数据库路径、输出目录、可选股票池 / 巡检日期 / 缓存目录 | `output/database_quality/*.csv` 与 `database_quality_report.md` |

数据库与 CSV / 输出目录的边界：

| 数据类型 | 推荐位置 | 说明 |
|----------|----------|------|
| 行情、财务、日频因子、公告、新闻、股票池快照 | SQLite | 长期复用、可增量更新、需要主键约束 |
| 净值图、绩效汇总、调仓日志、纸面交易日报、风险日报 | `output/` | 单次运行结果，作为实验档案保留 |

### 2.1 股票池：`data/stock_pool.xlsx` 或环境变量 `QUANT_STOCK_POOL_PATH`

**最低必需列**：

| 列名 | dtype | 说明 |
|------|--------|------|
| `股票代码` | str | 标的代码；推荐 Tushare 风格 `001309.SZ`，也支持 `600519` 等常见格式并在 `live.stock_pool` 规范化 |

股票池文件支持 `.xlsx` / `.xls` / `.csv`。本地真实股票池通常不进入 Git；默认 `.gitignore` 会忽略 `data/*.xlsx`、`data/*.xls` 与 `data/*.csv`。

### 2.2 行情：`data/stock_<ts_code中的数字部分>.csv` 或聚合多标的 `prices.csv`（二选一需在 `config` 中声明）

**最低必需列**：

| 列名 | dtype | 说明 |
|------|--------|------|
| `trade_date` | date | 交易日 |
| `ts_code` | str | 标的代码 |
| `open` | float64 | 开盘价 |
| `high` | float64 | 最高价 |
| `low` | float64 | 最低价 |
| `close` | float64 | 原始收盘价；用于真实成交、订单金额和持仓市值 |
| `adj_factor` | float64 | 复权因子；用于修正除权除息、送转、配股等公司行为造成的历史价格断点 |
| `adj_close` | float64 | 复权后的研究价格；用于动量、反转、波动率、历史收益和回测收益等研究口径 |
| `vol` 或 `volume` | float64 | 成交量；读入后可在 `data_feed` 层统一重命名为 `volume` |

可选：`amount`（成交额）、`pct_chg` 等；**缺列时由加载层报错或按 config 填充策略处理**。

### 2.3 财务：`data/finance_data.csv`

**最低必需列**（用于 PE、ROE、质量、成长与现金流因子；具体财报发布日对齐在因子层实现，契约只要求列存在）：

| 列名 | dtype | 说明 |
|------|--------|------|
| `ts_code` | str | 标的 |
| `ann_date` | date | 公告日；财务因子按它 backward 对齐到交易日，避免未来函数 |
| `eps` / `roe` / `grossprofit_margin` / `netprofit_margin` / `debt_to_assets` / `or_yoy` / `netprofit_yoy` / `fcff_ps` / `fcfe_ps` / `ocfps` / `cfps` / `ocf_to_profit` / `ocf_to_or` 等 | float64 | 由具体因子声明子集；缺失则该 (date, symbol) 因子为 NaN |

财务与行情对齐：**不做全局强制**；由 `factors/*` 在输出前将结果 reindex 到目标 `(date, symbol)` 网格。

---

## 3. 内存中的核心对象

### 3.1 `PanelLong`（长表面板）

- **类型**：`pd.Series` 或 `pd.DataFrame`，**索引**为 `pd.MultiIndex`，名称为 `["date", "symbol"]`。
- **语义**：每个 `(date, symbol)` 一条记录；用于因子值、信号、权重（单列用 Series，多列用 DataFrame）。

### 3.2 `PricePanel`（宽表，可选）

- **类型**：`pd.DataFrame`，索引 `date`，列 `symbol`，值为 `close`（或 OHLC 用 `pd.MultiIndex` columns，**若采用需在 config 固定**）。
- **用途**：部分回测/优化内部计算；**从磁盘加载后可在 `backtest_utils` 中转换为宽表**。

### 3.3 `NavSeries`（净值曲线）

- **类型**：`pd.Series`，索引 `date`，值 `float`，名称建议 `nav`。
- **约束**：严格递增索引、无重复日期；首日可为 1.0 或实际资金，**全项目统一在 `config`**。

### 3.4 `ICSeries`（optional）

- **类型**：`pd.Series`，索引 `date`，值为当日截面 IC（因子 vs 未来收益）。
- **默认实现**（`analysis.ic`）：截面 **Spearman** 相关；因子在日期 `t` 与前瞻收益 **close(t+h)/close(t)−1** 对齐，`h` 由 `config.ic_forward_days` 指定（默认 `1`）。因子在 `(t, symbol)` 处须仅依赖 **≤t 已公开数据**（与 §5.3 一致）；收益窗口起点为 `t` 收盘、不含更早未公开信息。

---

## 4. 模块级接口（函数签名契约）

命名与参数以代码中 docstring 为准；此处为**逻辑契约摘要**。

| 模块 | 函数 | 输入契约 | 输出契约 |
|------|------|-----------|-----------|
| `config` | `get_settings()` | 无 | 只读配置对象（路径、费率、回测区间、价格列名等） |
| `live.stock_pool` | `load_stock_pool(path, code_col="股票代码")` | Excel/CSV 股票池文件路径 | 去重后的 Tushare `ts_code` 列表 |
| `live.stock_pool` | `normalize_ts_code(value)` | 常见股票代码写法 | 规范化为 `000001.SZ` / `600519.SH` 等 Tushare 风格代码 |
| `live.data_feed` | `get_data_tushare(symbol, start, end, ...)` | 合法 `ts_code`、ISO 日期 | 满足 §2.1 列规范的 `pd.DataFrame`（可含额外列） |
| `live.data_feed` | `get_adj_factor_tushare(symbol, start, end, ...)` / `fetch_adj_factor_panel(...)` / `merge_adj_factor(...)` | 合法 `ts_code`、ISO 日期、日线行情长表 | 拉取并合并 `adj_factor`，生成研究用 `adj_close` |
| `live.data_feed` | `load_prices_from_csv(path_or_glob)` | 磁盘路径 | 长表或宽表 + 元数据说明（推荐返回 long 并标准化列名） |
| `live.cache_io` | `save_run_cache(settings, long_df, prices_wide, panel, panel_zscore=None)` | `Settings`、行情、原始因子面板与可选标准化面板 | 写 `output/cache/` 下 `prices_long.csv`、`factor_panel.csv`、`factor_panel_zscore.csv` 等 |
| `storage.warehouse` | `upsert_prices_daily`、`upsert_fina_indicator`、`upsert_factor_panel_daily`、`export_price_cache`、`export_factor_panel_cache` | 行情、财务、因子面板 DataFrame 与 SQLite 路径 | 按主键增量写入 SQLite，并导出当前 `main.py` / 日终纸面交易兼容缓存 |
| `storage.inspection` | `build_database_quality_report`、`save_database_quality_report` | SQLite 路径、可选股票池、巡检日期、缓存目录 | 生成数据库巡检明细与 Markdown 日报，只检查不改变交易逻辑 |
| `live.cache_io` | `save_run_config`、`save_performance_summary`、`save_rebalance_logs`、`save_decision_logs`、`save_turnover_logs`、`save_order_plans`、`save_order_checks`、`save_paper_trades`、`save_risk_exposure_logs`、`save_risk_exposure_summary`、`save_data_quality_reports`、`save_factor_diagnostics` | `Settings`、绩效 dict、回测 meta、换手表、订单计划、订单预检查结果、纸面交易日志、集中度表、数据质量表、因子诊断表 | 写 `run_config.json`、`performance_summary.csv`、`rebalance_logs/*.csv`、`decision_logs/*.csv`、`turnover_logs/*.csv`、`order_plans/*.csv`、`order_checks/*.csv`、`paper_trades/*.csv`、`risk_exposure/*.csv`、`data_quality/*.csv`、`factor_diagnostics/*.csv` |
| `factors.factor_*` | `calc_*(..., **kwargs)` | 行情/财务 DataFrame 或 PanelLong；财务扩展因子按 `ann_date` backward 对齐到交易日 | `PanelLong`（Series 或单列表 DataFrame） |
| `factors.factor_ml` | `forward_return_label(prices, forward_days=...)`、`build_ml_score_factor(panel, prices, settings, feature_cols=...)` | 基础因子面板、价格宽表、`Settings.ml_score_*` 配置；训练样本只允许使用预测日前已经能观察到完整 forward return 的历史样本 | `ML_SCORE` Series 与训练日志 DataFrame；`main` 会把 `ML_SCORE` 追加进因子面板，训练日志可写 `output/factor_diagnostics/ml_score_training_log.csv` |
| `factors.preprocess` | `winsorize_series`、`cross_sectional_zscore`、`preprocess_factor_panel` | 原始因子面板 | 清洗后的横截面 z-score 面板 |
| `backtest.backtest_utils` | `to_returns(prices, price_col="close", ...)` | 宽表或长表（需约定） | 宽表 `pct_change` 或与输入同型的收益 |
| `backtest.backtest_utils` | `align_panel(factor, prices, ...)` | 因子与价格时间轴 | 对齐后的联合索引，缺失为 NaN |
| `backtest.backtest_single` | `run_single_backtest(factor_name, ...)` | `factor_name` 或预计算因子、可选 `long_prices` / `liquidity_data` / `trade_status_data` / `industry_data`、`Settings.portfolio_weighting`（`equal` / `max_sharpe` / `risk_parity`）、`Settings.max_position_weight`、`Settings.max_industry_weight` / `industry_col`、`Settings.target_volatility`、`Settings.min_positions`、`Settings.max_rebalance_turnover`、`Settings.min_avg_volume` / `min_avg_amount`、`Settings.enable_trade_status_filter` | `NavSeries` + `meta`（含 `rebalance_log`；含 `decision_log`：逐股票 `factor_score/factor_rank/passed_liquidity_filter/selected_by_signal/industry/industry_cap_applied/volatility_target_applied/min_positions_applied/is_suspended/is_limit_up/is_limit_down/trade_block_reason/previous_weight/raw_target_weight/final_target_weight/action/decision_reason` 等） |
| `backtest.backtest_multi` | `run_multi_backtest(fused=..., prices=...)` 或 `run_multi_backtest(factors, weights=..., prices=...)` | 已融合得分 **或** 多列因子 + 线性权重 | `NavSeries` + `meta`（含 `multi_mode`：`pre_fused` / `linear_weight`） |
| `models.optimizer` | `maximize_sharpe` / `risk_parity` | `mu`、`cov` 与标的顺序一致（`risk_parity` 仅需 `cov`） | 权重向量；`maximize_sharpe` / `risk_parity` 在对应 `portfolio_weighting` 时由回测于再平衡日调用 |
| `models.fusion` | `fuse_equal_weight_zscore`、`fuse_ic_weighted_zscore`、`fuse_static_weight_zscore`、`fuse_models(...)` | 多列因子 Panel；`fuse_ic` 另需各列日 IC `Series`；静态融合另需 `{factor: weight}` | 单列综合得分 `PanelLong` |
| `models.factor_weighting` | `build_factor_weight_summary` | IC 分布表、IC rolling 稳定性表、分组收益汇总表、因子顺序 | `factor_score` / `fusion_weight` 表；全样本用于诊断，训练段用于 `FUSED_SCORE_WEIGHTED`，调仓日前滚动窗口用于 `FUSED_ROLLING_SCORE_WEIGHTED` |
| `analysis.ic` | `daily_ic_spearman`、`summarize_ic`、`ic_distribution_summary`、`ic_rolling_stability`、`save_ic_series`、`save_ic_diagnostics` | `PanelLong` 单列、价格宽表、`forward_days` / `Settings.ic_forward_days`、IC 序列字典 | `ICSeries`、基础汇总 dict、IC 分布表、滚动稳定性表、可选 `output/cache/ic_*.csv` 与 `output/ic_diagnostics/*.csv` |
| `analysis.factor_diagnostics` | `factor_long_only_nav`、`factor_long_excess_summary`、`batch_factor_long_excess`、`factor_group_return_detail`、`summarize_group_returns`、`batch_factor_group_returns` | 单列或多列因子面板、价格宽表、Top-K、分组数、再平衡频率 | 因子 Top-K 多头净值、相对股票池等权基准的超额摘要、分组收益明细、分组收益汇总与单调性评分 |
| `analysis.factor_validation` | `build_out_of_sample_validation`、`build_factor_decay_monitor`、`save_factor_validation_outputs` | 多列因子面板、价格宽表、`Settings`、因子列表、训练段比例 | 训练段 / 验证段 IC、多头超额、Top-Bottom 与单调性对照表；因子失效监控状态表；`output/factor_validation/*.csv` |
| `analysis.performance` | `summarize(nav, risk_free=0.0, periods=252)` | `NavSeries` | `dict`：`ann_return`, `ann_vol`, `sharpe`, `max_drawdown`, … |
| `analysis.data_quality` | `price_coverage`、`factor_coverage`、`factor_daily_coverage`、`rebalance_coverage` | 价格宽表、因子面板、调仓日序列 | 价格/因子/调仓日覆盖率报告 |
| `analysis.benchmark` | `equal_weight_benchmark_nav`、`summarize_excess`、`excess_nav_frame` | 价格宽表 / 策略净值 / 基准净值 | 股票池等权基准、超额收益指标、超额净值宽表 |
| `analysis.turnover` | `turnover_frame`、`summarize_turnover`、`turnover_wide` | `meta["rebalance_log"]`、手续费率 | 逐期换手表、换手/成本汇总、换手宽表 |
| `analysis.risk_exposure` | `concentration_frame`、`summarize_concentration`、`effective_n_wide` | `meta["rebalance_log"]` | 逐期集中度表、集中度汇总、effective_n 宽表 |
| `analysis.plotting` | `plot_nav`、`plot_ic`、`plot_weights`、`plot_turnover`、`plot_effective_n`、`plot_factor_coverage`、`rebalance_log_to_weights_frame` | 净值；日 IC；权重宽表；换手宽表；effective_n 宽表；覆盖率表 | `save_path` 有值则 Agg 写 PNG，否则 `show` |
| `live.signal_system` | `generate_signals(fused_score, rules, ...)` | `PanelLong` | `PanelLong` 取值 ∈ {-1, 0, 1} 或连续仓位 |
| `live.order_builder` | `build_order_plan(target_weights, current_positions, latest_prices, ...)` | 目标权重、当前持仓、最新价格、现金或总资产、手数与最小订单金额 | 订单计划 DataFrame，列含 `date/symbol/side/current_shares/target_shares/delta_shares/price/estimated_amount/current_weight/target_weight/trade_reason` |
| `live.order_builder` | `build_order_plan_from_rebalance_meta(meta, current_positions, latest_prices, ...)` | 回测 `meta["rebalance_log"]` 最近一期、当前持仓、最新价格 | 从最近一期目标权重生成订单计划 |
| `live.order_precheck` | `precheck_order_plan(order_plan, cash, current_positions=None, trade_status=None, ...)` | 订单计划、可用现金、当前持仓 / 可用股数、停牌 / 涨跌停状态、手数、最小订单金额、现金缓冲 | 订单检查 DataFrame，列含 `check_status/check_reason/cash_before/cash_after/available_shares/is_suspended/is_limit_up/is_limit_down` |
| `live.paper_trading` | `run_paper_trading(orders=..., order_checks=..., current_positions=..., ...)` | 订单计划、订单预检查结果、当前持仓、虚拟现金、手续费率 | 纸面成交 / 跳过日志 DataFrame，列含 `date/symbol/side/qty/price/gross_amount/commission/net_cash_flow/cash_before/cash_after/position_before/position_after/fill_status/fill_reason` |
| `live.paper_trading` | `paper_account_snapshot(trades, latest_prices, current_positions=None)` | 纸面成交日志、最新价格、可选初始持仓 | 账户快照 dict：`cash/market_value/total_asset/n_positions` |
| `live.account_state` | `save_account_state(settings, strategy, cash, positions, snapshot=None, trade_date=None)` | 纸面账户现金、持仓、账户快照、日期 | 写 `output/paper_account/<strategy>/account.csv`、`positions.csv`、`snapshots.csv` |
| `live.account_state` | `load_account_state(settings, strategy, default_cash=0.0)` | 策略名与默认现金 | 返回 `(cash, positions_df)`；状态不存在时返回默认现金和空持仓 |
| `live.account_state` | `positions_from_trades(trades, current_positions=None, updated_at=None)` | 纸面交易日志和初始持仓 | 最新持仓 DataFrame，列含 `symbol/shares/available_shares/updated_at` |
| `live.paper_runner` | `run_daily_paper_trade(settings, strategy, target_weights, latest_prices, trade_date, execution_mode=..., ...)` | `Settings`、策略名、目标权重、最新价格、交易日期、可选交易状态；内部读取纸面账户状态；执行模式支持 `paper_trading` / `simulated_broker` | 单日纸面运行结果 dict，含订单计划、订单预检查、券商订单回报、纸面交易日志、最新现金、最新持仓、账户快照与落盘路径 |
| `live.paper_report` | `build_daily_paper_report(result)`、`save_daily_paper_report(settings, result)` | 单日纸面运行结果 dict；可读取 `paper_account/<strategy>/snapshots.csv` 计算上一快照变化；若 result 含 `factor_decay_monitor`，展示因子健康状态 | Markdown 日报文本或 `output/paper_reports/<strategy>/<date>.md`；若存在 `broker_orders`，会展示统一券商订单回报；若存在因子失效监控表，会展示 `OK/WATCH/DEGRADED/FAILED` 状态和关键验证指标 |
| `live.manual_confirmation` | `build_manual_confirmation_sheet(result, factor_monitor=...)`、`save_manual_confirmation(settings, result, factor_monitor=...)` | 单日纸面运行结果 dict；可选 `factor_decay_monitor.csv` 读入表 | 人工确认 CSV / Markdown，默认写 `output/live_orders/<strategy>/<date>_manual_confirm.csv/.md`；只辅助人工复核，不自动下单 |
| `live.execution_feedback` | `build_execution_feedback(manual_confirmation)`、`save_execution_feedback(settings, detail, summary)` | 人工确认 CSV / DataFrame，需含 `date/strategy/symbol/side/delta_shares/price/estimated_amount`，可选 `executed_qty/executed_price/operator/confirmed_at/execution_note` | 逐笔执行偏差表、汇总表和 Markdown 报告，默认写 `output/execution_feedback/<strategy>/<date>_execution_feedback.csv`、`*_execution_summary.csv`、`*_execution_feedback.md` |
| `live.performance_attribution` | `build_performance_attribution(snapshots, strategy, trade_date, prices=None, positions=None, execution_feedback=None)`、`save_performance_attribution(settings, strategy, trade_date, summary, stock)` | 账户快照需含 `date/total_asset`，建议含 `cash/market_value/n_positions`；持仓需含 `symbol(or ts_code)/shares`；价格支持宽表或含 `date/symbol/close(or adj_close/price)` 的长表；执行回填可选 | 归因汇总表、逐股票贡献表和 Markdown 报告，默认写 `output/performance_attribution/<strategy>/<date>_performance_attribution_summary.csv`、`<date>_stock_contribution.csv`、`<date>_performance_attribution.md` |
| `live.deviation_analysis` | `build_live_deviation_analysis(...)`、`save_live_deviation_analysis(settings, strategy, trade_date, summary, position)` | 目标权重 Series / dict / DataFrame；纸面账户快照含 `date/total_asset`；纸面持仓含 `symbol(or ts_code)/shares`；价格支持宽表或长表；券商持仓和执行回填可选 | 偏差汇总表、逐股票偏差表和 Markdown 报告，默认写 `output/live_deviation/<strategy>/<date>_deviation_summary.csv`、`<date>_position_deviation.csv`、`<date>_deviation_report.md` |
| `live.paper_guard` | `validate_daily_inputs(...)`、`validate_daily_result(result)`、`raise_on_guard_errors(issues)` | 目标权重、最新价格、运行日期、目标权重日期、价格日期、单日纸面运行结果 dict | `GuardIssue` 列表；ERROR 级问题可抛出 `DailyPaperGuardError`，WARNING 级问题供摘要和日报展示 |
| `live.paper_run_control` | `load_trading_calendar_from_prices(path)`、`validate_daily_run_control(...)`、`has_paper_snapshot(...)` | 价格宽表缓存、策略名、运行日期、交易日日历、是否允许非交易日和重复运行 | 非交易日或重复运行时抛出 `DailyPaperRunControlError`；只读运行 `persist_outputs=False` 不阻断重复快照 |
| `live.daily_paper_cli` / `scripts/run_daily_paper.py` | `run_daily_paper_from_outputs(...)` / CLI | 已有 `output/rebalance_logs/<strategy>.csv` 与 `output/cache/prices_wide_close.csv`，可选交易状态 CSV 和因子失效监控 CSV，可选关闭日报、人工确认单、guard 或 run control，可选 `--execution-mode simulated_broker` | 调用运行控制、异常检查与 `run_daily_paper_trade`，打印日终摘要，并按配置写订单、检查、成交、账户状态、Markdown 日报和人工确认单；因子健康状态会进入摘要、日报和人工确认单 |
| `live.paper_scheduler` / `scripts/run_scheduled_daily_paper.py` | `run_scheduled_daily_paper(settings, daily_args=None, log_date=None)` / CLI | `Settings`、可选日志日期、透传给日终纸面交易的 CLI 参数 | 执行一次日终纸面交易，写 `output/scheduler_logs/<date>.log`，返回/退出码与日终纸面交易一致 |
| `live.broker` | `BrokerAdapter` 协议 | 真实券商 / 模拟券商实现统一方法：`sync/get_account/get_cash/get_positions/get_orders/submit_order/cancel_order` | 交易适配器标准插座；上层不依赖具体券商 API |
| `live.broker` | `SimulatedBroker` | 初始现金、当前持仓、最新价格、手续费率；可单笔 `submit_order`，也可 `submit_order_plan(order_plan, order_checks=...)` | 模拟立即成交，返回统一 `BrokerOrder` / 订单表；用于验证券商接口协议，不连接真实券商 |
| `live.broker` | `RealBrokerConfig`、`RealBrokerReadOnlyAdapter` | 券商名、账户标识、只读模式；可注入账户 / 持仓 / 订单快照，未来真实券商 adapter 可覆盖 `sync` | 真实券商只读接入骨架；允许查询账户、持仓、订单，禁止 `submit_order/cancel_order` |
| `live.broker_factory` | `create_broker_adapter(settings, ...)`、`build_broker_config(settings)` | `Settings.broker_mode`、`broker_provider`、`broker_account_id`，以及可选现金、持仓、价格、账户和订单快照 | 按配置创建 `SimulatedBroker` 或只读 Adapter；真实交易模式当前明确报错，未来 QMT / PTrade / 掘金 Adapter 在这里注册 |
| `live.broker_reconcile` | `reconcile_paper_with_broker(...)`、`save_reconciliation_outputs(...)` | `Settings`、策略名、只读 `BrokerAdapter`、对账日期和容忍阈值 | 账户差异表、持仓差异表、问题列表，以及 `output/broker_reconciliation/<strategy>/` 下的 CSV / Markdown 报告 |
| `scripts/reconcile_paper_broker.py` | CLI | 纸面账户状态、券商账户 CSV、券商持仓 CSV | 读取外部导出的只读券商快照，与纸面账户做对账，不下单、不撤单 |
| `scripts/build_execution_feedback.py` | CLI | 人工确认 CSV；默认读取 `output/live_orders/<strategy>/<date>_manual_confirm.csv`，也可用 `--manual-confirm` 指定 | 调用 `live.execution_feedback`，生成真实成交回填与执行偏差报告；不连接券商、不修改账户 |
| `scripts/build_live_performance_attribution.py` | CLI | 默认读取 `output/paper_account/<strategy>/snapshots.csv`、`positions.csv`、`output/cache/prices_wide_close.csv` 和可选当天执行回填 CSV；也可用参数显式指定 | 调用 `live.performance_attribution`，生成账户收益、基准收益、主动收益、个股贡献、执行滑点和残差归因报告 |
| `scripts/build_live_deviation_analysis.py` | CLI | 默认读取 `output/rebalance_logs/<strategy>.csv`、纸面账户快照、纸面持仓、价格缓存和可选当天执行回填；可额外传 `--broker-positions` | 调用 `live.deviation_analysis`，生成目标跟踪、纸面 / 券商持仓同步、未成交和滑点偏差报告 |

### 4.1 统一券商接口字段

| 数据结构 | 字段 |
|---|---|
| `BrokerAccount` | `cash`、`market_value`、`total_asset`、`updated_at` |
| `BrokerPosition` | `symbol`、`shares`、`available_shares`、`market_value`、`price`、`updated_at` |
| `BrokerOrder` | `order_id`、`date`、`symbol`、`side`、`qty`、`price`、`status`、`reason`、`filled_qty`、`avg_price`、`gross_amount`、`commission`、`cash_after`、`position_after`、`submitted_at` |

`status` 当前约定为 `NEW`、`FILLED`、`REJECTED`、`CANCELLED`。真实券商 adapter 可在内部映射券商原始状态，但暴露给本工程时应转换为上述统一状态或兼容扩展状态。

### 4.2 券商模式

| 模式 | 含义 |
|------|------|
| `simulated` | 默认模拟模式，不连接真实券商 |
| `real_readonly` | 真实券商只读模式，只允许查资金、查持仓、查订单 |
| `real_trading` | 预留真实交易模式；开启前必须完成只读验证、人工确认和更严格风控 |

`RealBrokerReadOnlyAdapter` 固定只接受 `real_readonly`，收到 `submit_order` 或 `cancel_order` 会抛出 `BrokerReadOnlyError`。

---

## 5. 对齐与缺失值

1. **再平衡日**：由 `config.rebalance_freq`（pandas offset 字符串，**月末建议 `ME`**；pandas 3 起 `M` 已弃用，`backtest_single` 内会将 `M` 映射为 `ME`）或显式 `rebalance_dates` 提供；回测模块仅在再平衡日更新目标权重。
2. **停牌/缺失价**：该日该标的不参与交易；若因子为 NaN，**默认剔除该标的于该截面**（或在单因子回测中记为「无效」，由 `backtest_utils` 统一策略）。
3. **流动性过滤**：若 `min_avg_volume` / `min_avg_amount` 为正，回测会在 Top-K 前使用 `long_prices` / `liquidity_data` 中的 `volume`、`amount` 或 `turnover` 字段计算过去窗口均值；缺少对应数据时该期无法通过该过滤。
4. **行业权重约束**：若 `max_industry_weight` 在 `(0, 1)`，回测会读取 `long_prices` / `industry_data` 中 `industry_col` 指定的列（默认 `industry`），用调仓日之前最近可用行业分类限制单个行业目标权重。缺少行业数据时记录 `industry_missing_data`，单票行业缺失时记为 `UNKNOWN`。
5. **波动率目标**：若 `target_volatility > 0`，回测会使用调仓日及之前的价格收益协方差估算组合年化波动。若估算值超过目标，则按比例降低股票目标仓位，剩余仓位作为现金；缺少足够历史样本时记录 `volatility_target_missing_data` 并保留原目标权重。
6. **最小持仓数量**：若 `min_positions > 0` 且有效目标持仓数少于阈值，回测会把股票总仓位缩到 `min_positions_exposure`，剩余保留现金。该规则用于避免候选数不足、过滤过严或交易受限时硬满仓。
7. **交易状态约束**：若 `enable_trade_status_filter=True`，回测会读取 `is_suspended` / `is_limit_up` / `is_limit_down`。停牌不能买卖，涨停不能买入 / 加仓，跌停不能卖出 / 减仓；缺少字段时默认不阻断但会记录 `trade_status_missing_data`。
8. **未来函数**：因子 `calc_*` 的输出在日期 `t` 必须**仅依赖 ≤ t 的公开数据**；标签（供 fusion 中 ML 使用）在单独函数中计算，**不得**与因子同文件混写而不标注。

---

## 6. 配置与安全

### 6.1 `Settings` 中与主流程强相关的字段（默认值以 `config.py` 为准）

| 字段 | 说明 |
|------|------|
| `project_root` / `data_dir` / `output_dir` | 路径；缓存默认 `output_dir/cache/` |
| `stock_pool_path` / `stock_pool_code_col` | 股票池文件路径与代码列名；默认 `data/stock_pool.xlsx` 和 `股票代码`，可用 `QUANT_STOCK_POOL_PATH` 改路径 |
| `database_path` | SQLite 数据库路径；默认 `data/quant_strategy.db`，可用 `QUANT_DATABASE_PATH` 覆盖 |
| `tushare_price_cache_path` | Tushare 日线行情本地缓存路径；默认 `data/prices_tushare_cache.csv`，可用 `QUANT_TUSHARE_PRICE_CACHE` 改路径 |
| `backtest_start` / `backtest_end` | 回测区间（字符串 ISO 日期） |
| `rebalance_freq` | 再平衡频率，默认 `ME`（月末） |
| `top_k` | 每期多头只数 |
| `commission_rate` | 单边手续费率 |
| `momentum_lookback` / `momentum_long_lookback` / `reversal_lookback` / `volume_ratio_window` / `vol_window` | 量价因子的默认窗口：短动量、长动量、短反转、成交量放大、低波 |
| `portfolio_weighting` | `equal` / `max_sharpe` / `risk_parity`（Top-K 内等权、夏普最大化或风险平价 ERC；后两者样本不足时回退等权） |
| `max_position_weight` | 单票目标权重上限；默认 `0.4`，目标权重超过上限时裁剪并重新分配，若因持仓数过少不可行则保留原归一权重 |
| `max_industry_weight` | 单个行业目标权重上限；默认 `0` 表示关闭，设为 `(0, 1)` 后限制行业暴露 |
| `industry_col` | 行业分类字段名；默认 `industry`，可来自 `long_prices` 或 `industry_data` |
| `factor_standardize_by_industry` | 因子研究面板是否按行业内标准化；默认 `True`，缺行业数据时回退普通横截面标准化 |
| `factor_industry_min_count` | 行业内标准化最小有效样本数；行业样本不足时该行业回退全股票池横截面 z-score |
| `target_volatility` | 组合目标年化波动；默认 `0` 表示关闭，开启后只在估算波动超目标时降低股票仓位 |
| `volatility_target_lookback_days` / `volatility_target_min_obs` | 目标波动估算使用的历史收益窗口和最少样本数 |
| `min_positions` / `min_positions_exposure` | 最小有效目标持仓数；不足时把股票总仓位缩到该 exposure，剩余保留现金 |
| `order_lot_size` / `min_order_amount` / `order_cash_buffer` | 订单生成 / 预检查使用的最小交易单位、最小订单金额与买入后现金缓冲；A 股默认 `order_lot_size=100` |
| `paper_initial_cash` | 纸面交易虚拟账户默认初始资金 |
| `broker_mode` | 券商模式，默认 `simulated`；真实接入先使用 `real_readonly` |
| `broker_provider` | 券商提供方名称，例如 `qmt`、`ptrade`；不存放账号密码 |
| `broker_account_id` | 非敏感账户标识；真实密钥、密码、Token 不应写入仓库 |
| `max_rebalance_turnover` | 单次再平衡目标权重变化上限；默认 `1.0`，首次建仓不节流，`0` 表示关闭 |
| `liquidity_lookback_days` | 可交易性过滤使用的成交量 / 成交额均值窗口 |
| `min_avg_volume` / `min_avg_amount` | Top-K 前的最小平均成交量 / 成交额过滤；默认 `0` 表示关闭 |
| `enable_trade_status_filter` | 停牌 / 涨跌停交易状态约束；默认关闭 |
| `optimizer_return_window` / `optimizer_min_obs` | 夏普配权用历史收益窗口与最少样本数 |
| `ic_forward_days` | IC 前瞻收益 horizon（交易日） |
| `ic_rolling_windows` | IC 稳定性诊断窗口；默认 `(20, 60)`，用于滚动均值、滚动波动、滚动正值比例 |
| `factor_group_count` | 因子分组收益诊断的分组数；默认 `5`，`G1` 为低分组，`G5` 为高分组 |
| `enable_ml_score` / `ml_score_model` | 是否生成机器学习打分因子；模型后端可选 `lightgbm` / `catboost` / `xgboost` / `hist_gradient_boosting` / `auto`，缺少可选依赖时回退到 sklearn 实现 |
| `ml_score_forward_days` / `ml_score_train_lookback_days` / `ml_score_min_train_days` / `ml_score_min_train_rows` / `ml_score_refit_every_days` | `ML_SCORE` 的标签前瞻天数、滚动训练窗口、最少训练日期、最少训练样本数和模型重训频率 |

### 6.2 股票池管理与实盘目标池确认

| 函数 / 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `live.stock_pool.load_stock_pool_frame` | Excel/CSV 股票池，含 `股票代码`，可选 `股票简称`、`主题`、`子行业`、`分类`、`是否启用` | 标准化 DataFrame：`symbol/name/theme/sub_industry/enabled/raw_symbol/source_path`；若无 `子行业` 但有 `分类`，会把 `分类` 写入 `sub_industry` | 保留人工研究池元信息，并为行业内标准化 / 行业约束提供行业字段 |
| `live.stock_pool.build_stock_pool_filter_report` | 股票池 DataFrame、价格数据、可选交易状态、日期、覆盖率/流动性阈值 | 逐股票过滤报告：`active/exclude_reason/latest_price/price_coverage/avg_volume/avg_amount/is_suspended/is_limit_up/is_limit_down` | 把人工研究池过滤成当日可用池 |
| `live.stock_pool.active_universe_from_report` | 过滤报告 | active universe DataFrame | 只保留 `active=True` 的股票 |
| `scripts/build_live_universe.py` | `--stock-pool`、`--prices`、可选 `--trade-status`、`--trade-date` | `output/live_universe/.../stock_pool_filter_report_<date>.csv` 与 `active_universe_<date>.csv` | 券商接口前的目标池确认入口 |
| `fusion_use_ic_weights` | `True`（默认）时融合用 `fuse_ic_weighted_zscore`；`False` 时用等权 `fuse_equal_weight_zscore` |
| `fusion_ic_rolling_window` / `fusion_ic_min_periods` | IC 列权：对 `ic.shift(1)` 做 rolling 均值时的窗口与最少样本数 |
| `factor_weight_train_ratio` | 静态综合权重融合的训练样本占比；训练段计算 `fusion_weight`，验证段生成 `FUSED_SCORE_WEIGHTED` |
| `rolling_factor_weight_lookback_days` / `rolling_factor_weight_min_days` | 滚动综合权重每个调仓日前可用的历史窗口与最少历史样本 |
| `rolling_factor_weight_min_weight` / `rolling_factor_weight_max_weight` | 滚动综合权重的单因子权重下限与上限 |
| `rolling_factor_weight_smoothing` | 滚动综合权重新旧权重平滑系数 |
| `persist_run_outputs` | 是否写 `output/cache/` 下行情、面板、IC CSV、运行配置，以及 `output/` 下绩效汇总、因子诊断、数据质量报告、调仓日志、换手日志、集中度日志、净值/超额净值/换手/集中度图表等 |

### 6.2 Token 与路径

- **API Token**：优先环境变量 `TUSHARE_TOKEN`；当前工程在 `config.py` 中允许**本地回退**（便于本机跑通）。**含密钥的 `config.py` 勿推送到远程仓库**。
- **股票池与行情缓存**：`QUANT_STOCK_POOL_PATH` 可指向本机 Excel/CSV 股票池；`QUANT_TUSHARE_PRICE_CACHE` 可指定行情缓存 CSV。默认 `data/*.csv`、`data/*.xlsx`、`data/*.xls` 均不进 Git，避免误公开真实股票池与行情数据。
- **路径**：通过 `get_settings()` 的 `data_dir`、`output_dir` 访问 `data/`、`output/`，避免硬编码散落。

---

## 7. 版本与演进

- 若从宽表改为长表为主，**优先在 `backtest_utils` 增加转换函数**，而非修改所有因子文件。
- 新增因子时：在 `factors/__init__.py` 或注册表中登记 `FACTOR_REGISTRY[name] = callable`，供 `run_single_backtest("NAME")` 解析（`main` 当前对手传 `factor_values` 路径可不调注册表）。
