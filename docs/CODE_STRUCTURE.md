# 代码结构说明（Code Structure）

本文档说明仓库内**各目录与文件在整体流水线中的职责**：解决什么问题、与谁协作、实现时建议放在哪一层。  
**数据形态与函数入参出参**以 [INTERFACE_AND_CONTRACTS.md](./INTERFACE_AND_CONTRACTS.md) 为准；本文侧重「为什么有这一层」而不是字段级契约。

---

## 1. 整体设计思路

项目按**研究 → 回测 → 接近实盘**的顺序拆模块，目标是：

- **因子、回测、优化、数据源**可独立替换，减少「一个脚本里全写满」的耦合；
- **同一套数据结构**（尤其是长表面板）从因子一直贯通到绩效与作图；
- **配置集中**（路径、费率、区间），密钥不进仓库。

逻辑上的数据流可以概括为：

```mermaid
flowchart LR
  data[(data/ 磁盘)]
  storage[storage/database<br/>SQLite 基础数据层]
  pool[live/stock_pool]
  feed[live/data_feed]
  fac[factors/]
  bt[backtest/]
  opt[models/]
  sig[live/signal_system]
  order[live/order_builder]
  capacity[live/capacity_impact]
  precheck[live/order_precheck]
  limits[live/risk_limits]
  dd[live/drawdown_control]
  stress[live/stress_test]
  rcontrol[live/risk_control_report]
  broker[live/broker]
  paper[live/paper_trading]
  state[live/account_state]
  runner[live/paper_runner]
  health[live/factor_health_report]
  confirm[live/manual_confirmation]
  feedback[live/execution_feedback]
  attribution[live/performance_attribution]
  deviation[live/deviation_analysis]
  checklist[live/semi_auto_checklist]
  guard[live/paper_guard]
  control[live/paper_run_control]
  scheduler[live/paper_scheduler + scripts/run_scheduled_daily_paper.py]
  cli[live/daily_paper_cli + scripts/run_daily_paper.py]
  perf[analysis/performance]
  dq[analysis/data_quality]
  diag[analysis/factor_diagnostics]
  bench[analysis/benchmark]
  turn[analysis/turnover]
  risk[analysis/risk_exposure]
  val[analysis/factor_validation]
  plot[analysis/plotting]

  data --> pool
  data --> storage
  storage --> data
  pool --> feed
  data --> feed
  data --> fac
  feed --> data
  fac --> bt
  fac --> dq
  fac --> diag
  fac --> val
  diag --> health
  val --> health
  opt --> bt
  bt --> perf
  bt --> bench
  bt --> turn
  bt --> risk
  bench --> perf
  turn --> perf
  risk --> perf
  bt --> plot
  bt --> dd
  dd --> order
  order --> capacity
  capacity --> precheck
  order --> limits
  capacity --> limits
  precheck --> limits
  order --> stress
  guard --> rcontrol
  dd --> rcontrol
  capacity --> rcontrol
  precheck --> rcontrol
  limits --> rcontrol
  stress --> rcontrol
  rcontrol --> cli
  precheck --> broker
  fac --> opt
  opt --> sig
  precheck --> paper
  broker --> paper
  paper --> state
  state --> runner
  runner --> confirm
  confirm --> feedback
  feedback --> attribution
  state --> attribution
  state --> deviation
  feedback --> deviation
  attribution --> checklist
  deviation --> checklist
  cli --> control
  cli --> guard
  control --> runner
  guard --> runner
  scheduler --> cli
  health --> cli
  stress --> cli
  runner --> guard
  runner --> order
  cli --> runner
  sig --> paper
  paper --> perf
```

`main.py` 负责按你的研究习惯**串联**上述步骤；具体调用顺序不必与上图箭头一一相同（例如融合可能在回测内部每期调用）。

---

## 2. 根目录文件

| 文件 | 作用 |
|------|------|
| `config.py` | **全局参数**：项目根、`data/` 路径、股票池路径、Tushare 行情缓存路径、默认价格列、手续费、再平衡频率、回测起止日、因子行业内标准化开关、单票/行业/波动率/最小持仓/换手约束、订单手数/最小订单金额/现金缓冲、纸面账户初始资金、年化用交易日数等；`get_tushare_token()` 从环境变量读取 Token，避免写死在代码里。 |
| `main.py` | **MVP 程序入口**：加载本地 demo / 行情缓存 / 股票池 Tushare 数据 → 多因子面板 → 行业内标准化研究面板 → 数据质量报告 → 可选落盘 → IC 与稳定性诊断 → 因子诊断（Top-K 多头超额 + 分组收益单调性）→ 多因子权重建议 / 训练段权重 / 滚动权重日志 → 可选 IC CSV 与图 → 多列单因子回测（内部可做可交易性 / 流动性过滤与决策审计）→ **IC 列权或等权**融合 / **训练段静态综合权重**融合 / **调仓日前滚动综合权重**融合 → `run_multi_backtest`（同样复用过滤与审计）→ 股票池等权基准与超额收益 → 换手率与预估成本 → 风险暴露与集中度 → 净值/超额净值/IC/权重/换手/集中度/覆盖率图。复杂逻辑在子包中实现。 |
| `requirements.txt` | **Python 依赖**列表，供虚拟环境一键安装。 |
| `README.md` | 快速开始、目录总览、文档索引。 |

---

## 3. `data/`：原始数据落盘

| 内容 | 作用 |
|------|------|
| 股票池 Excel/CSV、行情缓存、财务 CSV、SQLite 数据库等 | **离线研究的单一事实来源**；股票池由 `live/stock_pool` 规范化为 Tushare 代码，行情与 `live/data_feed` 拉取后的列名对齐后，因子与回测应优先读这里，保证可复现。SQLite 数据库默认 `data/quant_strategy.db`，用于长期基础数据；`output/` 仍保存单次实验结果。 |

约定哪些文件名、哪些列属于「契约」的一部分，见 `INTERFACE_AND_CONTRACTS.md` §2。  
当前仓库用 `.gitkeep` 占位；真实数据通常体积大且包含研究资产，默认 `.gitignore` 会忽略 `data/*.csv`、`data/*.xlsx`、`data/*.xls`。

## 3.1 `storage/`：SQLite 基础数据层

| 文件 | 作用 |
|------|------|
| `database.py` | **SQLite 表结构初始化**：定义 `prices_daily`、`fina_indicator`、`factor_panel_daily`、`announcement_events`、`news_sentiment`、`universe_snapshot` 和 `storage_metadata`，并为财务现金流字段提供轻量迁移。 |
| `warehouse.py` | **SQLite 数据读写层**：支持行情、财务和因子面板 upsert、读取与导出 `prices_long.csv`、`prices_wide_close.csv`、可选 `prices_wide_adj_close.csv`、`factor_panel.csv`，让数据库能服务现有回测缓存。 |
| `price_adjustment.py` | **复权价格工具**：把 `close + adj_factor` 转成研究用 `adj_close`，支持前复权 / 后复权口径。 |
| `inspection.py` | **SQLite 数据巡检层**：检查核心表行数、行情新鲜度、股票池覆盖、财务字段覆盖、因子覆盖率和导出缓存文件状态，并生成 CSV / Markdown 巡检日报。 |

---

## 4. `factors/`：因子计算

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出各 `calc_*`，并维护 **`FACTOR_REGISTRY`**：用字符串名称（如 `"ROE"`）映射到计算函数，供单因子回测按名调用。 |
| `factor_momentum.py` | **动量类因子**（如过去 N 日收益）：输入以价格宽表为主，输出规范为长表 `PanelLong`。 |
| `factor_reversal.py` | **短期反转因子**：过去 N 日收益取负，数值越大表示近期跌得越多。 |
| `factor_volume.py` | **成交量因子**：成交量相对过去窗口均量的放大程度。 |
| `factor_volatility.py` | **波动率因子**：基于收益宽表的滚动波动等，输出长表。 |
| `factor_pe.py` | **市盈率类因子**：需要行情与财报字段对齐，输出长表。 |
| `factor_roe.py` | **ROE 类因子**：依赖财务表与报告期/公告日规则，输出长表。 |
| `factor_finance.py` | **质量、成长与现金流类财务因子**：毛利率、净利率、低资产负债率、营收增长、利润增长、自由现金流收益率代理、经营现金流质量；按公告日向后对齐，避免未来函数。 |
| `factor_events.py` | **公告事件因子**：读取本地公告事件 CSV/XLSX，支持中文列名和显式 `event_score`，没有分数字段时用标题关键词生成粗略正负分，并按公告日向后衰减成 `ANNOUNCEMENT_EVENT_SCORE`；也支持把回购、减持、问询处罚、分红、合同项目等公告拆成类型分层事件因子。 |
| `factor_ml.py` | **机器学习打分因子**：用已有因子面板滚动训练梯度提升类模型，预测未来收益并输出 `ML_SCORE`；只作为候选因子进入 IC、分组收益、样本外验证和回测。 |
| `preprocess.py` | **因子清洗与标准化**：按交易日横截面做 winsorize、z-score，也支持行业内 z-score；行业样本不足时回退全截面结果。主流程默认用标准化后的研究面板进入 IC、诊断和回测。 |

**本层不负责**仓位、手续费、优化；只负责「在合法信息集下算出每个 `(date, symbol)` 上的因子值」。  
新增因子时：新建模块实现 `calc_xxx`。若要支持 `run_single_backtest("NAME")` 自动重算，需要在 `FACTOR_REGISTRY` 注册；若像质量 / 成长 / 现金流财务因子、公告事件因子一样依赖外部表与行情长表共同对齐，也可以先通过 `panel_builder` 统一生成，再由 `main` 以预计算 `factor_values` 传入回测。`ML_SCORE` 属于二阶因子：它依赖基础因子面板训练得到，因此在 `main` 中基础面板生成后追加。

---

## 5. `backtest/`：回测引擎与工具

| 文件 | 作用 |
|------|------|
| `backtest_utils.py` | **公共工具**：价格 ↔ 收益、长表 ↔ 宽表、因子与行情对齐等。避免在 `backtest_single` 与 `backtest_multi` 里重复写 pivot/stack、对齐索引。 |
| `backtest_single.py` | **单因子回测**：给定因子名（查注册表）或已算好的因子序列，执行分层/Top-K、Top-K 前可交易性 / 流动性过滤、停牌 / 涨跌停交易约束、再平衡、交易成本、单票权重上限、行业权重上限、波动率目标与现金仓位、最小持仓数量、单次换手上限等，输出**净值序列**及可选元信息（换手、持仓、过滤前后候选数、行业暴露、目标波动缩放、最小持仓检查、逐股票决策审计日志）。 |
| `backtest_multi.py` | **多因子回测入口**：`run_multi_backtest(fused=...)` 将已融合的一列得分交给 `run_single_backtest`；或 `run_multi_backtest(factors, weights=...)` 先做**列线性加权**再回测。`main` 中融合路径使用前者。 |

**本层**是「策略逻辑 + 时间轴 + 约束」的核心实现处之一；`models` 产出的权重或得分通常在这里被消费。

---

## 6. `models/`：组合优化与多模型融合

| 文件 | 作用 |
|------|------|
| `optimizer.py` | **组合优化**：`maximize_sharpe`、`risk_parity`（numpy）；**`backtest_single` 在 `portfolio_weighting` 为 `max_sharpe` 或 `risk_parity` 时再平衡日调用对应函数**。 |
| `fusion.py` | **多因子融合**：复用 `factors.preprocess.cross_sectional_zscore`，提供 `fuse_equal_weight_zscore`、**`fuse_ic_weighted_zscore`（IC 滞后滚动列权）**、**`fuse_static_weight_zscore`（训练段静态综合权重）**；`fuse_models` 仅部分 `method`。 |
| `factor_weighting.py` | **因子权重建议**：把 IC 分布、rolling IC、Top-Bottom 与单调性等评价指标合成 `factor_score` / `fusion_weight`；全样本用于诊断，训练段用于 `FUSED_SCORE_WEIGHTED`，调仓日前滚动窗口用于 `FUSED_ROLLING_SCORE_WEIGHTED`。 |

**本层**偏重「数学/优化问题」；日历、停牌、最小成交单位等**回测细节**仍建议在 `backtest` 或 `live` 处理。

---

## 7. `analysis/`：绩效与可视化

| 文件 | 作用 |
|------|------|
| `data_quality.py` | **数据质量与覆盖率**：统计价格覆盖率、因子覆盖率、每日覆盖率、调仓日有效截面规模。 |
| `performance.py` | **绩效指标**：由净值序列计算年化收益、波动、夏普、最大回撤等；与回测输出直接对接，便于统一口径。 |
| `benchmark.py` | **基准与超额收益**：构造股票池等权基准，计算超额收益、跟踪误差、信息比率，并生成超额净值宽表。 |
| `factor_diagnostics.py` | **因子诊断**：构造每个因子的 Top-K 等权多头腿，计算相对股票池等权基准的超额收益；同时计算分组收益、Top-Bottom 和单调性评分。 |
| `factor_validation.py` | **样本外验证与因子失效监控**：把 IC、多头超额、Top-Bottom 和单调性按训练段 / 验证段拆开比较，并输出 `OK/WATCH/DEGRADED/FAILED` 状态表。 |
| `turnover.py` | **换手率与成本**：从 `meta["rebalance_log"]` 计算逐期换手、预估成本和汇总指标。 |
| `risk_exposure.py` | **风险暴露与集中度**：从 `meta["rebalance_log"]` 计算 HHI、effective_n、Top 权重、持仓数和汇总指标。 |
| `plotting.py` | **图表**：`plot_nav`、`plot_ic`、`plot_weights`、`plot_turnover`、`plot_effective_n`、`plot_factor_coverage`；`rebalance_log_to_weights_frame` 将 `meta["rebalance_log"]` 转为权重宽表。 |
| `ic.py` | **截面 IC 与稳定性诊断**：日频 Spearman（因子 vs 前瞻收益）、基础汇总、分布分位数、正负占比、滚动稳定性与可选 CSV 落盘；**不参与**回测调仓。 |

**本层**应尽量**无业务状态**：输入 Series/DataFrame，输出指标 dict 或保存图片，方便单元测试与脚本复用。

---

## 8. `live/`：数据接入、信号、模拟交易

| 文件 | 作用 |
|------|------|
| `data_feed.py` | **行情接入**：Tushare/AkShare 拉取或读本地 CSV，输出列名与契约对齐，供因子与回测使用。 |
| `stock_pool.py` | **股票池管理与实盘目标池确认**：从 Excel/CSV 读取人工研究池，规范化 Tushare `ts_code`，保留简称、主题、子行业、启用状态；基于价格覆盖、最新价格、流动性和停牌 / 涨跌停状态生成过滤报告与 active universe。 |
| `cache_io.py` | **缓存与实验记录**：保存行情长表、收盘价宽表、因子面板、数据质量报告、因子诊断、训练段权重、滚动权重日志、运行配置、绩效汇总、调仓日志、换手日志、订单计划、订单预检查结果、纸面交易日志、集中度日志等，形成可复现实验档案。 |
| `account_state.py` | **纸面账户状态**：保存 / 读取虚拟账户现金、持仓和每日快照，让纸面交易可以跨天连续运行。 |
| `order_builder.py` | **订单生成**：把目标权重、当前持仓、现金 / 总资产和最新价格转换成 `BUY/SELL`、目标股数、调整股数、预估金额与交易原因。只生成订单计划，不连接券商、不模拟成交。 |
| `capacity_impact.py` | **容量与冲击成本**：读取订单计划和日频成交额历史，估算单笔订单参与率、冲击成本 bps、冲击成本金额和容量空间；缺少流动性数据时输出 `NA`，不当作通过。 |
| `order_precheck.py` | **订单预检查**：检查订单计划的现金、可卖数量、买入手数、最小金额、停牌、涨跌停和风险黑名单约束，输出 `PASS/BLOCK` 与原因。只做检查，不修改订单、不撮合成交。 |
| `risk_blacklist.py` | **风险预警与黑名单**：读取 CSV/XLSX、DataFrame、字典或代码列表，标准化 `symbol/reason/severity/source/active/created_at/expires_at`，并把当前有效黑名单交给订单预检查和纸面交易日报。 |
| `announcement_source.py` | **真实公告数据源**：把 Tushare 公告接口返回值统一成 `announcement_events.csv`，让公告事件因子和公告风险过滤都复用同一张事件表。 |
| `news_source.py` | **新闻 / 舆情数据源**：把 AkShare 个股新闻、未来 Tushare 新闻或商业新闻源统一成 `news_sentiment` 表，保留发布时间、标题、正文、来源、链接和情绪分。 |
| `event_risk_filter.py` | **公告事件风险过滤**：从公告事件表中识别问询、处罚、立案、诉讼、退市风险等负面事件，生成 `BLACKLIST/WATCH` 风险候选，并可转换为黑名单文件。 |
| `negative_sentiment_filter.py` | **新闻 / 舆情入口与负面过滤**：读取外部新闻 / 舆情 CSV/XLSX，统一股票代码、发布时间、标题、正文、来源、链接和情绪分，并按情绪分或负面关键词生成 `BLACKLIST/WATCH` 风险候选，可转换为黑名单文件。 |
| `risk_gate.py` | **统一风险门禁**：合并人工黑名单、公告风险候选和负面舆情候选，按 `BLOCK > WATCH > PASS` 输出统一门禁表，并可导出订单预检查可读取的 `risk_blacklist`。 |
| `drawdown_control.py` | **回撤止损与降仓控制**：读取纸面账户历史快照和当前持仓估值，按历史峰值计算回撤，并在订单生成前缩放目标权重；默认 5% 回撤降至 70% 仓位、10% 降至 50%、15% 转现金。 |
| `risk_limits.py` | **统一风险限额表**：把单票、Top3、effective_n、最低持仓数、现金缓冲、行业权重、换手、风险门禁和订单阻断等组合层指标统一检查成 `PASS/WATCH/BLOCK/NA`。 |
| `stress_test.py` | **组合压力测试**：对目标组合施加市场下跌、第一大持仓下跌、前三大持仓下跌和第一大行业下跌等情景，估算组合损失并输出 `PASS/WATCH/BLOCK/NA`。 |
| `risk_control_report.py` | **风险总控日报**：汇总运行检查、统一风险门禁、风险黑名单、回撤止损与降仓、容量与冲击成本、订单预检查、组合风险限额和组合压力测试，按 `BLOCK > WATCH > NA > PASS` 给出当天总控状态。 |
| `broker.py` | **统一券商接口协议**：定义 `BrokerAdapter`、`BrokerAccount`、`BrokerPosition`、`BrokerOrder`、`SimulatedBroker` 与 `RealBrokerReadOnlyAdapter`，把查资金、查持仓、查订单、下单、撤单抽象成统一方法。模拟券商用于验证协议；真实券商先用只读 adapter 验证账户、持仓和订单读取。 |
| `broker_factory.py` | **券商通道选择入口**：根据 `broker_mode/broker_provider/account_id` 创建模拟或只读 Adapter；后续 QMT、PTrade、掘金接入都应注册到这一层，真实交易模式当前明确阻断。 |
| `broker_reconcile.py` | **纸面 / 真实账户只读对账**：比较纸面账户与只读券商账户的现金、总资产、持仓股数和可用股数差异，输出账户差异、持仓差异和 Markdown 对账报告。 |
| `signal_system.py` | **信号生成**：将因子得分或融合结果变成离散买卖信号（或目标仓位），规则可与回测层对齐以减少「回测一套、实盘一套」。 |
| `paper_trading.py` | **纸面交易**：按订单计划与预检查结果更新虚拟现金和持仓，记录 `FILLED/SKIPPED`、手续费、现金变化与持仓变化；用于在接近实盘的流程下验证逻辑，**不等同**于已接入券商 API 的真实下单。 |
| `paper_runner.py` | **每日纸面运行器**：读取纸面账户状态，串联订单生成、订单预检查、执行模式选择、成交回报兼容、持仓更新、账户快照和落盘；默认走旧纸面成交，也可通过 `simulated_broker` 走统一券商接口。 |
| `paper_report.py` | **纸面交易日报**：把单日纸面运行结果整理成 Markdown，包含运行摘要、执行模式、账户快照、因子失效监控、增强因子健康总览、风格暴露、风险总控日报、统一风险门禁、风险黑名单、回撤止损与降仓、容量与冲击成本、组合风险限额、组合压力测试、订单、阻断原因、成交、券商订单回报、持仓和输出文件路径。 |
| `factor_health_report.py` | **增强因子健康总览**：读取样本外失效、滚动样本外、因子准入、冗余、权重漂移和牛熊市分段 CSV，压缩成日终纸面交易日报可读的健康摘要；只展示，不重新计算研究指标。 |
| `manual_confirmation.py` | **小资金人工确认实盘单**：基于订单计划、预检查和可选因子失效监控生成 CSV / Markdown 确认单，预留真实执行回填字段；只辅助人工下单，不自动连接券商。 |
| `execution_feedback.py` | **真实成交回填与执行偏差分析**：读取人工确认单中的真实成交回填字段，对比系统建议数量、价格、金额和实际执行结果，输出逐笔偏差、成交状态和汇总报告。 |
| `performance_attribution.py` | **实盘表现归因**：读取账户快照、当前持仓、价格缓存和真实成交回填，拆解账户收益、股票池等权基准收益、主动收益、个股贡献、执行滑点和未解释残差。 |
| `deviation_analysis.py` | **实盘偏差分析**：比较目标权重、纸面持仓、可选券商持仓和真实成交回填，输出目标跟踪偏差、持仓同步偏差、成交未完成比例和滑点提示。 |
| `semi_auto_checklist.py` | **半自动实盘执行清单**：汇总冻结清单、运行监控、风险总控、人工确认单、纸面日报、成交回填、表现归因和偏差分析，输出人工下单前总决策。 |
| `paper_guard.py` | **运行失败 / 异常检查**：在日终纸面运行前后检查目标权重、价格、日期、现金、持仓、订单检查和成交日志；ERROR 阻断，WARNING 进入摘要和日报。 |
| `paper_run_control.py` | **交易日日历 / 重复运行保护**：从价格缓存提取交易日日历，默认阻断非交易日运行；检查同日纸面账户快照，默认阻断重复覆盖。 |
| `paper_scheduler.py` | **每日调度封装**：运行一次日终纸面交易并记录 stdout、stderr、参数和退出码，供 cron / launchd / 服务器调度器调用。 |
| `daily_paper_cli.py` | **日终纸面交易辅助逻辑**：从 `output/rebalance_logs` 和 `output/cache/prices_wide_close.csv` 读取最近目标权重与最新价格，调用运行控制、异常检查和每日纸面运行器，生成命令行摘要，并默认写 Markdown 日报、增强因子健康总览、风险黑名单摘要、容量与冲击成本、组合风险限额、组合压力测试和风险总控日报；支持 `--risk-blacklist`、`--risk-limits`、`--drawdown-rules`、`--capacity-rules`、`--liquidity-history`、`--stress-scenarios` 与 `--execution-mode simulated_broker`。 |

## 8.1 `scripts/`：日常运行入口

| 文件 | 作用 |
|------|------|
| `run_daily_paper.py` | **日终纸面交易脚本**：薄命令行入口，调用 `live.daily_paper_cli.main`。默认使用 `FUSED_ROLLING_SCORE_WEIGHTED`，支持 `--strategy`、`--trade-date`、`--trade-status`、`--risk-gate`、`--risk-blacklist`、`--drawdown-rules`、`--capacity-rules`、`--liquidity-history`、`--factor-decay-monitor`、`--execution-mode`、`--no-persist`、`--no-report`、`--no-manual-confirm`、`--no-guard`、`--max-price-age-days`、`--allow-non-trading-day`、`--allow-rerun`。 |
| `init_database.py` | **SQLite 初始化入口**：调用 `storage.database.initialize_database`，生成 `data/quant_strategy.db` 及核心基础数据表。 |
| `update_database_cache.py` | **数据库缓存更新入口**：把本地行情、财务和因子 CSV 增量写入 SQLite，并导出 `output/cache/` 下主流程兼容缓存。 |
| `build_database_quality_report.py` | **数据库巡检入口**：调用 `storage.inspection`，生成 `output/database_quality/` 下的巡检 CSV 和 Markdown 报告。 |
| `run_scheduled_daily_paper.py` | **每日调度入口**：薄命令行入口，调用 `live.paper_scheduler.run_scheduled_daily_paper`，把未识别参数透传给日终纸面交易 CLI，并写 `output/scheduler_logs/<date>.log`。 |
| `reconcile_paper_broker.py` | **纸面 / 券商只读对账入口**：读取外部券商账户和持仓 CSV，构造只读 adapter，并与纸面账户状态生成差异报告。 |
| `build_execution_feedback.py` | **真实成交回填入口**：读取人工确认单 CSV 中的 `executed_qty`、`executed_price` 等字段，生成执行偏差 CSV 与 Markdown 报告。 |
| `build_live_performance_attribution.py` | **实盘表现归因入口**：默认读取纸面账户快照、当前持仓、价格缓存和当天执行回填，生成归因汇总、逐股票贡献和 Markdown 报告。 |
| `build_live_deviation_analysis.py` | **实盘偏差分析入口**：默认读取目标权重、纸面账户快照、纸面持仓、价格缓存和可选券商持仓 / 成交回填，生成偏差汇总、逐股票偏差和 Markdown 报告。 |
| `build_semi_auto_checklist.py` | **半自动实盘执行清单入口**：默认读取冻结清单、运行监控、风险总控、人工确认单、纸面日报、成交回填、表现归因和偏差分析，生成执行清单和总决策。 |
| `fetch_tushare_announcements.py` | **真实公告源接入入口**：读取股票池或显式股票代码，从 Tushare 拉取公告并保存为统一事件表。 |
| `fetch_akshare_stock_news.py` | **AkShare 个股新闻入口**：按股票池或显式代码拉取东方财富个股最近新闻，统一保存为 `news_sentiment` 表，并支持和既有缓存合并去重。 |
| `build_news_sentiment_smoke_backtest.py` | **新闻 / 舆情烟雾回测入口**：读取统一新闻表和近期行情，构造 `NEWS_*` 日频因子，并比较等权基线与负面舆情过滤版的短窗口表现。 |
| `build_a50_event_news_weekly_smoke_backtest.py` | **A50 公告 + 新闻短窗口周频回测**：用 Tushare A50 行情、AkShare 公告和 AkShare 新闻构造四组对比，验证信息因子作为 alpha 加入后是否改善短期组合表现。 |
| `build_event_risk_filter.py` | **公告事件风险过滤入口**：读取公告事件 CSV/XLSX，输出风险候选表，并可选导出日终纸面交易可读取的黑名单文件。 |
| `build_announcement_event_type_analysis.py` | **公告类型分层诊断入口**：把公告表拆成事件类型因子，并输出覆盖率、IC、多头超额、分组收益和建议标签。 |
| `build_announcement_event_type_backtest.py` | **公告类型因子组合回测入口**：比较不用公告、公告总分、公告类型收益因子、公告类型收益+风险混合等方案的滚动综合权重回测表现。 |
| `build_announcement_event_type_risk_filter_backtest.py` | **公告类型风险过滤回测入口**：正向公告类型继续做 alpha，负面公告类型作为调仓前候选股门禁，并输出过滤前后净值、绩效和风险命中日志。 |
| `build_negative_sentiment_filter.py` | **负面舆情过滤入口**：读取新闻 / 舆情 CSV/XLSX，输出负面风险候选表，并可选导出日终纸面交易可读取的黑名单文件。 |
| `build_negative_sentiment_filter_backtest.py` | **负面舆情风险门禁回测**：不把新闻作为 alpha 加分，而是在调仓日前把仍处于 `BLACKLIST/WATCH` 有效期的股票从候选中剔除，对比过滤前后组合表现。 |
| `build_unified_risk_gate.py` | **统一风险门禁入口**：读取股票池、公告风险候选、负面舆情候选和人工黑名单，输出 `risk_gate_<date>.csv`、风险明细和可选 `risk_blacklist_<date>.csv`。 |
| `build_drawdown_control.py` | **回撤控制检查入口**：读取纸面账户快照、当前账户状态和最新价格，输出账户级回撤止损与降仓检查表。 |
| `build_capacity_impact.py` | **容量与冲击成本入口**：读取订单计划和日频流动性历史，输出逐订单参与率、冲击成本、容量空间和 Markdown 摘要。 |
| `build_portfolio_risk_limits.py` | **统一风险限额检查入口**：读取目标权重、可选当前权重、行业映射、风险门禁和订单预检查结果，输出组合层 `PASS/WATCH/BLOCK/NA` 风险限额检查表和 Markdown 摘要。 |
| `build_portfolio_stress_tests.py` | **组合压力测试入口**：读取目标权重、可选行业映射和压力测试情景表，输出压力测试 CSV 与 Markdown 摘要。 |

**本层**是「研究与生产之间的缓冲带」：接口稳定后，真实实盘可在同结构下替换撮合与下单实现。

---

## 9. `docs/`：设计文档

| 文件 | 作用 |
|------|------|
| `INTERFACE_AND_CONTRACTS.md` | **接口与数据契约**：长表索引、CSV 列、各函数输入输出约定、缺失值与 Token 约定。 |
| `CODE_STRUCTURE.md` | **本文档**：模块职责与协作关系，偏架构与导读。 |
| `ENGINEERING_OVERVIEW.md` | **工程总览**：端到端行为、公式级说明、与 `main` 步骤对齐。 |
| `FLOW_AND_MODULES.md` | **主流程图**（Mermaid）与逐步说明表。 |

---

## 10. 阅读与改代码的顺序建议

1. `config.py` → 路径、费率、`factor_standardize_by_industry`、`portfolio_weighting`、`max_position_weight`、`max_industry_weight`、`target_volatility`、`min_positions`、`max_rebalance_turnover`、IC 与优化窗口等。
2. `docs/FLOW_AND_MODULES.md` 或 `ENGINEERING_OVERVIEW.md` → 主流程。  
3. `factors/panel_builder.py` + `backtest/backtest_utils.py` → 面板与对齐。  
4. `backtest/backtest_single.py` → 单策略闭环（含 Top-K 与等权 / 夏普 / 风险平价）。  
5. `analysis/plotting.py` → `plot_nav` / `plot_ic` / `plot_weights` 与 `rebalance_log_to_weights_frame`。  
6. `backtest/backtest_multi.py` + `models/fusion.py` → 多因子接入回测。  
7. `analysis/ic.py`、`analysis/data_quality.py`、`analysis/factor_diagnostics.py`、`analysis/performance.py`、`analysis/benchmark.py`、`analysis/turnover.py`、`analysis/risk_exposure.py` → IC 分布稳定性、数据质量、因子多头超额、分组收益、绩效、基准、超额收益、换手与成本、集中度。
8. `storage/` → SQLite 表结构和后续数据库读写层；长期基础数据进数据库，实验结果继续进 `output/`。
9. `live/` → 数据接入、订单生成、容量与冲击成本、订单预检查、统一风险限额检查、压力测试、风险总控日报、券商通道选择、纸面交易、账户状态、每日纸面运行器、纸面交易日报、增强因子健康总览、人工确认单、真实成交回填、运行异常检查、交易日日历 / 重复运行保护、每日调度封装与日终脚本辅助逻辑；信号生成仍是占位。
10. `scripts/` → 日常运行入口，例如数据库初始化、日终纸面交易命令和真实成交回填报告命令。

**文档与代码**需人工同步；无 CI 自动 diff。改 `main` 或契约时请更新 `docs/` 与 `README.md`。
