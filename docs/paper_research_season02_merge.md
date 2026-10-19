# 第二季论文成果：主分支选择性集成与验收

实施日期：2026-10-19。主工作区分支：`main`。

这次已经把可复用代码整理到主工程，并实际运行了测试、六策略长期缓冲对照和论文匹配基准复现。不是只写建议，也不是把整个论文分支直接合并覆盖。

本记录描述主分支的代码集成与验收；提交和远端同步状态以Git记录为准。集成时保留了原有的数据扩展、K值与IC实验，没有覆盖这些工作。没有修改六策略前向纸面账户、默认因子权重、股票池或生产风险门禁。

## 1. 已集成的能力与实际入口

| 能力 | 实现与调用位置 | 当前可以做什么 | 不代表什么 |
|---|---|---|---|
| 成交成本诊断 | `analysis/portfolio_diagnostics.py` → 原有 `scripts/run_research_audit.py` | 读取规范化成交，分开新进、退出、旧仓再平衡，汇总费用、金额换手与成交记录数 | 不从成交记录推断订单数，不凭空增加税费或最低佣金 |
| 组合诊断 | 同一模块及审计入口 | 股票仓位／现金、股票内归一化有效持仓数、资金重叠、可选行业分布、原始／主动收益相关性 | 不自动合并六个账户，不自动调整风险上限 |
| 特征匹配基准 | `analysis/characteristic_benchmarks.py`、`scripts/run_characteristic_audit.py` | 根据规模、账面市值比、动量构造对照；输出匹配质量、目标、原策略和对照净值 | 非完整DGTW复现，匹配后剩余收益不是已经确认的Alpha |
| 财务年度配对检查 | `analysis/quality_asof.py` | 复用现有财务日期解析，匹配已知年度版本和准确上一年；最新指标缺失／无效不偷偷改选旧年 | 是显式调用的研究接口，没有替换全部生产财务因子；不构成完整历史版本数据库 |
| 训练时间检查 | `analysis/training_time.py` | 成熟标签切分、训练验证边界、日期均衡权重、归档拟合边界审计 | 不启用ML交易，不宣称审计过所有历史模型 |
| 可选持仓缓冲 | `strategies/holding_buffer.py` → `run_single_backtest(rank_buffer_exit=...)` | 根据实际持仓保留符合排名条件的股票，生成保留／进入记录 | 默认关闭，仍经过已有权重、行业、换手和交易检查 |
| 明确现金语义的研究回放 | `backtest/research_replay.py` | 下一市场交易日执行；HOLD、CASH、部分现金仓位明确区分；输出目标、成交、费用和净值 | 不是券商执行，未完整模拟整手、涨跌停、税费和冲击 |

代码可以按需调用，不等于每日流程已经自动调用全部新功能。第一批实际运行已覆盖六策略回测结果的诊断；纸面账本需要映射到规范输入，尚未把新的诊断自动插入每日流水线。

组合评分相关性、统一资金账户内部的订单抵消、完整质量评分注册、组合层融合等，没有因为这次集成就默认启用。原有融合、IC接口继续保留。

## 2. 如何使用

### 2.1 已有净值、成交和持仓的统一审计

沿用原入口，在原JSON规格里增加可选`diagnostics`。不提供它时，旧审计内容与调用约定保持不变。

```json
{
  "diagnostics": {
    "trades": {"path": "trades.csv", "sha256": "文件实际SHA-256"},
    "allocations": {"path": "targets.csv", "sha256": "文件实际SHA-256"},
    "allocation_kind": "target",
    "return_correlations": true
  }
}
```

这只是原规格的新增部分，仍需原有`nav`、`pairs`、日频、小数单位、费用口径和候选家族声明。

- 成交必需列：`date,strategy,symbol,side,notional,fee,shares_before,shares_after,price,pre_nav`。`side`为BUY／SELL；货币单位必须一致，同一策略同一日期使用相同成交前净值。只支持非负多头持仓，容忍不超过1e-12的浮点股数残差，真正负持仓仍拒绝。
- 持仓必需列：`date,strategy,symbol,weight`；可选`industry`。股票权重相对于整个账户净值，剩余为现金，禁止超过100%。目标与实际持仓分开调用，不能混在一个文件里。
- 全现金组合没有股票行，当前组合快照输入不能自动认定其存在；缺少快照的两两比较标为未评估，不假装已经测得分散度。
- 每个输入文件校验哈希，日期必须在声明的净值日历内。输出目录必须是新的，不覆盖历史报告。当前审计附表用普通CSV；匹配入口另支持gzip。

已经生成的真实例子：

- [输入规格](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/audit_spec.json)
- [六策略回测审计报告](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/audit/README.md)
- [详细组合／成本输出](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/audit/report.json)

```bash
.venv/bin/python scripts/run_research_audit.py \
  --spec output/season02_integration_20261019_final/audit_spec.json \
  --output output/season02_audit_repeat_new
```

### 2.2 特征匹配与对照回放

```bash
.venv/bin/python scripts/run_characteristic_audit.py \
  --spec output/season02_archive_validation_20261019_final/spec.json \
  --output output/characteristic_audit_repeat_new
```

规格要求三个带SHA-256的输入：

1. `features`：`date,symbol,total_mv_yuan,VALUE_BP,MOM_12_1`。
2. `targets`：`date,execution_date,strategy,symbol,weight`。当前匹配器要求每个策略目标股票权重之和为1，不偷偷把部分现金组合归一化。
3. `prices`：`trade_date,ts_code,adj_close`。

还需显式指定`start,end,bins,minimum_peers,fee`。分组只读取当时的特征；输入快照是否正确构造仍须由上游保证，接口不会把快照自动认证为不可变的历史数据。

输出包括匹配目标、同类数量／降级记录、原策略和匹配基准的交易前及交易后净值、成交汇总。下游可把交易后净值传给现有研究审计入口，以“原策略→匹配基准”的映射计算统计区间；不要混用交易前和交易后口径。

本轮验证的是论文中的六个质量组合，不是声称六个MVP策略的长期特征匹配已经全部完成。对MVP六策略还需要提供同日期、同股票池、同字段定义的完整特征输入；不能拿论文质量截面直接替换它们的适用池。

### 2.3 可选持仓缓冲

现有单策略回测增加可选参数：

```python
nav, meta = run_single_backtest(
    factor_name,
    factor_values=scores,
    prices=prices,
    settings=settings,
    top_k=50,
    rank_buffer_exit=75,
)
```

省略`rank_buffer_exit`或传入`None`时不启用；参数必须为不小于K的整数。该入口沿用原回测的信号执行日约定，不自动给分数加一日滞后；真正下一交易日的对照使用下述研究入口。

保留规则只在当前有效、流动性筛选后的候选上执行，读取实际股数持仓。随后继续走已有权重、行业和交易限制。启用时候选少于K会明确报错，不默默改成另一种预算；`meta['rank_buffer_log']`记录原排名和保留／进入原因。这里是选择阶段记录，最终成交仍以交易限制为准。

研究回放接口`replay_ranked(prices,scores,schedule,top_k=50,exit_rank=75)`适合冻结历史评分后的下一交易日对照；`exit_rank=None`为原TopK。候选评分必须由调用方完成股票池与风险资格过滤，回放器不重新下载数据或推断生产风险配置。

更通用的`replay`接受目标生成函数：返回`None`表示不交易，空权重Series表示尝试清仓，正权重和小于1表示保留现金。缺报价不能成交；全体非空目标均缺报价时明确HOLD并记录标志，与旧研究对照一致。已有受阻退出仍保留持仓，缺价新目标不把预算分给其他股票。

旧回测空信号的历史处理没有在本轮顺手改变。需要明确CASH语义的实验应使用新研究回放入口，不把旧函数的空列表当成可靠的清仓指令。

### 2.4 财务与标签检查

`quality_asof.paired_annual_features`接收由`financial_asof.normalize_statements`规范化的利润表、资产负债表，以及`quality_asof.normalize_indicators`规范化的年度指标。输出当前年度、准确上一年度、有效性和变化值；调用方必须检查`pair_valid`，不能对无效项填成正常分数。

`training_time.chronological_split`要求明确`validation_periods`和`minimum_train_periods`，没有把短样本26周默认值带进长期训练。只使用标签结束日期严格早于拟合日的样本，训练标签还必须早于验证起点。`balanced_period_weights`使各历史期训练总权重相同。`validate_training_folds`只审计归档边界，不等于重训模型。

这些是集成后的可调用工具；本轮没有将其强制改接到全部旧财务因子或ML模型上，以免未验证地改变历史评分。

## 3. 实际做了哪些验证

| 验证 | 结果 | 边界 |
|---|---|---|
| 改动前全量测试 | 434项通过 | 原工作区已有的测试基线 |
| 集成后全量测试 | 470项通过，新增36项 | 测试正确性不等于策略通过 |
| 主回测默认路径对比HEAD版本 | 等权、换手限制、风险平价、单票限制四组：净值及决策记录精确一致 | 固定随机种子的合成回归用例 |
| 六策略真实历史原版复现 | 最大逐日净值误差约1.51e-14 | 复现此前已验证的研究执行口径，不是实际券商成交 |
| 六策略缓冲成对回测 | 12条3bp曲线，另12条13bp压力曲线；分段结果与六组配对统计均保存 | K50与退出75固定，不搜索K或阈值；已研究历史 |
| 特征匹配代码迁移验证 | 86期、六个质量组合、1,101,681条匹配权重完全一致 | 不是MVP六策略长期匹配验证 |
| 原策略／匹配基准回放 | 12条净值对照，最大误差约4.25e-14 | 两种调用口径一致，不是真实成交保证 |
| 机器学习归档边界 | 142次拟合记录通过成熟时间与边界检查 | 没有重新拟合142次模型，也未认证全部原始特征 |
| 纸面账户隔离 | 运行前后initial/plans/days文件哈希一致 | 不回填、不替换已有前向记录 |

各验证目录的`complete.json`／`result.json`记录输入、代码、输出哈希及状态。最终验收以带`_final`的目录为准；早期失败尝试和中间验证保留用于排查，不作为最终验收。

## 4. 缓冲不是全策略共同的升级答案

对照区间：2011-01-04初始现金至2026-10-15，沿用原六策略共同日历与有效信号；周度信号，下一市场交易日收盘研究撮合。下表为单边3bp简化费用，累计双边换手不是年化换手。

| 策略 | 原版累计收益 | 缓冲累计收益 | 原版最大回撤 | 缓冲最大回撤 | 原版／缓冲累计双边换手 |
|---|---:|---:|---:|---:|---:|
| 基础池多因子 | 216.45% | 219.92% | -58.02% | -54.52% | 537.85／357.77 |
| 大盘多因子 | 136.05% | 130.44% | -48.89% | -48.23% | 479.95／295.53 |
| 中小盘多因子 | 364.54% | 378.62% | -53.51% | -55.22% | 570.82／397.66 |
| 价值 | 379.13% | 320.19% | -42.73% | -43.96% | 264.07／149.45 |
| 成长 | 212.44% | 145.74% | -57.70% | -58.68% | 244.89／166.06 |
| 科技成长主题 | 337.16% | 375.02% | -59.68% | -59.07% | 450.44／269.52 |

换手均降低，但三个策略累计收益提高、三个降低，中小盘收益提高的同时回撤扩大。六组“缓冲减原版”的正向主动收益检验，经Holm调整后p值均为1；这不是证明两者相同，而是没有足够证据支持统一启用缓冲。

因此：**保留功能，默认关闭；不根据这次历史对比切换任何日报账户。** 原策略与缓冲策略的回撤也仍然较大，不能拿累计收益为实盘放行。

完整结果：

- [绩效指标](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/performance.csv)
- [事先固定的分段指标](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/subperiods.csv)
- [缓冲相对原版的六组配对检验](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/paired_audit/README.md)
- [六策略验收清单](/Users/dev/Documents/quant_strategy/output/season02_integration_20261019_final/complete.json)
- [匹配基准与训练边界验收](/Users/dev/Documents/quant_strategy/output/season02_archive_validation_20261019_final/complete.json)
- [主回测默认行为验收](/Users/dev/Documents/quant_strategy/output/season02_default_parity_20261019/result.json)

## 5. 如何重新验证

全部使用现有本地缓存，不需要Token。输出目录必须更换为尚不存在的路径。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_season02_integration.py \
  --output output/season02_integration_repeat_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_season02_archives.py \
  --paper-root /Users/dev/Documents/quant_strategy_paper_lab \
  --output output/season02_archive_repeat_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_season02_default_parity.py \
  --output output/season02_default_repeat_new
```

最后一个脚本与运行时HEAD比较；本轮结果已记录当时提交ID。未来提交改变后，它不再自动代表最初集成前的版本，应以本轮记录的提交为历史证据。

## 6. 仍未做的事

没有完整合并论文分支，没有默认启用质量P＋G、组合融合、ML或历史冠军切换；没有取消原风险门禁；没有创建真实订单或改变纸面账户。

也没有宣称解决全部数据版本和真实执行问题：供应商历史修订、整手、最低佣金、税费、涨跌停和冲击模型仍是边界。统一的纸面账本自动诊断适配、MVP六策略长期匹配输入、全量生产特征时间审计，可以在现有接口上继续接入，但不能把它们写成已经完成。

本轮完成的是可运行代码与实证验收，不是实盘资格认证。
