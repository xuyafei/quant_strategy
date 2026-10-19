# 当前 MVP 的持仓数敏感性实验

固定六策略现有因子配方，比较 K=10、15、20、30、50。K20 是主候选，K50 为现有方案对照。本工具不修改生产配置、策略计算模块或纸面账本。

协议：`research_protocols/20261017_mvp_holdings_count_v1.json`。父协议与六策略前向账户冻结协议内容一致。主数据为 2010—2026 年已下载的行情、财务、证券状态及行业区间；先保留预热，评价从 2011 年开始。

## 运行

```bash
# 一次性补充默认 VIP 字段集漏掉的现金利润质量列。
# token 从 TUSHARE_TOKEN 或隐藏输入读取，不写入文件。
.venv/bin/python scripts/fetch_mvp_history_finance.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/quant_strategy_mpl .venv/bin/python scripts/run_mvp_holdings_count.py --stage prepare
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/quant_strategy_mpl .venv/bin/python scripts/run_mvp_holdings_count.py --stage backtest
MPLCONFIGDIR=/tmp/quant_strategy_mpl .venv/bin/python scripts/report_mvp_holdings_count.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_mvp_holdings_replay.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_holdings_count
```

默认结果目录：`output/mvp_holdings_count_20261017_complete_factors`，详细结果见其中 `REPORT.md`。缓存按年保存，输入或计算代码变化会拒绝复用准备缓存。需要改变口径时另建协议和输出目录，不覆盖既有对照。

## 准确解释

- 30 条确定性 TopK 路径，30 条更高成本路径，3000 条从同一高分候选池抽取的完整随机路径。
- 股票目标等权，持有期间权重漂移；不是日度再平衡，不是 IC 加权。
- 下一交易日执行，不根据执行日行情补选替代股票。
- 多因子配方、行业预处理、基础池和策略适用池直接复用主分支函数。不是把论文分支的五个研究特征冒充当前 MVP。
- 快速批量回放单独放在 `analysis/holdings_count.py`，对现有 `_rebalance_to_target_weights` 做数值对账。输出使用交易后净值，首次和末次费用都入账。
- 随机分布反映选择敏感性，不是独立样本外验证或置信区间。数据修订、行业分类版本、退市价值和成交约束等限制必须与结果一起阅读。
- 不足 50 个有效候选的日期对所有 K 组同时保持持仓，避免小 K 独享早期交易日期。所有策略仍共用同一评价日历。
- 结果不能直接触发实盘，也不修改当前 K50 的前向账户。
