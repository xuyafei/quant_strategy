# 六策略固定权重 / IC权重 × 五档持仓数

本实验不修改六策略前向账户。协议在结果生成前冻结：
`research_protocols/20261018_six_strategy_ic_comparison_v1.json`。

## 主问题

同样Top50时，组内滚动IC加权是否改善固定融合？其次，固定与IC各自的K=10/15/20/30/50如何变化？不根据最高全期收益自动决定实盘K。

四个多因子账户沿用13个因子、五类各20%；价值3个、成长2个。只用传统因子，不用消息、公告或ML。股票权重保持等权。

IC是策略自身当日股票池内、生产预处理后因子与下一交易日收盘起五个交易日收益的Spearman相关。每日因子截面至少30个有效股票；窗口是最近252个已成熟的交易日，每个因子至少120个有效IC。收益终点严格早于本次信号。组内权重=50%等权+50%正平均IC归一化；不足样本或全组非正时整组等权。单因子组不变，不反转方向，不自动淘汰。

相邻收益标签重叠，不能把有效日数当独立样本数。初期等权回退与首个动态调整日单列。风险类别仍固定20%，本轮不研究类别之间的动态预算。

## 运行

```sh
MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u scripts/run_six_strategy_ic_comparison.py --stage all
MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/report_six_strategy_ic_comparison.py
.venv/bin/python -m unittest tests.test_ic_weight_comparison tests.test_holdings_count -v
```

可分阶段运行`prepare`、`rank`、`backtest`。按季度校验中间文件，冻结代码或输入变化即停止，使用新目录重跑；禁止改旧结果冒充同一协议。

## 核对

- 生产固定评分与新管线每个周度信号对账。
- 固定Top100排名与上一轮长历史研究精确对账。
- 30条固定权重净值指标与上一轮一致。
- 历史2011/2018/2026三个阶段、六策略两版本五档K，与原生产撮合器独立核对。
- 成熟标签、权重归一、正权重、现金费用恒等式校验。
- 前向账户`initial/plans/days`文件运行前后哈希不变。

## 边界

已有历史不是真正未知留出集。复权价可分割份额引擎不等于整手实盘撮合；缺完整历史涨跌停、印花税、最低佣金、冲击成本，另报成本压力和末期缺报价资产敏感性。财务是当前供应商快照按公告日对齐，非原始版本档案。现金流因子仍是经营现金流代理。没有重新跑随机子集，不对其分布作新增结论。
