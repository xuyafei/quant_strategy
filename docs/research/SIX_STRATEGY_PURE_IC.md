# 六策略纯 IC 补充实验

原组内收缩实验保留，新增 `PURE_IC`。研究脚本不会写入前向账本。

## 冻结规则

- 四个多因子策略的13因子全局归一，价值仍3因子，成长仍2因子；无消息/公告 alpha、无ML。
- 最近252个已成熟交易日，单因子至少120个有效RankIC；收益终点必须严格早于信号日。
- `w = max(meanIC,0) / sum(max(meanIC,0))`，不足样本者置零。无固定50%、五类20%、下限、上限、平滑或负向反转。
- 直接加权生产预处理后的单因子，取消类别二次标准化；缺失值按可用权重归一。
- 全部权重为零时现金目标；缺报价无法卖出的持仓保留估值，后续规定调仓日重试。空目标不是保持持仓。
- 所有版本股票目标等权；K=10/15/20/30/50，同K比较，主观察K50及预先指定K20。
- 原始全日历补充与共同预热结束后重置现金的主比较分别输出，避免初期纯IC等待历史造成伪改善。

## 运行

```sh
MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_six_strategy_pure_ic.py
MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/validate_pure_ic_cash.py
MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/report_six_strategy_pure_ic.py
.venv/bin/python -m unittest tests.test_pure_ic_comparison
```

协议见 `research_protocols/20261018_six_strategy_pure_ic_v1.json`。输出 `output/six_strategy_pure_ic_20261018/REPORT.md`。可以用 `--stage rank/backtest` 分段运行；冻结文件变化后必须使用新的 `--output`，不覆盖旧结果。报告脚本默认读取上述目录。

复用已封存的逐季度点时点因子与每日IC，不重新下载数据。验证源代码、原始数据、输入分块、原结果和账本哈希。新增现金路径单元测试，正常目标与旧引擎真实历史片段对账，逐笔核对费用现金，核对标签成熟日期及净值重算指标。

本轮为已经看过历史上的政策比较，不是未知样本外验证；正IC截断是隐含因子筛选，而不是原 pass/watch/fail 制度。继承复权可分割份额、历史财务非原始版本档案及交易摩擦不完整等限制。不会自动根据最高历史收益改动生产权重或选择持仓数。
