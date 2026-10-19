#!/usr/bin/env python3
"""Render the pure-IC supplement; no parameter fitting or account writes."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.run_six_strategy_pure_ic import OUTPUT
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.report_mvp_holdings_count import LABELS

VARIANTS=['FIXED','IC_SHRUNK','PURE_IC']
COLORS={'FIXED':'#737b85','IC_SHRUNK':'#16708b','PURE_IC':'#b66323'}


def report(out=OUTPUT):
    completed=json.loads((out/'completed.json').read_text())
    cash_validation=json.loads((out/'cash_path_validation.json').read_text())
    if cash_validation['status']!='PASS':raise ValueError('cash path validation missing')
    for name,digest in completed['files'].items():
        if sha(out/name)!=digest:raise ValueError('result checksum mismatch: '+name)
    perf=pd.read_csv(out/'performance.csv');main=perf[perf.scope.eq('common')]
    sub=pd.read_csv(out/'subperiods.csv');sub=sub[sub.scope.eq('common')]
    nav=pd.read_csv(out/'common_nav.csv',index_col='date',parse_dates=True)
    bench=pd.read_csv(out/'common_benchmark_nav.csv',index_col='date',parse_dates=True)
    weights=pd.read_csv(out/'weights.csv.gz',parse_dates=['signal_date'])
    weights=weights[weights.signal_date.ge(completed['common_signal'])]
    actions=pd.read_csv(out/'actions.csv',parse_dates=['date'])
    actions=actions[actions.date.ge(completed['common_signal'])]
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    kinds={'nav_k50':'K50 post-cost NAV','drawdown_k50':'K50 drawdown','cagr_by_k':'Annualized return by holdings count',
        'sharpe_by_k':'Sharpe by holdings count','stress_cagr_by_k':'Higher-cost annualized return by holdings count',
        'recent_k50':'K50 annualized return by historical subperiod'}
    for kind,title in kinds.items():
        fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
        for ax,(name,(_,en)) in zip(axes.ravel(),LABELS.items()):
            for j,v in enumerate(VARIANTS):
                if kind in ['nav_k50','drawdown_k50']:
                    s=nav[f'{name}__{v}__K50'];y=s if kind=='nav_k50' else 100*(s/s.cummax()-1)
                    ax.plot(y.index,y,label=v,color=COLORS[v],lw=1.2)
                    ax.set(xlabel='Date',ylabel='NAV (log; start=1)' if kind=='nav_k50' else 'Drawdown (%)')
                    if kind=='nav_k50':ax.set_yscale('log')
                elif kind=='recent_k50':
                    g=sub[sub.strategy.eq(name)&sub.variant.eq(v)&sub.k.eq(50)].sort_values('period')
                    x=np.arange(len(g));ax.bar(x+(j-1)*.24,g.cagr*100,width=.23,label=v,color=COLORS[v])
                    ax.set_xticks(x,g.period,fontsize=8);ax.set(xlabel='Period (first starts after warmup)',ylabel='CAGR (%)')
                else:
                    metric={'cagr_by_k':'cagr','sharpe_by_k':'sharpe','stress_cagr_by_k':'stress_cagr'}[kind]
                    g=main[main.strategy.eq(name)&main.variant.eq(v)].sort_values('k')
                    ax.plot(g.k,g[metric]*(1 if metric=='sharpe' else 100),marker='o',label=v,color=COLORS[v])
                    ax.set_xticks([10,15,20,30,50]);ax.set(xlabel='Target stock count K',ylabel='Sharpe (zero RF)' if metric=='sharpe' else 'CAGR (%)')
            if kind=='nav_k50':ax.plot(bench.index,bench[name],label='Own PIT universe',ls='--',color='#b6bbc0',lw=.8)
            ax.set_title(en);ax.grid(alpha=.2);ax.legend(fontsize=8)
        cost='13 bps each side' if kind=='stress_cagr_by_k' else '3 bps each side'
        fig.suptitle(f'{title}\nSame cash start / local historical PIT inputs | {nav.index[0].date()} to {nav.index[-1].date()} | {cost}',fontsize=13)
        fig.savefig(figdir/f'{kind}.png',dpi=140);plt.close(fig)
    # Family totals are observed outcomes, not imposed budgets.
    family=weights.groupby(['strategy','signal_date','family']).weight.sum().reset_index()
    family.to_csv(out/'family_weights.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for ax,(name,(_,en)) in zip(axes.ravel(),LABELS.items()):
        g=actions[actions.strategy.eq(name)]
        ax.plot(g.date,g.max_factor_weight*100,label='Largest individual factor weight',color=COLORS['PURE_IC'])
        ax.set(title=en,xlabel='Signal date',ylabel='Largest factor weight (%)',ylim=(0,105));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Pure IC has no factor cap and can concentrate\nWeekly historical weights | {nav.index[0].date()} to {nav.index[-1].date()} | zero when no positive IC',fontsize=13)
    fig.savefig(figdir/'factor_concentration.png',dpi=140);plt.close(fig)
    summary=[]
    for name in LABELS:
        a=actions[actions.strategy.eq(name)];w=weights[weights.strategy.eq(name)]
        summary.append({'strategy':name,'mean_positive_factor_count':a.positive_factors.mean(),
            'mean_effective_factor_count':a.effective_factor_count.mean(),'mean_max_weight':a.max_factor_weight.mean(),
            'maximum_factor_weight':a.max_factor_weight.max(),'cash_target_signals':int(a.action.eq('CASH').sum()),
            'pure_insufficient_score_holds':int(a.reason.eq('fewer_than_50_scores').sum()),
            'mean_family_weights':family[family.strategy.eq(name)].groupby('family').weight.mean().to_dict()})
    write_json(out/'weight_summary.json',summary)
    paired=[]
    for scope in ['common','full']:
        frame=perf[perf.scope.eq(scope)].set_index(['strategy','k','variant'])
        for name in LABELS:
            for k in [10,15,20,30,50]:
                pure=frame.loc[(name,k,'PURE_IC')]
                for control in ['FIXED','IC_SHRUNK']:
                    base=frame.loc[(name,k,control)]
                    paired.append({'scope':scope,'strategy':name,'k':k,'control':control,
                        **{'delta_'+f:pure[f]-base[f] for f in ['cagr','sharpe','max_drawdown','stress_cagr','annual_two_way_turnover']}})
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired_differences.csv',index=False)
    wins={c:int((paired.scope.eq('common')&paired.control.eq(c)&paired.delta_cagr.gt(0)).sum()) for c in ['FIXED','IC_SHRUNK']}
    lines=['# 六策略纯 IC 加权补充实验：取消固定份额与类别预算','',
        '这不是替换原实验，也没有调整前向日报账户。本轮增加 PURE_IC，与原来的 FIXED（固定融合）和 IC_SHRUNK（组内半固定、半 IC）并列。K=10、15、20、30、50 全部保留；主比较仍是相同 K50，K20 为预先指定的观察点，不按回测最高收益选赢家。','',
        '## 一、纯 IC 到底取消了什么','',
        '四个多因子策略的 13 个传统因子直接竞争一个总预算，价值策略仍限于原来的 3 个因子、成长策略仍限于 2 个。没有增加消息、公告或机器学习因子。股票池与因子方向不变。','',
        '`因子权重 = max(过去平均 RankIC, 0) / 所有合格因子正平均 RankIC 之和`','',
        '“合格”在这里仅指有足够历史：最近 252 个已成熟交易日中至少有 120 个有效 IC，否则该因子权重为零。不再保留 50% 固定份额，不再规定五类各 20%，没有最低权重、上限或时间平滑，也不会将负 IC 因子反向使用。','',
        '这是一种明确的纯正均值 IC 加权定义，不是唯一的“传统 IC”定义；它不是 ICIR 加权，也不是用因子协方差优化组合。正 IC 截断本身就是动态筛选，不能理解为所有因子永远参与。','',
        '分数直接由生产预处理后的单因子加权合成，不再先算类别分数再对类别做一次标准化。每只股票按可用因子权重重新归一；价值/成长保留原有原始因子数量要求。因而本轮测试的是完整的“全因子 IC 融合政策”，不是仅改一个系数、其他计算都完全不动。','',
        '因子权重不等于股票仓位。选出 TopK 后，股票目标资金仍等权。','',
        '## 二、哪些情况下空仓，为什么增加共同起点','',
        '如果所有因子平均 IC 都非正，或都没有足够历史，下一次规定的执行时点转为现金目标，绝不偷偷回退到等权。没有报价的股票不能卖出，仍保留旧估值；后续现金目标会再次尝试卖出。原控制组本来无法形成有效股票池的日期，所有组保持原规则。纯 IC 有正权重但有效评分不足 50 只，则该策略所有 K 保持持仓，并单独记录。','',
        f'共同起点由历史样本是否够用自动确定：{completed["common_signal"]} 形成信号，{completed["common_execution"]} 下一交易日收盘首次执行。三组均在共同信号日从现金净值 1 起步，结束日为 {nav.index[-1].date()}。这不是挑一个表现好的日期，而是要求六策略所有配置因子都有至少 120 个已成熟 IC，且均可形成排名。','',
        '另保存 2011 年初至 2026-10-15 的原始全区间三组结果。原来的两组直接复用，纯 IC 新跑。不能把早期纯 IC 等待历史而空仓、另外两组已经投资的差异，算成纯 IC 选股更好的证据。因此下面优先解释共同起点结果。','',
        '## 三、相同 K50 的三组结果','',
        '每项按“固定融合 → 半固定半 IC → 纯 IC”排序。年化收益为扣模拟费用后的复合年化收益；夏普使用零无风险利率。','']
    for name,(cn,_) in LABELS.items():
        g=main[main.strategy.eq(name)&main.k.eq(50)].set_index('variant').loc[VARIANTS]
        lines.append(f'- {cn}：年化收益 '+ ' → '.join(f'{x:.2%}' for x in g.cagr)+'；夏普 '+ ' → '.join(f'{x:.2f}' for x in g.sharpe)+'；最大回撤 '+ ' → '.join(f'{x:.2%}' for x in g.max_drawdown)+'。')
    lines+=['',f'在全部 30 个同策略、同 K 的配对中，纯 IC 年化收益高于固定融合 {wins["FIXED"]}/30，高于半固定半 IC {wins["IC_SHRUNK"]}/30。这只是描述性计数，不是显著性检验，30 个组合也不是相互独立的试验。','',
        '这里最需要警惕的是成长策略：它有 88 个现金目标信号，且单因子权重可以达到 100%。收益变化混合了因子重新分配、单因子切换和退出市场的效果，不能全部解释成“IC 更会选股票”。本轮没有做保持同一股票暴露的现金规则消融，因此不能定量拆出三者分别贡献多少。','',
        '![共同起点 K50 净值](figures/nav_k50.png)','',
        '图中虚线为各策略自身点时点股票池等权基准，不是沪深300指数。所有线均扣本轮模拟的费用。净值纵轴为对数，便于观察相对增速。','',
        '![共同起点 K50 回撤](figures/drawdown_k50.png)','',
        '## 四、缩小到 K20 后是否仍成立','']
    for name,(cn,_) in LABELS.items():
        g=main[main.strategy.eq(name)&main.k.eq(20)].set_index('variant').loc[VARIANTS]
        lines.append(f'- {cn}：年化收益 '+' → '.join(f'{x:.2%}' for x in g.cagr)+'；夏普 '+' → '.join(f'{x:.2f}' for x in g.sharpe)+'；最大回撤 '+' → '.join(f'{x:.2%}' for x in g.max_drawdown)+'。')
    lines+=['','![五档持仓数年化收益](figures/cagr_by_k.png)','','![五档持仓数夏普](figures/sharpe_by_k.png)','',
        '减少股票数量与改变因子权重是两个问题，不能把固定 K50 和纯 IC K10 的差异全部归因于 IC。本轮保留所有同 K 对照，不新增随机子集实验，也不据此宣称持仓分散程度已通过实盘验证。','',
        '## 五、不要只看十五年总收益：最近阶段与交易成本','',
        '2021—2026 年 K50 的年化收益同样按三组顺序列出。这是沿各自历史持仓切出的阶段表现，不是在 2021 年重新从现金建仓。','']
    for name,(cn,_) in LABELS.items():
        g=sub[sub.strategy.eq(name)&sub.k.eq(50)&sub.period.eq('2021-2026')].set_index('variant').loc[VARIANTS]
        lines.append(f'- {cn}：'+' → '.join(f'{x:.2%}' for x in g.cagr)+'。')
    lines+=['','![分阶段比较](figures/recent_k50.png)','',
        '基础成本为每笔买卖金额的万分之三；压力测试为每边万分之十三。两者都不是完整的真实券商费用模型。下面列出 K50“纯 IC / 固定融合 / 半固定半 IC”的压力年化，以及纯 IC 的年双边换手倍数（买入加卖出，不是单边口径）。','']
    for name,(cn,_) in LABELS.items():
        g=main[main.strategy.eq(name)&main.k.eq(50)].set_index('variant')
        lines.append(f'- {cn}：压力年化 {g.loc["PURE_IC","stress_cagr"]:.2%} / {g.loc["FIXED","stress_cagr"]:.2%} / {g.loc["IC_SHRUNK","stress_cagr"]:.2%}；纯 IC 年双边换手 {g.loc["PURE_IC","annual_two_way_turnover"]:.2f} 倍。')
    lines+=['','![压力成本比较](figures/stress_cagr_by_k.png)','',
        '## 六、权重到底有多集中','',
        '有效因子数 = 1 / Σ权重²，例如 10 个因子各 10% 时为 10；一个因子拿走 100% 时为 1。现金目标记为 0。这个指标只描述权重集中度，不代表因子之间相互独立。','']
    for s in summary:
        cn=LABELS[s['strategy']][0]
        lines.append(f'- {cn}：平均正权重因子 {s["mean_positive_factor_count"]:.2f} 个，平均有效因子数 {s["mean_effective_factor_count"]:.2f}，历史单因子权重峰值 {s["maximum_factor_weight"]:.1%}；现金目标 {s["cash_target_signals"]} 次，评分不足 50 的保持持仓 {s["pure_insufficient_score_holds"]} 次。')
    lines+=['','![单因子最大权重](figures/factor_concentration.png)','',
        '五类预算已取消，不代表类别暴露自动均衡：正 IC 因子较多或 IC 更高的类别会获得更多资金评分权重。13 个因子彼此相关，数值归一化不能自动消除重复暴露。完整逐周权重保存在 weights.csv.gz，类别合计保存在 family_weights.csv。','',
        '## 七、验证和边界','',
        f'- 本轮新跑 120 条基础成本路径和 120 条压力路径，另复用原来的 60 条控制路径。共同起点主比较为 90 条，完整日历附录也为 90 条；不是 180 次独立实验。',
        f'- {completed["original_engine_parity_checks"]} 次真实历史片段，与已有回放引擎比对正常建仓/调仓，净值最大误差 {completed["max_engine_nav_error"]:.3g}。原引擎已在前次实验与生产撮合器核对。空仓路径另用单元测试验证，包括费用与无报价卖出失败；原引擎的空列表代表保持持仓，不能冒充空仓校验。',
        f'- 成长策略 K20/K50、基础/压力两种费用，共 {cash_validation["full_history_paths"]} 条完整历史路径，用生产撮合器处理正常目标、独立逐股卖出逻辑处理现金目标再次核对；覆盖每条路径的 88 次现金目标，最大净值误差 {cash_validation["max_nav_error"]:.3g}。',
        f'- 所有使用的收益标签终点严格早于信号；净值重算指标最大误差 {completed["max_metric_error"]:.3g}。',
        '- 原实验文件、源数据、生产策略代码和前向账户输入/计划/账本均校验哈希；没有修改日报、实盘配置或原来的两组结果。',
        '- 末期无报价持仓仍按最近可用价格记账；performance.csv 提供末期缺报价资产占比以及将其归零的总收益敏感性，不将账面净值当成保证可变现的现金。',
        '- 财务数据按公告日对齐，但仍是当前供应商历史快照，不是原始版本档案。现金流因子仍是经营现金流代理，不是真正的自由现金流。行业历史覆盖存在边界。',
        '- 回放使用复权价、可分割份额；没有完整模拟整手、最低佣金、历史印花税、涨跌停与市场冲击。五日收益标签相互重叠，120 个有效日不等于 120 个独立样本。',
        '- 这些历史已经看过。本轮回答“取消约束后的结果怎样”，不能因此直接升级实盘；不追加新权重参数去追逐本轮最优值。','',
        '## 八、这次实验回答了什么','',
        '取消限制后，纯 IC 并没有成为六策略通用的更优替代：它在全阶段 K50 年化收益上普遍超过固定融合，但对半固定半 IC 只在全部 K 配对中赢了一半；缩小到 K20 后，基础池和中小盘明显弱于收缩版本。价值的收益小幅提高，不代表回撤改善。','',
        '2021—2026 年 K50，纯 IC 只在价值、成长两个策略上超过固定融合；科技主题虽然全期较强，最近阶段却落后。基础和中小盘的高换手也会吞噬优势。结论不是“IC 没用”，也不是“纯 IC 一定更好”，而是权重自由度增加带来的收益、集中、择时和交易成本必须一起评估。','',
        '本轮先保留这组三方证据，不修改日报。若继续研究，最有价值的问题是拆开成长策略的现金退出效果与因子权重效果，而不是再提高权重自由度或凭最高收益选 K。','',
        '## 文件与复现','',
        '协议：research_protocols/20261018_six_strategy_pure_ic_v1.json。依次运行 scripts/run_six_strategy_pure_ic.py、scripts/validate_pure_ic_cash.py、scripts/report_six_strategy_pure_ic.py。报告目录中的 performance.csv 包含两种时间口径的全部指标；common_nav.csv 为共同起点，full_nav.csv 为原全区间；paired_differences.csv 为同策略同 K 的纯 IC 配对差值。协议与实现变动后请换输出目录，不覆盖冻结结果。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    write_json(out/'summary.json',{'common_signal':completed['common_signal'],'common_execution':completed['common_execution'],
        'positive_cagr_pairs':wins,'k50':main[main.k.eq(50)].to_dict('records'),
        'k20':main[main.k.eq(20)].to_dict('records'),'weights':summary})
    print('Report and seven figures written:',out/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    report(parser.parse_args().output)
