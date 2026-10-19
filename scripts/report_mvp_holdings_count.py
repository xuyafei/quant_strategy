#!/usr/bin/env python3
"""Render the frozen K-comparison outputs without changing model or selections."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    'BASE_V2_MULTI_FACTOR': ('基础池多因子', 'Base multi-factor'),
    'LARGE_CAP_MULTI_FACTOR': ('大盘多因子', 'Large-cap multi-factor'),
    'MID_SMALL_CAP_MULTI_FACTOR': ('中小盘多因子', 'Mid/small multi-factor'),
    'VALUE_FOCUSED': ('价值策略', 'Value focused'),
    'GROWTH_FOCUSED': ('成长策略', 'Growth focused'),
    'TECH_GROWTH_THEME_MULTI_FACTOR': ('科技成长主题', 'Tech-growth theme'),
}
COLORS = {10:'#aab3bd',15:'#658598',20:'#176e9e',30:'#8a704f',50:'#303944'}


def render(out):
    p=pd.read_csv(out/'performance.csv')
    rs=pd.read_csv(out/'random_summary.csv')
    rp=pd.read_csv(out/'random_paths_metrics.csv')
    sub=pd.read_csv(out/'subperiods.csv')
    coverage=pd.read_csv(out/'coverage.csv')
    concentration=pd.read_csv(out/'concentration.csv')
    audit=pd.read_csv(out/'execution_audit.csv')
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True)
    protocol=json.loads((out/'protocol.json').read_text())
    rules=protocol['comparison_rules']
    assets=out/'figures';assets.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ['nav','drawdown']:
        fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
        for ax,(name,labels) in zip(axes.ravel(),LABELS.items()):
            for k in [10,15,20,30,50]:
                s=nav[f'{name}_K{k}']
                y=s if kind=='nav' else (s/s.cummax()-1)*100
                ax.plot(y.index,y,label=f'K={k}',color=COLORS[k],lw=1.8 if k==20 else 1.1)
            if kind=='nav':ax.set_yscale('log')
            ax.set(title=labels[1],xlabel='Date',ylabel='Post-cost NAV (log scale, start=1)' if kind=='nav' else 'Drawdown (%)')
            ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
        fig.suptitle(f'MVP fixed-score portfolios | {nav.index[0].date()} to {nav.index[-1].date()} | 3 bps each side\nSource: local Tushare history / MVP post-cost replay',fontsize=13)
        fig.savefig(assets/f'{kind}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for ax,(name,labels) in zip(axes.ravel(),LABELS.items()):
        g=p[p.strategy.eq(name)].sort_values('k')
        ax.plot(g.k,g.sharpe,marker='o',color=COLORS[20],label='Sharpe (zero RF)')
        ax2=ax.twinx();ax2.plot(g.k,-100*g.max_drawdown,marker='s',color=COLORS[50],ls='--',label='Max drawdown magnitude')
        ax.set(title=labels[1],xlabel='Target holdings K',ylabel='Annualized Sharpe');ax2.set_ylabel('Max drawdown magnitude (%)')
        ax.set_xticks(g.k);ax.grid(alpha=.2)
        ax.legend(loc='upper left',fontsize=8);ax2.legend(loc='upper right',fontsize=8)
    fig.suptitle(f'Return efficiency and peak-to-trough loss | same factors and costs\nSource: local Tushare history / MVP replay | {nav.index[0].date()} to {nav.index[-1].date()}',fontsize=13)
    fig.savefig(assets/'risk_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for ax,(name,labels) in zip(axes.ravel(),LABELS.items()):
        data=[100*rp.loc[rp.strategy.eq(name)&rp.k.eq(k),'cagr'].to_numpy() for k in [10,15,20,30,50]]
        ax.boxplot(data,tick_labels=['10','15','20','30','50'],whis=(5,95),showfliers=False,
            medianprops={'color':COLORS[20]})
        g=p[p.strategy.eq(name)].set_index('k')
        ax.scatter(range(1,6),g.loc[[10,15,20,30,50],'cagr']*100,marker='D',color=COLORS[50],s=20,label='Deterministic Top-K')
        ax.set(title=labels[1],xlabel='Target holdings K',ylabel='Full-period CAGR (%)')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'100 whole-path random subsets per K from each signal\'s top 100 (or all if <100)\nBoxes: P25-P75; whiskers: P5-P95. NOT confidence intervals.\nSource: local Tushare history / MVP replay | {nav.index[0].date()} to {nav.index[-1].date()}',fontsize=12)
    fig.savefig(assets/'random_subsets.png',dpi=150);plt.close(fig)
    decisions=[]
    for name in LABELS:
        g=p[p.strategy.eq(name)].set_index('k');r=rs[rs.strategy.eq(name)].set_index('k')
        similar=abs(g.loc[20,'sharpe']-g.loc[10,'sharpe'])<=rules['similar_sharpe_absolute_gap']
        dd=g.loc[20,'max_drawdown']-g.loc[10,'max_drawdown']>=rules['meaningful_drawdown_reduction_absolute']
        narrower=r.loc[20,'cagr_width']<=r.loc[10,'cagr_width']*(1-rules['meaningful_random_cagr_width_reduction_fraction'])
        k15=(g.loc[15,'sharpe']>=g.loc[20,'sharpe']-rules['k15_close_to_k20_sharpe_gap']
            and g.loc[15,'max_drawdown']>=g.loc[20,'max_drawdown']-rules['k15_close_to_k20_drawdown_gap']
            and r.loc[15,'cagr_width']<=r.loc[20,'cagr_width']*rules['k15_random_width_ratio_limit'])
        decisions.append({'strategy':name,'k20_similar_sharpe_to10':bool(similar),'k20_drawdown_reduction_3pp':bool(dd),
            'k20_random_width_narrower_10pct':bool(narrower),'all_three':bool(similar and dd and narrower),
            'k15_close_enough_to20':bool(k15)})
    pd.DataFrame(decisions).to_csv(out/'decision_checks.csv',index=False)
    lines=['# MVP 持仓数对比：10、15、20、30、50 只\n',
        f'评价区间：{nav.index[0].date()}—{nav.index[-1].date()}；行情原始历史始于 2010 年。采用当前六策略冻结配方，不启用 IC 加权，不重训模型，不改变现有纸面账户。\n',
        (out/'INTERPRETATION.md').read_text() if (out/'INTERPRETATION.md').exists() else '',
        '## 先理解这次比较\n',
        '同一策略、同一周、同一个股票池先生成一份排名，然后分别取前 10/15/20/30/50 名。组内现有因子配方不变；多因子策略五组各占 20%，价值和成长策略用现有聚焦配方。股票在调仓目标中等权，持有期间自然漂移，不是每天重新等权。\n',
        'K=20 是事先指定的主候选，K=50 是当前方案对照。不是跑完后挑夏普最高的一点。完整历史已用于研究，不属于新的样本外验证。\n',
        '## 六策略确定性结果\n']
    for name,(cn,en) in LABELS.items():
        lines.append(f'### {cn}\n')
        g=p[p.strategy.eq(name)].sort_values('k')
        for a in g.itertuples():
            lines.append(f'- K={a.k}：累计 {a.total_return:.2%}；年化 {a.cagr:.2%}；波动 {a.volatility:.2%}；夏普 {a.sharpe:.3f}；最大回撤 {a.max_drawdown:.2%}；年均双边换手 {a.annual_two_way_turnover:.2f} 倍；成本压力下年化 {a.stress_cagr:.2%}。')
        d=next(d for d in decisions if d['strategy']==name)
        lines.append(f'\n预设判据：20 对 10 的“夏普接近、回撤少至少 3 个百分点、随机年化区间宽度缩小至少 10%”三项同时成立：{"是" if d["all_three"] else "否"}；15 接近 20 的三项判据同时成立：{"是" if d["k15_close_enough_to20"] else "否"}。这些阈值是比较辅助，不是统计显著性检验。\n')
        lines.append('分阶段年化 / 最大回撤：\n')
        for period,h in sub[sub.strategy.eq(name)].groupby('period'):
            lines.append('- '+period+'：'+'；'.join(f'K{a.k} {a.cagr:.1%} / {a.max_drawdown:.1%}' for a in h.sort_values('k').itertuples()))
        lines.append('')
    for filename,title in [('nav','净值：对数纵轴'),('drawdown','从此前高点回撤'),('risk_comparison','夏普与最大回撤'),('random_subsets','随机子集敏感性')]:
        lines.append(f'## {title}\n\n![{title}](figures/{filename}.png)\n')
    lines += ['## 随机子集究竟验证什么\n',
        '每个调仓日，在该策略排名前 100 名中抽取 K 只；不足 100 名时使用全部候选，但至少须有 50 名。同一个随机排列取前 10/15/20/30/50，保证不同 K 配对比较。每个 K 重复 100 条完整交易路径，每条路径都保留真实日期顺序、持仓漂移、缺报价和手续费。共有 3000 条随机路径，另有 30 条确定性主路径和 30 条成本压力路径。\n',
        '随机组合不是另一个选股模型，也不是收益置信区间。它回答：在相同高分候选范围内，少持有或多持有一些股票，偶然选中某几只股票对整条路径有多大影响。分布变窄不代表收益变高，也不代表因子有可靠超额。\n',
        '## 数据与成交口径\n',
        '- 数据含历史退市证券，不是只用今天仍上市的股票。三个代码迁移链共六个代码在全部组中一致排除，未擅自拼接。',
        '- 基础池复用上市满 250 个交易日、非 ST、未退市、有当日报价、20 日至少 15 日有效行情、日均成交额不少于 5000 万，以及非正净资产、严重审计意见和退市整理风险过滤。',
        '- 大盘/中小盘按当时基础池流通市值分位区分；价值/成长为数据可用性池；主题按既有五个行业定义。没有新加盈利为正、行业禁令或 IC 准入。',
        '- 财务数据按公告日向后匹配；补齐了原始长历史默认字段集遗漏的 ocf_to_profit。沿用现有因子计算：其中 FREE_CASH_FLOW_YIELD 实际是经营现金流每股 / 股价的代理，不是真正 FCFF/FCFE 收益率。',
        '- 数据是目前提供商返回的历史快照，并非逐期保存的原始披露版本；行业 in/out 区间也不等于完整的历史分类标准档案。未知行业不向过去回填，使用当前工程已有的全市场标准化后备规则。因此不能把本实验称为已消除全部回看偏差的认证。',
        '- 每个已完成交易周最后一个交易日收盘生成排名，下一交易日收盘模拟执行。最后不完整的一周不人为提前调仓。K 组共用同一信号；不足 50 个有效得分时该策略所有 K 都保持原持仓。',
        '- 使用复权价格进行可分割份额研究回放。缺执行报价不补买下一名；目标现金保留，无法卖出的旧仓继续按最近报价估值。因而实际持仓数可能超过目标 K。',
        '- 主结果单边手续费 0.03%；压力结果单边总成本 0.13%。压力场景不是实际历史税费表。净值为每日交易后净值，包含首次与末次手续费；夏普以零无风险利率、252 日年化计算。',
        '- 没有模拟全部历史涨跌停、停牌状态、整数股/最小买入金额、最小佣金、印花税变迁、滑点和冲击。因此本结果不能直接当作百万元账户的可成交收益，也不授权实盘。',
        '- 末期无报价资产按零价值的终值敏感性另列于 performance.csv；它不是完整退市回收模拟，主结果的长期停牌/退市估值仍可能偏乐观。\n',
        '## 覆盖审计\n']
    for name,(cn,_) in LABELS.items():
        g=coverage[coverage.strategy.eq(name)]
        c=concentration[concentration.strategy.eq(name)&concentration.k.eq(20)]
        a=audit[audit.strategy.eq(name)&audit.k.eq(20)]
        lines.append(f'- {cn}：共 {len(g)} 个信号日期，得分不足 50 的日期 {int(g.scored_names.lt(50).sum())} 个；有效候选平均 {g.scored_names.mean():.0f}；已知行业覆盖均值 {g.industry_coverage.mean():.1%}；现金利润质量原始有效率均值 {g.valid_CASH_PROFIT_QUALITY.mean():.1%}。K20 目标最大行业占比均值 {c.max_industry_target_weight.mean():.1%}，实际持仓均值 {a.holdings.mean():.2f}。')
    lines += ['\n## 结果文件\n',
        '- performance.csv：30 组主结果、压力成本与末期无报价资产敏感性。',
        '- nav.csv / stress_nav.csv：逐日净值；annual.csv / subperiods.csv：逐年和分阶段数据。',
        '- random_summary.csv / random_paths_metrics.csv / random_daily_nav.npz：随机分布摘要与完整路径。',
        '- execution_audit.csv：每次调仓现金、手续费、换手、实际持仓、未成交目标及无法卖出仓位。',
        '- concentration.csv：目标行业集中度；coverage.csv：逐周股票池与因子覆盖。',
        '- inputs_YYYY/ranked_top100.csv.gz：当时排名与行业，可恢复每个 K 的目标名单。',
        '- protocol.json / started.json / prepared.json / completed.json：冻结口径、输入及代码哈希。\n',
        '## 验证与使用边界\n',
        '测试覆盖现有撮合函数、完整引擎的正常报价路径、缺报价和手续费处理、随机抽样的嵌套可重复性、基础股票池等价性、按年分块因子一致性，以及未来价格/财报不能改变过去结果。真实历史独立对账结果见 real_data_engine_parity.json；运行方法见 docs/research/MVP_HOLDINGS_COUNT.md。\n',
        '即使某个 K 更好，也只能提出新的研究候选。应先在原冻结账户旁建立独立对照，不能用事后选定 K 重写已有前向账本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT AND FIGURES COMPLETE')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path('output/mvp_holdings_count_20261017_complete_factors'))
    render(a.parse_args().output)
