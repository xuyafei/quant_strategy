#!/usr/bin/env python3
"""Read-only result interpretation and plots; no strategy tuning or ledger writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.report_mvp_holdings_count import LABELS
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_six_strategy_ic_comparison import DEFAULT_OUTPUT,load_prices
from analysis.holdings_count import replay_batch,metrics
from backtest.backtest_single import _rebalance_to_target_weights


def validate(out):
    completed=json.loads((out/'completed.json').read_text())
    for name,digest in completed['files'].items():
        if sha(out/name)!=digest:raise ValueError('completed result changed: '+name)
    p=json.loads((out/'protocol.json').read_text())
    prices=load_prices(p)
    ranks=pd.read_csv(out/'rankings.csv.gz',parse_dates=['date'])
    records=[]
    for (strategy,variant),g in ranks.groupby(['strategy','variant']):
        # First, middle, and last years, including genuinely dynamic-weight years.
        for year in [2011,2018,2026]:
            groups=[v.sort_values('rank') for _,v in g[g.date.dt.year.eq(year)].groupby('date')][:4]
            signal=pd.DatetimeIndex([v.date.iloc[0] for v in groups])
            begin=prices.index.get_loc(signal[0]);end=prices.index.get_loc(signal[-1])+5
            for k in p['k_values']:
                names=pd.Index(sorted(set(s for v in groups for s in v.head(k).symbol)))
                px=prices.iloc[begin:end].reindex(columns=names)
                ex=px.index.get_indexer(signal)+1
                plans=np.stack([names.get_indexer(v.head(k).symbol) for v in groups])[:,None,:]
                nav,_,_=replay_batch(px,ex,plans,p['commission_rate'])
                marks=px.ffill();cash=1.;shares=pd.Series(0.,index=names);actual=[]
                for i,date in enumerate(px.index):
                    if i in ex:
                        w=list(ex).index(i)
                        selected=[s for s in groups[w].head(k).symbol if pd.notna(px.loc[date,s])]
                        cash,shares=_rebalance_to_target_weights(cash,shares,px.loc[date],selected,[1/k]*len(selected),p['commission_rate'],valuation_px=marks.loc[date])
                    actual.append(cash+(shares*marks.loc[date]).sum())
                error=float(np.max(np.abs(nav[:,0]-np.asarray(actual))))
                if error>1e-10:raise AssertionError('matcher parity failed')
                records.append({'strategy':strategy,'variant':variant,'k':k,'year':year,'signals':len(groups),'max_nav_error':error})
    weights=pd.read_csv(out/'weights.csv.gz',parse_dates=['signal_date','latest_outcome_end'])
    mature=weights.latest_outcome_end.isna()|weights.latest_outcome_end.lt(weights.signal_date)
    sums=weights.groupby(['strategy','signal_date','family']).within_family_weight.sum()
    budgets=weights.groupby(['strategy','signal_date']).nominal_weight.sum()
    if not mature.all() or not np.allclose(sums,1) or not np.allclose(budgets,1) or (weights.within_family_weight<=0).any():
        raise AssertionError('weight audit failed')
    perf=pd.read_csv(out/'performance.csv')
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True)
    max_metric_error=0.
    for r in perf.itertuples():
        key=f'{r.strategy}__{r.variant}__K{r.k}'
        recalculated=metrics(nav[key].to_numpy())
        for metric,value in recalculated.items():
            max_metric_error=max(max_metric_error,abs(float(value[0])-getattr(r,metric)))
    if max_metric_error>1e-10:raise AssertionError('metric recomputation mismatch')
    # Account for weeks with no eligible rows at all, not just <50 ranked names.
    calendar=json.loads((out/'calendar.json').read_text())
    all_signals=pd.DatetimeIndex(calendar['signals'])
    unavailable=[]
    for strategy in sorted(ranks.strategy.unique()):
        active=set(pd.DatetimeIndex(ranks.loc[ranks.strategy.eq(strategy),'date']))
        for signal in all_signals:
            if signal not in active:
                unavailable.append({'date':signal,'strategy':strategy,'action':'ALL_ARMS_HOLD_OR_INITIAL_CASH'})
    pd.DataFrame(unavailable,columns=['date','strategy','action']).to_csv(out/'unavailable_signals.csv',index=False)
    result={'status':'PASS','production_matcher_checks':len(records),'max_nav_error':max(r['max_nav_error'] for r in records),
        'max_metric_recompute_error':max_metric_error,'all_weights_positive':True,'all_labels_matured':True,
        'family_and_total_budgets_sum_one':True,'fixed_prior_max_metric_error':completed['fixed_prior_max_metric_error'],
        'paper_account_unchanged':completed['paper_account_unchanged'],
        'unavailable_strategy_signals':len(unavailable),'checks':records}
    write_json(out/'validation.json',result)
    print('VALIDATION PASS:',len(records),'historical matcher checks',flush=True)


def report(out):
    p=json.loads((out/'protocol.json').read_text())
    perf=pd.read_csv(out/'performance.csv');diff=pd.read_csv(out/'paired_differences.csv')
    sub=pd.read_csv(out/'subperiods.csv');yearly=pd.read_csv(out/'annual.csv')
    weights=pd.read_csv(out/'weights.csv.gz',parse_dates=['signal_date'])
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True)
    bench=pd.read_csv(out/'benchmark_nav.csv',index_col='date',parse_dates=True)
    skipped=pd.read_csv(out/'unavailable_signals.csv')
    concentration=pd.read_csv(out/'concentration.csv')
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ['nav_k50','drawdown_k50','cagr_by_k','sharpe_by_k','drawdown_by_k','weight_shift','subperiod_ic_gain']:
        fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
        for ax,(name,(_,en)) in zip(axes.ravel(),LABELS.items()):
            if kind in ['nav_k50','drawdown_k50']:
                for v,color in [('FIXED','#4b5563'),('IC_SHRUNK','#16708b')]:
                    s=nav[f'{name}__{v}__K50']
                    y=s if kind=='nav_k50' else 100*(s/s.cummax()-1)
                    ax.plot(y.index,y,label=v,color=color,lw=1.2)
                if kind=='nav_k50':
                    ax.plot(bench.index,bench[name],label='Own PIT universe',color='#a6abb1',lw=.8,ls='--')
                    ax.set_yscale('log')
                ax.set(xlabel='Date',ylabel='Post-cost NAV (log, start=1)' if kind=='nav_k50' else 'Drawdown (%)')
            elif kind=='subperiod_ic_gain':
                g=sub[sub.strategy.eq(name)&sub.k.eq(50)].pivot(index='period',columns='variant',values='cagr')
                delta=100*(g.IC_SHRUNK-g.FIXED)
                ax.bar(delta.index,delta,color=['#16708b' if x>0 else '#6b7280' for x in delta],label='IC CAGR minus FIXED CAGR')
                ax.axhline(0,color='#a6abb1',lw=.8)
                ax.set(xlabel='Historical subperiod',ylabel='Annualized return difference (pp)')
            elif kind=='weight_shift':
                w=weights[weights.strategy.eq(name)].copy()
                n=w.groupby(['signal_date','family']).factor.transform('size')
                w['difference']=(w.nominal_weight-w.family_budget/n).abs()
                distance=w.groupby('signal_date').difference.sum()
                ax.plot(distance.index,100*distance,color='#16708b',label='L1 distance to fixed weights')
                ax.set(xlabel='Weekly signal date',ylabel='Nominal factor weight L1 distance (pp)')
            else:
                metric={'cagr_by_k':'cagr','sharpe_by_k':'sharpe','drawdown_by_k':'max_drawdown'}[kind]
                for v,color in [('FIXED','#4b5563'),('IC_SHRUNK','#16708b')]:
                    g=perf[perf.strategy.eq(name)&perf.variant.eq(v)].sort_values('k')
                    ax.plot(g.k,g[metric]*(1 if metric=='sharpe' else 100),marker='o',color=color,label=v)
                ax.set_xticks(p['k_values'])
                ax.set(xlabel='Target number of stocks K',ylabel={'cagr':'CAGR (%)','sharpe':'Annualized Sharpe (zero RF)','max_drawdown':'Maximum drawdown (%)'}[metric])
            ax.set_title(en);ax.grid(alpha=.2);ax.legend(fontsize=8)
        title={'nav_k50':'Primary comparison: K50 post-cost NAV','drawdown_k50':'Primary comparison: K50 drawdown',
            'cagr_by_k':'Holdings sensitivity: annualized return','sharpe_by_k':'Holdings sensitivity: Sharpe',
            'drawdown_by_k':'Holdings sensitivity: maximum drawdown','weight_shift':'IC changes factor weights, not stock position weights',
            'subperiod_ic_gain':'K50: full-history improvement does not imply recent improvement'}[kind]
        fig.suptitle(f'{title}\nLocal Tushare history / frozen MVP recipes | {nav.index[0].date()} to {nav.index[-1].date()} | 3 bps each side',fontsize=13)
        fig.savefig(figdir/f'{kind}.png',dpi=150);plt.close(fig)
    ws=weights.groupby(['strategy','family','factor']).agg(mean_nominal_weight=('nominal_weight','mean'),min_nominal_weight=('nominal_weight','min'),max_nominal_weight=('nominal_weight','max'),mean_ic=('mean_ic','mean')).reset_index()
    ws.to_csv(out/'weight_summary.csv',index=False)
    coverage=pd.concat([pd.read_csv(f) for f in sorted(out.glob('inputs_*/coverage.csv')) if f.stat().st_size>1],ignore_index=True)
    covlong=coverage.melt(id_vars=['date','strategy','eligible_names'],var_name='factor',value_name='raw_available_fraction').dropna(subset=['raw_available_fraction'])
    covlong['factor']=covlong.factor.str.removeprefix('valid_')
    covsummary=covlong.groupby(['strategy','factor']).raw_available_fraction.agg(['mean','min','max']).reset_index()
    covsummary.to_csv(out/'coverage_summary.csv',index=False)
    fallback=weights.groupby(['strategy','signal_date','family']).reason.first().reset_index()
    fallback.groupby(['strategy','reason']).size().rename('family_signal_count').reset_index().to_csv(out/'fallback_summary.csv',index=False)
    first_dynamic=weights[weights.reason.eq('positive_mean_ic_with_fixed_shrinkage')].groupby('strategy').signal_date.min()
    main=diff[diff.k.eq(50)]
    recent=sub[sub.k.eq(50)&sub.period.eq('2021-2026')].pivot(index='strategy',columns='variant',values='cagr')
    summary={'primary_k50':{'cagr_improved':int(main.ic_minus_fixed_cagr.gt(0).sum()),'sharpe_improved':int(main.ic_minus_fixed_sharpe.gt(0).sum()),'drawdown_improved':int(main.ic_minus_fixed_max_drawdown.gt(0).sum())},
        'recent_k50':{'period':'2021-2026','cagr_improved':int((recent.IC_SHRUNK>recent.FIXED).sum())},
        'stress_k50':{'cagr_improved':int(main.ic_minus_fixed_stress_cagr.gt(0).sum())},
        'all_30_pairs':{'cagr_improved':int(diff.ic_minus_fixed_cagr.gt(0).sum()),'sharpe_improved':int(diff.ic_minus_fixed_sharpe.gt(0).sum())},
        'first_dynamic_signal':{k:str(v.date()) for k,v in first_dynamic.items()}}
    write_json(out/'summary.json',summary)
    lines=['# 六策略：固定权重与滚动IC加权的长历史对照\n',
        f'评价区间：{nav.index[0].date()}—{nav.index[-1].date()}。60条主回放、60条成本压力回放、6条各自点时点股票池基准。当前日报账户不变。\n',
        '## 先看结论\n',
        f'主比较K50：IC版本在6个策略中有{summary["primary_k50"]["cagr_improved"]}个年化收益提高、{summary["primary_k50"]["sharpe_improved"]}个夏普提高、{summary["primary_k50"]["drawdown_improved"]}个最大回撤改善。全部30对同K比较中，年化收益改善{summary["all_30_pairs"]["cagr_improved"]}对，夏普改善{summary["all_30_pairs"]["sharpe_improved"]}对。这里只描述已见历史，不据此自动切换策略。\n',
        f'但更重要的限制是：2021年至2026年这一较近阶段，K50只有{summary["recent_k50"]["cagr_improved"]}个策略的年化收益改善，其他策略均低于固定版。全期改善不能解释为最近也在改善；本轮的主要增益明显依赖更早阶段。成本压力提高到单边0.13%后，K50仍有{summary["stress_k50"]["cagr_improved"]}个策略优于固定版，但基础池原本的收益优势消失。\n',
        '## 这次到底改变了什么\n',
        '四个多因子策略仍用13个传统因子、五类各20%；价值用3个因子，成长用2个因子。不使用消息、公告评分、成交量比率或ML，不恢复严格绩效淘汰。IC只改变类别内部的因子权重，低波动率独占的风险类别仍固定20%。每只入选股票仍等权。\n',
        'IC按各策略当日可用股票池计算，使用经过现有去极值和行业标准化的因子。信号日d的收益标签从d+1收盘到d+6收盘，即下一交易日执行后持有5个交易日；两个端点必须有报价，不用未来补值。权重只读取收益终点严格早于本次信号日的历史标签。5日标签是周频持有的近似，跨节假日不保证等于下次实际调仓间隔。\n',
        '在最近252个已成熟交易日里，每个因子至少有120个有效IC日、每天至少30个有效股票。组内权重=50%组内等权+50%正平均IC的归一化比例；负IC不翻转方向，所有因子保持正权重。任何组内因子样本不足，整组退回等权；全组平均IC不为正也退回等权。没有另加ICIR、参数搜索或时间平滑。相邻5日标签重叠，120个观测不等于120个独立样本，不用朴素显著性检验夸大证据。\n',
        '历史早期保留共同起点：IC证据不足时动态版等同固定版，不能把这段时期描述为已经在有效使用动态权重。各策略首次动态调整日期见下文。\n',
        '## 六策略逐项结果\n']
    for name,(cn,en) in LABELS.items():
        lines.append(f'### {cn}\n')
        lines.append(f'首次至少一个类别使用动态IC权重：{first_dynamic[name].date()}。\n')
        for k in p['k_values']:
            a=perf[(perf.strategy==name)&(perf.k==k)].set_index('variant')
            f,i=a.loc['FIXED'],a.loc['IC_SHRUNK']
            lines.append(f'- K={k}，固定 → IC：年化 {f.cagr:.2%} → {i.cagr:.2%}；夏普 {f.sharpe:.3f} → {i.sharpe:.3f}；最大回撤 {f.max_drawdown:.2%} → {i.max_drawdown:.2%}；年均双边换手 {f.annual_two_way_turnover:.2f} → {i.annual_two_way_turnover:.2f}倍。')
        lines.append('\nK50分阶段：\n')
        for period,g in sub[(sub.strategy==name)&(sub.k==50)].groupby('period'):
            g=g.set_index('variant');f,i=g.loc['FIXED'],g.loc['IC_SHRUNK']
            lines.append(f'- {period}：年化 {f.cagr:.2%} → {i.cagr:.2%}；回撤 {f.max_drawdown:.2%} → {i.max_drawdown:.2%}；相对各自股票池基准的年化收益差 {f.excess_cagr_difference:.2%} → {i.excess_cagr_difference:.2%}。')
        lines.append('\n')
    for filename,title in [('subperiod_ic_gain','分阶段看：全期改善不代表最近也改善'),('nav_k50','同样持有50只的净值与各自股票池基准'),('drawdown_k50','同样持有50只的回撤'),('cagr_by_k','不同持仓数的年化收益'),('sharpe_by_k','不同持仓数的夏普'),('drawdown_by_k','不同持仓数的最大回撤'),('weight_shift','因子权重究竟改变了多少')]:
        lines.append(f'## {title}\n\n![{title}]({figdir/filename}.png)\n')
    lines += ['## 怎样理解持仓数\n',
        '50只是现状对照，20只是预先约定的观察重点；10/15/30用于敏感性比较。没有根据全期最高收益自动选择K，也没有每周自动挑选持仓数量。本轮不重复随机子集实验，因此不能声称某个IC组合的随机子集分布更窄。历史回测能提示候选，不能替代实际本金、整数股和交易费用的可执行检查。\n',
        '本轮IC版大盘K20与K50夏普几乎相同，但K20回撤更深；中小盘K20年化略高于K50，回撤也更深。IC加权没有消除少持仓的集中风险，不能从本结果推导出六策略应统一缩为20只。\n',
        '## 成交、数据和局限\n',
        '沿用上一轮已经核对过的复权价、可分割份额研究引擎。每周最后一个已完成交易周信号，下一交易日收盘执行；每日按持仓漂移记账，只有调仓日恢复目标权重。每次目标名单在执行报价出现前形成。缺报价不补买下一名，未能买入的资金留在现金，无法卖出的旧仓保留并按此前价格估值。故实际持股可能超过K。\n',
        '主口径单边0.03%，压力口径单边0.13%；两者均非完整历史税费模型。没有完整模拟整手、最低佣金、印花税变迁、涨跌停排队、部分成交和冲击成本，不能直接称为小资金账户可实现收益。夏普按252交易日、零无风险利率计算；年化超额字段为策略CAGR减各自基准CAGR，不是回归alpha。\n',
        '财务数据按公告日向后匹配，但供应商当前历史快照不等于当年逐期保存的原始版本；历史行业分类也有覆盖局限。FREE_CASH_FLOW_YIELD沿用经营现金流每股/股价代理。原生产缺失值处理、可用因子重归一和行业后备标准化保持不变，配置13个因子不意味着每只股票都有13个有效原始字段。\n',
        '逐策略逐因子的历史原始可用率见coverage_summary.csv；部分现金利润质量截面的有效率低于80%。没有在IC组额外补值或另设股票池筛选，两组复用同一缺失值规则。本轮没有把第121篇的80%覆盖隔离阈值偷偷接入冻结日报策略。\n',
        '永久缺报价的旧持仓可能造成估值偏乐观。performance.csv保留末日缺报价资产比例，以及该部分按零计价的终值敏感性；这不是完整退市清算模拟。\n',
        f'两种权重、所有K共用信号不足50只时保持原仓规则，共记录{len(skipped)}个策略-信号跳过事件。不删除这些日期的日收益。\n',
        '这是一段已经反复研究过的历史，严格滚动不等于独立未知样本验证。本实验只能评价此处明确写出的“组内IC＋50%固定收缩”算法，不代表所有IC加权方式，也不证明固定或动态权重永远优越。\n',
        '## 复核入口\n',
        f'- 冻结协议：[protocol.json]({out}/protocol.json)。',
        f'- 全部60组指标：[performance.csv]({out}/performance.csv)。',
        f'- 同K差值：[paired_differences.csv]({out}/paired_differences.csv)。',
        f'- 逐年与分阶段：[annual.csv]({out}/annual.csv)、[subperiods.csv]({out}/subperiods.csv)。',
        f'- 每次因子权重：[weights.csv.gz]({out}/weights.csv.gz)；汇总：[weight_summary.csv]({out}/weight_summary.csv)。',
        f'- 独立核对：[validation.json]({out}/validation.json)。\n']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Compact real data for a single-file canvas; monthly last observations are display only.
    display_indices=np.unique(np.r_[0,np.flatnonzero(~nav.index.to_period('M').duplicated(keep='last'))])
    monthly=nav.iloc[display_indices];bm_month=bench.reindex(monthly.index)
    data={'summary':summary,'period':f'{nav.index[0].date()}—{nav.index[-1].date()}',
        'performance':perf.to_dict('records'),'differences':diff.to_dict('records'),
        'subperiods':sub.to_dict('records'),'weight_summary':ws.to_dict('records'),
        'labels':{k:v[0] for k,v in LABELS.items()},'dates':[str(d.date()) for d in monthly.index],
        'nav':{key:monthly[key].round(5).tolist() for key in monthly},
        'benchmark':{key:bm_month[key].round(5).tolist() for key in bm_month}}
    write_json(out/'canvas_data.json',data)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('REPORT AND SEVEN FIGURES COMPLETE',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    a.add_argument('--skip-validation',action='store_true');args=a.parse_args()
    if not args.skip_validation:validate(args.output)
    report(args.output)
