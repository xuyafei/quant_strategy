"""Read-only diagnostics for a declared family of daily, net-of-cost NAVs.

No imports from trading, no factor admission decisions, no automatic date joins.
The caller is responsible for declaring the entire tried family and data lineage.
"""
import numpy as np
import pandas as pd

from analysis.research_statistics import adjust_pvalues, hac_mean
from analysis.return_attribution import ols_hac
from analysis.selection_diagnostics import cscv, cscv_summary


def validate_nav(nav):
    if (not isinstance(nav, pd.DataFrame) or len(nav) < 2 or nav.shape[1] < 2
            or not isinstance(nav.index, pd.DatetimeIndex) or nav.index.hasnans
            or nav.index.has_duplicates or not nav.index.is_monotonic_increasing
            or nav.columns.has_duplicates
            or not all(isinstance(c, str) and c for c in nav.columns)
            or not np.isfinite(nav.to_numpy(dtype=float)).all() or (nav <= 0).any().any()):
        raise ValueError('NAV requires sorted unique dates, named columns and complete positive values')


def exact_window(frame, window):
    """Only an explicit, inclusive interval may reduce a sample; no inner join."""
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError('declare [start, end] explicitly')
    start, end = map(pd.Timestamp, window)
    if start not in frame.index or end not in frame.index or start > end:
        raise ValueError('window endpoints must exist and be ordered')
    return frame.loc[start:end]


def build_research_audit(nav, pairs, *, lags=5, factors=None,
                         attribution_window=None, cscv_window=None, blocks=10):
    """pairs maps each candidate to its own PIT-pool benchmark.

Daily units, 252-session annualization. Attribution consumes supplied explanatory
returns without silently subtracting a risk-free rate; report the convention.
Statistical failures are NOT passed tests. Invalid/missing NAV data fails loudly.
"""
    validate_nav(nav)
    if (not isinstance(pairs, dict) or not pairs or any(
            s not in nav or b not in nav or s == b for s, b in pairs.items())):
        raise ValueError('declare a nonempty candidate-to-benchmark mapping')
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 0:
        raise ValueError('HAC lag must be a nonnegative integer')
    returns = nav.pct_change(fill_method=None).iloc[1:]
    active = pd.DataFrame({s: returns[s] - returns[b] for s, b in pairs.items()})
    rows, correction_inputs = [], []
    for strategy, benchmark in pairs.items():
        total = float(nav[strategy].iloc[-1] / nav[strategy].iloc[0] - 1)
        bench = float(nav[benchmark].iloc[-1] / nav[benchmark].iloc[0] - 1)
        row = dict(strategy=strategy, benchmark=benchmark, total_return=total,
                   benchmark_return=bench, cumulative_gap_pp=(total-bench)*100)
        try:
            row.update(hac_mean(active[strategy], lags))
            row['status'] = 'ASSESSED'
            correction_inputs.append(row['p_positive'])
        except ValueError as error:
            row.update(status='NOT_ASSESSED', reason=str(error))
            # Keep the full declared family; never shrink multiplicity after errors.
            correction_inputs.append(1.)
        rows.append(row)
    adjusted = adjust_pvalues(correction_inputs)
    for index, row in enumerate(rows):
        row['adjusted_positive_p'] = (adjusted.iloc[index].to_dict()
                                      if row['status'] == 'ASSESSED' else None)

    attribution = {'status': 'NOT_REQUESTED'}
    if factors is not None:
        # Regression dates must exactly equal the explicitly declared return window.
        sample = exact_window(returns, attribution_window)
        if not factors.index.equals(sample.index):
            raise ValueError('attribution dates must align exactly; no automatic intersection')
        fitted, alpha_p = [], []
        for strategy in pairs:
            try:
                stats, coefficients, contributions = ols_hac(sample[strategy], factors, lags=lags)
                fitted.append(dict(strategy=strategy, status='ASSESSED', **stats,
                                   coefficients=coefficients.to_dict('records'),
                                   arithmetic_contributions=contributions))
                alpha_p.append(stats['alpha_p_value'])
            except ValueError as error:
                fitted.append(dict(strategy=strategy, status='NOT_ASSESSED', reason=str(error)))
                alpha_p.append(1.)
        corrected = adjust_pvalues(alpha_p).holm
        for index, row in enumerate(fitted):
            row['alpha_p_holm'] = float(corrected.iloc[index]) if row['status'] == 'ASSESSED' else None
        attribution = dict(status='ASSESSED' if all(r['status'] == 'ASSESSED' for r in fitted)
                           else 'PARTIAL_OR_NOT_ASSESSED', results=fitted,
                           start=str(sample.index[0].date()), end=str(sample.index[-1].date()),
                           observations=len(sample), convention='supplied proxies; no risk-free subtraction')
    elif attribution_window is not None:
        raise ValueError('attribution window specified without factor returns')

    selection = {'status': 'NOT_REQUESTED'}
    if cscv_window is not None:
        sample = exact_window(active, cscv_window)
        try:
            if len(sample) < 60:
                raise ValueError('at least 60 observations required for report CSCV')
            splits = cscv(sample, blocks)
            selection = dict(status='ASSESSED', **cscv_summary(splits), blocks=blocks,
                             observations=len(sample), excluded_rows=len(active)-len(sample),
                             start=str(sample.index[0].date()), end=str(sample.index[-1].date()))
        except ValueError as error:
            selection = dict(status='NOT_ASSESSED', reason=str(error), observations=len(sample))

    return dict(schema_version=1, diagnostic_only=True, changes_trading_rules=False,
                fresh_oos=False, live_approval=False, family_size=len(pairs),
                nav_start=str(nav.index[0].date()), nav_end=str(nav.index[-1].date()),
                observations=len(returns), hac_lags=lags, sessions_per_year=252,
                candidates=rows, attribution=attribution, selection=selection,
                point_in_time_data_check='NOT_ASSESSED_FROM_NAV',
                warnings=[
                    '历史复核不是新的样本外验证，报告不决定下单或淘汰因子。',
                    '只能检验声明的候选集；遗漏失败试验会低估多重尝试风险。',
                    '日主动收益=策略日收益减各自基准日收益；年化算术均值不等于复合收益。',
                    'HAC区间使用渐近正态近似，95%区间未做多重校正；p值不是策略有效概率。',
                    'R²解释日收益变动，不是利润占比或因果解释；本地代理不是官方Fama-French因子。',
                    'CSCV包含非时间顺序切分；PBO不是未来亏损概率；并列中位数另报。',
                    '仅凭净值不能证明股票池点时点、财务数据版本、费用或交易可实现性。',
                ])


def render_markdown(report):
    """Concise reader-facing companion; full coefficients stay in report.json."""
    lines = ['# 研究检查报告', '',
             f"区间：{report['nav_start']}—{report['nav_end']}；日收益 {report['observations']} 条；"
             f"声明候选 {report['family_size']} 个。", '',
             '本报告只读，不改持仓、订单或因子准入，不是实盘放行。', '',
             '## 与各自基准比较', '']
    for row in report['candidates']:
        lines += [f"### {row['strategy']}", '', f"基准：{row['benchmark']}。"
                  f"累计收益 {row['total_return']:.2%}，基准 {row['benchmark_return']:.2%}，"
                  f"累计差距 {row['cumulative_gap_pp']:+.2f} 个百分点。", '']
        if row['status'] == 'ASSESSED':
            lines += [f"年化算术主动收益 {row['annual_arithmetic_active']:+.2%}；"
                      f"未校正的95% HAC区间 [{row['ci_low_annual']:.2%}, {row['ci_high_annual']:.2%}]。"
                      f"正向单侧 p={row['p_positive']:.4f}，Holm调整后 "
                      f"{row['adjusted_positive_p']['holm']:.4f}。", '']
        else:
            lines += [f"统计证据未评估：{row['reason']}。不是通过，也不自动禁用。", '']
    lines += ['## 收益来源', '']
    attribution = report['attribution']
    if 'results' not in attribution:
        lines += ['未提供解释因子，本次未做收益归因。', '']
    else:
        lines += [f"归因区间：{attribution['start']}—{attribution['end']}；"
                  '使用传入代理，未扣除无风险利率。', '']
        for row in attribution['results']:
            if row['status'] == 'ASSESSED':
                lines += [f"- {row['strategy']}：R² {row['r_squared']:.2%}；剩余均值项年化 "
                          f"{row['alpha_arithmetic_annual']:+.2%}；双侧alpha p经Holm调整 "
                          f"{row['alpha_p_holm']:.4f}。"]
            else:
                lines += [f"- {row['strategy']}：未评估，{row['reason']}。"]
        lines += ['']
    selection = report['selection']
    lines += ['## 历史挑选风险', '']
    if selection['status'] == 'ASSESSED':
        lines += [f"显式选取 {selection['start']}—{selection['end']}，{selection['observations']} 条；"
                  f"排除窗口外 {selection['excluded_rows']} 条。共 {selection['splits']} 个切分："
                  f"训练冠军在测试部分严格低于中位数 {selection['pbo_strict']:.2%}，"
                  f"恰好处于中位数 {selection['median_mass']:.2%}，"
                  f"含并列的PBO {selection['pbo_inclusive']:.2%}。", '']
    else:
        lines += [f"未评估：{selection.get('reason', '本次未要求CSCV检查')}。", '']
    lines += ['## 解释边界', ''] + ['- '+s for s in report['warnings']] + ['']
    return '\n'.join(lines)
