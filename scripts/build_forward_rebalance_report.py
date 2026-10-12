#!/usr/bin/env python3
"""Read-only close-rebalance reconciliation; never execute trades or edit a ledger."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from live.forward_ledger import LABELS, latest_day, session_config, sha, verify
from scripts.build_forward_holding_day_report import check_close


def reconcile_account(before, after, old, new, old_market, market, trades, dividend, initial):
    """Close execution: old shares earn the price move; new buys do not earn it."""
    for frame in [old, new, old_market, market]:
        if frame.index.has_duplicates:
            raise ValueError('duplicate accounting symbols')
    if not trades.fill_status.isin(['FILLED', 'SKIPPED']).all():
        raise ValueError('unknown fill status')
    filled = trades[trades.fill_status.eq('FILLED')]
    if not filled.side.isin(['BUY', 'SELL']).all() or (filled.qty <= 0).any():
        raise ValueError('invalid filled order')
    p0, p1 = old_market.close.reindex(old.index), market.close.reindex(old.index)
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError('missing valuation for old holdings')
    expected = old.shares.to_dict()
    cash_flow = 0.
    for row in filled.itertuples():
        check_close(row.price, market.loc[row.symbol, 'close'], 'execution must equal model close')
        check_close(row.gross_amount, row.qty*row.price, 'trade notional')
        sign = 1 if row.side == 'BUY' else -1
        expected[row.symbol] = expected.get(row.symbol, 0) + sign*row.qty
        if expected[row.symbol] < 0:
            raise ValueError('negative shares')
        cash_flow += -sign*row.gross_amount - row.commission
    expected = {s:q for s,q in expected.items() if q > 0}
    if expected != new.shares.to_dict():
        raise ValueError('position reconciliation failed')
    fee = float(filled.commission.sum())
    check_close(cash_flow, float(trades.net_cash_flow.sum()), 'trade cash flow')
    check_close(before.market_value, float((old.shares*p0).sum()), 'previous stock value')
    check_close(before.total_asset, before.cash+before.market_value, 'previous assets')
    end_prices = market.close.reindex(new.index)
    if not np.isfinite(end_prices).all():
        raise ValueError('missing valuation for new holdings')
    check_close(after.market_value, float((new.shares*end_prices).sum()), 'ending stock value')
    check_close(after.cash, before.cash+dividend+cash_flow, 'cash')
    check_close(after.commission, fee, 'fees')
    check_close(after.dividend_cash_gross, dividend, 'dividends')
    check_close(after.total_asset, after.cash+after.market_value, 'assets')
    check_close(after.net_value, after.total_asset/initial, 'NAV')
    check_close(after.n_positions, len(new), 'positions')
    check_close(after.filled_orders, len(filled), 'fills')
    price_pnl = float((old.shares*(p1-p0)).sum())
    pnl = float(after.total_asset-before.total_asset)
    check_close(pnl, price_pnl+dividend-fee, 'daily price PNL plus dividend less fees')
    buy = filled[filled.side.eq('BUY')]
    sell = filled[filled.side.eq('SELL')]
    nav_before_trades = before.total_asset+price_pnl+dividend
    if before.total_asset <= 0 or nav_before_trades <= 0:
        raise ValueError('nonpositive assets')
    return {'price_pnl': price_pnl, 'daily_pnl': pnl, 'daily_return_pct': pnl/before.total_asset*100,
            'cumulative_return_pct': (after.net_value-1)*100, 'buy_fills': len(buy), 'sell_fills': len(sell),
            'buy_amount': float(buy.gross_amount.sum()), 'sell_amount': float(sell.gross_amount.sum()),
            'filled_turnover_pct': float(filled.gross_amount.sum())/nav_before_trades*100,
            'cash_ratio_pct': after.cash/after.total_asset*100,
            'previous_positions': len(old)}


def build(root, date, out):
    root, out = root.resolve(), out.resolve()
    if out == root or (out.is_relative_to(root) and not out.is_relative_to(root/'reports')):
        raise ValueError('reports cannot overwrite sealed source directories')
    if out.exists():
        raise FileExistsError('choose a new report directory')
    latest_day(root)  # Verify the entire already committed ancestry.
    config = session_config(root)
    day = root/'days'/date.replace('-', '')
    plan = root/'plans'/date.replace('-', '')
    dm, pm = verify(day), verify(plan)
    earlier = sorted(p for p in (root/'days').glob('????????') if p.is_dir() and p.name < day.name)
    if not earlier:
        raise ValueError('use first-day report for cash-only initialization')
    prior = earlier[-1]
    prior_meta = verify(prior)
    if (dm['previous_manifest_sha256'] != sha(prior/'manifest.json')
        or dm['plan_sha256'] != sha(plan/'manifest.json')
        or dm['date'] != date or pm['execution_date'] != date):
        raise ValueError('plan/day/ancestry mismatch')
    source = verify(day/'market_source')
    if source['date'] != date:
        raise ValueError('wrong market date')
    if dm.get('corporate_action_manifest_sha256'):
        verify(day/'corporate_action_source')
        if sha(day/'corporate_action_source/manifest.json') != dm['corporate_action_manifest_sha256']:
            raise ValueError('corporate action mismatch')
    before = pd.read_csv(prior/'summary.csv').set_index('strategy')
    after = pd.read_csv(day/'summary.csv').set_index('strategy')
    strategies = config['strategies']
    if (before.index.has_duplicates or after.index.has_duplicates
        or set(before.index) != set(strategies) or set(after.index) != set(strategies)
        or not after.action.eq('REBALANCE').all() or not after.date.eq(date).all()
        or not before.date.eq(prior_meta['date']).all()):
        raise ValueError('invalid strategy coverage, dates or action')
    old_market = pd.read_csv(prior/'market.csv').set_index('symbol')
    market = pd.read_csv(day/'market.csv').set_index('symbol')
    dividends = pd.read_csv(day/'dividends.csv')
    if dividends.duplicated(['strategy','event_id']).any():
        raise ValueError('duplicate dividend history')
    today = dividends[dividends.pay_date.eq(date)]
    rows, blocked, fills_all, positions_all = [], [], [], []
    for strategy in strategies:
        old = pd.read_csv(prior/'paper_account'/strategy/'positions.csv').set_index('symbol')
        new = pd.read_csv(day/'paper_account'/strategy/'positions.csv').set_index('symbol')
        trades = pd.read_csv(day/'paper_trades'/f'{strategy}.csv')
        checks = pd.read_csv(day/'order_checks'/f'{strategy}.csv')
        if checks.duplicated(['symbol','side']).any() or trades.duplicated(['symbol','side']).any():
            raise ValueError('duplicate orders')
        merged = trades.merge(checks[['symbol','side','check_status','check_reason']],
                              on=['symbol','side'], how='outer', validate='one_to_one', indicator=True)
        if not merged._merge.eq('both').all() or not merged.check_status.isin(['PASS','BLOCK']).all():
            raise ValueError('trade and precheck coverage mismatch')
        if (merged.fill_status.eq('FILLED') & merged.check_status.ne('PASS')).any():
            raise ValueError('blocked order filled')
        gross = float(today.loc[today.strategy.eq(strategy), 'cash_gross'].sum())
        metrics = reconcile_account(before.loc[strategy], after.loc[strategy], old, new,
                                    old_market, market, trades, gross, config['initial_cash'])
        no = checks[checks.check_status.eq('BLOCK')].copy()
        no['strategy'] = strategy
        blocked.append(no)
        fills_all.append(merged.drop(columns='_merge').assign(strategy=strategy))
        positions_all.append(new.reset_index().assign(strategy=strategy))
        rows.append({**after.loc[strategy].to_dict(), **metrics, 'strategy': strategy,
                     'label': LABELS.get(strategy, strategy), 'blocked_orders': len(no)})
    comparison = pd.DataFrame(rows)
    blocks = pd.concat(blocked,ignore_index=True)
    fills = pd.concat(fills_all,ignore_index=True)
    positions = pd.concat(positions_all,ignore_index=True)
    facts = {'date': date, 'previous_date': prior_meta['date'], 'signal_date': pm['signal_date'],
             'plan_created_at_utc': pm['created_at_utc'], 'market_retrieved_at_utc': source['retrieved_at_utc'],
             'completed_at_utc': dm['completed_at_utc'], 'initial_total_cash': config['initial_cash']*len(strategies),
             'previous_total_asset': float(before.total_asset.sum()), 'unique_held_symbols': int(positions.symbol.nunique()),
             'required_market_symbols': source['required_symbols'], 'blacklist': source['blacklist'],
             'news_risk': source['news_risk'], 'live_approval': False,
             'plan_manifest_sha256': sha(plan/'manifest.json'), 'day_manifest_sha256': sha(day/'manifest.json'),
             'prior_manifest_sha256': sha(prior/'manifest.json'), 'report_code_sha256': sha(Path(__file__))}
    for name in ['total_asset','cash','market_value','price_pnl','daily_pnl','dividend_cash_gross','commission',
                 'buy_fills','sell_fills','blocked_orders','buy_amount','sell_amount','n_positions']:
        facts['sum_'+name] = float(comparison[name].sum())
    facts['aggregate_daily_return_pct'] = facts['sum_daily_pnl']/facts['previous_total_asset']*100
    facts['aggregate_cumulative_return_pct'] = (facts['sum_total_asset']/facts['initial_total_cash']-1)*100
    out.mkdir(parents=True)
    comparison.to_csv(out/'rebalance_comparison.csv',index=False)
    blocks.to_csv(out/'blocked_orders.csv',index=False)
    fills.to_csv(out/'all_order_outcomes.csv',index=False)
    figures(comparison,date,prior_meta['date'],out)
    facts['report_output_hashes'] = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    verify(day);verify(plan);verify(prior)
    return facts


def figures(df,date,previous,out):
    plt.rcParams.update({'font.family':'Arial Unicode MS','axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    caption=f'来源：{previous}与{date}封存账本、订单与行情｜当日收盘价模拟，非真实券商成交'
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for axis,column,title,color in [(axes[0],'daily_return_pct','相对上一交易日的收益','#356b89'),
                                    (axes[1],'cumulative_return_pct','相对初始100万元的累计收益','#9b825b')]:
        bars=axis.barh(df.label,df[column],color=color)
        axis.invert_yaxis();axis.set_xlabel('账户收益率（%）');axis.set_title(title)
        axis.axvline(0,color='#888888',ls='--');axis.margins(x=.3)
        for bar,value in zip(bars,df[column]):
            axis.text(value,bar.get_y()+bar.get_height()/2,f' {value:+.3f}%',va='center',fontsize=10)
    fig.suptitle(f'{date} 六策略调仓日收益')
    fig.text(.5,.015,caption,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(out/'01_daily_vs_cumulative.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    x=np.arange(len(df))
    axes[0].bar(x-.2,df.buy_fills,.4,label='模拟买入笔数',color='#356b89')
    axes[0].bar(x+.2,df.sell_fills,.4,label='模拟卖出笔数',color='#9b825b')
    axes[0].scatter(x,df.blocked_orders,color='#a25f50',label='被阻断笔数',zorder=3)
    axes[0].set_ylabel('订单数量（笔）');axes[0].set_title('实际成交与被阻断订单');axes[0].legend(fontsize=9)
    bars=axes[1].bar(x,df.cash_ratio_pct,color='#356b89')
    axes[1].set_ylabel('收盘现金占总资产比例（%）');axes[1].set_title('调仓后仍有不同程度的现金');axes[1].margins(y=.18)
    for bar,value in zip(bars,df.cash_ratio_pct):
        axes[1].text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}%',ha='center',va='bottom')
    for axis in axes:
        axis.set_xticks(x,df.label,rotation=20,ha='right')
    fig.text(.5,.015,caption,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(out/'02_orders_and_cash.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--date',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.session,a.date,a.output),ensure_ascii=False,indent=2))
