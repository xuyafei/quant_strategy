#!/usr/bin/env python3
"""Read-only reconciliation of a rebalance day followed by a holding day."""
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
from scripts.build_forward_rebalance_report import reconcile_account


def period_returns(start, middle, end):
    values = np.asarray([start, middle, end], float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('positive finite assets required')
    return dict(first_return_pct=(middle/start-1)*100,
                second_return_pct=(end/middle-1)*100,
                period_return_pct=(end/start-1)*100,
                first_pnl=middle-start, second_pnl=end-middle, period_pnl=end-start)


def build(root, first, second, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a new report directory')
    if output == root or (output.is_relative_to(root) and not output.is_relative_to(root/'reports')):
        raise ValueError('reports cannot overwrite source ledger directories')
    latest_day(root)
    config = session_config(root)
    days = sorted(p for p in (root/'days').glob('????????') if p.is_dir())
    dates = [p.name for p in days]
    index = dates.index(first.replace('-', ''))
    if index == 0 or index+1 >= len(days) or days[index+1].name != second.replace('-', ''):
        raise ValueError('need consecutive rebalance and holding days, with a prior close')
    paths = days[index-1:index+2]
    manifests = [verify(p) for p in paths]
    for i in [1, 2]:
        if manifests[i]['previous_manifest_sha256'] != sha(paths[i-1]/'manifest.json'):
            raise ValueError('broken ancestry')
    summaries = [pd.read_csv(p/'summary.csv').set_index('strategy') for p in paths]
    for summary, manifest in zip(summaries, manifests):
        if (summary.index.has_duplicates or set(summary.index) != set(config['strategies'])
                or not summary.date.eq(manifest['date']).all()):
            raise ValueError('invalid strategy coverage/date')
    if not summaries[1].action.eq('REBALANCE').all() or not summaries[2].action.eq('HOLD_MARK').all():
        raise ValueError('expected rebalance then holding day')
    markets = [pd.read_csv(p/'market.csv').set_index('symbol') for p in paths]
    sources = {str(p/'manifest.json'): sha(p/'manifest.json') for p in paths}
    rows, contribution_rows = [], []
    for strategy in config['strategies']:
        positions = [pd.read_csv(p/'paper_account'/strategy/'positions.csv').set_index('symbol') for p in paths]
        metrics = []
        for i in [1, 2]:
            trade_path = paths[i]/'paper_trades'/f'{strategy}.csv'
            trades = pd.read_csv(trade_path) if trade_path.exists() else pd.DataFrame(columns=[
                'symbol', 'fill_status', 'side', 'qty', 'price', 'gross_amount', 'commission', 'net_cash_flow'])
            dividends = pd.read_csv(paths[i]/'dividends.csv')
            paid = dividends[dividends.strategy.eq(strategy) & dividends.pay_date.eq(manifests[i]['date'])]
            metrics.append(reconcile_account(summaries[i-1].loc[strategy], summaries[i].loc[strategy],
                           positions[i-1], positions[i], markets[i-1], markets[i], trades,
                           float(paid.cash_gross.sum()), config['initial_cash']))
        before, middle, after = [s.loc[strategy] for s in summaries]
        rates = period_returns(before.total_asset, middle.total_asset, after.total_asset)
        rows.append(dict(strategy=strategy, label=LABELS[strategy], **rates,
                         total_asset=after.total_asset, cash=after.cash, positions=int(after.n_positions),
                         cash_ratio_pct=after.cash/after.total_asset*100,
                         cumulative_return_pct=(after.total_asset/config['initial_cash']-1)*100,
                         first_buy_fills=metrics[0]['buy_fills'], first_sell_fills=metrics[0]['sell_fills'],
                         first_commission=middle.commission, second_fills=int(after.filled_orders),
                         second_commission=after.commission, first_dividend=middle.dividend_cash_gross,
                         second_dividend=after.dividend_cash_gross))
        held = positions[1]
        changes = held.shares*(markets[2].close.reindex(held.index)-markets[1].close.reindex(held.index))
        for symbol, amount in changes.items():
            contribution_rows.append(dict(strategy=strategy, symbol=symbol, price_pnl=float(amount)))
    frame = pd.DataFrame(rows)
    totals = [float(s.total_asset.sum()) for s in summaries]
    aggregate = period_returns(*totals)
    initial = len(config['strategies'])*config['initial_cash']
    facts = dict(first=first, second=second, baseline=manifests[0]['date'], **aggregate,
                 baseline_total=totals[0], first_total=totals[1], second_total=totals[2],
                 initial_total=initial, cumulative_return_pct=(totals[2]/initial-1)*100,
                 cumulative_pnl=totals[2]-initial, cash=float(frame.cash.sum()),
                 position_records=int(frame.positions.sum()),
                 unique_symbols=len(set(r['symbol'] for r in contribution_rows)),
                 first_buy_fills=int(frame.first_buy_fills.sum()), first_sell_fills=int(frame.first_sell_fills.sum()),
                 second_fills=int(frame.second_fills.sum()), first_commission=float(frame.first_commission.sum()),
                 second_commission=float(frame.second_commission.sum()),
                 first_dividend=float(frame.first_dividend.sum()), second_dividend=float(frame.second_dividend.sum()),
                 source_manifest_hashes=sources, code_sha256=sha(Path(__file__)), simulation_only=True)
    output.mkdir(parents=True)
    frame.to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(contribution_rows).to_csv(output/'holding_price_contributions.csv', index=False)
    charts(frame, first, second, manifests[0]['date'], output)
    for path, checksum in sources.items():
        if sha(Path(path)) != checksum:
            raise ValueError('source changed during reporting')
    for path in paths:
        verify(path)
    facts['output_hashes'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output/'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return facts


def charts(frame, first, second, baseline, output):
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(frame))
    for offset, column, label, color in [(-.19, 'first_return_pct', first+' 调仓日', '#9baab4'),
                                       (.19, 'second_return_pct', second+' 持有日', '#356b89')]:
        bars = ax.bar(x+offset, frame[column], .36, label=label, color=color)
        ax.bar_label(bars, fmt='%+.3f', padding=4, fontsize=9)
    ax.set_xticks(x, frame.label)
    ax.set_xlabel('六个独立虚拟账户')
    ax.set_ylabel('相对上一交易日总资产的收益（%）')
    ax.set_title('节后两天：调仓日五跌一涨，随后持有日六账户上涨')
    ax.axhline(0, color='#777777', lw=.8)
    ax.margins(y=.22); ax.legend()
    fig.text(.5,.025,f'来源：{baseline}、{first}、{second}封存账本｜含简化佣金及税前分红｜不是实盘业绩',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(output/'01_two_daily_returns.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,6.5))
    for ax, column, title, color in [(axes[0],'period_return_pct',f'节后两日：相对{baseline}收盘','#356b89'),
                                   (axes[1],'cumulative_return_pct','建仓以来：相对初始100万元','#9c805a')]:
        bars=ax.barh(frame.label,frame[column],color=color)
        ax.bar_label(bars,fmt='%+.3f',padding=4,fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel('账户收益（%）'); ax.set_title(title)
        ax.set_ylabel('独立虚拟账户'); ax.axvline(0,color='#777777',lw=.8); ax.margins(x=.28)
    fig.suptitle('今天上涨、节后两日盈利、建仓以来盈利，是三个问题')
    fig.text(.5,.025,f'来源：截至{second}封存账本｜两日收益按期末资产/期初资产计算，不直接相加日收益｜非实盘',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94)); fig.savefig(output/'02_period_vs_cumulative.png',dpi=180); plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--first',required=True); p.add_argument('--second',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(build(a.session,a.first,a.second,a.output),ensure_ascii=False,indent=2))
