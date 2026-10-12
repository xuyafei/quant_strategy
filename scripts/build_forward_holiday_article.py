#!/usr/bin/env python3
"""Read sealed September 30 ledgers/plans and create article-only figures."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from live.forward_ledger import verify


def build(session, refresh=False):
    session = session.resolve()
    day = session / 'days/20260930'
    prior = session / 'days/20260929'
    old_plan = session / 'plans/20260928'
    new_plan = session / 'plans/20261008'
    for folder in (day, prior, old_plan, new_plan):
        verify(folder)
    out = session / 'reports/20260930_article'
    if out.exists() and not refresh:
        raise FileExistsError(f'Report already exists: {out}')
    report = session / 'reports/20260930_close'
    data = pd.read_csv(report / 'holding_day_comparison.csv')
    current = pd.read_csv(day / 'summary.csv').set_index('strategy')
    previous = pd.read_csv(prior / 'summary.csv').set_index('strategy')
    indexed = data.set_index('strategy')
    assert set(indexed.index) == set(current.index) == set(previous.index)
    np.testing.assert_allclose(indexed.total_asset, current.total_asset.reindex(indexed.index), atol=1e-6, rtol=0)
    np.testing.assert_allclose(indexed.daily_pnl, (current.total_asset - previous.total_asset).reindex(indexed.index), atol=1e-6, rtol=0)
    np.testing.assert_allclose(data.daily_pnl, data.price_pnl + data.dividend_cash_gross, atol=1e-6, rtol=0)
    assert current.filled_orders.eq(0).all()
    old = pd.read_csv(old_plan / 'targets_combined.csv')
    new = pd.read_csv(new_plan / 'targets_combined.csv')
    changes = []
    for row in data.itertuples():
        a = set(old.loc[old.strategy.eq(row.strategy), 'symbol'])
        b = set(new.loc[new.strategy.eq(row.strategy), 'symbol'])
        assert len(a) == len(b) == 50
        changes.append({'strategy': row.strategy, 'label': row.label,
                        'retained': len(a & b), 'replaced': len(b - a)})
    changes = pd.DataFrame(changes)
    out.mkdir(parents=True, exist_ok=refresh)
    changes.to_csv(out / 'target_list_changes.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False})
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for offset, col, label, color in [(-.23, 'price_pnl', '持仓价格损益', '#35677e'),
                                     (0, 'dividend_cash_gross', '税前现金分红', '#aa8958'),
                                     (.23, 'daily_pnl', '账户当日总损益', '#59615d')]:
        bars = ax.barh(y + offset, data[col], height=.21, label=label, color=color)
        for bar, value in zip(bars, data[col]):
            ax.annotate(f'{value:+,.2f}', (value, bar.get_y()+bar.get_height()/2),
                        xytext=(4 if value >= 0 else -4, 0), textcoords='offset points',
                        ha='left' if value >= 0 else 'right', va='center', fontsize=9)
    ax.set_yticks(y, data.label)
    ax.invert_yaxis()
    ax.set_xlim(-3000, 9000)
    ax.set_xlabel('当日损益或现金收入（元）')
    ax.set_ylabel('独立虚拟账户')
    ax.axvline(0, color='#888888', lw=.8)
    ax.set_title('9月30日收益拆分 价格损益加税前分红等于账户总损益', pad=18)
    ax.legend(loc='upper right', frameon=False)
    fig.text(.5, .025, '来源：2026年9月30日封存账本及收益核对表｜当日无成交、无佣金｜税前纸面结果，非实盘', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(out / 'pnl_components.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    ax.barh(y, changes.retained, label='保留旧目标股票', color='#35677e', height=.6)
    ax.barh(y, changes.replaced, left=changes.retained, label='新进入目标名单', color='#aa8958', height=.6)
    for i, row in enumerate(changes.itertuples()):
        ax.text(row.retained/2, i, str(row.retained), ha='center', va='center', color='white')
        ax.text(row.retained+row.replaced/2, i, str(row.replaced), ha='center', va='center', color='white')
    ax.set_yticks(y, changes.label)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.set_xlabel('目标股票数量（只）')
    ax.set_ylabel('策略目标名单')
    ax.set_title('10月8日冻结目标与9月28日旧目标的名单对比', pad=18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    fig.text(.5, .025, '来源：两期封存目标名单｜新计划信号日为2026年9月30日｜名单替换数不等于实际买卖或成交笔数', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(out / 'target_list_changes.png', dpi=170)
    plt.close(fig)
    print(changes[['label', 'retained', 'replaced']].to_string(index=False))
    print(f'Article-only figures saved: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true', help='Regenerate only the three article report files')
    args = parser.parse_args()
    build(args.session, refresh=args.refresh)
