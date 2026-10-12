#!/usr/bin/env python3
"""Render article 132 figures from the completed migration acceptance record."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def build(source, output):
    raw = source.read_bytes()
    result = json.loads(raw)
    if result['status'] != 'PASSED' or result['new_backtest_run'] or not result['diagnostic_only']:
        raise ValueError('expected completed, read-only migration verification')
    if output.exists():
        raise FileExistsError('choose a new figure folder')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False,
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    control = result['null_control']
    fig, ax = plt.subplots(figsize=(10, 6))
    values = [control['unadjusted_false_any']*100, control['holm_false_any']*100]
    bars = ax.bar(['不做多重校正', 'Holm校正'], values, width=.48, color=['#a2adb5', '#356b89'])
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=7, fontsize=16)
    ax.axhline(5, color='#666666', linestyle='--', linewidth=1, label='5%检验水平参考线')
    ax.set_ylim(0, 80)
    ax.set_ylabel('至少误判一个方案有效的重复实验比例（%）')
    ax.set_xlabel('检验处理方式')
    ax.set_title('所有方案都没有真实优势，反复尝试仍会产生“赢家”', pad=18)
    ax.legend(loc='upper right', frameon=False)
    fig.text(.5, .035, f"来源：本次迁移验收的已知无优势模拟｜{control['trials']:,}次、每次{control['candidates']}个独立正态统计量\n"
             f"固定种子{control['seed']}，单侧检验｜不是市场回测，没有交易日期区间", ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(output/'01_false_discovery_control.png', dpi=180)
    plt.close(fig)

    selection = result['cscv']
    shares = [selection['pbo_strict'], selection['median_mass'], 1-selection['pbo_inclusive']]
    counts = [round(v*selection['splits']) for v in shares]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(['严格低于中位数', '恰好处于中位数', '高于中位数'],
                   [s*100 for s in shares], color=['#356b89', '#a2adb5', '#ccd2d6'], height=.55)
    ax.bar_label(bars, labels=[f'{v:.2%}（{n}次）' for v, n in zip(shares, counts)], padding=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('该类结果占全部252个切分的比例（%）')
    ax.set_ylabel('训练冠军在测试部分的排名位置')
    ax.set_title('迁移后重现原结果：历史冠军经常落到测试排名的后半段', pad=18)
    fig.text(.5, .035, f"来源：13个论文实验候选，{selection['start']}—{selection['end']}，{selection['observations']}条日主动收益\n"
             '10块CSCV、252个切分；按信息比率排名｜不是严格前向检验，也不是未来亏损概率',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(output/'02_selection_stability.png', dpi=180)
    plt.close(fig)
    manifest = {'source': str(source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.png'))}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'output/paper_research_merge_20261009_verified/verification.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/articles/assets/article132')
    args = parser.parse_args()
    print(build(args.source, args.output))
