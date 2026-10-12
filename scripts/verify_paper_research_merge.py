#!/usr/bin/env python3
"""Migration acceptance: compare the new read-only entry point with archived lessons.

This adapter alone knows the historical lesson paths; production audit APIs do not.
No fetching, refitting trading signals, modifying source NAVs, or ledger writes.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import norm
from scripts.run_research_audit import digest, run


def verify_merge(paper_root, output):
    paper_root, output = Path(paper_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a new verification folder')
    sources = {}

    def archived(folder, name):
        folder = paper_root/'output'/folder
        manifest_path = folder/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        path = folder/name
        checksum = digest(path)
        if manifest['output_hashes'].get(name) != checksum:
            raise ValueError(f'archived output hash mismatch: {path}')
        sources[str(path)] = checksum
        sources[str(manifest_path)] = digest(manifest_path)
        return path

    p2 = 'paper02_style_attribution_20260930_verified'
    nav2 = archived(p2, 'archived_nav_evaluation.csv')
    factors2 = archived(p2, 'local_factor_returns.csv')
    own = pd.read_csv(archived(p2, 'own_pool_comparison.csv'))
    regression = pd.read_csv(archived(p2, 'regression_summary.csv'))
    factors = pd.read_csv(factors2, index_col='date', parse_dates=True)
    output.mkdir(parents=True)

    def declaration(path):
        return dict(path=str(path), sha256=digest(path))

    def audit(name, spec):
        path = output/f'{name}_spec.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return run(path, output/name)

    common = dict(frequency='daily', return_units='decimal',
                  cost_convention='archived research net NAV; simplified one-way commission 0.03%; not live fills')
    six = audit('six_strategy_attribution', dict(
        **common, family_description='six original strategy profiles; not the later 13 lesson candidates',
        nav=declaration(nav2), factors=declaration(factors2),
        pairs=dict(zip(own.strategy, own.benchmark)),
        options=dict(attribution_window=[str(factors.index[0].date()), str(factors.index[-1].date())])))
    old = regression[(regression.period == 'FULL') & (regression.rf_annual == 0)
                     & (regression.model == 'THREE_LOCAL_PROXIES')].set_index('strategy')
    max_attribution_error = 0.
    for row in six['attribution']['results']:
        for key in ['r_squared', 'alpha_arithmetic_annual', 'alpha_ci_low_annual',
                    'alpha_ci_high_annual', 'alpha_p_value']:
            difference = abs(row[key]-old.loc[row['strategy'], key])
            max_attribution_error = max(max_attribution_error, difference)
            np.testing.assert_allclose(row[key], old.loc[row['strategy'], key], rtol=0, atol=1e-10)
        np.testing.assert_allclose(row['alpha_p_holm'], old.loc[row['strategy'], 'alpha_p_holm_six_primary'],
                                   rtol=0, atol=1e-10)
    for row in six['candidates']:
        reference = own.set_index('strategy').loc[row['strategy']]
        np.testing.assert_allclose(row['total_return'], reference.total_return, rtol=0, atol=1e-10)
        np.testing.assert_allclose(row['cumulative_gap_pp'], reference.return_gap_pp, rtol=0, atol=1e-10)

    p6 = 'paper06_multiple_testing_20261006'
    protocol = json.loads(archived(p6, 'protocol.json').read_text())
    original_tests = pd.read_csv(archived(p6, 'tests.csv'))
    original_tests = original_tests[(original_tests.cost == 'net') & (original_tests.lags == 5)].copy()
    original_tests['candidate'] = original_tests.lecture+'_'+original_tests.arm
    original_tests = original_tests.set_index('candidate')
    navs, pairs = [], {}
    original_navs = {}
    for lesson, suffix in [('paper03', 'momentum'), ('paper04', 'profitability'), ('paper05', 'constraints')]:
        path = archived(f'{lesson}_{suffix}_20260930', 'nav_net.csv')
        nav = pd.read_csv(path, index_col='date', parse_dates=True)
        original_navs[lesson] = nav
        if navs and not nav.index.equals(navs[0].index):
            raise ValueError('source NAV calendars differ')
        chosen = nav[protocol['family'][lesson]+['BENCHMARK']].add_prefix(lesson+'_')
        pairs.update({lesson+'_'+arm: lesson+'_BENCHMARK' for arm in protocol['family'][lesson]})
        navs.append(chosen)
    np.testing.assert_allclose(original_navs['paper04'].VALUE_GP_BLEND,
                               original_navs['paper05'].BASE_TOP50, atol=1e-12, rtol=0)
    combined = pd.concat(navs, axis=1)
    combined_path = output/'thirteen_nav.csv'; combined.to_csv(combined_path, index_label='date')
    p7 = 'paper07_overfitting_20261006_final'
    retained = pd.read_csv(archived(p7, 'retained_active_returns.csv'), index_col='date', parse_dates=True)
    old_cscv = pd.read_csv(archived(p7, 'cscv_summary.csv'))
    thirteen = audit('thirteen_candidate_audit', dict(
        **common, family_description='13 deduplicated lessons 03-05 candidates, declared in lesson 06 protocol',
        nav=declaration(combined_path), pairs=pairs,
        options=dict(cscv_window=[str(retained.index[0].date()), str(retained.index[-1].date())], blocks=10)))
    daily = combined.pct_change(fill_method=None).iloc[1:]
    active = pd.DataFrame({s: daily[s]-daily[b] for s, b in pairs.items()})
    pd.testing.assert_frame_equal(active.loc[retained.index, retained.columns], retained, atol=1e-12, rtol=0)
    max_mean_error = 0.
    for row in thirteen['candidates']:
        reference = original_tests.loc[row['strategy']]
        for key in ['annual_arithmetic_active', 'hac_se', 'p_positive', 'ci_low_annual',
                    'ci_high_annual', 'cumulative_gap_pp']:
            difference = abs(row[key]-reference[key])
            max_mean_error = max(max_mean_error, difference)
            np.testing.assert_allclose(row[key], reference[key], rtol=0, atol=1e-10)
        for method in ['raw', 'bonferroni', 'holm', 'bh', 'by']:
            np.testing.assert_allclose(row['adjusted_positive_p'][method], reference[method], atol=1e-10, rtol=0)
    reference = old_cscv[(old_cscv.family == 'all_13') & (old_cscv.blocks == 10)].iloc[0]
    for key in ['splits', 'pbo_inclusive', 'pbo_strict', 'median_mass',
                'negative_test_active_rate', 'mean_train_ir', 'mean_test_ir']:
        np.testing.assert_allclose(thirteen['selection'][key], reference[key], rtol=0, atol=1e-10)

    # Controlled null: all 20 candidates have exactly zero expected advantage.
    pvalues = norm.sf(np.random.default_rng(42).normal(size=(10000, 20)))
    raw_false_any = float((pvalues <= .05).any(axis=1).mean())
    # First rejection under Holm is identical to Bonferroni in the all-null setting.
    holm_false_any = float((pvalues.min(axis=1) <= .05/20).mean())
    assert abs(raw_false_any-(1-.95**20)) < .02
    assert holm_false_any < .065

    for path, checksum in sources.items():
        if digest(Path(path)) != checksum:
            raise RuntimeError('archive mutated during verification')
    result = dict(status='PASSED', new_backtest_run=False, source_hashes=sources,
                  attribution_max_abs_error=max_attribution_error, active_statistics_max_abs_error=max_mean_error,
                  six_adjusted_alpha_discoveries=int(sum(r['alpha_p_holm'] <= .05 for r in six['attribution']['results'])),
                  thirteen_positive_mean_count=int(sum(r['annual_arithmetic_active'] > 0 for r in thirteen['candidates'])),
                  thirteen_holm_discoveries=int(sum(r['adjusted_positive_p']['holm'] <= .05 for r in thirteen['candidates'])),
                  cscv=thirteen['selection'], null_control=dict(seed=42, trials=10000, candidates=20,
                  unadjusted_false_any=raw_false_any, holm_false_any=holm_false_any),
                  diagnostic_only=True, live_approval=False)
    (output/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_merge(args.paper_root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, ensure_ascii=False, indent=2))
