#!/usr/bin/env python3
"""Read a declared NAV family and write a new, read-only research audit folder."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from analysis.research_audit import build_research_audit, render_markdown


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(spec_path, output):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a new report directory; existing results are never overwritten')
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    inputs = {str(spec_path): hashlib.sha256(spec_bytes).hexdigest()}

    def load(entry, *, indexed=True):
        path = (spec_path.parent / entry['path']).resolve()
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        if checksum != entry['sha256']:
            raise ValueError(f'input hash mismatch: {path}')
        inputs[str(path)] = checksum
        # pandas otherwise renames duplicate column headers silently.
        header = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
        if len(header) != len(set(header)) or 'date' not in header:
            raise ValueError('unique CSV headers and a date column required')
        return pd.read_csv(io.BytesIO(data), index_col='date' if indexed else None, parse_dates=['date'])

    nav = load(spec['nav'])
    factors = load(spec['factors']) if 'factors' in spec else None
    if spec.get('frequency') != 'daily' or spec.get('return_units') != 'decimal':
        raise ValueError('explicit daily frequency and decimal return units required')
    if not spec.get('family_description') or not spec.get('cost_convention'):
        raise ValueError('declare candidate family and cost convention')
    result = build_research_audit(nav, spec['pairs'], factors=factors, **spec.get('options', {}))
    if 'diagnostics' in spec:
        from analysis.portfolio_diagnostics import cost_summary, portfolio_diagnostics, return_correlations
        extra = spec['diagnostics']
        if not isinstance(extra, dict) or set(extra)-{'trades', 'allocations', 'allocation_kind', 'return_correlations'}:
            raise ValueError('unknown diagnostics input')
        result['diagnostics'] = {}
        for key in ['trades', 'allocations']:
            if key not in extra:
                continue
            frame = load(extra[key], indexed=False)
            if frame.date.isna().any() or not frame.date.isin(nav.index).all():
                raise ValueError('diagnostic dates must belong to declared NAV interval')
            if key == 'trades':
                if 'strategy' not in frame or frame.strategy.isna().any() or not frame.strategy.isin(spec['pairs']).all():
                    raise ValueError('trades need declared strategy names')
                result['diagnostics']['costs'] = {s:cost_summary(g) for s,g in frame.groupby('strategy')}
            else:
                if extra.get('allocation_kind') not in {'target', 'actual'}:
                    raise ValueError('declare target or actual allocation_kind')
                if 'strategy' not in frame or not frame.strategy.isin(spec['pairs']).all():
                    raise ValueError('allocation strategies must be declared')
                result['diagnostics']['allocations'] = portfolio_diagnostics(frame)
                result['diagnostics']['allocation_kind'] = extra['allocation_kind']
        if extra.get('return_correlations'):
            result['diagnostics']['return_correlations'] = return_correlations(nav, spec['pairs'])
    result['input_declarations'] = {k: spec[k] for k in
                                   ['frequency', 'return_units', 'family_description', 'cost_convention']}
    result['input_hashes'] = inputs
    names = ['analysis/research_audit.py', 'analysis/research_statistics.py',
             'analysis/return_attribution.py', 'analysis/selection_diagnostics.py',
             'scripts/run_research_audit.py']
    if 'diagnostics' in spec:
        names.append('analysis/portfolio_diagnostics.py')
    result['code_hashes'] = {name: digest(ROOT / name) for name in names}
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    markdown = render_markdown(result)
    for path, checksum in inputs.items():
        if digest(Path(path)) != checksum:
            raise ValueError('input changed during report generation')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(encoded+'\n', encoding='utf-8')
    (output / 'README.md').write_text(markdown, encoding='utf-8')
    (output / 'input_spec.json').write_bytes(spec_bytes)
    manifest = {name: digest(output / name) for name in ['report.json', 'README.md', 'input_spec.json']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.spec, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'family_size': report['family_size'],
                      'observations': report['observations'], 'diagnostic_only': True}, ensure_ascii=False))
