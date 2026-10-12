"""Retrospective CSCV ranking diagnostics, not a live strategy or an alpha test."""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def validate_returns(frame):
    if not isinstance(frame, pd.DataFrame) or len(frame) < 4 or frame.shape[1] < 2:
        raise ValueError('need at least four observations and two candidates')
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError('need valid dates')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('dates must be unique and sorted')
    if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
        raise ValueError('candidate names must be unique strings')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('returns must be finite; do not silently fill missing data')


def information_ratio(values):
    """Annualized arithmetic active mean / sample standard deviation; daily input."""
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('invalid return array')
    std = values.std(axis=0, ddof=1)
    if np.any(std <= 1e-14):
        raise ValueError('zero tracking error: IR is undefined')
    return values.mean(axis=0) / std * np.sqrt(252)


def cscv(frame, blocks=10):
    """All half-block combinations; average test ranks; lexical train tie break.

    Rank 1 is worst. Both strict-below and at-or-below median are retained,
    since an odd number of candidates has positive mass exactly at the median.
    Blocks preserve row order but are not chronological walk-forward splits.
    """
    validate_returns(frame)
    if (isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2
            or blocks > len(frame) or blocks % 2 or len(frame) % blocks):
        raise ValueError('even block count must divide observations exactly')
    if comb(blocks, blocks // 2) > 20000:
        raise ValueError('too many CSCV combinations; explicit computational bound is 20000')
    data = frame.sort_index(axis=1)
    values = data.to_numpy(dtype=float)
    block_rows = np.arange(len(data)).reshape(blocks, -1)
    n = data.shape[1]
    records = []
    for train_blocks in combinations(range(blocks), blocks // 2):
        train_rows = block_rows[list(train_blocks)].ravel()
        mask = np.ones(len(data), dtype=bool)
        mask[train_rows] = False
        train_ir = information_ratio(values[train_rows])
        test_ir = information_ratio(values[mask])
        winner = int(np.argmax(train_ir))
        rank = float(rankdata(test_ir, method='average')[winner])
        relative_rank = rank / (n + 1)
        records.append({'train_blocks': ','.join(map(str, train_blocks)),
                        'winner': data.columns[winner], 'train_ir': train_ir[winner],
                        'test_ir': test_ir[winner], 'test_rank': rank,
                        'relative_rank': relative_rank,
                        'logit': np.log(relative_rank / (1-relative_rank)),
                        'strict_below_median': rank < (n+1)/2,
                        'at_median': rank == (n+1)/2,
                        'at_or_below_median': rank <= (n+1)/2,
                        'negative_test_active': test_ir[winner] < 0})
    return pd.DataFrame(records)


def cscv_summary(result):
    return {'splits': len(result),
            'pbo_inclusive': float(result.at_or_below_median.mean()),
            'pbo_strict': float(result.strict_below_median.mean()),
            'median_mass': float(result.at_median.mean()),
            'negative_test_active_rate': float(result.negative_test_active.mean()),
            'mean_train_ir': float(result.train_ir.mean()),
            'mean_test_ir': float(result.test_ir.mean())}
