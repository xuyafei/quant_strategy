"""Multiplicity diagnostics for archived research, with no trading side effects."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def adjust_pvalues(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('p values must be a finite nonempty vector in [0,1]')
    count = len(values)
    order = np.argsort(values, kind='stable')
    ordered = values[order]
    ranks = np.arange(1, count + 1)
    harmonic = np.sum(1 / ranks)
    result = {'raw': values.copy(), 'bonferroni': np.minimum(1, count * values)}
    procedures = {
        'holm': np.maximum.accumulate(ordered * (count + 1 - ranks)),
        'bh': np.minimum.accumulate((ordered * count / ranks)[::-1])[::-1],
        'by': np.minimum.accumulate((ordered * count * harmonic / ranks)[::-1])[::-1],
    }
    for method, adjusted in procedures.items():
        restored = np.empty(count)
        restored[order] = np.minimum(1, adjusted)
        result[method] = restored
    return pd.DataFrame(result)


def hac_mean(values, lags=5):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 60 or not np.isfinite(values).all():
        raise ValueError('at least 60 complete observations required')
    count = len(values)
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or not 0 <= lags < count:
        raise ValueError('invalid HAC lag')
    residual = values - values.mean()
    meat = float(residual @ residual)
    for lag in range(1, lags + 1):
        meat += 2 * (1 - lag / (lags + 1)) * float(residual[lag:] @ residual[:-lag])
    variance = meat / count**2 * count / (count - 1)
    if variance <= 1e-30:
        raise ValueError('degenerate uncertainty estimate')
    standard_error = np.sqrt(variance)
    statistic = values.mean() / standard_error
    return {'observations': count, 'mean_daily': values.mean(), 'hac_se': standard_error,
            'statistic': statistic, 'p_positive': norm.sf(statistic),
            'p_two_sided': 2 * norm.sf(abs(statistic)),
            'annual_arithmetic_active': values.mean() * 252,
            'ci_low_annual': (values.mean() - norm.ppf(.975) * standard_error) * 252,
            'ci_high_annual': (values.mean() + norm.ppf(.975) * standard_error) * 252}


def paired_returns(nav):
    if (not isinstance(nav.index, pd.DatetimeIndex) or nav.index.hasnans or nav.shape[1] < 1
            or nav.index.has_duplicates or not nav.index.is_monotonic_increasing or nav.columns.has_duplicates
            or len(nav) < 61 or not np.isfinite(nav.to_numpy()).all() or (nav <= 0).any().any()):
        raise ValueError('NAV must have ordered unique dates, complete positive values')
    return nav.pct_change(fill_method=None).iloc[1:]
