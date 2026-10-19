"""Outcome-maturity checks, independent of any ML estimator or admission policy."""
import numpy as np
import pandas as pd


def chronological_split(panel, cutoff, *, validation_periods, minimum_train_periods):
    """Strict boundaries; callers must declare windows, no short-sample default.

    Rows require signal_date, symbol, label_end, y. Missing outcomes are ineligible;
    malformed dates and labels ending before/equal to the signal are errors.
    """
    for n in [validation_periods, minimum_train_periods]:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError('positive integer period counts required')
    p = panel.copy()
    for c in ['signal_date', 'label_end']:
        p[c] = pd.to_datetime(p[c], errors='raise')
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or p.signal_date.isna().any() or p.symbol.isna().any():
        raise ValueError('missing time or symbol')
    if p.duplicated(['signal_date', 'symbol']).any():
        raise ValueError('duplicate feature keys')
    if (p.label_end.notna() & p.label_end.le(p.signal_date)).any():
        raise ValueError('label must end after feature date')
    eligible = p.signal_date.lt(cutoff) & p.label_end.lt(cutoff) & np.isfinite(p.y)
    dates = sorted(p.loc[eligible, 'signal_date'].unique())
    if len(dates) < validation_periods + minimum_train_periods:
        raise ValueError('insufficient mature periods')
    train_dates, validation_dates = dates[:-validation_periods], dates[-validation_periods:]
    train = p.loc[eligible & p.signal_date.isin(train_dates) & p.label_end.lt(validation_dates[0])].copy()
    validation = p.loc[eligible & p.signal_date.isin(validation_dates)].copy()
    if train.signal_date.nunique() < minimum_train_periods:
        raise ValueError('insufficient training after boundary purge')
    return train, validation


def balanced_period_weights(train):
    if train.empty or train.signal_date.isna().any() or train.symbol.isna().any() or train.duplicated(['signal_date', 'symbol']).any():
        raise ValueError('invalid training keys')
    weights = 1 / train.groupby('signal_date').symbol.transform('size')
    return weights / weights.mean()


def validate_training_folds(folds, *, minimum_train_periods):
    """Audit archived boundary metadata, NOT raw feature or fitted-model validation."""
    f=folds.copy()
    for c in ['fit_date','train_first','train_last_label','validation_first','validation_last_label']:
        f[c]=pd.to_datetime(f[c],errors='raise')
        if f[c].isna().any():raise ValueError('missing fold boundary')
    if f.empty or f.fit_date.duplicated().any():raise ValueError('empty/duplicate fits')
    valid=(f.train_first.lt(f.train_last_label)&f.train_last_label.lt(f.validation_first)
           &f.validation_first.lt(f.validation_last_label)&f.validation_last_label.lt(f.fit_date)
           &f.train_weeks.ge(minimum_train_periods))
    if not valid.all():raise ValueError('invalid training/validation boundary')
    return dict(status='ASSESSED',folds=len(f),minimum_train_periods=minimum_train_periods,
                scope='archived boundary metadata only; no model refit or vintage certification')
