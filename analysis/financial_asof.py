"""Annual profitability research with explicit disclosure-time joins.

Provider snapshots cannot establish immutable vintage history. This module prevents
known later disclosures from leaking backwards but does not claim to repair that gap.
"""
import numpy as np
import pandas as pd


def _date(series):
    if pd.api.types.is_datetime64_any_dtype(series): return pd.to_datetime(series)
    text=series.astype('string').str.replace(r'\.0$','',regex=True)
    parsed = pd.to_datetime(text,format='mixed',errors='coerce')
    if (text.notna() & text.str.strip().ne('') & parsed.isna()).any():
        raise ValueError('malformed statement date; cannot infer availability')
    return parsed


def normalize_statements(raw, kind):
    if kind not in ['income','balance']: raise ValueError('unknown statement kind')
    values=['revenue','oper_cost'] if kind=='income' else ['total_assets','total_hldr_eqy_exc_min_int']
    required=['ts_code','ann_date','f_ann_date','end_date','report_type','comp_type']+values
    if not set(required).issubset(raw): raise ValueError('missing statement fields')
    f=raw[required].copy()
    for col in ['ann_date','f_ann_date','end_date']: f[col]=_date(f[col])
    for col in values+['report_type','comp_type']: f[col]=pd.to_numeric(f[col],errors='coerce')
    f['available_date']=f[['ann_date','f_ann_date']].max(axis=1)
    valid=f.report_type.eq(1)&f.end_date.dt.month.eq(12)&f.end_date.dt.day.eq(31)
    valid &= f.ts_code.notna() & f.available_date.notna() & f.end_date.le(f.available_date)
    f=f[valid].copy()
    keys=['ts_code','end_date','available_date']
    conflict=f.groupby(keys)[values+['comp_type']].nunique(dropna=False).gt(1).any(axis=1).rename('ambiguous')
    f=f.sort_values(keys,kind='stable').drop_duplicates(keys).merge(conflict,on=keys,validate='one_to_one')
    f.loc[f.ambiguous,values]=np.nan
    return f,{'input_rows':len(raw),'kept_annual_versions':len(f),'filtered_rows_before_dedup':int((~valid).sum()),
              'ambiguous_versions':int(f.ambiguous.sum()),'kind':kind}


def annual_asof(income,balance,date):
    """Both statements must be public BEFORE signal day; same annual end required."""
    date=pd.Timestamp(date)
    if pd.isna(date):
        raise ValueError('invalid signal date')
    parts=[]
    for f,kind in [(income,'income'),(balance,'balance')]:
        known=f[f.available_date.lt(date)].sort_values(['ts_code','end_date','available_date'],kind='stable')
        known=known.drop_duplicates(['ts_code','end_date'],keep='last')
        parts.append(known.rename(columns={col:f'{kind}_{col}' for col in known if col not in ['ts_code','end_date']}))
    joined=parts[0].merge(parts[1],on=['ts_code','end_date'],how='inner',validate='one_to_one')
    joined=joined.sort_values(['ts_code','end_date']).drop_duplicates('ts_code',keep='last').set_index('ts_code')
    joined['report_age_days']=(date-joined.end_date).dt.days
    joined.index.name='symbol'
    return joined
