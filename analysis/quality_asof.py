"""Annual quality inputs only; no new default factors. Extracted from paper10."""
import numpy as np
import pandas as pd
from analysis.financial_asof import _date


def normalize_indicators(raw):
    cols=['ts_code','ann_date','end_date','roe','debt_to_assets']
    if not set(cols).issubset(raw):raise ValueError('missing indicator fields')
    f=raw[cols].copy()
    for c in ['ann_date','end_date']:f[c]=_date(f[c])
    for c in ['roe','debt_to_assets']:f[c]=pd.to_numeric(f[c],errors='coerce')
    valid=f.ts_code.notna()&f.ann_date.notna()&f.end_date.le(f.ann_date)&f.end_date.dt.month.eq(12)&f.end_date.dt.day.eq(31)
    f=f[valid].copy();keys=['ts_code','end_date','ann_date']
    conflict=f.groupby(keys)[['roe','debt_to_assets']].nunique(dropna=False).gt(1).any(axis=1).rename('ambiguous')
    f=f.sort_values(keys,kind='stable').drop_duplicates(keys).merge(conflict,on=keys,validate='one_to_one')
    f.loc[f.ambiguous,['roe','debt_to_assets']]=np.nan
    return f,dict(raw_rows=len(raw),annual_versions=len(f),ambiguous_versions=int(f.ambiguous.sum()))


def annual_versions_asof(income,balance,indicators,date):
    """All periods and latest known versions; no validity-based fallback."""
    date=pd.Timestamp(date);parts=[]
    for f,prefix,col in [(income,'income','available_date'),(balance,'balance','available_date'),(indicators,'indicator','ann_date')]:
        known=f[f[col].lt(date)].sort_values(['ts_code','end_date',col],kind='stable')
        known=known.drop_duplicates(['ts_code','end_date'],keep='last')
        parts.append(known.rename(columns={c:prefix+'_'+c for c in known if c not in ['ts_code','end_date']}))
    a=parts[0].merge(parts[1],on=['ts_code','end_date'],how='inner',validate='one_to_one')
    # Missing latest indicators must invalidate that period, not select an older year.
    a=a.merge(parts[2],on=['ts_code','end_date'],how='left',validate='one_to_one')
    values=['income_revenue','income_oper_cost','balance_total_assets','balance_total_hldr_eqy_exc_min_int','indicator_roe','indicator_debt_to_assets']
    valid=np.isfinite(a[values]).all(axis=1)
    valid &= a.income_comp_type.eq(1)&a.balance_comp_type.eq(1)&~a.income_ambiguous&~a.balance_ambiguous&~a.indicator_ambiguous.fillna(True).astype(bool)
    valid &= a.income_revenue.gt(0)&a.income_oper_cost.ge(0)&a.balance_total_assets.gt(0)&a.balance_total_hldr_eqy_exc_min_int.gt(0)&a.indicator_debt_to_assets.ge(0)
    a['valid']=valid
    a['GP']=(a.income_revenue-a.income_oper_cost)/a.balance_total_assets
    a['GM']=(a.income_revenue-a.income_oper_cost)/a.income_revenue
    a['ROE']=a.indicator_roe/100
    a['LEV']=a.indicator_debt_to_assets/100
    return a




def paired_annual_features(income, balance, indicators, date):
    """Latest common annual period and exact preceding year; invalid stays invalid."""
    date = pd.Timestamp(date)
    if pd.isna(date):
        raise ValueError('invalid asof date')
    annual = annual_versions_asof(income, balance, indicators, date)
    current = annual.sort_values(['ts_code','end_date']).drop_duplicates('ts_code', keep='last').copy()
    current['prior_end_date'] = current.end_date - pd.DateOffset(years=1)
    prior = annual.rename(columns={c:'prior_'+c for c in annual if c != 'ts_code'})
    result = current.merge(prior, on=['ts_code','prior_end_date'], how='left', validate='one_to_one')
    result['pair_valid'] = result.valid & result.prior_valid.eq(True)
    for c in ['GP','GM','ROE']:
        result['delta_'+c] = (result[c]-result['prior_'+c]).where(result.pair_valid)
    return result
