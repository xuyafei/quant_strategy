"""Descriptive return attribution extracted from paper 02; not causal alpha proof."""
import numpy as np
import pandas as pd
from scipy.stats import norm

def ols_hac(y, factors, *, lags=5, min_obs=60):
    """OLS with Bartlett Newey-West covariance, n/(n-k) correction, normal CI."""
    if (not isinstance(y, pd.Series) or not isinstance(factors, pd.DataFrame)
            or not isinstance(y.index, pd.DatetimeIndex) or y.index.hasnans
            or factors.shape[1] < 1 or factors.columns.has_duplicates
            or not all(isinstance(c, str) and c and c != 'intercept' for c in factors)
            or isinstance(lags, bool) or not isinstance(lags, (int, np.integer))
            or isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 60):
        raise ValueError('invalid regression schema, lag or minimum observations')
    if not y.index.equals(factors.index) or y.index.has_duplicates or not y.index.is_monotonic_increasing:
        raise ValueError("regression dates must align exactly")
    x=np.column_stack([np.ones(len(y)), factors.to_numpy(float)])
    yv=y.to_numpy(float)
    n,k=x.shape
    if n<max(min_obs,k+1) or not 0<=lags<n or not np.isfinite(x).all() or not np.isfinite(yv).all():
        raise ValueError("invalid/incomplete regression sample")
    if np.linalg.matrix_rank(x)<k:
        raise ValueError("rank-deficient regression")
    coefficients=np.linalg.lstsq(x,yv,rcond=None)[0]
    resid=yv-x@coefficients
    bread=np.linalg.inv(x.T@x)
    z=x*resid[:,None]
    meat=z.T@z
    for lag in range(1,lags+1):
        cross=z[lag:].T@z[:-lag]
        meat+=(1-lag/(lags+1))*(cross+cross.T)
    covariance=(bread@meat@bread)*(n/(n-k))
    se=np.sqrt(np.maximum(np.diag(covariance),0))
    if (se<=1e-15).any():
        raise ValueError("degenerate uncertainty estimate")
    stat=coefficients/se
    total=float(((yv-yv.mean())**2).sum())
    if total<=0:
        raise ValueError("constant dependent variable")
    r2=1-float(resid@resid)/total
    rows=[]
    for j,term in enumerate(["intercept"]+factors.columns.tolist()):
        rows.append({"term":term,"coefficient":coefficients[j],"hac_se":se[j],"statistic":stat[j],
                     "p_value":2*norm.sf(abs(stat[j])),"ci_low":coefficients[j]-norm.ppf(.975)*se[j],
                     "ci_high":coefficients[j]+norm.ppf(.975)*se[j]})
    result={"observations":n,"r_squared":r2,"adjusted_r_squared":1-(1-r2)*(n-1)/(n-k),
            "residual_daily_mean":float(resid.mean()),"condition_number":float(np.linalg.cond(x)),
            "alpha_arithmetic_annual":coefficients[0]*252,"alpha_ci_low_annual":(coefficients[0]-norm.ppf(.975)*se[0])*252,
            "alpha_ci_high_annual":(coefficients[0]+norm.ppf(.975)*se[0])*252,
            "alpha_p_value":rows[0]["p_value"]}
    contributions={"intercept":coefficients[0]*252}
    contributions.update({col:coefficients[j+1]*float(factors[col].mean())*252 for j,col in enumerate(factors)})
    if not np.isclose(sum(contributions.values()),yv.mean()*252,atol=1e-10):
        raise ValueError("arithmetic attribution failed")
    return result,pd.DataFrame(rows),contributions
