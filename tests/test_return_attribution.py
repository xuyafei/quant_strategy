import unittest
import numpy as np
import pandas as pd

from analysis.return_attribution import ols_hac


class PaperStyleTest(unittest.TestCase):
    def test_hac_matches_direct_bartlett_matrix_and_mean_reconciles(self):
        rng=np.random.default_rng(3)
        dates=pd.bdate_range("2020-01-01",periods=100)
        factors=pd.DataFrame(rng.normal(0,.02,(100,3)),index=dates,columns=["M","S","V"])
        y=pd.Series(.001+factors.to_numpy()@np.array([1.2,.4,-.2])+rng.normal(0,.002,100),index=dates)
        stats,coefs,components=ols_hac(y,factors)
        x=np.column_stack([np.ones(100),factors])
        beta=np.linalg.lstsq(x,y,rcond=None)[0]
        residual=y.to_numpy()-x@beta
        z=x*residual[:,None]
        weight=np.maximum(0,1-np.abs(np.arange(100)[:,None]-np.arange(100)[None,:])/6)
        bread=np.linalg.inv(x.T@x)
        expected=bread@(z.T@weight@z)@bread*(100/96)
        np.testing.assert_allclose(coefs.hac_se,np.sqrt(np.diag(expected)),rtol=1e-10)
        self.assertAlmostEqual(sum(components.values()),y.mean()*252)
        self.assertGreater(stats["r_squared"],.95)
        np.testing.assert_allclose(coefs.coefficient.iloc[1:],[1.2,.4,-.2],atol=.05)

    def test_bad_regression_inputs_do_not_drop_rows_silently(self):
        dates=pd.bdate_range("2020-01-01",periods=100)
        y=pd.Series(np.linspace(0,.1,100),index=dates)
        f=pd.DataFrame({"A":y,"B":y},index=dates)
        with self.assertRaisesRegex(ValueError,"rank"):
            ols_hac(y,f)
        f=f[["A"]].copy();f.iloc[1]=np.nan
        with self.assertRaisesRegex(ValueError,"incomplete"):
            ols_hac(y,f)
        with self.assertRaisesRegex(ValueError,"align"):
            ols_hac(y.iloc[1:],f)


if __name__=="__main__":
    unittest.main()
