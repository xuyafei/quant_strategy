import unittest
import numpy as np
import pandas as pd
from analysis.pure_ic_comparison import pure_ic_weights, pure_score, replay_actions
from analysis.holdings_count import replay_batch


class PureICTests(unittest.TestCase):
    def test_global_weights_and_zero_floor(self):
        d=pd.bdate_range('2020-01-01',periods=20)
        x=pd.DataFrame({'a':.1,'b':.3,'c':-.2,'risk':0.},index=d)
        w,_=pure_ic_weights(x,pd.Series(d,index=d),d[-1],lookback=10,minimum=5)
        np.testing.assert_allclose(w,[.25,.75,0,0])

    def test_insufficient_factor_does_not_revert_other_factors(self):
        d=pd.bdate_range('2020-01-01',periods=20)
        x=pd.DataFrame({'a':.1,'b':np.nan},index=d)
        w,_=pure_ic_weights(x,pd.Series(d,index=d),d[-1],lookback=10,minimum=5)
        np.testing.assert_allclose(w,[1,0])

    def test_no_positive_no_equal_fallback(self):
        d=pd.bdate_range('2020-01-01',periods=20)
        x=pd.DataFrame({'a':-.1,'b':-.2},index=d)
        w,_=pure_ic_weights(x,pd.Series(d,index=d),d[-1],lookback=10,minimum=5)
        self.assertEqual(w.sum(),0)
        self.assertTrue(pure_score(x,w).empty)

    def test_maturity_and_future_mutation(self):
        d=pd.bdate_range('2020-01-01',periods=30)
        x=pd.DataFrame({'a':.1,'b':.2},index=d);end=pd.Series(d,index=d).shift(-6)
        w,log=pure_ic_weights(x,end,d[20],lookback=10,minimum=5)
        x.loc[d[14]:,'b']=999
        after,_=pure_ic_weights(x,end,d[20],lookback=10,minimum=5)
        pd.testing.assert_series_equal(w,after)
        self.assertLess(log[0]['latest_outcome_end'],d[20])

    def test_score_no_family_restandardization(self):
        x=pd.DataFrame({'a':[1.,2.,np.nan],'b':[4.,1.,3.]})
        s=pure_score(x,pd.Series({'a':.25,'b':.75}))
        np.testing.assert_allclose(s,[3.25,1.25,3.])
        self.assertEqual(len(pure_score(x,pd.Series({'a':.25,'b':.75}),pd.Series([2,2,1]),2)),2)

    def test_cash_is_not_hold(self):
        px=np.array([[10.],[10.],[20.],[40.]])
        cash,log,_=replay_actions(px,[0,1],[[0],[]],fee=.01)
        hold,_,_=replay_actions(px,[0,1],[[0],None],fee=.01)
        np.testing.assert_allclose(cash[1:],.99/1.01)
        self.assertGreater(hold[-1],cash[-1]*3)
        self.assertEqual(log[1]['holdings'],0)

    def test_unquoted_cash_exit_retry_and_reentry(self):
        px=np.array([[10.,10.],[np.nan,10.],[20.,10.],[20.,10.]])
        nav,log,_=replay_actions(px,[0,1,2,3],[[0,1],[],[],[1]],fee=0)
        np.testing.assert_allclose(nav,[1,1,1.5,1.5])
        self.assertEqual(log[1]['holdings'],1)
        self.assertEqual(log[2]['holdings'],0)
        self.assertEqual(log[3]['holdings'],1)

    def test_nonempty_matches_existing_engine(self):
        rng=np.random.default_rng(9)
        px=10*np.exp(np.cumsum(rng.normal(0,.02,(30,6)),axis=0))
        px[10:15,0]=np.nan;px[16,2:4]=np.nan
        picks=[[0,1],[2,3],[0,4],[2,5]];ex=[1,10,16,24]
        actual,_,_=replay_actions(px,ex,picks)
        expected,_,_=replay_batch(px,ex,np.asarray(picks)[:,None,:])
        np.testing.assert_allclose(actual,expected[:,0],atol=1e-12)

    def test_invalid_indices_rejected(self):
        with self.assertRaises(ValueError):replay_actions(np.ones((3,2)),[1],[[-1]])
        with self.assertRaises(ValueError):replay_actions(np.ones((3,2)),[1,1],[[0],[]])


if __name__=='__main__':unittest.main()
