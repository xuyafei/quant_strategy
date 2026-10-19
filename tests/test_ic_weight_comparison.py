import unittest
import numpy as np
import pandas as pd

from analysis.ic_weight_comparison import execution_labels, daily_rank_ic, rolling_weights, combine_scores, replay_universe
from analysis.holdings_count import replay_batch
from factors.preprocess import preprocess_factor_panel
from strategies.style import build_fixed_family_score, build_focused_style_score


class ICWeightComparisonTests(unittest.TestCase):
    def test_label_execution_lag_and_missing(self):
        p = pd.DataFrame({'a': np.arange(10)+10.}, index=pd.bdate_range('2020-01-01', periods=10))
        label, end = execution_labels(p)
        self.assertAlmostEqual(label.iloc[0,0], 16/11-1)
        self.assertEqual(end.iloc[0], p.index[6])
        p.iloc[1,0] = np.nan
        self.assertTrue(np.isnan(execution_labels(p)[0].iloc[0,0]))

    def test_ic_pairwise_rank_and_constant(self):
        idx = pd.MultiIndex.from_product([pd.to_datetime(['2020-01-01']), list('abcd')], names=['date','symbol'])
        x = pd.DataFrame({'a':[4,3,2,np.nan], 'b':[1]*4}, index=idx)
        y = pd.Series([1,2,3,100],index=idx)
        ic,n = daily_rank_ic(x,y,3)
        self.assertAlmostEqual(ic.a.iloc[0], -1)
        self.assertEqual(n.a.iloc[0],3)
        self.assertTrue(np.isnan(ic.b.iloc[0]))

    def test_matured_only_and_future_mutation(self):
        d = pd.bdate_range('2020-01-01',periods=30)
        ic = pd.DataFrame({'a': .1,'b':-.1},index=d)
        end = pd.Series(d,index=d).shift(-6)
        w,log = rolling_weights(ic,end,d[20],{'f':['a','b']},lookback=10,minimum=5)
        self.assertAlmostEqual(w.a,.75); self.assertAlmostEqual(w.b,.25)
        self.assertLess(log[0]['latest_outcome_end'],d[20])
        ic.loc[d[14]:,'b'] = 10
        after,_ = rolling_weights(ic,end,d[20],{'f':['a','b']},lookback=10,minimum=5)
        pd.testing.assert_series_equal(w,after)

    def test_fallbacks_and_single_factor(self):
        d = pd.bdate_range('2020-01-01',periods=10)
        x = pd.DataFrame({'a':-.1,'b':-.2,'c':np.nan},index=d)
        for minimum, cols, reason in [(2,['a','b'],'no_positive_mean_ic'),(20,['a','b'],'insufficient_history'),(2,['c'],'single_component')]:
            w,log = rolling_weights(x,pd.Series(d,index=d),d[-1],{'f':cols},lookback=25,minimum=minimum)
            self.assertTrue(np.allclose(w,1/len(cols))); self.assertEqual(log[0]['reason'],reason)

    def test_equal_weights_match_production(self):
        rng=np.random.default_rng(4)
        idx=pd.MultiIndex.from_product([pd.bdate_range('2020-01-01',periods=4),list('abcdefghij')],names=['date','symbol'])
        raw=pd.DataFrame(rng.normal(size=(40,4)),index=idx,columns=list('ABCD'))
        raw.loc[idx[3],'B']=np.nan
        mask=pd.Series(True,index=idx);mask.iloc[7]=False
        ind=pd.Series(['x']*5+['y']*5,index=idx[:10]).reindex(idx).fillna('')
        families={'one':['A','B'],'two':['C','D']}
        z=preprocess_factor_panel(raw.where(mask,axis=0),industry=ind,by_industry=True,min_industry_count=3).where(mask,axis=0)
        w=pd.Series(.5,index=list('ABCD'))
        actual=combine_scores(z,families,w).where(mask).dropna()
        expected,_=build_fixed_family_score(raw,mask,ind,families)
        pd.testing.assert_series_equal(actual,expected,atol=1e-12,rtol=1e-12)
        actual=combine_scores(z,{'f':['A','B']},w,focused=True,raw_count=raw[['A','B']].notna().sum(axis=1),minimum_components=2).where(mask).dropna()
        expected,_=build_focused_style_score(raw,mask,ind,['A','B'],minimum_components=2)
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_variable_benchmark_matches_fixed_engine(self):
        rng=np.random.default_rng(7)
        px=10*np.exp(np.cumsum(rng.normal(0,.02,(30,6)),axis=0))
        px[10:15,0]=np.nan;px[16,2:4]=np.nan
        picks=[np.array(x) for x in [[0,1],[2,3],[0,4],[2,5]]]
        ex=[1,10,16,24]
        nav,_,_=replay_batch(px,ex,np.array(picks)[:,None,:])
        np.testing.assert_allclose(replay_universe(px,ex,picks),nav[:,0],atol=1e-12)

    def test_window_drops_old_evidence_and_weights_have_floor(self):
        d=pd.bdate_range('2020-01-01',periods=30)
        ic=pd.DataFrame({'a':.1,'b':-.2,'c':-.1},index=d)
        ic.loc[d[:10],'b']=100
        w,log=rolling_weights(ic,pd.Series(d,index=d),d[-1],{'f':['a','b','c']},lookback=10,minimum=5)
        np.testing.assert_allclose(w,[2/3,1/6,1/6])
        self.assertEqual(log[0]['window_start'],d[-11])

    def test_one_insufficient_component_reverts_whole_family(self):
        d=pd.bdate_range('2020-01-01',periods=30)
        ic=pd.DataFrame({'a':.1,'b':np.nan,'risk':.2},index=d)
        w,log=rolling_weights(ic,pd.Series(d,index=d),d[-1],{'one':['a','b'],'risk':['risk']},lookback=20,minimum=10)
        np.testing.assert_allclose(w,[.5,.5,1.])
        self.assertEqual(log[0]['reason'],'insufficient_history')
        self.assertAlmostEqual(sum(v['nominal_weight'] for v in log),1)


if __name__ == '__main__':
    unittest.main()
