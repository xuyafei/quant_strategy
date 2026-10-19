import unittest
import numpy as np
import pandas as pd
from analysis.characteristic_benchmarks import bins_for,memberships,matching_plan,arithmetic_decomposition


class CharacteristicTests(unittest.TestCase):
    @staticmethod
    def frame(n=270):
        rng=np.random.default_rng(23)
        return pd.DataFrame({'total_mv_yuan':np.arange(n)+100.,'VALUE_BP':rng.uniform(.1,2,n),
            'MOM_12_1':rng.normal(size=n)},index=pd.Index([f's{i:04d}' for i in range(n)],name='symbol'))

    def test_bins_balanced(self):
        b=bins_for(self.frame().total_mv_yuan,3)
        self.assertEqual(b.value_counts().tolist(),[90,90,90])

    def test_ties_deterministic(self):
        x=pd.Series(1.,index=['b','c','a'])
        self.assertEqual(bins_for(x,3).to_dict(),{'b':2,'c':3,'a':1})

    def test_permutation_invariant(self):
        f=self.frame();a=memberships(f,3);b=memberships(f.sample(frac=1,random_state=5),3)
        pd.testing.assert_frame_equal(a,b)

    def test_bad_size(self):
        f=self.frame();f.iloc[0,0]=0
        with self.assertRaises(ValueError):memberships(f,3)

    def test_bad_bp(self):
        f=self.frame();f.iloc[0,1]=np.nan
        with self.assertRaises(ValueError):memberships(f,3)

    def test_duplicate_symbol(self):
        f=self.frame();f.index=['a']*len(f)
        with self.assertRaises(ValueError):memberships(f,3)

    def test_future_outcome_column_ignored(self):
        f=self.frame();a=memberships(f,3);w=pd.Series(1.,index=[f.index[0]])
        p,x=matching_plan(a,w,minimum_peers=5)
        f['future_return']=np.arange(len(f))*1000
        q,y=matching_plan(memberships(f,3),w,minimum_peers=5)
        pd.testing.assert_series_equal(p,q);pd.testing.assert_frame_equal(x,y)

    def test_self_exclusion(self):
        f=memberships(self.frame(),3);s=f.index[0]
        p,a=matching_plan(f,pd.Series(1.,index=[s]),minimum_peers=5)
        self.assertNotIn(s,p.index);self.assertEqual(a.level.iloc[0],'SBM')
        self.assertTrue(np.allclose(p,1/len(p)))

    def test_missing_momentum_fallback_not_deletion(self):
        f=self.frame();s=f.index[0];f.loc[s,'MOM_12_1']=np.nan
        p,a=matching_plan(memberships(f,3),pd.Series(1.,index=[s]),minimum_peers=5)
        self.assertEqual(a.level.iloc[0],'SB');self.assertTrue(a.missing_target_momentum.iloc[0])
        self.assertAlmostEqual(p.sum(),1);self.assertNotIn(s,p.index)

    def test_small_cell_fallback(self):
        f=memberships(self.frame(),3)
        _,a=matching_plan(f,pd.Series(1.,index=[f.index[0]]),minimum_peers=20)
        self.assertEqual(a.level.iloc[0],'SB')

    def test_size_fallback(self):
        f=memberships(self.frame(),3)
        _,a=matching_plan(f,pd.Series(1.,index=[f.index[0]]),minimum_peers=40)
        self.assertEqual(a.level.iloc[0],'S')

    def test_global_fallback(self):
        f=memberships(self.frame(),3)
        _,a=matching_plan(f,pd.Series(1.,index=[f.index[0]]),minimum_peers=100)
        self.assertEqual(a.level.iloc[0],'ALL')

    def test_fail_insufficient_global(self):
        f=memberships(self.frame(21),3)
        with self.assertRaises(ValueError):matching_plan(f,pd.Series(1.,index=[f.index[0]]),minimum_peers=21)

    def test_threshold_counts_after_self_exclusion(self):
        f=memberships(self.frame(21),3)
        p,a=matching_plan(f,pd.Series(1.,index=[f.index[0]]),minimum_peers=20)
        self.assertEqual(len(p),20);self.assertEqual(a.peers.iloc[0],20)

    def test_other_target_can_remain_peer(self):
        f=memberships(self.frame(),3)
        g=next(g for _,g in f.groupby(['size_bin','bp_bin','momentum_bin']) if len(g)>5)
        w=pd.Series(.5,index=g.index[:2]);p,_=matching_plan(f,w,minimum_peers=5)
        self.assertTrue(set(w.index).issubset(p.index))

    def test_linearity(self):
        f=memberships(self.frame(),3);names=f.index[:2]
        a,_=matching_plan(f,pd.Series(1.,index=names[:1]),minimum_peers=5)
        b,_=matching_plan(f,pd.Series(1.,index=names[1:]),minimum_peers=5)
        c,_=matching_plan(f,pd.Series([.2,.8],index=names),minimum_peers=5)
        expected=(a*.2).add(b*.8,fill_value=0)
        pd.testing.assert_series_equal(c,expected.sort_index())

    def test_invalid_budget(self):
        f=memberships(self.frame(),3)
        with self.assertRaises(ValueError):matching_plan(f,pd.Series(.2,index=f.index[:2]))

    def test_full_pool_negative_control(self):
        f=memberships(self.frame(),3)
        p,_=matching_plan(f,pd.Series(1/len(f),index=f.index),minimum_peers=5)
        np.testing.assert_allclose(p,1/len(f),atol=1e-14)

    def test_rank_gap_matches_explicit_peers(self):
        f=memberships(self.frame(),3);s=f.index[0]
        p,a=matching_plan(f,pd.Series(1.,index=[s]),minimum_peers=5)
        for c in ['total_mv_yuan','VALUE_BP','MOM_12_1']:
            self.assertAlmostEqual(a[c+'_peer_rank'].iloc[0],float((p*f.loc[p.index,c+'_rank']).sum()))

    def test_arithmetic_identity(self):
        i=pd.date_range('2025-01-01',periods=3)
        d=arithmetic_decomposition(pd.Series([1,1.1,.99],index=i),pd.Series([1,1.04,1.01],index=i),pd.Series([1,1.02,1.03],index=i))
        np.testing.assert_allclose(d.total_active,d.matched_residual+d.matched_style)

    def test_unaligned_nav_rejected(self):
        with self.assertRaises(ValueError):arithmetic_decomposition(pd.Series([1,2]),pd.Series([1,2],index=[1,2]),pd.Series([1,2]))


if __name__=='__main__':unittest.main()
