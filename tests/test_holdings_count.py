import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from analysis.holdings_count import eligible_base, replay_batch, metrics, nested_random_plans
from backtest.backtest_single import _rebalance_to_target_weights
from backtest.backtest_multi import run_multi_backtest
from config import get_settings
from universe.dynamic import build_dynamic_universe_report
from scripts.build_dynamic_universe_comparison import build_weekly_raw_factor_panel


class HoldingsCountTests(unittest.TestCase):
    def test_batch_matches_existing_matcher_with_missing_quotes(self):
        rng = np.random.default_rng(17)
        px = 10*np.exp(np.cumsum(rng.normal(0,.02,(40,8)),axis=0))
        px[8:20,0] = np.nan
        px[23,4:7] = np.nan
        px[24:,2] = np.nan
        ex = np.array([1,8,15,23,30])
        plans = np.array([[[0,1,2],[5,6,7]],[[3,4,5],[0,1,2]],
            [[0,2,4],[1,3,5]],[[4,5,6],[4,5,6]],[[1,3,4],[1,4,6]]])
        nav, audit, stranded = replay_batch(px,ex,plans)
        marks = pd.DataFrame(px).ffill()
        for r in range(2):
            cash,shares = 1.,pd.Series(0.,index=range(8))
            expected=[]
            for day in range(len(px)):
                if day in ex:
                    w = list(ex).index(day)
                    picks=[int(i) for i in plans[w,r] if np.isfinite(px[day,i])]
                    cash,shares = _rebalance_to_target_weights(cash,shares,pd.Series(px[day]),
                        picks,[1/3]*len(picks),.0003,valuation_px=marks.loc[day])
                expected.append(cash+(shares*marks.loc[day]).sum())
            np.testing.assert_allclose(nav[:,r],expected,atol=1e-12)
            self.assertAlmostEqual(stranded[r],(shares*marks.iloc[-1]*pd.isna(px[-1])).sum())
        self.assertTrue(all((a['cash_fraction'] >= -1e-10).all() for a in audit))

    def test_full_engine_parity_with_complete_quotes(self):
        dates=pd.bdate_range('2024-01-01',periods=20)
        rng=np.random.default_rng(2)
        px=pd.DataFrame(10*np.exp(np.cumsum(rng.normal(0,.02,(20,5)),axis=0)),index=dates,columns=list('ABCDE'))
        ex=np.array([1,6,11,16])
        plans=np.array([[[0,1]],[[2,3]],[[0,4]],[[3,4]]])
        nav,log,_=replay_batch(px,ex,plans)
        records=[]
        for w,day in enumerate(ex):
            for rank,i in enumerate(plans[w,0]):
                records.append((dates[day],px.columns[i],2-rank))
        score=pd.DataFrame(records,columns=['date','symbol','score']).set_index(['date','symbol']).score
        settings=replace(get_settings(),backtest_start=str(dates[0].date()),backtest_end=str(dates[-1].date()),
            rebalance_freq='D',force_final_rebalance=False,top_k=2,portfolio_weighting='equal',
            commission_rate=.0003,max_position_weight=0.,max_rebalance_turnover=0.,min_avg_volume=0.,
            min_avg_amount=0.,enable_trade_status_filter=False,max_industry_weight=0.,
            max_tech_growth_weight=0.,target_volatility=0.,min_positions=0)
        actual,_=run_multi_backtest(fused=score,prices=px,settings=settings,top_k=2,empty_signal_policy='hold')
        before=nav[:,0].copy()
        for a in log:
            before[a['execution_index']]=a['pre'][0]
        np.testing.assert_allclose(actual.to_numpy(),before,atol=1e-12)

    def test_random_nested_and_reproducible(self):
        pools=[np.arange(100),np.arange(20,120)]
        a=nested_random_plans(pools,[10,20,50],10,17)
        b=nested_random_plans(pools,[10,20,50],10,17)
        np.testing.assert_array_equal(a[10],a[50][:,:,:10])
        np.testing.assert_array_equal(a[20],b[20])
        np.testing.assert_array_equal(a[50][:,0],np.stack([p[:50] for p in pools]))
        self.assertTrue(all(len(np.unique(r))==50 for w in a[50] for r in w))

    def test_base_matches_production(self):
        dates=pd.bdate_range('2020-01-01',periods=300)
        symbols=['000001.SZ','000002.SZ','600001.SH']
        close=pd.DataFrame(10.,index=dates,columns=symbols)
        amount=pd.DataFrame(60e6,index=dates,columns=symbols)
        close.loc[dates[-2],symbols[0]]=np.nan
        amount.loc[:,symbols[1]]=40e6
        master=pd.DataFrame({'ts_code':symbols,'list_date':['19910101']*3,'delist_date':[None,None,dates[-1].strftime('%Y%m%d')]})
        names=pd.DataFrame({'ts_code':['600001.SH'],'name':['ST Test'],
            'start_date':[dates[-4].strftime('%Y%m%d')],'end_date':[dates[-3].strftime('%Y%m%d')],
            'ann_date':[dates[-4].strftime('%Y%m%d')]})
        decisions=dates[-5:]
        fast=eligible_base(close,amount,master,names,decisions)
        long=close.stack().rename('close').reset_index()
        long.columns=['trade_date','ts_code','close'];long['amount_yuan']=amount.to_numpy().ravel()
        slow=build_dynamic_universe_report(long,master,names,dates,decisions)
        expected=slow.loc[slow.eligible,['date','symbol','listed_sessions','valid_days_20','adv20_yuan']].reset_index(drop=True)
        pd.testing.assert_frame_equal(fast[expected.columns],expected,check_dtype=False)

    def test_future_prices_do_not_change_past_nav(self):
        px=np.ones((20,3))*10
        plans=np.array([[[0,1]],[[1,2]],[[0,2]]])
        before,_,_=replay_batch(px,[1,6,11],plans)
        px[12:]*=2
        after,_,_=replay_batch(px,[1,6,11],plans)
        np.testing.assert_array_equal(before[:12],after[:12])

    def test_first_fee_and_terminal_fee_included(self):
        nav,audit,_=replay_batch(np.ones((10,4))*10,[1,9],np.array([[[0,1]],[[2,3]]]),fee=.001)
        self.assertLess(nav[1,0],1)
        self.assertLess(nav[-1,0],nav[-2,0])
        self.assertAlmostEqual(metrics(nav)['total_return'][0],nav[-1,0]-1)

    def test_rejects_duplicate_plan(self):
        with self.assertRaises(ValueError):
            replay_batch(np.ones((5,2)),[1],np.array([[[0,0]]]))

    def test_year_slice_preserves_factor_values(self):
        dates=pd.bdate_range('2023-01-01',periods=320)
        rng=np.random.default_rng(9)
        values=10*np.exp(np.cumsum(rng.normal(0,.01,(320,3)),axis=0))
        prices=pd.DataFrame({'trade_date':np.repeat(dates,3),
            'ts_code':list('ABC')*len(dates),'close':values.ravel(),'adj_close':values.ravel()})
        finance=pd.DataFrame({'ts_code':list('ABC')*2,'ann_date':[20220101]*3+[20260101]*3,
            'end_date':[20211231]*3+[20251231]*3,'eps':[1]*3+[100]*3,
            'ocfps':[2]*3+[200]*3,'ocf_to_profit':[.5]*3+[50]*3})
        decisions=dates[250::5]
        whole=build_weekly_raw_factor_panel(prices,finance,decisions)
        cut=build_weekly_raw_factor_panel(prices[prices.trade_date.ge(dates[160])],finance,decisions)
        pd.testing.assert_frame_equal(whole,cut,atol=1e-12,rtol=1e-12)
        past_only=build_weekly_raw_factor_panel(prices,finance[finance.ann_date.lt(20260101)],decisions)
        pd.testing.assert_frame_equal(whole,past_only)


if __name__ == '__main__':
    unittest.main()
