import json
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from backtest.research_replay import replay, replay_ranked
from backtest.backtest_single import _rebalance_to_target_weights, run_single_backtest
from config import get_settings
from strategies.holding_buffer import select_buffer
from analysis.portfolio_diagnostics import cost_summary, portfolio_diagnostics
from analysis.training_time import chronological_split, balanced_period_weights
from analysis.quality_asof import normalize_indicators, paired_annual_features
from analysis.financial_asof import normalize_statements


class Season02Tests(unittest.TestCase):
    def prices(self):
        return pd.DataFrame([[10.,20.,30.],[11.,19.,31.],[12.,18.,32.],[13.,17.,33.],[14.,16.,34.]],
                            index=pd.bdate_range('2025-01-01',periods=5),columns=['a','b','c'])

    def schedule(self,p):
        return pd.DataFrame({'execution_date':p.index[[1,3]]}, index=p.index[[0,2]])

    def test_buffer_uses_actual_eligible_holdings(self):
        score=pd.Series([4.,3.,2.,1.],index=list('abcd'))
        self.assertEqual(select_buffer(score,['c','removed'],2,3),['a','c'])
        self.assertEqual(select_buffer(score,['c','d'],2,2),['a','b'])
        with self.assertRaises(ValueError):select_buffer(score,[],2.5,3)
        with self.assertRaises(ValueError):select_buffer(score,[],5,6)

    def test_reference_execution_parity_and_cash_accounting(self):
        p=self.prices();s=self.schedule(p)
        targets={p.index[0]:pd.Series([.5,.5],index=['a','b']),p.index[2]:pd.Series([.2,.4],index=['b','c'])}
        r=replay(p,s,lambda d,h:targets[d])
        cash=1.;shares=pd.Series(0.,index=p.columns)
        for signal,row in s.iterrows():
            w=targets[signal];day=row.execution_date
            cash,shares=_rebalance_to_target_weights(cash,shares,p.loc[day],list(w.index),list(w),.0003,valuation_px=p.loc[day])
            expected=cash+float((shares*p.loc[day]).sum())
            self.assertAlmostEqual(expected,r['daily'].loc[day,'post_nav'])
        costs=cost_summary(r['trades'])
        self.assertAlmostEqual(costs['fees'],r['executions'].fees.sum())
        self.assertAlmostEqual(costs['cumulative_two_way_turnover'],r['executions'].two_way_turnover.sum())
        self.assertIsNone(costs['order_count'])

    def test_hold_cash_and_partial_cash_differ(self):
        p=self.prices();s=self.schedule(p)
        def run(second):return replay(p,s,lambda d,h:pd.Series(.5,index=['a']) if d==p.index[0] else second)
        hold=run(None);cash=run(pd.Series(dtype=float))
        self.assertEqual(hold['executions'].iloc[-1].action,'HOLD')
        self.assertEqual(cash['executions'].iloc[-1].holdings,0)
        self.assertGreater(hold['daily'].iloc[-1].post_nav,cash['daily'].iloc[-1].post_nav)
        self.assertGreater(hold['executions'].iloc[0].cash_fraction,.49)

    def test_missing_targets_no_renormalization_and_blocked_cash_exit(self):
        p=self.prices();s=self.schedule(p);p.loc[p.index[1],'b']=np.nan;p.loc[p.index[3],'a']=np.nan
        r=replay(p,s,lambda d,h:pd.Series([.5,.5],index=['a','b']) if d==p.index[0] else pd.Series(dtype=float))
        self.assertGreater(r['executions'].iloc[0].cash_fraction,.49)
        self.assertGreater(r['executions'].iloc[-1].blocked_exit_value,0)
        self.assertEqual(r['executions'].iloc[-1].holdings,1)
        self.assertFalse(r['trades'].price.isna().any())

    def test_all_missing_target_is_explicit_hold(self):
        p=self.prices();s=self.schedule(p);p.loc[p.index[3],'c']=np.nan
        r=replay(p,s,lambda d,h:pd.Series(1.,index=['a'] if d==p.index[0] else ['c']))
        self.assertTrue(r['executions'].iloc[-1].all_targets_missing_hold)
        self.assertEqual(r['executions'].iloc[-1].two_way_turnover,0)

    def test_invalid_inputs_fail_closed(self):
        p=self.prices();s=self.schedule(p)
        for w in [pd.Series([.8,.8],index=['a','b']),pd.Series(-1.,index=['a']),pd.Series(1.,index=['unknown'])]:
            with self.assertRaises(ValueError):replay(p,s,lambda d,h:w)
        s.iloc[0,0]=s.index[0]
        with self.assertRaises(ValueError):replay(p,s,lambda d,h:None)

    def test_future_prices_cannot_change_past_targets(self):
        p=self.prices();s=self.schedule(p)
        scores=pd.concat({p.index[0]:pd.Series([3,2,1],index=p.columns),p.index[2]:pd.Series([1,3,2],index=p.columns)})
        a=replay_ranked(p,scores,s,top_k=2,exit_rank=3)
        p.iloc[-1]*=10000
        b=replay_ranked(p,scores,s,top_k=2,exit_rank=3)
        pd.testing.assert_frame_equal(a['targets'],b['targets'])
        pd.testing.assert_frame_equal(a['daily'].iloc[:-1],b['daily'].iloc[:-1])

    def test_portfolio_cash_and_overlap(self):
        w=pd.DataFrame({'date':['2025-01-01']*4,'strategy':['a','a','b','b'],
                        'symbol':['x','y','y','z'],'weight':[.25,.25,.5,.5]})
        r=portfolio_diagnostics(w)
        self.assertEqual(r['snapshots'][0]['cash_weight'],.5)
        self.assertEqual(r['snapshots'][0]['effective_stock_count'],2)
        self.assertEqual(r['overlap'][0]['common_capital_weight'],.25)
        json.dumps(r,allow_nan=False)
        with self.assertRaises(ValueError):portfolio_diagnostics(pd.concat([w,w]))

    def test_time_split_maturity_purge_and_date_balance(self):
        dates=pd.date_range('2020-01-01',periods=12,freq='7D')
        panel=pd.DataFrame({'signal_date':dates,'symbol':'a','label_end':dates+pd.Timedelta(days=8),'y':np.arange(12.)})
        train,val=chronological_split(panel,dates[-1],validation_periods=2,minimum_train_periods=4)
        self.assertTrue(train.label_end.lt(val.signal_date.min()).all())
        self.assertTrue(val.label_end.lt(dates[-1]).all())
        train=pd.concat([train,train.iloc[:2].assign(symbol='b')])
        w=balanced_period_weights(train)
        np.testing.assert_allclose(w.groupby(train.signal_date).sum(),w.groupby(train.signal_date).sum().iloc[0])
        panel.loc[0,'label_end']=panel.loc[0,'signal_date']
        with self.assertRaises(ValueError):chronological_split(panel,dates[-1],validation_periods=2,minimum_train_periods=4)

    def financial(self):
        base=pd.DataFrame({'ts_code':['a']*3,'ann_date':['20220401','20230401','20240401'],
            'f_ann_date':['20220401','20230401','20240401'],'end_date':['20211231','20221231','20231231'],
            'report_type':[1]*3,'comp_type':[1]*3,'revenue':[100.]*3,'oper_cost':[50.]*3,
            'total_assets':[200.]*3,'total_hldr_eqy_exc_min_int':[100.]*3,'roe':[10.]*3,'debt_to_assets':[20.]*3})
        return base

    def test_annual_pair_strict_dates_and_no_invalid_fallback(self):
        b=self.financial();inc,_=normalize_statements(b,'income');bal,_=normalize_statements(b,'balance');ind,_=normalize_indicators(b)
        a=paired_annual_features(inc,bal,ind,'2024-04-01')
        self.assertEqual(a.end_date.iloc[0],pd.Timestamp('2022-12-31'))
        b.loc[2,'roe']=np.nan;ind,_=normalize_indicators(b)
        a=paired_annual_features(inc,bal,ind,'2024-04-02')
        self.assertEqual(a.end_date.iloc[0],pd.Timestamp('2023-12-31'))
        self.assertFalse(a.pair_valid.iloc[0])
        self.assertTrue(a.delta_ROE.isna().all())

    def test_exact_previous_annual_period_required(self):
        b=self.financial().drop(index=1);inc,_=normalize_statements(b,'income');bal,_=normalize_statements(b,'balance');ind,_=normalize_indicators(b)
        a=paired_annual_features(inc,bal,ind,'2024-04-02')
        self.assertFalse(a.pair_valid.iloc[0])

    def test_missing_latest_indicator_does_not_select_old_year(self):
        b=self.financial();inc,_=normalize_statements(b,'income');bal,_=normalize_statements(b,'balance');ind,_=normalize_indicators(b.iloc[:2])
        a=paired_annual_features(inc,bal,ind,'2024-04-02')
        self.assertEqual(a.end_date.iloc[0],pd.Timestamp('2023-12-31'))
        self.assertFalse(a.pair_valid.iloc[0])

    def test_negative_real_position_rejected_but_float_dust_allowed(self):
        p=self.prices();r=replay(p,self.schedule(p),lambda d,h:pd.Series(1.,index=['a']) if d==p.index[0] else pd.Series(dtype=float))
        t=r['trades'].copy();t.loc[t.side.eq('SELL'),'shares_after']=-1e-17
        cost_summary(t)
        t.loc[t.side.eq('SELL'),'shares_after']=-.01
        with self.assertRaises(ValueError):cost_summary(t)

    def test_main_backtest_buffer_is_opt_in_and_retains_risk_path(self):
        p=self.prices()
        settings=replace(get_settings(),rebalance_freq='D',backtest_start='2025-01-01',backtest_end='2025-01-07',
            portfolio_weighting='equal',force_final_rebalance=False,max_position_weight=0.,max_industry_weight=0.,
            max_tech_growth_weight=0.,min_positions=0,target_volatility=0.,max_rebalance_turnover=0.,
            min_avg_volume=0.,min_avg_amount=0.,enable_trade_status_filter=False)
        scores=pd.concat({d:pd.Series([3,2,1] if i==0 else [1,2,3],index=p.columns) for i,d in enumerate(p.index)})
        a,ma=run_single_backtest('test',factor_values=scores,prices=p,settings=settings,top_k=2)
        b,mb=run_single_backtest('test',factor_values=scores,prices=p,settings=settings,top_k=2,rank_buffer_exit=None)
        pd.testing.assert_series_equal(a,b)
        # Legacy metadata contains NaN estimates (NaN != NaN); compare serialization.
        self.assertEqual(json.dumps(ma,sort_keys=True,default=str),json.dumps(mb,sort_keys=True,default=str))
        _,mc=run_single_backtest('test',factor_values=scores,prices=p,settings=settings,top_k=2,rank_buffer_exit=3)
        self.assertEqual(mc['rebalance_log'][1]['selected_picks'],['b','a'])
        self.assertTrue(any(r['signal_rank']==3 and r['reason']=='retained' for r in mc['rank_buffer_log']))
        with self.assertRaises(ValueError):run_single_backtest('test',factor_values=scores,prices=p,settings=settings,top_k=2,rank_buffer_exit=1)


if __name__=='__main__':unittest.main()
