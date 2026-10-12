from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd
from scripts.build_forward_rebalance_report import reconcile_account, build


class RebalanceReportTest(unittest.TestCase):
    def inputs(self):
        before=SimpleNamespace(cash=1000.,market_value=1000.,total_asset=2000.)
        after=SimpleNamespace(cash=1008.,market_value=1100.,total_asset=2108.,
                              commission=2.,dividend_cash_gross=10.,net_value=1.054,n_positions=2,filled_orders=2)
        old=pd.DataFrame({'shares':[100.]},index=['A'])
        new=pd.DataFrame({'shares':[50.,50.]},index=['A','B'])
        p0=pd.DataFrame({'close':[10.,1.]},index=['A','B'])
        p1=pd.DataFrame({'close':[11.,11.]},index=['A','B'])
        trades=pd.DataFrame([{'symbol':'A','side':'SELL','qty':50,'price':11.,'gross_amount':550.,'commission':1.,'net_cash_flow':549.,'fill_status':'FILLED'},
                             {'symbol':'B','side':'BUY','qty':50,'price':11.,'gross_amount':550.,'commission':1.,'net_cash_flow':-551.,'fill_status':'FILLED'}])
        return before,after,old,new,p0,p1,trades,10.,2000.

    def test_only_old_shares_earn_price_move(self):
        result=reconcile_account(*self.inputs())
        self.assertEqual(result['price_pnl'],100.)
        self.assertEqual(result['daily_pnl'],108.)
        self.assertEqual(result['buy_fills'],1)
        self.assertEqual(result['sell_fills'],1)
        self.assertAlmostEqual(result['filled_turnover_pct'],1100/2110*100)

    def test_incorrect_close_price_rejected(self):
        args=list(self.inputs());args[6].loc[0,'price']=10.
        with self.assertRaisesRegex(ValueError,'execution'):
            reconcile_account(*args)

    def test_missing_price_rejected(self):
        args=list(self.inputs());args[5].loc['A','close']=float('nan')
        with self.assertRaisesRegex(ValueError,'valuation'):
            reconcile_account(*args)

    def test_wrong_cash_or_positions_rejected(self):
        args=list(self.inputs());args[1].cash+=1
        with self.assertRaisesRegex(ValueError,'cash'):
            reconcile_account(*args)
        args=list(self.inputs());args[3].loc['B','shares']=49
        with self.assertRaisesRegex(ValueError,'position'):
            reconcile_account(*args)

    def test_skipped_trade_does_not_change_holdings(self):
        args=list(self.inputs())
        skipped=args[6].iloc[[0]].assign(symbol='C',qty=100,fill_status='SKIPPED',net_cash_flow=0.,commission=0.)
        args[6]=pd.concat([args[6],skipped],ignore_index=True)
        self.assertEqual(reconcile_account(*args)['daily_pnl'],108.)

    def test_protect_source_directory(self):
        with self.assertRaisesRegex(ValueError,'source'):
            build(Path('/tmp/report_fixture'), '2026-10-08',Path('/tmp/report_fixture/days/20261008'))


if __name__=='__main__':
    unittest.main()
