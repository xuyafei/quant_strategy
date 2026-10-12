#!/usr/bin/env python3
"""Export sealed paper holdings and trades as daily Markdown, without trading."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from live.forward_ledger import LABELS, latest_day, locked, sha, verify


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def close(actual, expected):
    if not np.isfinite([actual, expected]).all() or abs(actual-expected) > 1e-6:
        raise ValueError('daily record accounting mismatch')


def names_asof(root, compact):
    names = {}
    for plan in sorted((root/'plans').glob('????????')):
        if plan.name > compact:
            continue
        verify(plan)
        targets = pd.read_csv(plan/'targets_combined.csv')
        if 'name' in targets:
            names.update({r.symbol: str(r.name) for r in targets.itertuples()
                          if pd.notna(r.name)})
    return names


def render_day(root, day, prior, config):
    meta = verify(day)
    if meta['previous_manifest_sha256'] != sha(prior/'manifest.json'):
        raise ValueError('broken daily ancestry')
    date = meta['date']
    summary = pd.read_csv(day/'summary.csv').set_index('strategy')
    if summary.index.has_duplicates or set(summary.index) != set(config['strategies']):
        raise ValueError('strategy coverage mismatch')
    if not summary.date.eq(date).all():
        raise ValueError('wrong summary date')
    prices = pd.read_csv(day/'market.csv').set_index('symbol')
    if prices.index.has_duplicates or not prices.date.eq(date).all():
        raise ValueError('invalid daily market')
    names = names_asof(root, day.name)
    label = lambda s: names.get(s, '名称未记录')
    initial = float(config['initial_cash'])
    trades_all, positions_all, overview, sections = [], [], [], []
    dividend_file = day/'dividends.csv'
    dividends = pd.read_csv(dividend_file) if dividend_file.exists() else pd.DataFrame()
    if not dividends.empty:
        if dividends.duplicated(['strategy','event_id']).any():
            raise ValueError('duplicate dividends')
        dividends = dividends[dividends.pay_date.eq(date)]
    for strategy in config['strategies']:
        row = summary.loc[strategy]
        current = pd.read_csv(day/'paper_account'/strategy/'positions.csv').set_index('symbol')
        old = pd.read_csv(prior/'paper_account'/strategy/'positions.csv').set_index('symbol')
        if current.index.has_duplicates or old.index.has_duplicates or (current.shares <= 0).any():
            raise ValueError('invalid positions')
        quotes = prices.reindex(current.index)
        if not np.isfinite(quotes.close).all():
            raise ValueError('missing holding valuation')
        values = current.shares*quotes.close
        close(values.sum(), row.market_value)
        close(row.cash+row.market_value, row.total_asset)
        close(row.net_value, row.total_asset/initial)
        close(len(current), row.n_positions)
        old_cash = float(pd.read_csv(prior/'paper_account'/strategy/'account.csv').cash.iloc[0])
        new_cash = float(pd.read_csv(day/'paper_account'/strategy/'account.csv').cash.iloc[0])
        close(new_cash, row.cash)
        trade_file = day/'paper_trades'/f'{strategy}.csv'
        check_file = day/'order_checks'/f'{strategy}.csv'
        if row.action == 'REBALANCE':
            trades = pd.read_csv(trade_file)
            checks = pd.read_csv(check_file)
            if trades.duplicated(['symbol','side']).any() or checks.duplicated(['symbol','side']).any():
                raise ValueError('duplicate orders')
            combined = trades.merge(checks[['symbol','side','check_status','check_reason']],
                                    on=['symbol','side'],how='outer',validate='one_to_one',indicator=True)
            if not combined._merge.eq('both').all():
                raise ValueError('order/check mismatch')
            if not combined.fill_status.isin(['FILLED','SKIPPED']).all():
                raise ValueError('unknown fill status')
            if (combined.fill_status.eq('FILLED') & combined.check_status.ne('PASS')).any():
                raise ValueError('blocked trade was filled')
        elif row.action == 'HOLD_MARK':
            if trade_file.exists() or check_file.exists():
                raise ValueError('unexpected trading files on holding day')
            combined = pd.DataFrame(columns=['symbol','side','qty','price','gross_amount','commission',
                                              'net_cash_flow','fill_status','check_status','check_reason'])
        else:
            raise ValueError('unknown account action')
        filled = combined[combined.fill_status.eq('FILLED')]
        skipped = combined[combined.fill_status.ne('FILLED')]
        expected = old.shares.to_dict()
        for t in filled.itertuples():
            if t.side not in ['BUY','SELL'] or t.qty <= 0:
                raise ValueError('invalid fill')
            expected[t.symbol] = expected.get(t.symbol,0) + (t.qty if t.side=='BUY' else -t.qty)
            close(t.gross_amount, t.qty*t.price)
            close(t.net_cash_flow, (t.gross_amount if t.side=='SELL' else -t.gross_amount)-t.commission)
        if {s:q for s,q in expected.items() if q != 0} != current.shares.to_dict():
            raise ValueError('position changes differ from fills')
        payments = dividends[dividends.strategy.eq(strategy)] if not dividends.empty else dividends
        gross = float(payments.cash_gross.sum()) if not payments.empty else 0.
        close(row.get('dividend_cash_gross',0.),gross)
        close(row.commission,filled.commission.sum())
        close(row.filled_orders,len(filled))
        close(skipped.net_cash_flow.sum(),0.)
        close(row.cash,old_cash+gross+filled.net_cash_flow.sum())
        buys, sells = int(filled.side.eq('BUY').sum()), int(filled.side.eq('SELL').sum())
        overview.append([LABELS.get(strategy,strategy),f'{row.total_asset:,.2f}',f'{row.cash:,.2f}',
                         f'{row.market_value:,.2f}',str(len(current)),f'{(row.net_value-1)*100:+.3f}%',
                         str(buys),str(sells),str(len(skipped))])
        sections.extend([f'## {LABELS.get(strategy,strategy)}', '', f'策略标识：`{strategy}`。', '',
                         f'当日佣金：{row.commission:,.6f}元；税前分红：{gross:,.2f}元。', '',
                         '### 收盘实际持仓', ''])
        holding_rows=[]
        for symbol in values.sort_values(ascending=False).index:
            holding_rows.append([symbol,label(symbol),f'{current.loc[symbol,"shares"]:,.0f}',
                                 f'{current.loc[symbol,"shares"]-old.shares.get(symbol,0):+,.0f}',
                                 f'{quotes.loc[symbol,"close"]:.2f}',f'{values[symbol]:,.2f}',
                                 f'{values[symbol]/row.total_asset*100:.2f}%',
                                 str(quotes.loc[symbol,'price_date'])])
        sections += [table(['代码','名称','持股数','较上一日增减','估值价（元）','市值（元）','占总资产','价格日期'],holding_rows)
                     if holding_rows else '收盘无股票持仓。', '', '### 当日模拟成交', '']
        trade_rows=[[t.symbol,label(t.symbol),'买入' if t.side=='BUY' else '卖出',f'{t.qty:,.0f}',
                     f'{t.price:.2f}',f'{t.gross_amount:,.2f}',f'{t.commission:.6f}',f'{t.net_cash_flow:+,.6f}']
                    for t in filled.itertuples()]
        sections += [table(['代码','名称','方向','股数','模拟价（元）','成交额（元）','佣金（元）','净现金流（元）'],trade_rows)
                     if trade_rows else '当日无成交。持仓估值不等于买卖，未生成虚构交易。', '', '### 未成交／阻断订单', '']
        no_rows=[[t.symbol,label(t.symbol),'买入' if t.side=='BUY' else '卖出',f'{t.qty:,.0f}',
                  str(t.check_status),str(t.check_reason)] for t in skipped.itertuples()]
        sections += [table(['代码','名称','方向','计划股数','预检查','原因'],no_rows) if no_rows else '无。', '']
        if not payments.empty:
            sections += ['### 当日分红入账','',table(['代码','名称','登记日','权益股数','税前入账（元）'],
                [[t.symbol,label(t.symbol),t.record_date,f'{t.eligible_shares:,.0f}',f'{t.cash_gross:,.2f}']
                 for t in payments.itertuples()]),'']
        trades_all.append(filled)
        positions_all.extend(current.index)
    fills = pd.concat(trades_all,ignore_index=True)
    plan_hash = meta.get('plan_sha256') or '无：当日只估值'
    text = [f'# {date} 六策略持仓与交易记录','',
            '本文件从已封存的纸面账本导出，不是股票推荐、目标名单或真实券商成交单。','',
            '持仓为当日收盘实际虚拟持仓；已卖光股票仅在成交表中。股数增减相对上一已结算日，首日相对初始空仓。',
            '名称取截至当日已有冻结计划，不查询未来名单；名称缺失时保留代码。未记录成本价，因此不编造持仓浮盈亏。',
            '成交价格沿用收盘模拟模型；费用为简化佣金，分红为税前口径。未成交订单不当作成交，亦不会由本文自动追单。','',
            '## 账户总览','', table(['账户','总资产（元）','现金（元）','股票市值（元）','持仓数','累计收益','买入笔数','卖出笔数','未成交'],overview),'',
            f'合计总资产：{summary.total_asset.sum():,.2f}元；持仓记录{len(positions_all)}条，去重{len(set(positions_all))}只；模拟成交{len(fills)}笔。',
            '账户合计不是额外验证过的资金配置组合；金额展示舍入，不改变底层账本。',''] + sections + [
            '## 来源与完整性','', f'- 原始账本：`{day.relative_to(ROOT) if day.is_relative_to(ROOT) else day}`。',
            f'- 实际处理时间（UTC）：`{meta.get("completed_at_utc","未记录")}`。',
            f'- 当日账本摘要：`{sha(day/"manifest.json")}`。',
            f'- 上一账户摘要：`{meta["previous_manifest_sha256"]}`。',
            f'- 当日执行计划摘要：`{plan_hash}`。','', '[返回每日目录](README.md)','']
    return '\n'.join(text), {'date':date,'assets':float(summary.total_asset.sum()),
                            'holdings':len(positions_all),'fills':len(fills)}


def index_text(root, records):
    lines=['# 每日持仓与交易记录','',f'会话：`{root.name}`。只包含已经提交的真实日期纸面账本，不生成休市日或未结算日。','']
    if records:
        last=records[-1]['date']
        lines += [f'最新记录：[查看 {last}]({last}.md)','',table(['日期','六账户总资产（元）','持仓记录数','模拟成交笔数'],
                    [[f'[{r["date"]}]({r["date"]}.md)',f'{r["assets"]:,.2f}',r['holdings'],r['fills']]
                     for r in reversed(records)]),'']
    lines += ['每个交易日一份文档，按账户列出全部收盘持仓、当日模拟成交、未成交原因及分红。',
              '由日常纸面流程在账本提交后生成；没有安装自动调度，也不读取未来行情或触发交易。',
              '此目录为自动导出的阅读副本，请将个人笔记另存，避免后续导出发生内容冲突。','']
    return '\n'.join(lines)


def atomic_text(path, text):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as f:
        f.write(text)
        temporary=Path(f.name)
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def export(root):
    root=Path(root).resolve()
    out=root/'reports/daily_records'
    with locked(root):
        latest_day(root)
        config=json.loads((root/'initial/session.json').read_text())
        prior=root/'initial'
        rendered,records={},[]
        for day in sorted((root/'days').glob('????????')):
            if not day.is_dir():
                continue
            content,record=render_day(root,day,prior,config)
            rendered[out/(record['date']+'.md')]=content
            records.append(record)
            prior=day
        if not records:
            return {'status':'NO_COMMITTED_DAYS','days':0}
        # Validate existing generated content before any writes; never erase manual edits.
        for path,content in rendered.items():
            if path.exists() and path.read_text()!=content:
                raise FileExistsError(f'daily document differs; preserve and review: {path}')
        index=out/'README.md'
        previous_records=[r for r in records if (out/(r['date']+'.md')).exists()]
        if index.exists() and index.read_text() not in [index_text(root,previous_records),index_text(root,records)]:
            raise FileExistsError('daily index differs; preserve and review')
        out.mkdir(parents=True,exist_ok=True)
        for path,content in rendered.items():
            if not path.exists():
                atomic_text(path,content)
        atomic_text(index,index_text(root,records))
    return {'status':'EXPORTED','days':len(records),'latest_date':records[-1]['date'],'index':str(index)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    print(json.dumps(export(p.parse_args().session),ensure_ascii=False,indent=2))
