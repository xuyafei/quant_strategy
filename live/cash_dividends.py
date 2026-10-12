"""Conservative cash-only corporate actions for the idealized paper ledger.

Only previous-session record dates with same-day ex/payment dates are supported.
Amounts are GROSS: dividend withholding/holding-period tax is NOT simulated.
Unsupported events and unexplained adjustment changes fail closed.
"""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import math

import pandas as pd

POLICY = "CASH_ONLY_SAME_DAY_GROSS_V1"
FIELDS = ("ts_code,end_date,ann_date,imp_ann_date,div_proc,stk_div,stk_bo_rate,"
          "stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate")
EVENT_COLUMNS = ["event_id", "symbol", "record_date", "ex_date", "pay_date", "cash_per_share_gross"]
PAYMENT_COLUMNS = EVENT_COLUMNS + ["strategy", "eligible_shares", "cash_gross", "tax_policy"]


def _date(value):
    if pd.isna(value) or not str(value).strip():
        raise RuntimeError("corporate action has missing date")
    text = str(value).strip()
    return pd.to_datetime(text, format="%Y%m%d" if len(text) == 8 else "%Y-%m-%d", errors="raise").strftime("%Y-%m-%d")


def normalize_events(raw, held, previous, market, prior_date, date):
    """Validate vendor events against record-date holdings and factor movement."""
    if not set(FIELDS.split(",")).issubset(raw.columns):
        raise RuntimeError("corporate action response schema incomplete")
    rows = []
    relevant = raw[raw.ts_code.isin(held) & raw.div_proc.eq("实施")]
    if relevant.ts_code.duplicated().any():
        raise RuntimeError("duplicate/conflicting implemented corporate actions")
    for rec in relevant.to_dict("records"):
        symbol = rec["ts_code"]
        record, ex, pay = (_date(rec[k]) for k in ["record_date", "ex_date", "pay_date"])
        if record != prior_date or ex != date or pay != date:
            raise RuntimeError("unsupported record/ex/payment dates: " + symbol)
        if _date(rec["ann_date"]) > date or (pd.notna(rec["imp_ann_date"]) and _date(rec["imp_ann_date"]) > date):
            raise RuntimeError("future corporate action announcement")
        # Aggregate stk_div must explicitly confirm zero. Components may be absent.
        if pd.isna(rec["stk_div"]) or float(rec["stk_div"]) != 0:
            raise RuntimeError("stock distribution is not supported: " + symbol)
        for col in ["stk_bo_rate", "stk_co_rate"]:
            if pd.notna(rec[col]) and float(rec[col]) != 0:
                raise RuntimeError("stock distribution is not supported: " + symbol)
        cash = float(rec["cash_div_tax"])
        if not math.isfinite(cash) or cash <= 0:
            raise RuntimeError("invalid gross cash dividend: " + symbol)
        old, new = previous.loc[symbol], market.loc[symbol]
        if _date(old.price_date) != prior_date or bool(new.is_suspended):
            raise RuntimeError("suspended/stale corporate action requires review: " + symbol)
        close = float(old.close)
        if cash >= close:
            raise RuntimeError("invalid cash dividend relative to reference price")
        expected = close / (close - cash)
        actual = float(new.adj_factor) / float(old.adj_factor)
        # Fixed tolerance for vendor rounding; never infer cash from the factor.
        if not math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-8):
            raise RuntimeError("cash dividend does not explain factor change: " + symbol)
        key = "|".join([symbol, _date(rec["end_date"]), record, ex, pay])
        rows.append(dict(event_id=hashlib.sha256(key.encode()).hexdigest(), symbol=symbol,
                         record_date=record, ex_date=ex, pay_date=pay, cash_per_share_gross=cash))
    covered = {r["symbol"] for r in rows}
    for symbol in held:
        if symbol not in previous.index or symbol not in market.index:
            raise RuntimeError("held stock missing from corporate action history")
        if not math.isclose(float(previous.loc[symbol, "adj_factor"]),
                            float(market.loc[symbol, "adj_factor"]), rel_tol=1e-10) and symbol not in covered:
            raise RuntimeError("unexplained corporate action: " + symbol)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def payments_for(events, positions, strategy, already_paid):
    shares = positions.set_index("symbol").shares
    rows = []
    for rec in events.to_dict("records"):
        quantity = float(shares.get(rec["symbol"], 0))
        if quantity <= 0:
            continue
        if rec["event_id"] in already_paid:
            raise RuntimeError("dividend already credited: " + rec["event_id"])
        if not math.isfinite(quantity) or not quantity.is_integer():
            raise RuntimeError("invalid record-date share quantity")
        amount = (Decimal(str(quantity)) * Decimal(str(rec["cash_per_share_gross"]))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        rows.append({**rec, "strategy": strategy, "eligible_shares": quantity,
                     "cash_gross": float(amount), "tax_policy": "GROSS_DIVIDEND_TAX_NOT_MODELED"})
    return pd.DataFrame(rows, columns=PAYMENT_COLUMNS)
