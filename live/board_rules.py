"""A-share auction quantity rules used by the explicitly opt-in paper protocol.

STAR: >=200 shares, increments of 1, limit-order maximum 100,000.
Main/ChiNext: buys in 100-share lots; residual holdings may be liquidated.
These are not brokerage permission checks. Unsupported instruments fail closed.
"""
from __future__ import annotations

import math


def board(symbol: str) -> str:
    code, exchange = symbol.split(".")
    if exchange == "SH" and code.startswith("688"):
        return "STAR"
    if exchange == "SZ" and code.startswith(("300", "301")):
        return "CHINEXT"
    if (exchange == "SH" and code.startswith(("600", "601", "603", "605"))) or (
        exchange == "SZ" and code.startswith(("000", "001", "002", "003"))
    ):
        return "MAIN"
    raise ValueError("unsupported instrument: " + symbol)


def target_shares(symbol: str, value: float, price: float) -> int:
    units = max(0, math.floor(value / price))
    if board(symbol) == "STAR":
        return units if units >= 200 else 0
    return units // 100 * 100


def quantity_error(symbol: str, side: str, quantity: int, held: int) -> str:
    kind = board(symbol)
    if side == "HOLD":
        return ""
    maximum = 100_000 if kind == "STAR" else (300_000 if kind == "CHINEXT" else 1_000_000)
    if quantity > maximum:
        return "above_single_order_limit"
    if kind == "STAR":
        if quantity < 200 and not (side == "SELL" and quantity == held):
            return "below_star_minimum_200"
    elif quantity % 100 and not (side == "SELL" and quantity == held):
        return "not_lot_size"
    return ""
