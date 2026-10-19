"""Optional rank retention policy. Call only with the already eligible universe.

Extracted from paper08; never imports live execution or changes defaults.
"""
import numpy as np
import pandas as pd


def ordered_scores(scores):
    if scores.index.has_duplicates or scores.index.isna().any():
        raise ValueError('unique nonmissing symbols required')
    scores = scores.astype(float)
    if not np.isfinite(scores).all():
        raise ValueError('finite scores required')
    frame = scores.rename('score').rename_axis('symbol').reset_index()
    return frame.sort_values(['score', 'symbol'], ascending=[False, True], kind='stable').set_index('symbol').score


def select_buffer(scores, held, top_k, exit_rank):
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) for n in [top_k, exit_rank]):
        raise ValueError('integer ranks required')
    if not 1 <= top_k <= exit_rank:
        raise ValueError('require 1 <= top_k <= exit_rank')
    ranked = ordered_scores(scores)
    if len(ranked) < top_k:
        raise ValueError('insufficient eligible names; no silent budget change')
    actual = set(held)
    keep = [s for s in ranked.index[:exit_rank] if s in actual][:top_k]
    selected = set(keep)
    for symbol in ranked.index[:top_k]:
        if len(selected) == top_k:
            break
        selected.add(symbol)
    return [s for s in ranked.index[:exit_rank] if s in selected]
