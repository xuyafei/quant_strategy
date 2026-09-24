"""Point-in-time style strategies shared by research, paper, and live paths."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from factors.preprocess import cross_sectional_zscore, preprocess_factor_panel


def _mask_for_panel(raw_panel: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    if not isinstance(raw_panel.index, pd.MultiIndex) or raw_panel.index.nlevels != 2:
        raise TypeError("raw_panel must use a two-level (date, symbol) MultiIndex")
    panel = raw_panel.copy()
    panel.index = panel.index.set_names(["date", "symbol"])
    mask = eligible.copy()
    if not isinstance(mask.index, pd.MultiIndex) or mask.index.nlevels != 2:
        raise TypeError("eligible must use a two-level (date, symbol) MultiIndex")
    mask.index = mask.index.set_names(["date", "symbol"])
    return mask.reindex(panel.index).fillna(False).astype(bool)


def build_fixed_family_score(
    raw_panel: pd.DataFrame,
    eligible: pd.Series,
    industry: pd.Series,
    families: Mapping[str, Sequence[str]],
    *,
    require_all_families: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build the project's fixed-equal multi-family score inside one universe."""
    missing = sorted(
        {factor for factors in families.values() for factor in factors} - set(raw_panel.columns)
    )
    if missing:
        raise ValueError("raw_panel missing factors: %s" % missing)
    mask = _mask_for_panel(raw_panel, eligible)
    masked = raw_panel.where(mask, axis=0)
    standardized = preprocess_factor_panel(
        masked,
        industry=industry,
        by_industry=True,
        min_industry_count=3,
    )
    family_raw = pd.DataFrame(
        {
            str(family): standardized[list(factors)].mean(axis=1, skipna=True)
            for family, factors in families.items()
        },
        index=standardized.index,
    )
    family_scores = cross_sectional_zscore(family_raw.where(mask, axis=0))
    available = family_scores.notna().sum(axis=1)
    required = len(families) if require_all_families else 1
    score = family_scores.mean(axis=1, skipna=True).where(available.ge(required))
    return score.where(mask).dropna().rename("score"), family_scores


def build_focused_style_score(
    raw_panel: pd.DataFrame,
    eligible: pd.Series,
    industry: pd.Series,
    factors: Sequence[str],
    *,
    minimum_components: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build an equal-weight focused value or growth score inside its data-ready pool."""
    selected = [str(factor) for factor in factors]
    if not selected:
        raise ValueError("focused style requires at least one factor")
    if minimum_components < 1 or minimum_components > len(selected):
        raise ValueError("invalid minimum_components")
    if missing := sorted(set(selected) - set(raw_panel.columns)):
        raise ValueError("raw_panel missing focused-style factors: %s" % missing)
    mask = _mask_for_panel(raw_panel, eligible)
    standardized = preprocess_factor_panel(
        raw_panel[selected].where(mask, axis=0),
        industry=industry,
        by_industry=True,
        min_industry_count=3,
    )
    component_count = raw_panel[selected].where(mask, axis=0).notna().sum(axis=1)
    composite = standardized.mean(axis=1, skipna=True).where(
        component_count.ge(int(minimum_components))
    )
    score = cross_sectional_zscore(composite.to_frame("style"))["style"]
    return score.where(mask).dropna().rename("score"), standardized
