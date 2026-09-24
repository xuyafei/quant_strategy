"""Runtime exports that connect strategy universes to paper/live order workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def equal_weight_rebalance_log(
    execution_score: pd.Series,
    *,
    strategy: str,
    universe: str,
    top_k: int,
) -> pd.DataFrame:
    """Convert an execution-dated score into the standard rebalance-log schema."""
    if not isinstance(execution_score.index, pd.MultiIndex) or execution_score.index.nlevels != 2:
        raise TypeError("execution_score must use a two-level (date, symbol) MultiIndex")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    score = execution_score.copy()
    score.index = score.index.set_names(["date", "symbol"])
    rows: list[dict[str, Any]] = []
    for date, values in score.groupby(level="date", sort=True):
        ranked = values.droplevel("date").dropna().sort_values(ascending=False).head(int(top_k))
        if ranked.empty:
            continue
        weight = 1.0 / float(len(ranked))
        for rank, (symbol, value) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "symbol": str(symbol),
                    "weight": weight,
                    "rank": rank,
                    "selected": True,
                    "selected_rank": rank,
                    "score": float(value),
                    "strategy": str(strategy),
                    "universe": str(universe),
                    "weighting": "equal",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "weight",
            "rank",
            "selected",
            "selected_rank",
            "score",
            "strategy",
            "universe",
            "weighting",
        ],
    )


def latest_universe_snapshot(
    report: pd.DataFrame,
    *,
    universe: str,
) -> pd.DataFrame:
    """Return the latest eligible point-in-time snapshot for one universe."""
    required = {"date", "symbol", "eligible"}
    if missing := required - set(report.columns):
        raise ValueError("strategy report missing columns: %s" % sorted(missing))
    work = report.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    latest = work["date"].max()
    column = "eligible" if universe == "BASE_V2" else "eligible_" + str(universe).upper()
    if column not in work.columns:
        raise ValueError("strategy report missing eligibility column %s" % column)
    selected = work[work["date"].eq(latest) & work[column].astype(bool)].copy()
    selected.insert(2, "universe", str(universe))
    selected["eligible_for_strategy"] = True
    preferred = [
        "date",
        "symbol",
        "universe",
        "eligible_for_strategy",
        "name",
        "industry_l1",
        "circ_mv_yuan",
        "adv20_yuan",
        "market_cap_percentile",
        "value_component_count",
        "growth_component_count",
    ]
    return selected[[column for column in preferred if column in selected.columns]].sort_values(
        "symbol"
    ).reset_index(drop=True)


def save_runtime_exports(
    output_dir: Path,
    *,
    report: pd.DataFrame,
    shifted_scores: dict[str, pd.Series],
    universe_by_strategy: dict[str, str],
    top_k_by_strategy: dict[str, int],
    protocol_sha256: str,
) -> dict[str, Path]:
    """Save paper/live-compatible logs and latest universe snapshots."""
    base = Path(output_dir)
    rebalance_dir = base / "rebalance_logs"
    snapshot_dir = base / "universe_snapshots"
    rebalance_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for strategy, score in shifted_scores.items():
        universe = universe_by_strategy[strategy]
        log = equal_weight_rebalance_log(
            score,
            strategy=strategy,
            universe=universe,
            top_k=int(top_k_by_strategy[strategy]),
        )
        path = rebalance_dir / (strategy + ".csv")
        log.to_csv(path, index=False)
        paths["rebalance_log_" + strategy] = path
    for universe in sorted(set(universe_by_strategy.values())):
        snapshot = latest_universe_snapshot(report, universe=universe)
        path = snapshot_dir / (universe + ".csv")
        snapshot.to_csv(path, index=False)
        paths["universe_snapshot_" + universe] = path
    manifest = {
        "protocol_sha256": protocol_sha256,
        "strategies": sorted(shifted_scores),
        "universes": sorted(set(universe_by_strategy.values())),
        "rebalance_log_schema": "date,symbol,weight compatible with live.daily_paper_cli",
        "execution_timing": "NEXT_SESSION_CLOSE",
        "live_approval": False,
    }
    manifest_path = base / "runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths
