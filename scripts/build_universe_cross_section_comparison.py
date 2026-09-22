#!/usr/bin/env python3
"""Compare strict weekly walk-forward results across A50 and CSI300 universes."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_nav(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    frame.index.name = "date"
    return frame


def _normalize(frame: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    out = frame.loc[frame.index >= start].copy()
    return out.div(out.iloc[0])


def _admission_summary(path: Path, universe: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    active_dates = frame.loc[frame["decision"].eq("PASS"), "as_of_date"].nunique()
    passed = frame[frame["decision"].eq("PASS")].groupby("factor").size().rename("pass_count")
    selected_flag = frame["selected_after_redundancy"].astype(str).str.lower().eq("true")
    selected = frame[selected_flag].groupby("factor").size().rename("selected_count")
    out = pd.concat([passed, selected], axis=1).fillna(0).astype(int).reset_index()
    out.insert(0, "universe", universe)
    out["active_rebalances"] = int(active_dates)
    out["selected_rate"] = out["selected_count"] / max(int(active_dates), 1)
    return out


def run(
    a50_sensitivity_dir: Path,
    csi300_sensitivity_dir: Path,
    a50_walk_forward_dir: Path,
    csi300_walk_forward_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    a50 = _load_nav(a50_sensitivity_dir / "nav_comparison.csv")
    csi = _load_nav(csi300_sensitivity_dir / "nav_comparison.csv")
    a50_perf = pd.read_csv(a50_sensitivity_dir / "performance_summary.csv")
    csi_perf = pd.read_csv(csi300_sensitivity_dir / "performance_summary.csv")
    common = a50.index.intersection(csi.index)
    start = max(pd.Timestamp(a50_perf["start"].min()), pd.Timestamp(csi_perf["start"].min()))
    end = pd.Timestamp(common.max())
    common = common[(common >= start) & (common <= end)]
    a50 = _normalize(a50.loc[common], start)
    csi = _normalize(csi.loc[common], start)

    paths = {
        "chart": output_dir / "cross_universe_nav_excess.png",
        "performance": output_dir / "cross_universe_performance.csv",
        "admission": output_dir / "factor_admission_comparison.csv",
        "admission_chart": output_dir / "factor_admission_comparison.png",
    }

    a50_perf.insert(0, "universe", "A50")
    csi_perf.insert(0, "universe", "CSI300")
    pd.concat([a50_perf, csi_perf], ignore_index=True).to_csv(paths["performance"], index=False)

    admission = pd.concat(
        [
            _admission_summary(a50_walk_forward_dir / "walk_forward" / "factor_decisions.csv", "A50"),
            _admission_summary(csi300_walk_forward_dir / "walk_forward" / "factor_decisions.csv", "CSI300"),
        ],
        ignore_index=True,
    )
    admission.to_csv(paths["admission"], index=False)

    admission_plot = admission.pivot(index="factor", columns="universe", values="selected_rate").fillna(0.0)
    admission_plot = admission_plot.reindex(columns=["A50", "CSI300"], fill_value=0.0)
    admission_plot = admission_plot.sort_values(
        by=["CSI300", "A50"], ascending=[True, True]
    )
    fig_admission, ax_admission = plt.subplots(figsize=(10, 7))
    admission_plot.plot.barh(ax=ax_admission, width=0.78)
    ax_admission.set_title("Factor admission frequency: A50 versus CSI300")
    ax_admission.set_xlabel("Share of 51 active weekly rebalances")
    ax_admission.set_ylabel("")
    ax_admission.set_xlim(0.0, 1.05)
    ax_admission.grid(axis="x", alpha=0.25)
    ax_admission.legend(title="Point-in-time universe", loc="lower right")
    fig_admission.tight_layout()
    fig_admission.savefig(paths["admission_chart"], dpi=180, bbox_inches="tight")
    plt.close(fig_admission)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(a50.index, a50["A50_TOP05"], label="A50 Top5", linewidth=2)
    axes[0].plot(csi.index, csi["CSI300_TOP05"], label="CSI300 Top5", linewidth=2)
    axes[0].plot(a50.index, a50["BENCH_EQUAL_WEIGHT"], label="A50 PIT benchmark", linestyle="--")
    axes[0].plot(csi.index, csi["BENCH_EQUAL_WEIGHT"], label="CSI300 PIT benchmark", linestyle="--")
    axes[0].set_title("Same strict weekly strategy, different point-in-time universes")
    axes[0].set_ylabel("NAV (rebased to 1.0)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", ncol=2)

    excess_series = {
        "A50 Top5 excess": a50["A50_TOP05"] / a50["BENCH_EQUAL_WEIGHT"],
        "A50 Top10 excess": a50["A50_TOP10"] / a50["BENCH_EQUAL_WEIGHT"],
        "CSI300 Top5 excess": csi["CSI300_TOP05"] / csi["BENCH_EQUAL_WEIGHT"],
        "CSI300 Top10 excess": csi["CSI300_TOP10"] / csi["BENCH_EQUAL_WEIGHT"],
    }
    for label, series in excess_series.items():
        axes[1].plot(series.index, series, label=label, linewidth=1.8)
    axes[1].axhline(1.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_title("Strategy wealth relative to its own point-in-time benchmark")
    axes[1].set_ylabel("Relative NAV")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", ncol=2)
    fig.text(
        0.01,
        0.01,
        f"Common active period: {start:%Y-%m-%d} to {end:%Y-%m-%d} · weekly rebalance · costs included",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(paths["chart"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a50-sensitivity-dir", type=Path, required=True)
    parser.add_argument("--csi300-sensitivity-dir", type=Path, required=True)
    parser.add_argument("--a50-walk-forward-dir", type=Path, required=True)
    parser.add_argument("--csi300-walk-forward-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run(
        args.a50_sensitivity_dir,
        args.csi300_sensitivity_dir,
        args.a50_walk_forward_dir,
        args.csi300_walk_forward_dir,
        args.output_dir,
    ).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
