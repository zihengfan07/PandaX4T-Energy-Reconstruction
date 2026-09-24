#!/usr/bin/env python3
"""Plot event-aligned original formulas against the v12_safe output."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "results" / "v12_safe_20260803"


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def spike(values, lo=500.0, hi=3800.0, width=5.0):
    counts, edges = np.histogram(finite(values), bins=np.arange(lo, hi + width, width))
    best = (0.0, None, None)
    for index in range(3, len(counts) - 3):
        baseline = np.median(np.r_[counts[index-3:index], counts[index+1:index+4]])
        if baseline >= 20:
            ratio = counts[index] / baseline
            if ratio > best[0]:
                best = (float(ratio), float((edges[index] + edges[index+1]) / 2), int(counts[index]))
    return {"ratio": best[0], "center_kev": best[1], "count": best[2]}


def main():
    frame = pd.read_csv(ROOT / "original_vs_v12_events.csv")
    series = {
        "Raw S1/S2 formula": frame.original_energy_kev.to_numpy(float),
        "Given cubic Energy_cor": frame.original_energy_cor_kev.to_numpy(float),
        "v12_safe": frame.v12_safe_energy_kev.to_numpy(float),
    }
    colors = {
        "Raw S1/S2 formula": "#777777",
        "Given cubic Energy_cor": "#2455c3",
        "v12_safe": "#d62728",
    }
    edges = np.arange(500.0, 3805.0, 5.0)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, values in series.items():
        ax.hist(values, bins=edges, histtype="step", linewidth=1.15, color=colors[name], label=name)
    ax.set(xlabel="Energy [keV]", ylabel="Events / 5 keV", title="Same background events: original formulas versus v12_safe")
    ax.legend(frameon=False)
    ax.set_xlim(500, 3800)
    fig.tight_layout()
    fig.savefig(ROOT / "original_formula_vs_v12_full.png", dpi=200)
    plt.close(fig)

    foundation = frame.v12_foundation_energy_kev.to_numpy(float)
    candidate = frame.v12_safe_energy_kev.to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[1.15, 1.0])
    for ax, limits, width in [
        (axes[0], (1000, 3200), 5.0),
        (axes[1], (2500, 3000), 2.5),
    ]:
        local_edges = np.arange(limits[0], limits[1] + width, width)
        ax.hist(foundation, bins=local_edges, histtype="step", linewidth=1.2, color="#2455c3", label="v12 robust foundation")
        ax.hist(candidate, bins=local_edges, histtype="step", linewidth=1.2, color="#d62728", label="v12_safe output")
        ax.set_xlim(*limits)
        ax.set_ylabel("Events / {:.1f} keV".format(width))
        ax.legend(frameon=False)
    axes[0].set_title("Spectrum-shape safety check")
    axes[1].set_xlabel("Energy [keV]")
    for value in [2600.0, 2887.5]:
        axes[1].axvline(value, color="0.45", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "v12_problem_regions_zoom.png", dpi=200)
    plt.close(fig)

    valid = np.isfinite(frame.original_energy_cor_kev) & np.isfinite(frame.v12_safe_energy_kev)
    x = frame.loc[valid, "original_energy_cor_kev"].to_numpy(float)
    y = frame.loc[valid, "v12_safe_energy_kev"].to_numpy(float)
    select = (x >= 500) & (x <= 3800) & (y >= 500) & (y <= 3800)
    selected_x = x[select]
    selected_y = y[select]
    rng = np.random.RandomState(20260804)
    sample = rng.choice(len(selected_x), size=min(15000, len(selected_x)), replace=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(selected_x[sample], selected_y[sample], s=2, alpha=0.12, color="#2455c3", rasterized=True)
    bin_edges = np.arange(500.0, 3850.0, 50.0)
    bin_id = np.digitize(selected_x, bin_edges)
    centers, medians = [], []
    for index in range(1, len(bin_edges)):
        local = selected_y[bin_id == index]
        if len(local) >= 20:
            centers.append((bin_edges[index-1] + bin_edges[index]) / 2.0)
            medians.append(np.median(local))
    ax.plot(centers, medians, color="#d62728", linewidth=1.5, label="binned median")
    ax.plot([500, 3800], [500, 3800], "--", color="0.35", linewidth=1.0, label="y = x")
    ax.set(xlim=(500, 3800), ylim=(500, 3800), xlabel="Given cubic Energy_cor [keV]", ylabel="v12_safe [keV]", title="Event-by-event energy comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "event_by_event_energy_comparison.png", dpi=200)
    plt.close(fig)

    ratio = candidate / foundation - 1.0
    ratio = ratio[np.isfinite(ratio)]
    audit = {
        "rows": int(len(frame)),
        "finite_v12": int(np.isfinite(candidate).sum()),
        "median_kev": {name: float(np.nanmedian(values)) for name, values in series.items()},
        "spike_audit_5kev": {name: spike(values) for name, values in series.items()},
        "v12_change_relative_to_robust_foundation": {
            "q01": float(np.quantile(ratio, 0.01)),
            "median": float(np.quantile(ratio, 0.50)),
            "q99": float(np.quantile(ratio, 0.99)),
        },
        "spearman_energy_cor_vs_v12": float(pd.Series(x).rank().corr(pd.Series(y).rank())),
    }
    (ROOT / "original_vs_v12_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
