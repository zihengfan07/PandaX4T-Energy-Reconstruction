#!/usr/bin/env python3
"""Create the remaining event-by-event comparison and numerical audit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "results" / "v12_safe_20260803"


def spike(values, lo=500.0, hi=3800.0, width=5.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=np.arange(lo, hi + width, width))
    best = (0.0, None, None)
    for index in range(3, len(counts) - 3):
        baseline = np.median(np.r_[counts[index-3:index], counts[index+1:index+4]])
        if baseline >= 20 and counts[index] / baseline > best[0]:
            best = (
                float(counts[index] / baseline),
                float((edges[index] + edges[index+1]) / 2),
                int(counts[index]),
            )
    return {"ratio": best[0], "center_kev": best[1], "count": best[2]}


def main():
    frame = pd.read_csv(ROOT / "original_vs_v12_events.csv")
    x = frame.original_energy_cor_kev.to_numpy(float)
    y = frame.v12_safe_energy_kev.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    display = (x >= 500) & (x <= 3800) & (y >= 500) & (y <= 3800)
    dx, dy = x[display], y[display]
    rng = np.random.RandomState(20260804)
    sample = rng.choice(len(dx), size=min(12000, len(dx)), replace=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(dx[sample], dy[sample], s=2, alpha=0.12, color="#2455c3", rasterized=True)
    edges = np.arange(500.0, 3850.0, 50.0)
    ids = np.digitize(dx, edges)
    centers, medians = [], []
    for index in range(1, len(edges)):
        local = dy[ids == index]
        if len(local) >= 20:
            centers.append((edges[index-1] + edges[index]) / 2.0)
            medians.append(np.median(local))
    ax.plot(centers, medians, color="#d62728", linewidth=1.5, label="binned median")
    ax.plot([500, 3800], [500, 3800], "--", color="0.35", linewidth=1.0, label="y = x")
    ax.set(xlim=(500, 3800), ylim=(500, 3800), xlabel="Given cubic Energy_cor [keV]", ylabel="v12_safe [keV]", title="Event-by-event energy comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "event_by_event_energy_comparison.png", dpi=160)
    plt.close(fig)

    foundation = frame.v12_foundation_energy_kev.to_numpy(float)
    candidate = frame.v12_safe_energy_kev.to_numpy(float)
    ratio = candidate / foundation - 1.0
    ratio = ratio[np.isfinite(ratio)]
    named = {
        "Raw S1/S2 formula": frame.original_energy_kev.to_numpy(float),
        "Given cubic Energy_cor": frame.original_energy_cor_kev.to_numpy(float),
        "v12_safe": candidate,
    }
    audit = {
        "rows": int(len(frame)),
        "finite_v12": int(np.isfinite(candidate).sum()),
        "median_kev": {name: float(np.nanmedian(values)) for name, values in named.items()},
        "spike_audit_5kev": {name: spike(values) for name, values in named.items()},
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
