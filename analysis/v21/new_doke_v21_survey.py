#!/usr/bin/env python3
"""Initial survey for the new Kr and 2615 keV Doke data.

This script is deliberately diagnostic only. It uses the teacher-supplied
combined-energy formula, treats Kr as the single-energy control sample, and
does not train an event-level model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


G1 = 0.128755
G2B = 8.6125
KEV_PER_QUANTA = 0.0137
VARS = [
    "dt", "wS2CDF_max",
    "xS2max_desImageMCPAF_firstS2", "yS2max_desImageMCPAF_firstS2",
]
SIGNALS = ["qS1_PCs", "qS2Bdes_PCs"]


def read_tree(path, columns):
    tree = uproot.open(path)["out_tree"]
    return tree.arrays(columns, library="np"), tree.num_entries


def energy(a):
    return KEV_PER_QUANTA * (a["qS1_PCs"] / G1 + a["qS2Bdes_PCs"] / G2B)


def peak_window(e, center, half_width):
    return np.isfinite(e) & (e >= center - half_width) & (e <= center + half_width)


def run_normalize(values, runs, mask):
    out = np.full(len(values), np.nan)
    for run in np.unique(runs[mask]):
        take = mask & (runs == run) & np.isfinite(values) & (values > 0)
        if take.sum() < 100:
            continue
        out[take] = values[take] / np.median(values[take])
    return out


def binned(x, y, bins=24):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    edges = np.unique(np.quantile(x, np.linspace(.01, .99, bins + 1)))
    centers, med, low, high, count = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (x >= lo) & (x < hi if hi != edges[-1] else x <= hi)
        if take.sum() < 100:
            continue
        q = np.quantile(y[take], [.16, .5, .84])
        centers.append(np.median(x[take])); low.append(q[0]); med.append(q[1]); high.append(q[2]); count.append(take.sum())
    return tuple(np.asarray(v) for v in (centers, med, low, high, count))


def safe_name(name):
    return name.replace("_max", "").replace("desImageMCPAF", "image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kr", required=True)
    ap.add_argument("--th2615", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    kr_cols = SIGNALS + VARS + ["runNumber", "qS1_nn_factor", "qS2_nn_factor"]
    kr, kr_entries = read_tree(args.kr, kr_cols)
    th_cols = SIGNALS + VARS + ["runNumber"]
    th, th_entries = read_tree(args.th2615, th_cols)
    e_kr, e_th = energy(kr), energy(th)
    valid_kr = np.isfinite(e_kr) & (e_kr > 0)
    valid_th = np.isfinite(e_th) & (e_th > 0)
    # The narrow central Kr window removes continuum tails, while preserving a large sample.
    kr_peak = valid_kr & peak_window(e_kr, 41.25, 3.5)
    s1_norm = run_normalize(kr["qS1_PCs"], kr["runNumber"], kr_peak)
    s2_norm = run_normalize(kr["qS2Bdes_PCs"], kr["runNumber"], kr_peak)

    summary = {
        "formula": "Erec = 0.0137 * (qS1_PCs / 0.128755 + qS2Bdes_PCs / 8.6125)",
        "kr": {
            "entries": int(kr_entries), "valid_energy": int(valid_kr.sum()),
            "peak_window_events": int(kr_peak.sum()),
            "runs": int(np.unique(kr["runNumber"]).size),
            "energy_quantiles_kev": [float(v) for v in np.quantile(e_kr[valid_kr], [.01,.05,.16,.5,.84,.95,.99])],
        },
        "th2615": {
            "entries": int(th_entries), "valid_energy": int(valid_th.sum()),
            "runs": [int(v) for v in np.unique(th["runNumber"])],
            "energy_quantiles_kev": [float(v) for v in np.quantile(e_th[valid_th], [.01,.05,.16,.5,.84,.95,.99])],
        },
        "kr_dependence": {},
    }
    for var in VARS:
        x = kr[var].astype(float)
        record = {}
        for label, norm in (("S1", s1_norm), ("S2", s2_norm)):
            good = kr_peak & np.isfinite(x) & np.isfinite(norm)
            bx, med, lo, hi, count = binned(x[good], norm[good])
            record[label] = {
                "spearman": float(spearmanr(x[good], norm[good]).statistic),
                "median_span_percent": float(100 * (np.max(med) - np.min(med))),
                "binned_points": int(len(med)),
                "coverage_kr_p01_p99": [float(np.quantile(x[good], .01)), float(np.quantile(x[good], .99))],
                "coverage_2615_p01_p99": [float(np.quantile(th[var][valid_th], .01)), float(np.quantile(th[var][valid_th], .99))],
            }
        summary["kr_dependence"][var] = record

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(e_kr[valid_kr], bins=np.arange(10, 90.05, .1), histtype="step", lw=1.1)
    axes[0].axvspan(37.75, 44.75, color="tab:orange", alpha=.15, label="Kr control window")
    axes[0].set(xlabel="Erec [keV]", ylabel="events / 0.1 keV", title="Kr spectrum: 787,039 events")
    axes[0].legend()
    axes[1].hist(e_th[valid_th], bins=np.arange(2350, 3100.5, 2.5), histtype="step", lw=1.1)
    axes[1].axvline(2614.5, color="gray", ls=":", lw=1, label="2614.5 keV")
    axes[1].set(xlabel="Erec [keV]", ylabel="events / 2.5 keV", title="2615 dataset: 19,036 events")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "01_energy_spectra.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(2, len(VARS), figsize=(4.0 * len(VARS), 7.4), squeeze=False)
    for j, var in enumerate(VARS):
        x = kr[var].astype(float)
        for i, (label, norm, color) in enumerate((("S1", s1_norm, "tab:blue"), ("S2", s2_norm, "tab:orange"))):
            ax = axes[i, j]
            good = kr_peak & np.isfinite(x) & np.isfinite(norm)
            idx = np.flatnonzero(good)[::max(int(good.sum() / 7000), 1)]
            ax.scatter(x[idx], norm[idx], s=2, alpha=.08, color=color)
            bx, med, lo, hi, _ = binned(x[good], norm[good])
            ax.fill_between(bx, lo, hi, color=color, alpha=.18)
            ax.plot(bx, med, color=color, lw=2)
            ax.axhline(1, color="black", ls="--", lw=.7)
            ax.set(xlabel=var, ylabel="run-normalized " + label)
            ax.set_title(label + " response")
    fig.suptitle("Kr control sample: residual S1/S2 dependence after existing PCs", y=1.02, fontsize=15)
    fig.tight_layout(); fig.savefig(out / "02_kr_variable_dependence.png", dpi=190, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, len(VARS), figsize=(4.0 * len(VARS), 3.8), squeeze=False)
    for j, var in enumerate(VARS):
        ax = axes[0, j]
        ax.hist(kr[var][kr_peak], bins=70, histtype="step", density=True, label="Kr peak", lw=1.2)
        ax.hist(th[var][valid_th], bins=70, histtype="step", density=True, label="2615", lw=1.2)
        ax.set(xlabel=var, ylabel="density", title="Input coverage")
        if j == 0: ax.legend()
    fig.tight_layout(); fig.savefig(out / "03_coverage_kr_vs_2615.png", dpi=190); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
