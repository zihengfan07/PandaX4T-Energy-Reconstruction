#!/usr/bin/env python3
"""Create report-ready ratio-versus-input figures for the v15 S10 model."""

from __future__ import print_function

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLES = [
    ("dt", "Drift time", 0.001, "us"),
    ("wS2CDF_max", "S2 CDF width", 1.0, "raw unit"),
    ("yS2Tcor_max", "Corrected top y", 1.0, "mm"),
    ("xS2Bcor_max", "Corrected bottom x", 1.0, "mm"),
]


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def prepare(raw_path, energy_path, after_name):
    columns = [item[0] for item in VARIABLES]
    raw = pd.read_csv(raw_path, sep="\t", usecols=columns, low_memory=False)
    energy = pd.read_csv(energy_path, low_memory=False)
    if len(raw) != len(energy):
        raise ValueError("Row mismatch: {} versus {}".format(len(raw), len(energy)))
    ratio = energy[after_name].to_numpy(float) / energy["energy_cor_kev"].to_numpy(float)
    return raw, ratio


def binned_summary(x, ratio, mask, bins=18):
    edges = np.unique(np.quantile(x[mask], np.linspace(0.0, 1.0, bins + 1)))
    centers, low, median, high = [], [], [], []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            selected = mask & (x >= left) & (x <= right)
        else:
            selected = mask & (x >= left) & (x < right)
        if not np.any(selected):
            continue
        q16, q50, q84 = np.quantile(ratio[selected], [0.16, 0.50, 0.84])
        centers.append(float(np.median(x[selected])))
        low.append(float(q16)); median.append(float(q50)); high.append(float(q84))
    return np.asarray(centers), np.asarray(low), np.asarray(median), np.asarray(high)


def draw_panel(ax, x, ratio, title, unit, rng, max_points):
    valid = np.isfinite(x) & np.isfinite(ratio) & (ratio > 0)
    xlo, xhi = np.quantile(x[valid], [0.005, 0.995])
    visible = valid & (x >= xlo) & (x <= xhi)
    indices = np.flatnonzero(visible)
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    ax.scatter(x[indices], ratio[indices], s=7, alpha=0.20, color="#2E86DE",
               edgecolors="none", rasterized=True, label="events")
    bx, q16, q50, q84 = binned_summary(x, ratio, visible)
    ax.fill_between(bx, q16, q84, color="#F28E2B", alpha=0.20,
                    linewidth=0, label="16%-84%")
    ax.plot(bx, q50, color="#E66101", linewidth=2.0, label="binned median")
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ylo, yhi = np.quantile(ratio[valid], [0.003, 0.997])
    pad = 0.06 * (yhi - ylo)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo - pad, yhi + pad)
    applied = valid & (np.abs(ratio - 1.0) > 1.0e-12)
    rho_all = spearman(x[valid], ratio[valid])
    rho_applied = spearman(x[applied], ratio[applied])
    ax.set_title("{}\nSpearman rho = {:.3f}; applied only = {:.3f}".format(
        title, rho_all, rho_applied), fontsize=11)
    ax.set_xlabel("{} [{}]".format(title, unit))
    ax.set_ylabel(r"$E_{\mathrm{after}}/E_{\mathrm{before}}$")
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def make_set(raw, ratio, sample_name, output_dir, prefix, max_points):
    rng = np.random.RandomState(20260815)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), constrained_layout=True)
    for ax, (column, label, scale, unit) in zip(axes.flat, VARIABLES):
        draw_panel(ax, raw[column].to_numpy(float) * scale, ratio,
                   label, unit, rng, max_points)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("v15 S10: energy correction ratio versus model inputs\n{}".format(sample_name),
                 fontsize=15, y=1.065)
    combined = output_dir / (prefix + "_ratio_vs_inputs_2x2.png")
    fig.savefig(str(combined), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    paths = [combined]
    for number, (column, label, scale, unit) in enumerate(VARIABLES, start=1):
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        draw_panel(ax, raw[column].to_numpy(float) * scale, ratio,
                   label, unit, rng, max_points)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="best", frameon=False)
        fig.suptitle("{}: {}".format(sample_name, column), fontsize=13)
        path = output_dir / ("{}_{:02d}_ratio_vs_{}.png".format(prefix, number, column))
        fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig); paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-raw", required=True)
    parser.add_argument("--calibration-energy", required=True)
    parser.add_argument("--background-raw", required=True)
    parser.add_argument("--background-energy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    cal_raw, cal_ratio = prepare(args.calibration_raw, args.calibration_energy,
                                 "candidate_v15_oof_energy_kev")
    bg_raw, bg_ratio = prepare(args.background_raw, args.background_energy,
                               "candidate_v15_energy_kev")
    paths = []
    paths += make_set(cal_raw, cal_ratio, "Calibration OOF events", output,
                      "01_calibration", 3500)
    paths += make_set(bg_raw, bg_ratio, "Background events", output,
                      "02_background", 8000)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
