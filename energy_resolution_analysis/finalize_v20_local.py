#!/usr/bin/env python3
"""Create local QA figures for a v20 result downloaded from bl-0."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    args = ap.parse_args()
    out = Path(args.result)
    bg = pd.read_csv(out / "background_energies.txt", sep=r"\s+")
    given = bg["given_energy_kev"].to_numpy(float)
    v20 = bg["v20_physics_energy_kev"].to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    bins = np.arange(500, 3800 + 5, 5)
    axes[0].hist(given[np.isfinite(given)], bins=bins, histtype="step", lw=1.15,
                 label="given cubic energy")
    axes[0].hist(v20[np.isfinite(v20)], bins=bins, histtype="step", lw=1.15,
                 label="v20 separate S1/S2")
    axes[0].set(xlabel="energy [keV]", ylabel="events / 5 keV",
                title="Background spectrum safety check")
    axes[0].legend()
    bins = np.arange(2450, 3000 + 2.5, 2.5)
    axes[1].hist(given[np.isfinite(given)], bins=bins, histtype="step", lw=1.15,
                 label="given cubic energy")
    axes[1].hist(v20[np.isfinite(v20)], bins=bins, histtype="step", lw=1.15,
                 label="v20 separate S1/S2")
    axes[1].axvline(2600, color="gray", ls=":", lw=1)
    axes[1].set(xlabel="energy [keV]", ylabel="events / 2.5 keV",
                title="2450--3000 keV zoom")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "05_background_safety.png", dpi=190)
    plt.close(fig)

    good = np.isfinite(given) & np.isfinite(v20) & (given > 0)
    ratio = v20[good] / given[good]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hexbin(given[good], v20[good], gridsize=90, bins="log", mincnt=1)
    lim = [500, 3800]
    axes[0].plot(lim, lim, "k--", lw=.8)
    axes[0].set(xlim=lim, ylim=lim, xlabel="given energy [keV]",
                ylabel="v20 energy [keV]", title="Event-by-event background comparison")
    axes[1].hist(ratio[(ratio > .5) & (ratio < 1.5)], bins=160, histtype="step")
    axes[1].axvline(1, color="black", ls="--", lw=.8)
    axes[1].set(xlabel="v20 / given energy", ylabel="events",
                title="Background energy-ratio distribution")
    fig.tight_layout()
    fig.savefig(out / "06_background_event_comparison.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
