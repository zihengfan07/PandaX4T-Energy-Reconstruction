"""Create diagnostic figures for the ten v14 small-model comparison."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v14_smallmodels_20260814"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#2455C3"
RED = "#D62728"
GOLD = "#E6A700"
GRAY = "#8A94A3"
NAVY = "#17365D"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
})


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def comparison(table, summary):
    table = table.sort_values("name")
    names = table.name.str.replace("_", "\n", regex=False).to_list()
    safe = table.safe.astype(bool).to_numpy()
    colors = np.where(safe, GOLD, GRAY)
    sigma = 100.0 * table.relative_sigma_improvement.to_numpy(float)
    applied = 100.0 * table.background_applied.to_numpy(float)
    ood = 100.0 * table.background_ood.to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.1), sharex=True)
    x = np.arange(len(table))
    axes[0].bar(x, sigma, color=colors, edgecolor=NAVY, linewidth=0.5)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(ylabel=r"Relative $\sigma/\mu$ improvement [%]",
                title="Ten small models: time-OOF resolution result")
    for i, value in enumerate(sigma):
        axes[0].text(i, value + (0.035 if value >= 0 else -0.045), f"{value:.3f}",
                     ha="center", va="bottom" if value >= 0 else "top", fontsize=7)

    width = 0.36
    axes[1].bar(x - width / 2, applied, width, color=BLUE, label="Correction applied")
    axes[1].bar(x + width / 2, ood, width, color=RED, label="OOD")
    axes[1].axhline(10, color=NAVY, ls="--", lw=0.8, label="10% coverage gate")
    axes[1].set(ylabel="Background events [%]", title="Background coverage and OOD")
    axes[1].legend(ncol=3, loc="upper left")
    axes[1].set_xticks(x, names, rotation=0)
    fig.suptitle("v14 small-model screening (gold = all safety gates passed)", y=1.01)
    fig.tight_layout()
    save(fig, "v14_ten_model_comparison.png")


def spectrum(events, summary):
    before = events.energy_cor_kev.to_numpy(float)
    after = events.candidate_v14_energy_kev.to_numpy(float)
    valid = np.isfinite(before) & np.isfinite(after)
    before, after = before[valid], after[valid]

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.0))
    edges = np.arange(500, 3805, 5)
    axes[0].hist(before, bins=edges, histtype="step", lw=1.0, color=BLUE,
                 label=r"Before: cubic $E_{cor}$")
    axes[0].hist(after, bins=edges, histtype="step", lw=1.0, color=RED,
                 label=r"After: v14 M06")
    axes[0].set(xlim=(500, 3800), ylabel="Events / 5 keV",
                title="Winner background-spectrum safety check")
    axes[0].legend()

    zoom = np.arange(2500, 3002.5, 2.5)
    axes[1].hist(before, bins=zoom, histtype="step", lw=1.0, color=BLUE,
                 label=r"Before: cubic $E_{cor}$")
    axes[1].hist(after, bins=zoom, histtype="step", lw=1.0, color=RED,
                 label=r"After: v14 M06")
    axes[1].axvline(2600, color=GRAY, ls="--", lw=0.8)
    axes[1].axvline(2888, color=GRAY, ls="--", lw=0.8)
    axes[1].set(xlim=(2500, 3000), xlabel="Energy [keV]",
                ylabel="Events / 2.5 keV", title="High-energy zoom")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "v14_winner_spectrum.png")


def correction(events, summary):
    before = events.energy_cor_kev.to_numpy(float)
    after = events.candidate_v14_energy_kev.to_numpy(float)
    valid = np.isfinite(before) & np.isfinite(after) & (before > 0)
    before, after = before[valid], after[valid]
    rel = 100.0 * (after / before - 1.0)
    nz = np.abs(rel) > 1.0e-10
    rng = np.random.default_rng(140814)
    take = rng.choice(len(before), min(24000, len(before)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.55))
    axes[0].scatter(before[take], after[take], s=2, alpha=0.12, color=BLUE,
                    rasterized=True)
    axes[0].plot([500, 3800], [500, 3800], color="black", ls="--", lw=0.8)
    axes[0].set(xlim=(500, 3800), ylim=(500, 3800),
                xlabel=r"Before $E_{cor}$ [keV]", ylabel=r"After $E_{v14}$ [keV]",
                title="Event-by-event comparison")
    axes[1].hist(rel[nz], bins=np.linspace(-0.25, 0.25, 101),
                 histtype="stepfilled", color=RED, alpha=0.34, edgecolor=RED)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Relative correction [%]", ylabel="Corrected events",
                title="Applied correction amplitude")
    winner = summary["winner"]
    axes[1].text(0.04, 0.96,
                 "Applied: {:.2f}%\nOOD: {:.2f}%\nStrong fallback: {:.2f}%".format(
                     100 * winner["background_applied"],
                     100 * winner["background_ood"],
                     100 * winner["background_strong"],
                 ), transform=axes[1].transAxes, va="top")
    fig.tight_layout()
    save(fig, "v14_winner_correction.png")


def main():
    table = pd.read_csv(RESULTS / "ten_small_models.csv")
    events = pd.read_csv(RESULTS / "background_events.csv")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    comparison(table, summary)
    spectrum(events, summary)
    correction(events, summary)
    print("FIGURES=" + str(FIGURES))


if __name__ == "__main__":
    main()
