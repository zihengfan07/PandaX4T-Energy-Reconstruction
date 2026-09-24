#!/usr/bin/env python3
"""Create report figures from frozen v11 OOF outputs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = HERE.parent / "server_runs" / "run10972_foundation_environment_v11"
FIGURES = HERE / "figures"
DATA = ROOT / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#4C78A8"
ORANGE = "#F58518"
PURPLE = "#8F63B8"
GREEN = "#2CA02C"


def read_oof():
    with (RESULTS / "oof_events.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    def values(name):
        return np.asarray([float(row[name]) for row in rows], dtype=float)
    return rows, values


def energy_histogram(values):
    methods = [
        ("current_formula_energy", "Current linear formula", BLUE),
        ("legacy_energy_cor", "Existing cubic Energy_cor", ORANGE),
        ("selected_foundation_energy", "Selected image foundation", PURPLE),
        ("candidate_v11_energy", "candidate_v11", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    bins = np.linspace(2350, 3300, 105)
    for name, label, color in methods:
        ax.hist(values(name), bins=bins, density=True, histtype="step",
                linewidth=1.8, label=label, color=color)
    ax.axvline(2614.5, color="black", linestyle="--", linewidth=1,
               label="2614.5 keV")
    ax.set_xlabel("Reconstructed energy [keV]")
    ax.set_ylabel("Normalized density")
    ax.set_title("One-dimensional OOF energy comparison")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "v11_energy_1d.png", dpi=190)
    plt.close(fig)


def outer_blocks():
    with (RESULTS / "outer_block_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [f"{r['file_start']}-{int(r['file_stop'])-1}" for r in rows]
    series = [
        ("Current", [100 * float(r["current_sigma"]) for r in rows], BLUE),
        ("Legacy cubic", [100 * float(r["legacy_sigma"]) for r in rows], ORANGE),
        ("Selected foundation",
         [100 * float(r["foundation_sigma"]) for r in rows], PURPLE),
        ("candidate_v11",
         [100 * float(r["candidate_sigma"]) for r in rows], GREEN),
    ]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.3, 5.1))
    for index, (name, data, color) in enumerate(series):
        ax.bar(x + (index - 1.5) * width, data, width=width,
               label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Held-out fileNumber block")
    ax.set_ylabel(r"Median fit $\sigma/\mu$ [%]")
    ax.set_title("Performance on five unseen acquisition-file blocks")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "v11_outer_blocks.png", dpi=190)
    plt.close(fig)


def model_progress():
    labels = ["Current\nlinear", "Legacy\ncubic", "v10\n20 variables",
              "v11\nfoundation + env."]
    values = [4.1696, 4.1847, 2.7698, 0.9973]
    colors = [BLUE, ORANGE, "#72A0C1", GREEN]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.08,
                f"{value:.3f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 4.7)
    ax.set_ylabel(r"Merged OOF median $\sigma/\mu$ [%]")
    ax.set_title("Model-development result under grouped validation")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "v11_model_progress.png", dpi=190)
    plt.close(fig)


def correction(values):
    data = 100 * values("log_correction")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(data, bins=70, color=GREEN, alpha=0.75)
    ax.axvline(-10, color="black", linestyle="--", linewidth=1)
    ax.axvline(10, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Applied bounded log correction [%]")
    ax.set_ylabel("Events")
    ax.set_title("candidate_v11 bounded correction distribution")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "v11_correction.png", dpi=190)
    plt.close(fig)


def feature_importance():
    rows = json.loads(
        (RESULTS / "feature_importance.json").read_text(encoding="utf-8")
    )[:18]
    names = [row["feature"] for row in rows][::-1]
    values = [row["normalized_importance"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(9.1, 6.2))
    ax.barh(names, values, color=BLUE, alpha=0.9)
    ax.set_xlabel("Normalized grouped spline-coefficient norm")
    ax.set_title("candidate_v11 key-variable ranking")
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "v11_feature_importance.png", dpi=190)
    plt.close(fig)


def xy_bias(rows, values):
    wanted = {int(row["sourceRow"]) for row in rows}
    coordinates = {}
    with DATA.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if index in wanted:
                try:
                    coordinates[index] = (
                        float(row["xS2T_max"]), float(row["yS2T_max"])
                    )
                except ValueError:
                    coordinates[index] = (np.nan, np.nan)
    x = np.asarray([coordinates[int(row["sourceRow"])][0] for row in rows])
    y = np.asarray([coordinates[int(row["sourceRow"])][1] for row in rows])
    methods = [
        ("legacy_energy_cor", "Existing cubic Energy_cor"),
        ("selected_foundation_energy", "Selected image foundation"),
        ("candidate_v11_energy", "candidate_v11"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    for ax, (name, title) in zip(axes, methods):
        bias = values(name) / 2614.5 - 1.0
        scatter = ax.scatter(x, y, c=bias, s=7, cmap="coolwarm",
                             vmin=-0.08, vmax=0.08, alpha=0.75,
                             linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x_{\rm S2T}$ [mm]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel(r"$y_{\rm S2T}$ [mm]")
    fig.colorbar(scatter, ax=axes, shrink=0.82,
                 label="Per-event fractional energy bias")
    fig.suptitle("Two-dimensional OOF energy-bias comparison", y=1.02)
    fig.savefig(FIGURES / "v11_xy_bias.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def main():
    rows, values = read_oof()
    energy_histogram(values)
    outer_blocks()
    model_progress()
    correction(values)
    feature_importance()
    xy_bias(rows, values)
    print(json.dumps({"figures": 6, "events": len(rows)}))


if __name__ == "__main__":
    main()
