#!/usr/bin/env python3
"""Create report figures from frozen v10 CSV/JSON outputs only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "server_runs" / "run10972_sparse_grouped_v10"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
NOMINAL = 2614.5


def load():
    events = pd.read_csv(RESULTS / "oof_events.csv")
    diagnostic = pd.read_csv(RESULTS / "diagnostic_features.csv")
    outer = pd.read_csv(RESULTS / "outer_block_metrics.csv")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    nested = json.loads(
        (RESULTS / "nested_sparse_selection.json").read_text(encoding="utf-8")
    )
    merged = events.merge(diagnostic, on="sourceRow", how="left", validate="one_to_one")
    return merged, outer, summary, nested


def energy_histogram(data):
    methods = [
        ("current_formula_energy", "Current linear formula", "#4C78A8"),
        ("legacy_corrected_energy", "Existing cubic Energy_cor", "#F58518"),
        ("candidate_energy", "Sparse candidate_v10", "#2CA02C"),
    ]
    combined = np.concatenate([data[column].to_numpy() for column, _, _ in methods])
    low, high = np.quantile(combined[np.isfinite(combined)], [0.01, 0.99])
    bins = np.linspace(max(1800, low), min(3300, high), 100)
    fig, ax = plt.subplots(figsize=(9.3, 5.3))
    for column, label, color in methods:
        ax.hist(
            data[column],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.1,
            label=label,
            color=color,
        )
    ax.axvline(NOMINAL, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Reconstructed energy [keV]")
    ax.set_ylabel("Normalized event density")
    ax.set_title("One-dimensional corrected-energy comparison (grouped OOF)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_1d_comparison.png", dpi=190)
    plt.close(fig)


def outer_blocks(outer):
    x = np.arange(len(outer))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar(
        x - width,
        100 * outer["current_sigma"],
        width,
        label="Current linear",
        color="#4C78A8",
    )
    ax.bar(
        x,
        100 * outer["legacy_cor_sigma"],
        width,
        label="Existing cubic Energy_cor",
        color="#F58518",
    )
    ax.bar(
        x + width,
        100 * outer["candidate_sigma"],
        width,
        label="Sparse v10",
        color="#2CA02C",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["0-199", "200-399", "400-599", "600-799", "800-999"])
    ax.set_xlabel("Held-out fileNumber block")
    ax.set_ylabel(r"12-protocol median $\sigma/\mu$ [%]")
    ax.set_title("Performance on five unseen acquisition blocks")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "outer_block_performance.png", dpi=190)
    plt.close(fig)


def spatial_map(x, y, energy, edges):
    result = np.full((len(edges) - 1, len(edges) - 1), np.nan)
    xb = np.digitize(x, edges) - 1
    yb = np.digitize(y, edges) - 1
    residual = energy / NOMINAL - 1.0
    for ix in range(len(edges) - 1):
        for iy in range(len(edges) - 1):
            mask = (xb == ix) & (yb == iy)
            if np.sum(mask) >= 8:
                result[iy, ix] = np.median(residual[mask])
    return result


def xy_maps(data):
    # Raw S2 top-array reconstruction covers the physical cross-section more
    # uniformly than the supplied corrected-coordinate branch in this sample.
    x = data["xS2T_max"].to_numpy()
    y = data["yS2T_max"].to_numpy()
    legacy = data["legacy_corrected_energy"].to_numpy() / NOMINAL - 1.0
    candidate = data["candidate_energy"].to_numpy() / NOMINAL - 1.0
    panels = [
        (legacy, "Existing Energy_cor", -0.08, 0.08),
        (candidate, "Sparse candidate_v10", -0.08, 0.08),
        (candidate - legacy, "v10 minus existing", -0.05, 0.05),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))
    for ax, (values, title, vmin, vmax) in zip(axes, panels):
        image = ax.scatter(
            x,
            y,
            c=values,
            s=7,
            linewidths=0,
            alpha=0.75,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(r"$x_{\rm S2,raw}$ [mm]")
        ax.set_ylabel(r"$y_{\rm S2,raw}$ [mm]")
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Median fractional energy bias",
        )
    fig.suptitle("Two-dimensional x-y energy-response comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "xy_energy_bias_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def binned(values, energy, bins=10):
    finite = np.isfinite(values) & np.isfinite(energy)
    values = values[finite]
    energy = energy[finite]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    x, y = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values <= high)
        if np.sum(mask) >= 10:
            x.append(np.median(values[mask]))
            y.append(100 * np.median(energy[mask] / NOMINAL - 1))
    return np.asarray(x), np.asarray(y)


def variable_trends(data, summary):
    features = summary["final_selected_features"][:4]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1))
    for ax, feature in zip(axes.ravel(), features):
        for column, label, color in [
            ("legacy_corrected_energy", "Existing Energy_cor", "#F58518"),
            ("candidate_energy", "Sparse v10", "#2CA02C"),
        ]:
            x, y = binned(data[feature].to_numpy(), data[column].to_numpy())
            ax.plot(x, y, marker="o", linewidth=1.7, color=color, label=label)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel(feature)
        ax.set_ylabel("Median energy bias [%]")
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Top-variable trends before and after v10 correction", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "top_variable_trends.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def importance(summary):
    rows = summary["final_feature_ranking"][:15]
    labels = [row["feature"] for row in rows][::-1]
    values = np.asarray([row["importance"] for row in rows], dtype=float)
    values = (values / values.max())[::-1]
    frequency = summary["feature_selection_frequency"]
    freq = np.asarray([frequency[row["feature"]] for row in rows])[::-1]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4), sharey=True)
    axes[0].barh(y, values, color="#4C78A8")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("Normalized grouped coefficient importance")
    axes[0].grid(axis="x", alpha=0.22)
    axes[1].barh(y, freq, color="#72B7B2")
    axes[1].set_xlim(0, 5)
    axes[1].set_xlabel("Number of outer folds selecting feature")
    axes[1].grid(axis="x", alpha=0.22)
    fig.suptitle("Key-variable importance and selection stability")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def k_selection(nested):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for record in nested:
        k_values = []
        sigma_values = []
        for candidate in record["k_candidates"]:
            k_values.append(candidate["k"])
            sigma_values.append(
                100
                * np.median(
                    [row["candidate_sigma"] for row in candidate["fold_rows"]]
                )
            )
        ax.plot(
            k_values,
            sigma_values,
            marker="o",
            alpha=0.75,
            label="outer block {}".format(record["outer_block"]),
        )
        selected = record["winner_k"]
        selected_y = sigma_values[k_values.index(selected)]
        ax.scatter([selected], [selected_y], color="black", s=42, zorder=5)
    ax.set_xticks([3, 5, 8, 12, 20, 30])
    ax.set_xlabel("Number of retained variables")
    ax.set_ylabel(r"Inner-validation median $\sigma/\mu$ [%]")
    ax.set_title("Why the final model cannot be reduced to only a few variables")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "variable_count_selection.png", dpi=190)
    plt.close(fig)


def correction_distribution(data):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.hist(
        100 * data["log_correction"],
        bins=70,
        color="#59A14F",
        alpha=0.82,
    )
    ax.axvline(-10, color="black", linestyle="--", linewidth=1)
    ax.axvline(10, color="black", linestyle="--", linewidth=1)
    near = 100 * data["correction_near_cap"].mean()
    ax.text(
        0.98,
        0.94,
        "Near-bound fraction: {:.2f}%".format(near),
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.set_xlabel("Applied bounded log correction [%]")
    ax.set_ylabel("Events")
    ax.set_title("Smooth bounded correction distribution")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "correction_distribution.png", dpi=190)
    plt.close(fig)


def main():
    data, outer, summary, nested = load()
    energy_histogram(data)
    outer_blocks(outer)
    xy_maps(data)
    variable_trends(data, summary)
    importance(summary)
    k_selection(nested)
    correction_distribution(data)
    print("created figures in", FIGURES)


if __name__ == "__main__":
    main()
