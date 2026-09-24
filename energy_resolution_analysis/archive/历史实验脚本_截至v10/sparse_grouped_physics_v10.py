#!/usr/bin/env python3
"""Sparse grouped PandaX energy correction and teacher-requested diagnostics.

Feature ranking and the number of retained variables are selected only inside
each outer training pool.  Complete 200-file acquisition blocks remain the
smallest validation unit.  ``fileNumber`` is grouping metadata, never a model
input.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

from grouped_nested_physics_v9 import (
    BLOCK_WIDTH,
    CONFIGS,
    NOMINAL_ENERGY_KEV,
    OUTER_BLOCKS,
    SEED,
    base_energy,
    choose_features,
    evaluate_split,
    fit_predictor,
    fit_protocol,
    predict_with_state,
    prepare_frame,
    protocol_metrics,
    selection_score,
)


SPARSE_CONFIG = [
    config for config in CONFIGS
    if config["name"] == "spline_ridge_a100_c100_soft"
][0]
K_CANDIDATES = [3, 5, 8, 12, 20, 30]


def legacy_energy_cor(frame):
    energy = 0.0137 * base_energy(frame)
    return (
        -1.73706e-09 * energy ** 3
        + 7.98193e-06 * energy ** 2
        + 1.07904 * energy
        - 9.22086
    )


def training_scale(values):
    fit = fit_protocol(values, "linear", 0.18)
    center = fit["mu"] if fit.get("success") else float(np.median(values))
    return NOMINAL_ENERGY_KEV / center


def rank_features(frame, all_features, train_index, seed):
    x = frame.iloc[train_index][all_features].to_numpy(dtype=float)
    energy = base_energy(frame)[train_index]
    state = fit_predictor(x, energy, SPARSE_CONFIG, seed)
    coefficients = np.asarray(state["model"].coef_, dtype=float)
    n_features = len(all_features)
    if len(coefficients) % n_features != 0:
        raise RuntimeError("Unexpected spline coefficient layout")
    blocks = coefficients.reshape((-1, n_features))
    importance = np.sqrt(np.sum(blocks ** 2, axis=0))
    order = np.argsort(-importance)
    rows = [
        {
            "feature": all_features[int(index)],
            "importance": float(importance[int(index)]),
            "rank": int(rank + 1),
        }
        for rank, index in enumerate(order)
    ]
    return [row["feature"] for row in rows], rows


def select_k(frame, all_features, allowed_blocks, seed_offset):
    records = {k: [] for k in K_CANDIDATES}
    ranking_records = []
    for validation_block in allowed_blocks:
        training_blocks = [
            block for block in allowed_blocks if block != validation_block
        ]
        train_index = np.flatnonzero(
            frame["outerBlock"].isin(training_blocks).to_numpy()
        )
        test_index = np.flatnonzero(
            (frame["outerBlock"] == validation_block).to_numpy()
        )
        ranking, rows = rank_features(
            frame,
            all_features,
            train_index,
            SEED + seed_offset + 1000 + validation_block,
        )
        ranking_records.append(
            {
                "validation_block": int(validation_block),
                "training_blocks": training_blocks,
                "ranking": rows,
            }
        )
        for k in K_CANDIDATES:
            selected = ranking[: min(k, len(ranking))]
            result = evaluate_split(
                frame,
                selected,
                train_index,
                test_index,
                SPARSE_CONFIG,
                SEED + seed_offset + 100 * k + validation_block,
            )
            candidate = result["candidate"]
            baseline = result["baseline_metrics"]
            records[k].append(
                {
                    "validation_block": int(validation_block),
                    "features": selected,
                    "score": selection_score(result),
                    "candidate_sigma": candidate["sigma_median"],
                    "baseline_sigma": baseline["sigma_median"],
                    "candidate_r68": candidate["r68"],
                    "baseline_r68": baseline["r68"],
                    "candidate_r90": candidate["r90"],
                    "baseline_r90": baseline["r90"],
                    "center_bias_max": candidate["center_bias_max"],
                    "near_cap_fraction": float(np.mean(result["touched"])),
                }
            )
    summaries = []
    for k in K_CANDIDATES:
        fold_rows = records[k]
        eligible = all(
            np.isfinite(row["candidate_sigma"])
            and row["candidate_sigma"] < row["baseline_sigma"]
            and row["candidate_r90"] <= 1.02 * row["baseline_r90"]
            and row["center_bias_max"] <= 0.005
            for row in fold_rows
        )
        summaries.append(
            {
                "k": k,
                "median_score": float(np.median(
                    [row["score"] for row in fold_rows]
                )),
                "worst_score": float(np.max(
                    [row["score"] for row in fold_rows]
                )),
                "eligible": bool(eligible),
                "fold_rows": fold_rows,
            }
        )
    eligible = [row for row in summaries if row["eligible"]]
    pool = eligible if eligible else summaries
    winner = sorted(
        pool, key=lambda row: (row["median_score"], row["k"])
    )[0]
    return int(winner["k"]), summaries, ranking_records


def plot_energy_histograms(events, output):
    methods = [
        ("current_formula_energy", "Current linear formula", "#4C78A8"),
        ("legacy_corrected_energy", "Existing cubic Energy_cor", "#F58518"),
        ("candidate_energy", "candidate_v10", "#2CA02C"),
    ]
    values = np.concatenate(
        [events[column].to_numpy(dtype=float) for column, _, _ in methods]
    )
    low, high = np.quantile(values[np.isfinite(values)], [0.01, 0.99])
    low = max(1800.0, low)
    high = min(3300.0, high)
    bins = np.linspace(low, high, 95)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for column, label, color in methods:
        ax.hist(
            events[column],
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            label=label,
            color=color,
        )
    ax.axvline(NOMINAL_ENERGY_KEV, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Reconstructed energy [keV]")
    ax.set_ylabel("Normalized event density")
    ax.set_title("One-dimensional corrected-energy comparison (grouped OOF)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(str(output), dpi=180)
    plt.close(fig)


def spatial_median_map(x, y, energy, edges):
    result = np.full((len(edges) - 1, len(edges) - 1), np.nan)
    x_bin = np.digitize(x, edges) - 1
    y_bin = np.digitize(y, edges) - 1
    residual = energy / NOMINAL_ENERGY_KEV - 1.0
    for ix in range(len(edges) - 1):
        for iy in range(len(edges) - 1):
            mask = (x_bin == ix) & (y_bin == iy)
            if np.sum(mask) >= 8:
                result[iy, ix] = np.median(residual[mask])
    return result


def plot_xy_comparison(frame, events, output):
    x = frame["xS2Tcor_max"].to_numpy(dtype=float)
    y = frame["yS2Tcor_max"].to_numpy(dtype=float)
    edges = np.linspace(-400, 400, 17)
    legacy = spatial_median_map(
        x, y, events["legacy_corrected_energy"].to_numpy(dtype=float), edges
    )
    candidate = spatial_median_map(
        x, y, events["candidate_energy"].to_numpy(dtype=float), edges
    )
    difference = candidate - legacy
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [
        (legacy, "Existing Energy_cor", -0.08, 0.08, "coolwarm"),
        (candidate, "candidate_v10", -0.08, 0.08, "coolwarm"),
        (difference, "v10 minus existing", -0.05, 0.05, "coolwarm"),
    ]
    for ax, (values, title, vmin, vmax, cmap) in zip(axes, panels):
        image = ax.imshow(
            values,
            origin="lower",
            extent=[edges[0], edges[-1], edges[0], edges[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel(r"$x_{\rm S2,cor}$ [mm]")
        ax.set_ylabel(r"$y_{\rm S2,cor}$ [mm]")
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Median fractional energy bias",
        )
    fig.suptitle("Two-dimensional x-y comparison after correction", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output), dpi=180, bbox_inches="tight")
    plt.close(fig)


def binned_trend(values, energy, bins=10):
    finite = np.isfinite(values) & np.isfinite(energy)
    values = values[finite]
    energy = energy[finite]
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    centers = []
    medians = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values <= high)
        if np.sum(mask) < 10:
            continue
        centers.append(float(np.median(values[mask])))
        medians.append(float(
            np.median(energy[mask] / NOMINAL_ENERGY_KEV - 1.0)
        ))
    return np.asarray(centers), np.asarray(medians)


def plot_top_variable_trends(frame, events, top_features, output):
    selected = top_features[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    for ax, feature in zip(axes.ravel(), selected):
        values = frame[feature].to_numpy(dtype=float)
        for column, label, color in [
            ("legacy_corrected_energy", "Existing Energy_cor", "#F58518"),
            ("candidate_energy", "candidate_v10", "#2CA02C"),
        ]:
            x, y = binned_trend(
                values,
                events[column].to_numpy(dtype=float),
            )
            ax.plot(x, 100.0 * y, marker="o", linewidth=1.6, label=label, color=color)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Median energy bias [%]")
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Energy trend before and after v10 correction", y=1.01)
    fig.tight_layout()
    fig.savefig(str(output), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(rank_rows, selection_frequency, output):
    top = rank_rows[:12]
    labels = [row["feature"] for row in top][::-1]
    importance = np.asarray([row["importance"] for row in top], dtype=float)
    importance = (importance / np.max(importance))[::-1]
    frequency = np.asarray(
        [selection_frequency.get(row["feature"], 0) for row in top],
        dtype=float,
    )[::-1]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True)
    axes[0].barh(y, importance, color="#4C78A8")
    axes[0].set_xlabel("Normalized grouped coefficient importance")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].grid(axis="x", alpha=0.22)
    axes[1].barh(y, frequency, color="#72B7B2")
    axes[1].set_xlabel("Outer folds selecting feature")
    axes[1].set_xlim(0, OUTER_BLOCKS)
    axes[1].grid(axis="x", alpha=0.22)
    fig.suptitle("Key-variable ranking and outer-fold stability")
    fig.tight_layout()
    fig.savefig(str(output), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, rejected = prepare_frame(args.input)
    all_features = choose_features(frame)
    n = len(frame)
    current_oof = np.full(n, np.nan)
    legacy_oof = np.full(n, np.nan)
    candidate_oof = np.full(n, np.nan)
    correction_oof = np.full(n, np.nan)
    near_cap_oof = np.zeros(n, dtype=bool)
    selected_k_oof = np.zeros(n, dtype=int)
    outer_rows = []
    selection_records = []
    feature_frequency = {feature: 0 for feature in all_features}
    outer_selected_features = []
    all_energy = base_energy(frame)
    all_legacy = legacy_energy_cor(frame)

    for outer_block in range(OUTER_BLOCKS):
        allowed_blocks = [
            block for block in range(OUTER_BLOCKS) if block != outer_block
        ]
        winner_k, k_records, inner_rankings = select_k(
            frame,
            all_features,
            allowed_blocks,
            seed_offset=10000 * outer_block,
        )
        train_index = np.flatnonzero(
            (frame["outerBlock"] != outer_block).to_numpy()
        )
        test_index = np.flatnonzero(
            (frame["outerBlock"] == outer_block).to_numpy()
        )
        outer_ranking, outer_rank_rows = rank_features(
            frame,
            all_features,
            train_index,
            SEED + 50000 + outer_block,
        )
        selected = outer_ranking[:winner_k]
        outer_selected_features.append(selected)
        for feature in selected:
            feature_frequency[feature] += 1
        result = evaluate_split(
            frame,
            selected,
            train_index,
            test_index,
            SPARSE_CONFIG,
            SEED + 60000 + outer_block,
        )
        current_oof[test_index] = result["baseline"]
        candidate_oof[test_index] = result["prediction"]
        correction_oof[test_index] = result["correction"]
        near_cap_oof[test_index] = result["touched"]
        selected_k_oof[test_index] = winner_k
        legacy_oof[test_index] = (
            all_legacy[test_index] * training_scale(all_legacy[train_index])
        )
        candidate = result["candidate"]
        baseline = result["baseline_metrics"]
        legacy_metrics = protocol_metrics(legacy_oof[test_index])
        outer_rows.append(
            {
                "outer_block": outer_block,
                "file_low": outer_block * BLOCK_WIDTH,
                "file_high_exclusive": (outer_block + 1) * BLOCK_WIDTH,
                "events": int(len(test_index)),
                "selected_k": winner_k,
                "selected_features": ";".join(selected),
                "current_sigma": baseline["sigma_median"],
                "legacy_cor_sigma": legacy_metrics["sigma_median"],
                "candidate_sigma": candidate["sigma_median"],
                "candidate_r68": candidate["r68"],
                "candidate_r90": candidate["r90"],
                "candidate_center_bias_max": candidate["center_bias_max"],
                "candidate_near_cap_fraction": float(
                    np.mean(result["touched"])
                ),
            }
        )
        selection_records.append(
            {
                "outer_block": outer_block,
                "winner_k": winner_k,
                "outer_training_ranking": outer_rank_rows,
                "k_candidates": k_records,
                "inner_rankings": inner_rankings,
            }
        )
        print(
            "outer {} k={} current {:.5f} legacy {:.5f} v10 {:.5f}".format(
                outer_block,
                winner_k,
                baseline["sigma_median"],
                legacy_metrics["sigma_median"],
                candidate["sigma_median"],
            )
        )

    if not np.all(np.isfinite(candidate_oof)):
        raise RuntimeError("Incomplete sparse OOF prediction")

    identity = frame[
        ["sourceRow", "runNumber", "fileNumber", "eventNumber", "outerBlock"]
    ].copy()
    identity["current_formula_energy"] = current_oof
    identity["legacy_corrected_energy"] = legacy_oof
    identity["candidate_energy"] = candidate_oof
    identity["log_correction"] = correction_oof
    identity["correction_near_cap"] = near_cap_oof
    identity["selected_k"] = selected_k_oof
    identity.to_csv(output_dir / "oof_events.csv", index=False)
    diagnostic_columns = list(dict.fromkeys(
        ["sourceRow", "xS2Tcor_max", "yS2Tcor_max"] + all_features
    ))
    frame[diagnostic_columns].to_csv(
        output_dir / "diagnostic_features.csv", index=False
    )
    pd.DataFrame(outer_rows).to_csv(
        output_dir / "outer_block_metrics.csv", index=False
    )
    (output_dir / "nested_sparse_selection.json").write_text(
        json.dumps(selection_records, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Final deployable sparse identity selected without using a new run.
    final_k, final_k_records, final_inner_rankings = select_k(
        frame,
        all_features,
        list(range(OUTER_BLOCKS)),
        seed_offset=90000,
    )
    full_index = np.arange(n, dtype=int)
    final_ranking, final_rank_rows = rank_features(
        frame, all_features, full_index, SEED + 99000
    )
    final_features = final_ranking[:final_k]
    final_state = fit_predictor(
        frame[final_features].to_numpy(dtype=float),
        all_energy,
        SPARSE_CONFIG,
        SEED + 99999,
    )
    bundle = {
        "format_version": 3,
        "model_name": "PandaX_sparse_grouped_physics_v10",
        "training_run": 10972,
        "training_events": int(n),
        "feature_names": final_features,
        "selected_k": final_k,
        "feature_ranking": final_rank_rows,
        "group_key": "fileNumber",
        "group_key_is_predictor": False,
        "base_formula": "qS1ub_C/0.125 + qS2Bdesub_C/10.58",
        "baseline_energy_scale": training_scale(all_energy),
        "legacy_formula": (
            "E=0.0137*base; "
            "Energy_cor=-1.73706e-09*E^3+7.98193e-06*E^2+1.07904*E-9.22086"
        ),
        "legacy_energy_scale": training_scale(all_legacy),
        "nominal_energy_kev": NOMINAL_ENERGY_KEV,
        "selected_config": SPARSE_CONFIG,
        "predictor_state": final_state,
        "warning": (
            "Validated across complete file blocks inside run10972 only. "
            "Independent-run blind validation is required."
        ),
    }
    joblib.dump(bundle, str(output_dir / "candidate_v10.joblib"), compress=3)

    outer_table = pd.DataFrame(outer_rows)
    metrics = {
        "current_formula": protocol_metrics(current_oof),
        "legacy_energy_cor": protocol_metrics(legacy_oof),
        "candidate_v10": protocol_metrics(candidate_oof),
    }
    compact_metrics = {
        method: {
            key: value for key, value in result.items()
            if key != "protocol_rows"
        }
        for method, result in metrics.items()
    }
    summary = {
        "events": int(n),
        "files": int(frame["fileNumber"].nunique()),
        "rejected_source_rows": rejected,
        "all_features": all_features,
        "outer_selected_k": outer_table["selected_k"].astype(int).tolist(),
        "outer_selected_features": outer_selected_features,
        "feature_selection_frequency": feature_frequency,
        "final_selected_k": final_k,
        "final_selected_features": final_features,
        "final_feature_ranking": final_rank_rows,
        "merged_oof_metrics": compact_metrics,
        "outer_median_current_sigma": float(
            np.median(outer_table["current_sigma"])
        ),
        "outer_median_legacy_sigma": float(
            np.median(outer_table["legacy_cor_sigma"])
        ),
        "outer_median_candidate_sigma": float(
            np.median(outer_table["candidate_sigma"])
        ),
        "all_outer_candidate_better_than_current": bool(
            np.all(outer_table["candidate_sigma"] < outer_table["current_sigma"])
        ),
        "all_outer_candidate_better_than_legacy": bool(
            np.all(
                outer_table["candidate_sigma"]
                < outer_table["legacy_cor_sigma"]
            )
        ),
        "independent_run_validated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "final_sparse_selection.json").write_text(
        json.dumps(
            {
                "winner_k": final_k,
                "winner_features": final_features,
                "k_candidates": final_k_records,
                "inner_rankings": final_inner_rankings,
                "full_ranking": final_rank_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if HAS_MATPLOTLIB:
        plot_energy_histograms(identity, output_dir / "energy_1d_comparison.png")
        plot_xy_comparison(
            frame, identity, output_dir / "xy_energy_bias_comparison.png"
        )
        plot_top_variable_trends(
            frame,
            identity,
            final_features,
            output_dir / "top_variable_trends.png",
        )
        plot_feature_importance(
            final_rank_rows,
            feature_frequency,
            output_dir / "feature_importance.png",
        )
    else:
        print("matplotlib unavailable; numerical outputs completed without plots")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
