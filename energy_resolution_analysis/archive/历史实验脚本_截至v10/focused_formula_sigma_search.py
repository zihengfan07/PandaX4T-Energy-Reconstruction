#!/usr/bin/env python3
"""Focused nested search around the current PandaX energy formula.

Compatible with Python 3.6 / scikit-learn 0.24.  The search keeps one fixed
upstream-valid event sample and chooses every coefficient/model inside the
outer training data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingRegressor

from full_feature_sigma_search import (
    balanced_folds,
    fill_from_training,
    gaussian_core_fit,
)


SEED = 10972
S2_CANDIDATES = [
    "qS2Bub_C",
    "qS2Bdesub_C",
    "qS2Bub_C_stretch",
    "qS2Bdesub_C_stretch",
    "qS2BC_max",
    "qS2Bdes_max",
]
K_VALUES = np.round(np.linspace(0.75, 1.30, 56), 2)

GEOMETRY = [
    "dt",
    "xS2T_max",
    "yS2T_max",
    "xS2Tcor_max",
    "yS2Tcor_max",
]
SHAPE = [
    "wS1_max",
    "wS2_max",
    "wS2CDF_max",
    "wS2FWHM_max",
    "tDiffBottomTopS1_max",
    "tDiffBottomTopS2_max",
    "nPMTS1_max",
    "nPMTS2_max",
]


def formula(frame, s2_name, k_value):
    return (
        k_value * frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame[s2_name].to_numpy(dtype=float) / 10.58
    )


def metrics(values):
    values = np.asarray(values, dtype=float)
    fit = gaussian_core_fit(values)
    q05, q16, q50, q84, q95 = np.quantile(
        values[np.isfinite(values) & (values > 0)],
        [0.05, 0.15865, 0.5, 0.84135, 0.95],
    )
    return {
        "sigma_over_mu": fit["sigma_over_mu"],
        "fit_p": fit["p_value"],
        "r68": float((q84 - q16) / (2.0 * q50)),
        "r90": float((q95 - q05) / (2.0 * q50)),
    }


def is_better(candidate, baseline, min_sigma_gain=0.0):
    return (
        np.isfinite(candidate["sigma_over_mu"])
        and candidate["sigma_over_mu"]
        <= baseline["sigma_over_mu"] - min_sigma_gain
        and candidate["r68"] <= 1.002 * baseline["r68"]
        and candidate["r90"] <= 1.002 * baseline["r90"]
    )


def choose_formula(train, usable_s2):
    baseline = metrics(formula(train, "qS2Bdesub_C", 1.0))
    rows = []
    for s2_name in usable_s2:
        for k_value in K_VALUES:
            score = metrics(formula(train, s2_name, float(k_value)))
            row = {
                "s2": s2_name,
                "k": float(k_value),
                "sigma_over_mu": score["sigma_over_mu"],
                "fit_p": score["fit_p"],
                "r68": score["r68"],
                "r90": score["r90"],
                "tail_safe": (
                    score["r68"] <= 1.002 * baseline["r68"]
                    and score["r90"] <= 1.002 * baseline["r90"]
                ),
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    allowed = table[table["tail_safe"] & np.isfinite(table["sigma_over_mu"])]
    if allowed.empty:
        return {"s2": "qS2Bdesub_C", "k": 1.0}, table
    best = allowed.sort_values(
        ["sigma_over_mu", "r68", "r90", "s2", "k"]
    ).iloc[0]
    return {"s2": best["s2"], "k": float(best["k"])}, table


def fit_correction(train, target, train_energy, target_energy, features, leaves):
    train_x, target_x = fill_from_training(
        train[features].to_numpy(dtype=float),
        target[features].to_numpy(dtype=float),
    )
    center = np.median(train_energy)
    response = np.log(np.clip(train_energy / center, 1.0e-6, None))
    model = HistGradientBoostingRegressor(
        loss="least_absolute_deviation",
        learning_rate=0.035,
        max_iter=100,
        max_leaf_nodes=leaves,
        min_samples_leaf=140,
        l2_regularization=25.0,
        random_state=SEED,
    )
    model.fit(train_x, response)
    train_prediction = model.predict(train_x)
    target_prediction = model.predict(target_x)
    correction = np.exp(
        np.clip(
            target_prediction - np.median(train_prediction),
            -0.10,
            0.10,
        )
    )
    return target_energy / correction


def choose_correction(train, selected_formula, feature_groups, outer):
    inner_fold = balanced_folds(len(train), 4, SEED + 800 + outer)
    baseline_oof = np.full(len(train), np.nan)
    for inner in range(4):
        validation = np.flatnonzero(inner_fold == inner)
        baseline_oof[validation] = formula(
            train.iloc[validation],
            selected_formula["s2"],
            selected_formula["k"],
        )
    baseline_score = metrics(baseline_oof)
    rows = [
        {
            "group": "none",
            "leaves": 0,
            **baseline_score
        }
    ]
    for group_name, features in feature_groups.items():
        for leaves in [3, 5]:
            oof = np.full(len(train), np.nan)
            for inner in range(4):
                development_index = np.flatnonzero(inner_fold != inner)
                validation_index = np.flatnonzero(inner_fold == inner)
                development = train.iloc[development_index]
                validation = train.iloc[validation_index]
                development_energy = formula(
                    development,
                    selected_formula["s2"],
                    selected_formula["k"],
                )
                validation_energy = formula(
                    validation,
                    selected_formula["s2"],
                    selected_formula["k"],
                )
                oof[validation_index] = fit_correction(
                    development,
                    validation,
                    development_energy,
                    validation_energy,
                    features,
                    leaves,
                )
            score = metrics(oof)
            rows.append(
                {
                    "group": group_name,
                    "leaves": leaves,
                    **score
                }
            )
    table = pd.DataFrame(rows)
    allowed = table[
        (table["group"] != "none")
        & (table["sigma_over_mu"] <= baseline_score["sigma_over_mu"] - 0.0005)
        & (table["r68"] <= 1.002 * baseline_score["r68"])
        & (table["r90"] <= 1.002 * baseline_score["r90"])
    ]
    if allowed.empty:
        return {"group": "none", "leaves": 0}, table
    best = allowed.sort_values(
        ["sigma_over_mu", "r68", "r90", "leaves"]
    ).iloc[0]
    return {"group": best["group"], "leaves": int(best["leaves"])}, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    requested = (
        ["runNumber", "eventNumber", "qS1ub_C"]
        + S2_CANDIDATES
        + GEOMETRY
        + SHAPE
    )
    header = pd.read_csv(args.input, sep="\t", nrows=0)
    usecols = [name for name in requested if name in header.columns]
    frame = pd.read_csv(args.input, sep="\t", usecols=usecols, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    fixed_validity = (
        np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    rejected = frame.loc[~fixed_validity, "sourceRow"].astype(int).tolist()
    frame = frame.loc[fixed_validity].reset_index(drop=True)

    usable_s2 = []
    for name in S2_CANDIDATES:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=float)
        if np.isfinite(values).all() and (values > 0).all():
            usable_s2.append(name)

    features_geometry = [
        name for name in GEOMETRY
        if name in frame.columns and np.isfinite(frame[name]).all()
    ]
    features_shape = [
        name for name in SHAPE
        if name in frame.columns and np.isfinite(frame[name]).all()
    ]
    feature_groups = {
        "geometry": features_geometry,
        "geometry_shape": features_geometry + features_shape,
    }

    outer_fold = balanced_folds(len(frame), 5, SEED)
    current = formula(frame, "qS2Bdesub_C", 1.0)
    oof_formula = np.full(len(frame), np.nan)
    oof_corrected = np.full(len(frame), np.nan)
    fold_rows = []
    scan_tables = []
    correction_tables = []

    for outer in range(5):
        train_index = np.flatnonzero(outer_fold != outer)
        test_index = np.flatnonzero(outer_fold == outer)
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        selected_formula, scan = choose_formula(train, usable_s2)
        scan.insert(0, "outer_fold", outer)
        scan_tables.append(scan)
        selected_correction, correction_table = choose_correction(
            train, selected_formula, feature_groups, outer
        )
        correction_table.insert(0, "outer_fold", outer)
        correction_tables.append(correction_table)

        train_energy_raw = formula(
            train, selected_formula["s2"], selected_formula["k"]
        )
        test_energy_raw = formula(
            test, selected_formula["s2"], selected_formula["k"]
        )
        # Different outer folds may select slightly different coefficients.
        # Put every held-out fold onto a common relative scale using only the
        # corresponding outer-training median.
        formula_train_fit = gaussian_core_fit(train_energy_raw)
        formula_scale = formula_train_fit["mu"]
        if not np.isfinite(formula_scale) or formula_scale <= 0:
            formula_scale = np.median(train_energy_raw)
        test_energy = test_energy_raw / formula_scale
        if selected_correction["group"] == "none":
            corrected = test_energy.copy()
        else:
            corrected_raw = fit_correction(
                train,
                test,
                train_energy_raw,
                test_energy_raw,
                feature_groups[selected_correction["group"]],
                selected_correction["leaves"],
            )
            corrected_train_raw = fit_correction(
                train,
                train,
                train_energy_raw,
                train_energy_raw,
                feature_groups[selected_correction["group"]],
                selected_correction["leaves"],
            )
            corrected_train_fit = gaussian_core_fit(corrected_train_raw)
            corrected_scale = corrected_train_fit["mu"]
            if not np.isfinite(corrected_scale) or corrected_scale <= 0:
                corrected_scale = np.median(corrected_train_raw)
            corrected = corrected_raw / corrected_scale
        oof_formula[test_index] = test_energy
        oof_corrected[test_index] = corrected
        current_test = formula(test, "qS2Bdesub_C", 1.0)
        current_score = metrics(current_test)
        formula_score = metrics(test_energy)
        corrected_score = metrics(corrected)
        fold_rows.append(
            {
                "outer_fold": outer,
                "events": len(test),
                "s2": selected_formula["s2"],
                "k": selected_formula["k"],
                "correction_group": selected_correction["group"],
                "leaves": selected_correction["leaves"],
                "current_sigma_over_mu": current_score["sigma_over_mu"],
                "formula_sigma_over_mu": formula_score["sigma_over_mu"],
                "corrected_sigma_over_mu": corrected_score["sigma_over_mu"],
                "current_r68": current_score["r68"],
                "formula_r68": formula_score["r68"],
                "corrected_r68": corrected_score["r68"],
                "current_r90": current_score["r90"],
                "formula_r90": formula_score["r90"],
                "corrected_r90": corrected_score["r90"],
            }
        )

    summary_rows = []
    for name, values in [
        ("current_formula", current),
        ("nested_oof_weight_formula", oof_formula),
        ("nested_oof_weight_plus_correction", oof_corrected),
    ]:
        score = metrics(values)
        score["method"] = name
        score["events"] = int(np.isfinite(values).sum())
        summary_rows.append(score)

    events = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    events["outer_fold"] = outer_fold
    events["current_formula"] = current
    events["oof_weight_formula"] = oof_formula
    events["oof_weight_plus_correction"] = oof_corrected
    events.to_csv(output / "oof_events.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_results.csv", index=False)
    pd.concat(scan_tables, ignore_index=True).to_csv(
        output / "coefficient_scan.csv", index=False
    )
    pd.concat(correction_tables, ignore_index=True).to_csv(
        output / "correction_selection.csv", index=False
    )
    metadata = {
        "events": len(frame),
        "rejected_source_rows": rejected,
        "formula_family": (
            "E = 0.0137 * (k*qS1ub_C/0.125 + S2_candidate/10.58)"
        ),
        "outer_validation": "balanced seeded event-level five-fold OOF",
        "seed": SEED,
        "usable_s2": usable_s2,
        "feature_groups": feature_groups,
        "selection_constraints": (
            "lower sigma/mu with R68 and R90 no more than 0.2% relatively worse"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
