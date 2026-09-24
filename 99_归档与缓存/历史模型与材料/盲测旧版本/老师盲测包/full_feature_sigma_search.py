#!/usr/bin/env python3
"""Nested event-level search for a lower Gaussian-core sigma/mu.

The script is deliberately compatible with Python 3.6 and scikit-learn 0.24
so that it can run on the bl-0 HTCondor pool.  It never reads or uses
fileNumber.  All model and channel choices are made inside the outer training
sample; the final values are assembled from one out-of-fold prediction per
event.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


SEED = 10972

S1_CANDIDATES = [
    "qS1_max",
    "qS1C_max",
    "qS1ub_C",
    "qS1C_desImageMCPAF_maxS2",
    "qS1C_desImageMCPAF_maxS2forMS",
    "qS1C_ImageMCPAF_stretch_maxS2forMS",
    "qS1C_desImageMCPAF_firstS2forMS",
    "qS1C_ImageMCPAF_stretch_firstS2forMS",
]

S2_CANDIDATES = [
    "qS2B_max",
    "qS2BC_max",
    "qS2Bdes_max",
    "qS2Bub_C",
    "qS2Bdesub_C",
    "qS2Bub_C_stretch",
    "qS2Bdesub_C_stretch",
    "qS2BC_desImageMCPAF_maxS2",
    "qS2BdesC_desImageMCPAF_maxS2",
    "qS2BC_ImageMCPAF_total_stretch_maxS2forMS",
    "qS2BdesC_ImageMCPAF_total_stretch_maxS2forMS",
    "qS2BC_ImageMCPAF_total_stretch_firstS2forMS",
    "qS2BdesC_ImageMCPAF_total_stretch_firstS2forMS",
]

GEOMETRY_COLUMNS = [
    "dt",
    "xS2T_max",
    "yS2T_max",
    "xS2Tcor_max",
    "yS2Tcor_max",
    "xS2max_cdfTM",
    "yS2max_cdfTM",
    "xS2max_cdfTMs",
    "yS2max_cdfTMs",
    "xS2max_cdfPAF",
    "yS2max_cdfPAF",
    "xS2max_desCorCog_maxS2",
    "yS2max_desCorCog_maxS2",
    "xS2max_desImageMCPAF_maxS2",
    "yS2max_desImageMCPAF_maxS2",
]

SHAPE_COLUMNS = [
    "wS1_max",
    "wS2_max",
    "wS1CDF_max",
    "wS2CDF_max",
    "wS2FWHM_max",
    "widthTenS2_max",
    "tDiffBottomTopS1_max",
    "tDiffBottomTopS2_max",
    "pS1_max",
    "pS2_max",
    "hS1_max",
    "hS2_max",
    "nPMTS1_max",
    "nPMTS2_max",
    "rmsCogS1T_max",
    "rmsCogS1B_max",
]

TOPOLOGY_COLUMNS = [
    "nS1",
    "nPostS1",
    "nGoodS1",
    "nCandidateS1",
    "nS2",
    "nPostS2",
    "nRealPostS2",
    "nSignalBeforeS1max",
    "nSignalBetweenS1S2max",
    "nSignalAfterS2max",
    "ratioqS2PrePeak_max",
    "ratioqS2PrePeakSmr_max",
    "ratioqS1PrePeak_max",
    "ratioqS1PrePeakSmr_max",
    "qS1hitStdevTo1_max",
    "qS1channelStdevTo1_max",
    "qS2hitStdevTo1_max",
    "qS2channelStdevTo1_max",
]

ALPHAS = np.linspace(0.05, 0.95, 19)


def gaussian_plus_linear(x, amplitude, mu, sigma, offset, slope):
    return (
        amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        + offset
        + slope * (x - mu)
    )


def robust_width(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 100:
        return np.inf
    q16, q50, q84 = np.quantile(values, [0.15865, 0.5, 0.84135])
    return float((q84 - q16) / (2.0 * q50))


def gaussian_core_fit(values, bins=90):
    """Fit a Gaussian plus linear background in a fixed normalized window."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 200:
        return {
            "success": False,
            "sigma_over_mu": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "p_value": np.nan,
            "events": int(len(values)),
        }

    scale = np.median(values)
    normalized = values / scale
    counts0, edges0 = np.histogram(normalized, bins=160, range=(0.65, 1.35))
    centers0 = 0.5 * (edges0[:-1] + edges0[1:])
    mode = float(centers0[np.argmax(counts0)])
    half_window = 0.18
    low = max(0.55, mode - half_window)
    high = min(1.45, mode + half_window)
    counts, edges = np.histogram(normalized, bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts + 1.0)
    width0 = np.clip(robust_width(normalized), 0.012, 0.12)
    p0 = [
        max(float(np.max(counts) - np.median(counts)), 1.0),
        mode,
        width0,
        max(float(np.median(counts)), 0.0),
        0.0,
    ]
    bounds = (
        [0.0, low, 0.004, 0.0, -1.0e5],
        [np.inf, high, 0.25, np.inf, 1.0e5],
    )
    try:
        pars, _ = curve_fit(
            gaussian_plus_linear,
            centers,
            counts,
            p0=p0,
            sigma=errors,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=30000,
        )
        expected = gaussian_plus_linear(centers, *pars)
        expected = np.maximum(expected, 1.0e-9)
        chi2_value = float(np.sum((counts - expected) ** 2 / (expected + 1.0)))
        ndf = max(int(len(counts) - len(pars)), 1)
        p_value = float(chi2.sf(chi2_value, ndf))
        sigma_over_mu = float(abs(pars[2] / pars[1]))
        success = (
            np.isfinite(sigma_over_mu)
            and 0.004 < sigma_over_mu < 0.25
            and low < pars[1] < high
        )
        return {
            "success": bool(success),
            "sigma_over_mu": sigma_over_mu,
            "mu": float(pars[1] * scale),
            "sigma": float(abs(pars[2]) * scale),
            "p_value": p_value,
            "chi2": chi2_value,
            "ndf": ndf,
            "events": int(len(values)),
            "fit_low_normalized": low,
            "fit_high_normalized": high,
        }
    except Exception as exc:
        return {
            "success": False,
            "sigma_over_mu": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "p_value": np.nan,
            "events": int(len(values)),
            "error": str(exc),
        }


def normalized_combination(train, target, s1_name, s2_name, alpha):
    s1_scale = np.nanmedian(train[s1_name].to_numpy(dtype=float))
    s2_scale = np.nanmedian(train[s2_name].to_numpy(dtype=float))
    if not np.isfinite(s1_scale) or not np.isfinite(s2_scale):
        raise ValueError("Non-finite channel scale")
    if s1_scale <= 0 or s2_scale <= 0:
        raise ValueError("Non-positive channel scale")
    train_s1 = train[s1_name].to_numpy(dtype=float) / s1_scale
    train_s2 = train[s2_name].to_numpy(dtype=float) / s2_scale
    combination_scale = np.nanmedian(
        alpha * train_s1 + (1.0 - alpha) * train_s2
    )
    s1 = target[s1_name].to_numpy(dtype=float) / s1_scale
    s2 = target[s2_name].to_numpy(dtype=float) / s2_scale
    return (alpha * s1 + (1.0 - alpha) * s2) / combination_scale


def available_nonconstant(frame, names):
    result = []
    for name in names:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        # Candidate channels/features must be available for the complete fixed
        # analysis sample.  This prevents a seemingly better method from
        # gaining resolution by silently losing difficult events.
        if len(finite) == len(values) and np.nanstd(finite) > 0:
            result.append(name)
    return result


def add_derived_geometry(frame):
    frame = frame.copy()
    position_pairs = [
        ("raw", "xS2T_max", "yS2T_max"),
        ("cor", "xS2Tcor_max", "yS2Tcor_max"),
        ("cdf", "xS2max_cdfTM", "yS2max_cdfTM"),
        ("paf", "xS2max_cdfPAF", "yS2max_cdfPAF"),
        ("mcpaf", "xS2max_desImageMCPAF_maxS2", "yS2max_desImageMCPAF_maxS2"),
    ]
    for label, x_name, y_name in position_pairs:
        if x_name in frame.columns and y_name in frame.columns:
            x = frame[x_name].to_numpy(dtype=float)
            y = frame[y_name].to_numpy(dtype=float)
            frame["r2_" + label] = x * x + y * y
            frame["phi_" + label] = np.arctan2(y, x)
    return frame


def fill_from_training(train_x, target_x):
    train_x = np.asarray(train_x, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    medians = np.nanmedian(train_x, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    target_x = np.where(np.isfinite(target_x), target_x, medians)
    return train_x, target_x


def fit_residual_model(train, target, train_energy, target_energy, features):
    if not features:
        return target_energy.copy()
    train_x, target_x = fill_from_training(
        train[features].to_numpy(dtype=float),
        target[features].to_numpy(dtype=float),
    )
    center = np.median(train_energy)
    y = np.log(np.clip(train_energy / center, 1.0e-6, None))
    model = HistGradientBoostingRegressor(
        loss="least_absolute_deviation",
        learning_rate=0.045,
        max_iter=140,
        max_leaf_nodes=7,
        min_samples_leaf=80,
        l2_regularization=12.0,
        random_state=SEED,
    )
    model.fit(train_x, y)
    prediction_train = model.predict(train_x)
    prediction_target = model.predict(target_x)
    prediction_center = np.median(prediction_train)
    correction = np.exp(np.clip(prediction_target - prediction_center, -0.20, 0.20))
    return target_energy / correction


def balanced_folds(n_events, n_splits, seed):
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n_events)
    fold = np.empty(n_events, dtype=int)
    fold[permutation] = np.arange(n_events, dtype=int) % n_splits
    return fold


def select_base_candidate(train, s1_candidates, s2_candidates, outer_fold):
    inner_fold = balanced_folds(len(train), 4, SEED + 100 + outer_fold)
    rows = []
    for s1_name in s1_candidates:
        for s2_name in s2_candidates:
            for alpha in ALPHAS:
                oof = np.full(len(train), np.nan)
                valid = True
                for inner in range(4):
                    development = train.iloc[np.flatnonzero(inner_fold != inner)]
                    validation_index = np.flatnonzero(inner_fold == inner)
                    validation = train.iloc[validation_index]
                    try:
                        oof[validation_index] = normalized_combination(
                            development, validation, s1_name, s2_name, alpha
                        )
                    except ValueError:
                        valid = False
                        break
                if not valid or not np.isfinite(oof).all():
                    continue
                rows.append(
                    {
                        "s1": s1_name,
                        "s2": s2_name,
                        "alpha": float(alpha),
                        "inner_r68": robust_width(oof),
                    }
                )
    table = pd.DataFrame(rows).sort_values("inner_r68").reset_index(drop=True)
    if table.empty:
        raise RuntimeError("No valid S1/S2 candidate")

    # Perform the slower Gaussian fit only for the best robust-width candidates.
    finalists = []
    for _, row in table.head(30).iterrows():
        oof = np.full(len(train), np.nan)
        for inner in range(4):
            development = train.iloc[np.flatnonzero(inner_fold != inner)]
            validation_index = np.flatnonzero(inner_fold == inner)
            validation = train.iloc[validation_index]
            oof[validation_index] = normalized_combination(
                development,
                validation,
                row["s1"],
                row["s2"],
                float(row["alpha"]),
            )
        fit = gaussian_core_fit(oof)
        item = row.to_dict()
        item["inner_sigma_over_mu"] = fit["sigma_over_mu"]
        item["inner_fit_p"] = fit["p_value"]
        item["inner_fit_success"] = fit["success"]
        finalists.append(item)
    finalists = pd.DataFrame(finalists)
    accepted = finalists[
        finalists["inner_fit_success"]
        & np.isfinite(finalists["inner_sigma_over_mu"])
        & (finalists["inner_fit_p"] >= 0.01)
    ]
    if accepted.empty:
        accepted = finalists[
            finalists["inner_fit_success"]
            & np.isfinite(finalists["inner_sigma_over_mu"])
        ]
    selected = accepted.sort_values(
        ["inner_sigma_over_mu", "inner_r68"]
    ).iloc[0]
    return selected, table, finalists


def select_residual_group(train, selected, feature_groups, outer_fold):
    inner_fold = balanced_folds(len(train), 4, SEED + 500 + outer_fold)
    rows = []
    for group_name, features in feature_groups.items():
        oof = np.full(len(train), np.nan)
        for inner in range(4):
            development_index = np.flatnonzero(inner_fold != inner)
            validation_index = np.flatnonzero(inner_fold == inner)
            development = train.iloc[development_index]
            validation = train.iloc[validation_index]
            development_energy = normalized_combination(
                development,
                development,
                selected["s1"],
                selected["s2"],
                float(selected["alpha"]),
            )
            validation_energy = normalized_combination(
                development,
                validation,
                selected["s1"],
                selected["s2"],
                float(selected["alpha"]),
            )
            oof[validation_index] = fit_residual_model(
                development,
                validation,
                development_energy,
                validation_energy,
                features,
            )
        fit = gaussian_core_fit(oof)
        rows.append(
            {
                "group": group_name,
                "feature_count": len(features),
                "inner_r68": robust_width(oof),
                "inner_sigma_over_mu": fit["sigma_over_mu"],
                "inner_fit_p": fit["p_value"],
                "inner_fit_success": fit["success"],
            }
        )
    table = pd.DataFrame(rows)
    accepted = table[
        table["inner_fit_success"]
        & np.isfinite(table["inner_sigma_over_mu"])
        & (table["inner_fit_p"] >= 0.01)
    ]
    none_row = table[table["group"] == "none"].iloc[0]
    if accepted.empty:
        selected_group = none_row
    else:
        candidate = accepted.sort_values(
            ["inner_sigma_over_mu", "inner_r68", "feature_count"]
        ).iloc[0]
        # Require a visible inner-validation gain in sigma/mu and no R68
        # deterioration.  Otherwise retain the simpler uncorrected estimate.
        sigma_gain = (
            none_row["inner_sigma_over_mu"] - candidate["inner_sigma_over_mu"]
        )
        r68_ok = candidate["inner_r68"] <= 1.002 * none_row["inner_r68"]
        if (
            candidate["group"] != "none"
            and sigma_gain >= 0.001
            and r68_ok
        ):
            selected_group = candidate
        else:
            selected_group = none_row
    return selected_group, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(args.input, sep="\t", nrows=0)
    requested = set(
        ["runNumber", "eventNumber"]
        + S1_CANDIDATES
        + S2_CANDIDATES
        + GEOMETRY_COLUMNS
        + SHAPE_COLUMNS
        + TOPOLOGY_COLUMNS
    )
    usecols = [name for name in header.columns if name in requested]
    frame = pd.read_csv(args.input, sep="\t", usecols=usecols, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    # One input event has an enormous negative corrected S2 value, for which
    # the existing Energy formula is itself non-physical.  Define this fixed
    # upstream-validity condition before any model comparison and apply it to
    # every method equally.
    fixed_validity = (
        np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    rejected_source_rows = frame.loc[~fixed_validity, "sourceRow"].astype(int).tolist()
    frame = frame.loc[fixed_validity].reset_index(drop=True)
    frame = add_derived_geometry(frame)

    s1_candidates = available_nonconstant(frame, S1_CANDIDATES)
    s2_candidates = available_nonconstant(frame, S2_CANDIDATES)
    if not s1_candidates or not s2_candidates:
        raise RuntimeError("No usable S1 or S2 candidates")

    derived_geometry = [
        name for name in frame.columns
        if name.startswith("r2_") or name.startswith("phi_")
    ]
    geometry = available_nonconstant(frame, GEOMETRY_COLUMNS + derived_geometry)
    shape = available_nonconstant(frame, SHAPE_COLUMNS)
    topology = available_nonconstant(frame, TOPOLOGY_COLUMNS)
    feature_groups = {
        "none": [],
        "geometry": geometry,
        "geometry_shape": geometry + shape,
        "geometry_shape_topology": geometry + shape + topology,
    }

    n_events = len(frame)
    outer_fold = balanced_folds(n_events, 5, SEED)
    oof_base = np.full(n_events, np.nan)
    oof_corrected = np.full(n_events, np.nan)
    fold_rows = []
    base_finalist_tables = []
    residual_tables = []

    for outer in range(5):
        train_index = np.flatnonzero(outer_fold != outer)
        test_index = np.flatnonzero(outer_fold == outer)
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        selected, _, finalists = select_base_candidate(
            train, s1_candidates, s2_candidates, outer
        )
        finalists.insert(0, "outer_fold", outer)
        base_finalist_tables.append(finalists)
        selected_group, residual_table = select_residual_group(
            train, selected, feature_groups, outer
        )
        residual_table.insert(0, "outer_fold", outer)
        residual_tables.append(residual_table)

        train_energy = normalized_combination(
            train,
            train,
            selected["s1"],
            selected["s2"],
            float(selected["alpha"]),
        )
        test_energy = normalized_combination(
            train,
            test,
            selected["s1"],
            selected["s2"],
            float(selected["alpha"]),
        )
        corrected = fit_residual_model(
            train,
            test,
            train_energy,
            test_energy,
            feature_groups[selected_group["group"]],
        )
        oof_base[test_index] = test_energy
        oof_corrected[test_index] = corrected
        base_fit = gaussian_core_fit(test_energy)
        corrected_fit = gaussian_core_fit(corrected)
        fold_rows.append(
            {
                "outer_fold": outer,
                "events": int(len(test_index)),
                "s1": selected["s1"],
                "s2": selected["s2"],
                "alpha": float(selected["alpha"]),
                "residual_group": selected_group["group"],
                "base_sigma_over_mu": base_fit["sigma_over_mu"],
                "base_fit_p": base_fit["p_value"],
                "corrected_sigma_over_mu": corrected_fit["sigma_over_mu"],
                "corrected_fit_p": corrected_fit["p_value"],
                "base_r68": robust_width(test_energy),
                "corrected_r68": robust_width(corrected),
            }
        )

    current_energy = (
        frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    ) * 0.0137
    current_energy_cor = (
        -1.73706e-09 * current_energy ** 3
        + 7.98193e-06 * current_energy ** 2
        + 1.07904 * current_energy
        - 9.22086
    )

    methods = {
        "current_energy": current_energy,
        "current_energy_cor": current_energy_cor,
        "nested_oof_best_channels": oof_base,
        "nested_oof_residual_corrected": oof_corrected,
    }
    summary_rows = []
    fit_details = {}
    for name, values in methods.items():
        fit = gaussian_core_fit(values)
        fit_details[name] = fit
        summary_rows.append(
            {
                "method": name,
                "events": int(np.isfinite(values).sum()),
                "sigma_over_mu": fit["sigma_over_mu"],
                "fit_p_value": fit["p_value"],
                "fit_success": fit["success"],
                "r68": robust_width(values),
            }
        )

    event_output = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    event_output["outer_fold"] = outer_fold
    event_output["current_energy"] = current_energy
    event_output["current_energy_cor"] = current_energy_cor
    event_output["oof_best_channels"] = oof_base
    event_output["oof_residual_corrected"] = oof_corrected
    event_output.to_csv(output / "oof_events.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_results.csv", index=False)
    pd.concat(base_finalist_tables, ignore_index=True).to_csv(
        output / "base_finalists.csv", index=False
    )
    pd.concat(residual_tables, ignore_index=True).to_csv(
        output / "residual_group_selection.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    metadata = {
        "metric": "Gaussian-core sigma/mu from a Gaussian plus linear-background fit",
        "seed": SEED,
        "events": n_events,
        "fixed_validity_rule": "qS1ub_C > 0 and qS2Bdesub_C > 0, both finite",
        "rejected_source_rows": rejected_source_rows,
        "outer_validation": "balanced seeded event-level five-fold OOF",
        "ignored_identity": "fileNumber is not read, modeled, split on, diagnosed, or resampled",
        "s1_candidates": s1_candidates,
        "s2_candidates": s2_candidates,
        "feature_groups": feature_groups,
        "fit_details": fit_details,
        "warning": (
            "A small sigma/mu is reportable only when the fit shape is acceptable "
            "and the result survives an independent run. This run is exploratory."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
