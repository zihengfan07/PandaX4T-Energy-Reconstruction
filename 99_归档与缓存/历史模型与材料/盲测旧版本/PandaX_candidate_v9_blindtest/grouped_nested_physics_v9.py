#!/usr/bin/env python3
"""Grouped nested validation for a physics-constrained PandaX energy correction.

The acquisition ``fileNumber`` is used only as a grouping/audit key.  It is
never exposed to a predictor.  Five contiguous 200-file blocks form the outer
validation.  Candidate selection is repeated inside every outer training pool
using the remaining complete blocks.

Compatible with Python 3.6, NumPy/Pandas/SciPy and scikit-learn 0.24.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


SEED = 10972
NOMINAL_ENERGY_KEV = 2614.5
BLOCK_WIDTH = 200
OUTER_BLOCKS = 5

# These variables describe detector geometry, pulse shape, PMT participation,
# and event topology.  Absolute S1/S2 charge, run/file/event IDs and absolute
# time are deliberately absent.  Therefore, with these features held fixed,
# E_corrected remains monotonic in both qS1ub_C and qS2Bdesub_C.
RAW_FEATURES = [
    "dt",
    "xS2T_max",
    "yS2T_max",
    "xS2Tcor_max",
    "yS2Tcor_max",
    "xS2max_desImageMCPAF_maxS2",
    "yS2max_desImageMCPAF_maxS2",
    "wS1_max",
    "wS2_max",
    "wS1CDF_max",
    "wS2CDF_max",
    "wS2FWHM_max",
    "widthTenS2_max",
    "tDiffBottomTopS1_max",
    "tDiffBottomTopS2_max",
    "nPMTS1_max",
    "nPMTS2_max",
    "ratioqS2PrePeak_max",
    "ratioqS2PrePeakSmr_max",
    "ratioqS1PrePeak_max",
    "ratioqS1PrePeakSmr_max",
    "qS1hitStdevTo1_max",
    "qS1channelStdevTo1_max",
    "qS2hitStdevTo1_max",
    "qS2channelStdevTo1_max",
    "rmsCogS1T_max",
    "rmsCogS1B_max",
]

CONFIGS = [
    {
        "name": "linear_ridge_a10_c075_soft",
        "kind": "linear_ridge",
        "alpha": 10.0,
        "clip": 0.075,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_ridge_a30_c075_soft",
        "kind": "spline_ridge",
        "alpha": 30.0,
        "clip": 0.075,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_ridge_a100_c075_soft",
        "kind": "spline_ridge",
        "alpha": 100.0,
        "clip": 0.075,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_ridge_a100_c100_soft",
        "kind": "spline_ridge",
        "alpha": 100.0,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
    {
        "name": "hgb_conservative_c075_soft",
        "kind": "hgb",
        "leaves": 3,
        "min_leaf": 160,
        "l2": 35.0,
        "iterations": 120,
        "clip": 0.075,
        "cap_transform": "tanh",
    },
    {
        "name": "hgb_balanced_c100_soft",
        "kind": "hgb",
        "leaves": 5,
        "min_leaf": 130,
        "l2": 25.0,
        "iterations": 140,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
]

HALF_WINDOWS = [0.08, 0.12, 0.16, 0.20]
BACKGROUNDS = ["none", "constant", "linear"]


def base_energy(frame):
    return (
        frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    )


def add_geometry(frame):
    frame = frame.copy()
    for label, x_name, y_name in [
        ("raw", "xS2T_max", "yS2T_max"),
        ("cor", "xS2Tcor_max", "yS2Tcor_max"),
        (
            "mcpaf",
            "xS2max_desImageMCPAF_maxS2",
            "yS2max_desImageMCPAF_maxS2",
        ),
    ]:
        if x_name in frame.columns and y_name in frame.columns:
            x = frame[x_name].to_numpy(dtype=float)
            y = frame[y_name].to_numpy(dtype=float)
            frame["r2_" + label] = x * x + y * y
    return frame


def prepare_frame(path):
    header = pd.read_csv(path, sep="\t", nrows=0)
    required = [
        "runNumber",
        "fileNumber",
        "eventNumber",
        "qS1ub_C",
        "qS2Bdesub_C",
    ]
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    requested = required + RAW_FEATURES
    usecols = [name for name in requested if name in header.columns]
    frame = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    validity = (
        np.isfinite(frame["fileNumber"])
        & np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    rejected = frame.loc[~validity, "sourceRow"].astype(int).tolist()
    frame = frame.loc[validity].reset_index(drop=True)
    frame["fileNumber"] = frame["fileNumber"].astype(int)
    frame["outerBlock"] = np.clip(
        frame["fileNumber"].to_numpy(dtype=int) // BLOCK_WIDTH,
        0,
        OUTER_BLOCKS - 1,
    )
    return add_geometry(frame), rejected


def choose_features(frame):
    requested = list(RAW_FEATURES) + ["r2_raw", "r2_cor", "r2_mcpaf"]
    features = []
    for name in requested:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) > 0 and np.std(finite) > 0:
            features.append(name)
    return features


def robust_width(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 50:
        return np.nan
    q16, q50, q84 = np.quantile(values, [0.15865, 0.5, 0.84135])
    return float((q84 - q16) / (2.0 * q50))


def tail_metrics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 50:
        return {"r68": np.nan, "r90": np.nan}
    q05, q16, q50, q84, q95 = np.quantile(
        values, [0.05, 0.15865, 0.5, 0.84135, 0.95]
    )
    return {
        "r68": float((q84 - q16) / (2.0 * q50)),
        "r90": float((q95 - q05) / (2.0 * q50)),
    }


def gaussian_none(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_constant(x, amplitude, mu, sigma, offset):
    return gaussian_none(x, amplitude, mu, sigma) + offset


def gaussian_linear(x, amplitude, mu, sigma, offset, slope):
    return gaussian_none(x, amplitude, mu, sigma) + offset + slope * (x - mu)


def fit_protocol(values, background, half_window, bins=70):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 150:
        return {"success": False}
    scale = float(np.median(values))
    normalized = values / scale
    counts0, edges0 = np.histogram(normalized, bins=180, range=(0.60, 1.40))
    centers0 = 0.5 * (edges0[:-1] + edges0[1:])
    mode = float(centers0[np.argmax(counts0)])
    low = max(0.50, mode - half_window)
    high = min(1.50, mode + half_window)
    counts, edges = np.histogram(normalized, bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts + 1.0)
    width0 = np.clip(robust_width(normalized), 0.008, 0.15)
    amplitude = max(float(np.max(counts) - np.median(counts)), 1.0)
    offset = max(float(np.median(counts)), 0.0)
    if background == "none":
        function = gaussian_none
        p0 = [amplitude, mode, width0]
        bounds = ([0.0, low, 0.003], [np.inf, high, 0.25])
    elif background == "constant":
        function = gaussian_constant
        p0 = [amplitude, mode, width0, offset]
        bounds = ([0.0, low, 0.003, 0.0], [np.inf, high, 0.25, np.inf])
    else:
        function = gaussian_linear
        p0 = [amplitude, mode, width0, offset, 0.0]
        bounds = (
            [0.0, low, 0.003, 0.0, -1.0e5],
            [np.inf, high, 0.25, np.inf, 1.0e5],
        )
    try:
        pars, _ = curve_fit(
            function,
            centers,
            counts,
            p0=p0,
            sigma=errors,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=30000,
        )
        expected = np.maximum(function(centers, *pars), 1.0e-9)
        chi2_value = float(np.sum((counts - expected) ** 2 / (expected + 1.0)))
        ndf = max(int(len(counts) - len(pars)), 1)
        ratio = float(abs(pars[2] / pars[1]))
        mu = float(pars[1] * scale)
        success = (
            np.isfinite(ratio)
            and np.isfinite(mu)
            and 0.003 < ratio < 0.25
            and low < pars[1] < high
        )
        return {
            "success": bool(success),
            "sigma_over_mu": ratio,
            "mu": mu,
            "p_value": float(chi2.sf(chi2_value, ndf)),
            "background": background,
            "half_window": half_window,
        }
    except Exception as exc:
        return {
            "success": False,
            "background": background,
            "half_window": half_window,
            "error": str(exc),
        }


def protocol_metrics(values):
    rows = []
    for background in BACKGROUNDS:
        for half_window in HALF_WINDOWS:
            rows.append(fit_protocol(values, background, half_window))
    successful = [row for row in rows if row.get("success")]
    tail = tail_metrics(values)
    if not successful:
        return {
            "protocol_successes": 0,
            "sigma_median": np.nan,
            "sigma_worst": np.nan,
            "center_bias_max": np.nan,
            "fit_p_median": np.nan,
            "r68": tail["r68"],
            "r90": tail["r90"],
            "protocol_rows": rows,
        }
    widths = np.asarray([row["sigma_over_mu"] for row in successful])
    centers = np.asarray([row["mu"] for row in successful])
    p_values = np.asarray([row["p_value"] for row in successful])
    return {
        "protocol_successes": len(successful),
        "sigma_median": float(np.median(widths)),
        "sigma_worst": float(np.max(widths)),
        "center_bias_max": float(
            np.max(np.abs(centers / NOMINAL_ENERGY_KEV - 1.0))
        ),
        "fit_p_median": float(np.median(p_values)),
        "r68": tail["r68"],
        "r90": tail["r90"],
        "protocol_rows": rows,
    }


def peak_weights(energy):
    log_energy = np.log(np.clip(np.asarray(energy, dtype=float), 1.0e-9, None))
    center = float(np.median(log_energy))
    q16, q84 = np.quantile(log_energy, [0.15865, 0.84135])
    scale = max(float((q84 - q16) / 2.0), 1.0e-3)
    z = (log_energy - center) / scale
    weights = np.exp(-0.5 * (z / 2.5) ** 2)
    return np.maximum(weights, 0.05)


def weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))


def bounded_delta(raw_delta, clip_value, transform):
    if transform == "tanh":
        return clip_value * np.tanh(raw_delta / clip_value)
    return np.clip(raw_delta, -clip_value, clip_value)


def prepare_linear_features(train_x, test_x, spline):
    medians = np.nanmedian(train_x, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(train_x), train_x, medians)
    test = np.where(np.isfinite(test_x), test_x, medians)
    q16 = np.quantile(train, 0.15865, axis=0)
    q84 = np.quantile(train, 0.84135, axis=0)
    scales = (q84 - q16) / 2.0
    scales[~np.isfinite(scales) | (scales < 1.0e-9)] = 1.0
    centers = np.median(train, axis=0)
    z_train = np.clip((train - centers) / scales, -5.0, 5.0)
    z_test = np.clip((test - centers) / scales, -5.0, 5.0)
    if spline:
        train_parts = [z_train, z_train ** 2, z_train ** 3]
        test_parts = [z_test, z_test ** 2, z_test ** 3]
        for knot in [-1.5, -0.5, 0.5, 1.5]:
            train_parts.append(np.maximum(z_train - knot, 0.0) ** 3)
            test_parts.append(np.maximum(z_test - knot, 0.0) ** 3)
        design_train = np.concatenate(train_parts, axis=1)
        design_test = np.concatenate(test_parts, axis=1)
    else:
        design_train = z_train
        design_test = z_test
    design_center = np.mean(design_train, axis=0)
    design_scale = np.std(design_train, axis=0)
    design_scale[~np.isfinite(design_scale) | (design_scale < 1.0e-9)] = 1.0
    design_train = (design_train - design_center) / design_scale
    design_test = (design_test - design_center) / design_scale
    state = {
        "medians": medians,
        "centers": centers,
        "scales": scales,
        "design_center": design_center,
        "design_scale": design_scale,
        "spline": bool(spline),
    }
    return design_train, design_test, state


def transform_linear_features(x, state):
    values = np.where(np.isfinite(x), x, state["medians"])
    z = np.clip((values - state["centers"]) / state["scales"], -5.0, 5.0)
    if state["spline"]:
        parts = [z, z ** 2, z ** 3]
        for knot in [-1.5, -0.5, 0.5, 1.5]:
            parts.append(np.maximum(z - knot, 0.0) ** 3)
        design = np.concatenate(parts, axis=1)
    else:
        design = z
    return (design - state["design_center"]) / state["design_scale"]


def fit_predictor(train_x, train_energy, config, seed):
    weights = peak_weights(train_energy)
    log_energy = np.log(np.clip(train_energy, 1.0e-9, None))
    target_center = weighted_mean(log_energy, weights)
    target = log_energy - target_center
    kind = config["kind"]
    if kind in ("linear_ridge", "spline_ridge"):
        design, _, transform = prepare_linear_features(
            train_x, train_x[:0], kind == "spline_ridge"
        )
        model = Ridge(alpha=config["alpha"], fit_intercept=True)
        model.fit(design, target, sample_weight=weights)
        train_prediction = model.predict(design)
        state = {
            "kind": kind,
            "model": model,
            "transform": transform,
        }
    else:
        medians = np.nanmedian(train_x, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        design = np.where(np.isfinite(train_x), train_x, medians)
        model = HistGradientBoostingRegressor(
            loss="least_absolute_deviation",
            learning_rate=0.035,
            max_iter=config["iterations"],
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["min_leaf"],
            l2_regularization=config["l2"],
            random_state=seed,
        )
        model.fit(design, target)
        train_prediction = model.predict(design)
        state = {
            "kind": kind,
            "model": model,
            "medians": medians,
        }
    prediction_center = weighted_mean(train_prediction, weights)
    raw_delta = train_prediction - prediction_center
    bounded = bounded_delta(
        raw_delta, config["clip"], config.get("cap_transform", "hard")
    )
    correction = np.exp(bounded)
    corrected_train = train_energy / correction
    train_fit = fit_protocol(corrected_train, "linear", 0.18)
    if train_fit.get("success"):
        training_mu = train_fit["mu"]
    else:
        training_mu = float(np.median(corrected_train))
    state.update(
        {
            "prediction_center": prediction_center,
            "energy_scale": NOMINAL_ENERGY_KEV / training_mu,
            "clip": config["clip"],
            "cap_transform": config.get("cap_transform", "hard"),
            "config": dict(config),
        }
    )
    return state


def predict_with_state(state, x, energy):
    if state["kind"] in ("linear_ridge", "spline_ridge"):
        design = transform_linear_features(x, state["transform"])
    else:
        design = np.where(np.isfinite(x), x, state["medians"])
    raw_delta = state["model"].predict(design) - state["prediction_center"]
    bounded = bounded_delta(
        raw_delta, state["clip"], state.get("cap_transform", "hard")
    )
    correction = np.exp(bounded)
    output = energy / correction * state["energy_scale"]
    near_cap = np.abs(bounded) >= 0.95 * state["clip"]
    return output, bounded, near_cap


def calibrated_baseline(train_energy, test_energy):
    fit = fit_protocol(train_energy, "linear", 0.18)
    if fit.get("success"):
        train_mu = fit["mu"]
    else:
        train_mu = float(np.median(train_energy))
    return test_energy * (NOMINAL_ENERGY_KEV / train_mu)


def evaluate_split(frame, features, train_index, test_index, config, seed):
    energy = base_energy(frame)
    train_x = frame.iloc[train_index][features].to_numpy(dtype=float)
    test_x = frame.iloc[test_index][features].to_numpy(dtype=float)
    state = fit_predictor(train_x, energy[train_index], config, seed)
    prediction, correction, touched = predict_with_state(
        state, test_x, energy[test_index]
    )
    baseline = calibrated_baseline(energy[train_index], energy[test_index])
    candidate_metrics = protocol_metrics(prediction)
    baseline_metrics = protocol_metrics(baseline)
    finite_positive = bool(
        np.all(np.isfinite(prediction)) and np.all(prediction > 0)
    )
    return {
        "state": state,
        "prediction": prediction,
        "baseline": baseline,
        "correction": correction,
        "touched": touched,
        "candidate": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "finite_positive": finite_positive,
    }


def selection_score(result):
    candidate = result["candidate"]
    baseline = result["baseline_metrics"]
    if (
        not result["finite_positive"]
        or candidate["protocol_successes"] < 9
        or not np.isfinite(candidate["sigma_median"])
    ):
        return 1.0e6
    score = candidate["sigma_median"]
    score += 0.30 * candidate["r68"] + 0.10 * candidate["r90"]
    score += 8.0 * max(candidate["center_bias_max"] - 0.003, 0.0)
    near_cap_fraction = float(np.mean(result["touched"]))
    score += 1.5 * max(near_cap_fraction - 0.02, 0.0)
    score += 2.0 * max(
        candidate["r90"] - 1.02 * baseline["r90"], 0.0
    )
    return float(score)


def select_config(frame, features, allowed_blocks, seed_offset):
    energy = base_energy(frame)
    rows = []
    for config_index, config in enumerate(CONFIGS):
        fold_rows = []
        for validation_block in allowed_blocks:
            train_index = np.flatnonzero(
                frame["outerBlock"].isin(
                    [block for block in allowed_blocks if block != validation_block]
                ).to_numpy()
            )
            test_index = np.flatnonzero(
                (frame["outerBlock"] == validation_block).to_numpy()
            )
            result = evaluate_split(
                frame,
                features,
                train_index,
                test_index,
                config,
                SEED + seed_offset + 100 * config_index + validation_block,
            )
            candidate = result["candidate"]
            baseline = result["baseline_metrics"]
            fold_rows.append(
                {
                    "validation_block": int(validation_block),
                    "score": selection_score(result),
                    "candidate_sigma": candidate["sigma_median"],
                    "baseline_sigma": baseline["sigma_median"],
                    "candidate_r68": candidate["r68"],
                    "baseline_r68": baseline["r68"],
                    "candidate_r90": candidate["r90"],
                    "baseline_r90": baseline["r90"],
                    "center_bias_max": candidate["center_bias_max"],
                    "near_cap_fraction": float(np.mean(result["touched"])),
                    "protocol_successes": candidate["protocol_successes"],
                    "finite_positive": result["finite_positive"],
                }
            )
        valid_scores = [row["score"] for row in fold_rows]
        all_blocks_improve = all(
            np.isfinite(row["candidate_sigma"])
            and row["candidate_sigma"] < row["baseline_sigma"]
            for row in fold_rows
        )
        center_pass = all(
            np.isfinite(row["center_bias_max"])
            and row["center_bias_max"] <= 0.005
            for row in fold_rows
        )
        tail_pass = all(
            row["candidate_r90"] <= 1.02 * row["baseline_r90"]
            for row in fold_rows
        )
        rows.append(
            {
                "config": config["name"],
                "median_score": float(np.median(valid_scores)),
                "worst_score": float(np.max(valid_scores)),
                "all_blocks_improve": bool(all_blocks_improve),
                "center_pass": bool(center_pass),
                "tail_pass": bool(tail_pass),
                "eligible": bool(
                    all_blocks_improve and center_pass and tail_pass
                ),
                "fold_rows": fold_rows,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible if eligible else rows
    winner = sorted(
        pool, key=lambda row: (row["median_score"], row["worst_score"], row["config"])
    )[0]
    config_by_name = {config["name"]: config for config in CONFIGS}
    return config_by_name[winner["config"]], rows


def cluster_bootstrap_r68(events, iterations=1000):
    rng = np.random.RandomState(SEED + 9001)
    files = np.asarray(sorted(events["fileNumber"].unique()), dtype=int)
    groups = {
        file_number: events.loc[
            events["fileNumber"] == file_number,
            ["baseline_energy", "candidate_energy"],
        ].to_numpy(dtype=float)
        for file_number in files
    }
    deltas = []
    for _ in range(iterations):
        selected = rng.choice(files, size=len(files), replace=True)
        sample = np.concatenate([groups[int(number)] for number in selected], axis=0)
        delta = robust_width(sample[:, 1]) - robust_width(sample[:, 0])
        deltas.append(delta)
    return np.asarray(deltas, dtype=float)


def serializable_protocol_rows(rows, outer_block, method):
    output = []
    for row in rows:
        item = dict(row)
        item["outer_block"] = int(outer_block)
        item["method"] = method
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, rejected = prepare_frame(args.input)
    features = choose_features(frame)
    forbidden = {
        "runNumber",
        "fileNumber",
        "eventNumber",
        "t",
        "qS1ub_C",
        "qS2Bdesub_C",
    }
    overlap = sorted(forbidden.intersection(features))
    if overlap:
        raise RuntimeError("Forbidden predictor features found: " + repr(overlap))

    energies = base_energy(frame)
    oof_candidate = np.full(len(frame), np.nan)
    oof_baseline = np.full(len(frame), np.nan)
    oof_correction = np.full(len(frame), np.nan)
    oof_touched = np.zeros(len(frame), dtype=bool)
    selected_config = np.empty(len(frame), dtype=object)
    outer_rows = []
    selection_records = []
    protocol_rows = []

    for outer_block in range(OUTER_BLOCKS):
        allowed_blocks = [
            block for block in range(OUTER_BLOCKS) if block != outer_block
        ]
        winner, records = select_config(
            frame, features, allowed_blocks, seed_offset=10000 * outer_block
        )
        selection_records.append(
            {
                "outer_block": outer_block,
                "winner": winner["name"],
                "candidates": records,
            }
        )
        train_index = np.flatnonzero(
            (frame["outerBlock"] != outer_block).to_numpy()
        )
        test_index = np.flatnonzero(
            (frame["outerBlock"] == outer_block).to_numpy()
        )
        result = evaluate_split(
            frame,
            features,
            train_index,
            test_index,
            winner,
            SEED + 50000 + outer_block,
        )
        oof_candidate[test_index] = result["prediction"]
        oof_baseline[test_index] = result["baseline"]
        oof_correction[test_index] = result["correction"]
        oof_touched[test_index] = result["touched"]
        selected_config[test_index] = winner["name"]
        candidate = result["candidate"]
        baseline = result["baseline_metrics"]
        outer_rows.append(
            {
                "outer_block": outer_block,
                "file_low": int(outer_block * BLOCK_WIDTH),
                "file_high_exclusive": int((outer_block + 1) * BLOCK_WIDTH),
                "events": int(len(test_index)),
                "selected_config": winner["name"],
                "baseline_sigma_median": baseline["sigma_median"],
                "candidate_sigma_median": candidate["sigma_median"],
                "relative_sigma_improvement": (
                    1.0 - candidate["sigma_median"] / baseline["sigma_median"]
                ),
                "baseline_r68": baseline["r68"],
                "candidate_r68": candidate["r68"],
                "baseline_r90": baseline["r90"],
                "candidate_r90": candidate["r90"],
                "candidate_center_bias_max": candidate["center_bias_max"],
                "candidate_fit_p_median": candidate["fit_p_median"],
                "candidate_protocol_successes": candidate["protocol_successes"],
                "near_cap_fraction": float(np.mean(result["touched"])),
                "finite_positive": result["finite_positive"],
            }
        )
        protocol_rows.extend(
            serializable_protocol_rows(
                baseline["protocol_rows"], outer_block, "baseline"
            )
        )
        protocol_rows.extend(
            serializable_protocol_rows(
                candidate["protocol_rows"], outer_block, "candidate"
            )
        )
        print(
            "outer block {} winner {} sigma {:.6f} -> {:.6f}".format(
                outer_block,
                winner["name"],
                baseline["sigma_median"],
                candidate["sigma_median"],
            )
        )

    if not (
        np.all(np.isfinite(oof_candidate))
        and np.all(np.isfinite(oof_baseline))
    ):
        raise RuntimeError("OOF predictions are incomplete")

    events = frame[
        [
            "sourceRow",
            "runNumber",
            "fileNumber",
            "eventNumber",
            "outerBlock",
        ]
    ].copy()
    events["baseline_energy"] = oof_baseline
    events["candidate_energy"] = oof_candidate
    events["log_correction"] = oof_correction
    events["correction_near_cap"] = oof_touched
    events["selected_config"] = selected_config
    events.to_csv(output_dir / "oof_events.csv", index=False)

    outer_table = pd.DataFrame(outer_rows)
    outer_table.to_csv(output_dir / "outer_block_metrics.csv", index=False)
    pd.DataFrame(protocol_rows).to_csv(
        output_dir / "protocol_metrics.csv", index=False
    )
    (output_dir / "nested_selection.json").write_text(
        json.dumps(selection_records, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline_all = protocol_metrics(oof_baseline)
    candidate_all = protocol_metrics(oof_candidate)
    deltas = cluster_bootstrap_r68(events, iterations=args.bootstrap)
    pd.DataFrame({"delta_r68": deltas}).to_csv(
        output_dir / "file_cluster_bootstrap_r68.csv", index=False
    )
    bootstrap_quantiles = np.quantile(deltas, [0.025, 0.5, 0.975])

    # Choose the deployable identity using grouped CV over all five blocks.
    final_config, final_selection = select_config(
        frame,
        features,
        list(range(OUTER_BLOCKS)),
        seed_offset=90000,
    )
    full_state = fit_predictor(
        frame[features].to_numpy(dtype=float),
        energies,
        final_config,
        SEED + 99999,
    )
    baseline_full_fit = fit_protocol(energies, "linear", 0.18)
    if baseline_full_fit.get("success"):
        baseline_training_mu = baseline_full_fit["mu"]
    else:
        baseline_training_mu = float(np.median(energies))
    bundle = {
        "format_version": 2,
        "model_name": "PandaX_grouped_nested_physics_v9",
        "training_run": 10972,
        "training_events": int(len(frame)),
        "feature_names": features,
        "group_key": "fileNumber",
        "group_key_is_predictor": False,
        "outer_blocks": [
            [block * BLOCK_WIDTH, (block + 1) * BLOCK_WIDTH]
            for block in range(OUTER_BLOCKS)
        ],
        "base_formula": "qS1ub_C/0.125 + qS2Bdesub_C/10.58",
        "baseline_energy_scale": (
            NOMINAL_ENERGY_KEV / baseline_training_mu
        ),
        "nominal_energy_kev": NOMINAL_ENERGY_KEV,
        "monotonicity": (
            "By construction with correction features fixed: no absolute S1/S2 "
            "charge enters the correction model."
        ),
        "selected_config": final_config,
        "predictor_state": full_state,
        "warning": (
            "Validated only across grouped file blocks inside run10972. "
            "Independent-run blind validation is still required."
        ),
    }
    joblib.dump(bundle, str(output_dir / "candidate_v9.joblib"), compress=3)
    (output_dir / "final_selection.json").write_text(
        json.dumps(
            {
                "winner": final_config["name"],
                "candidates": final_selection,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    all_outer_improve = bool(
        np.all(
            outer_table["candidate_sigma_median"]
            < outer_table["baseline_sigma_median"]
        )
    )
    center_pass = bool(
        np.all(outer_table["candidate_center_bias_max"] <= 0.003)
    )
    tail_pass = bool(
        np.all(outer_table["candidate_r90"] <= outer_table["baseline_r90"])
    )
    finite_pass = bool(np.all(outer_table["finite_positive"]))
    correction_pass = bool(np.all(outer_table["near_cap_fraction"] <= 0.05))
    bootstrap_pass = bool(bootstrap_quantiles[2] < 0)
    gates = {
        "all_outer_blocks_improve": all_outer_improve,
        "center_bias_le_0p3_percent": center_pass,
        "r90_not_worse_in_every_block": tail_pass,
        "finite_and_positive": finite_pass,
        "correction_near_cap_le_5_percent": correction_pass,
        "file_cluster_bootstrap_r68_upper_lt_zero": bootstrap_pass,
        "monotonic_s1_s2_by_construction": True,
    }
    summary = {
        "events": int(len(frame)),
        "files": int(frame["fileNumber"].nunique()),
        "rejected_source_rows": rejected,
        "features": features,
        "forbidden_predictor_overlap": overlap,
        "outer_validation": (
            "Five contiguous complete file blocks; candidate selection only "
            "inside each outer training pool."
        ),
        "baseline_outer_median_sigma_over_mu": float(
            np.median(outer_table["baseline_sigma_median"])
        ),
        "candidate_outer_median_sigma_over_mu": float(
            np.median(outer_table["candidate_sigma_median"])
        ),
        "relative_improvement_at_outer_medians": float(
            1.0
            - np.median(outer_table["candidate_sigma_median"])
            / np.median(outer_table["baseline_sigma_median"])
        ),
        "merged_oof_baseline": {
            key: value
            for key, value in baseline_all.items()
            if key != "protocol_rows"
        },
        "merged_oof_candidate": {
            key: value
            for key, value in candidate_all.items()
            if key != "protocol_rows"
        },
        "bootstrap_delta_r68_2p5": float(bootstrap_quantiles[0]),
        "bootstrap_delta_r68_median": float(bootstrap_quantiles[1]),
        "bootstrap_delta_r68_97p5": float(bootstrap_quantiles[2]),
        "final_frozen_config": final_config,
        "acceptance_gates": gates,
        "all_acceptance_gates_pass": bool(all(gates.values())),
        "independent_run_validated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
