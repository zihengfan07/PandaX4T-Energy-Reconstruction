#!/usr/bin/env python3
"""Repeated cross-fitted ensemble for sigma/mu reduction.

Every prediction used in the final spectrum comes from models that excluded
that event. Repeating the split and geometrically averaging OOF predictions
reduces model and fold-calibration variance.
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
REPEATS = 8

BASES = {
    "current_k100": 1.00,
    "candidate_k104": 1.04,
}

RAW_FEATURES = [
    "dt",
    "xS2T_max",
    "yS2T_max",
    "xS2Tcor_max",
    "yS2Tcor_max",
    "xS2max_cdfTM",
    "yS2max_cdfTM",
    "xS2max_cdfPAF",
    "yS2max_cdfPAF",
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
    "rmsCogS1T_max",
    "rmsCogS1B_max",
    "qS1C_max",
    "qS1Tub_C",
    "qS1Bub_C",
    "qS1T_max",
    "qS1B_max",
    "qS2BC_max",
    "qS2ub_C",
    "qS2Bub_C",
    "qS2desub_C",
    "qS2T_max",
    "qS2B_max",
    "qS2Bub_C_stretch",
    "qS2Bdesub_C_stretch",
    "qS1hitStdevTo1_max",
    "qS1channelStdevTo1_max",
    "qS2hitStdevTo1_max",
    "qS2channelStdevTo1_max",
    "ratioqS2PrePeak_max",
    "ratioqS2PrePeakSmr_max",
    "ratioqS1PrePeak_max",
    "ratioqS1PrePeakSmr_max",
]

MODEL_CONFIGS = [
    {
        "name": "conservative_lad",
        "loss": "least_absolute_deviation",
        "leaves": 3,
        "min_leaf": 160,
        "l2": 35.0,
        "iterations": 120,
        "clip": 0.10,
    },
    {
        "name": "balanced_lad",
        "loss": "least_absolute_deviation",
        "leaves": 5,
        "min_leaf": 130,
        "l2": 25.0,
        "iterations": 140,
        "clip": 0.12,
    },
    {
        "name": "flexible_lad",
        "loss": "least_absolute_deviation",
        "leaves": 7,
        "min_leaf": 105,
        "l2": 18.0,
        "iterations": 160,
        "clip": 0.14,
    },
    {
        "name": "aggressive_lad",
        "loss": "least_absolute_deviation",
        "leaves": 11,
        "min_leaf": 80,
        "l2": 14.0,
        "iterations": 180,
        "clip": 0.16,
    },
    {
        "name": "balanced_l2",
        "loss": "least_squares",
        "leaves": 5,
        "min_leaf": 130,
        "l2": 30.0,
        "iterations": 140,
        "clip": 0.12,
    },
    {
        "name": "flexible_l2",
        "loss": "least_squares",
        "leaves": 9,
        "min_leaf": 95,
        "l2": 20.0,
        "iterations": 170,
        "clip": 0.15,
    },
]


def energy(frame, k_value):
    return (
        k_value * frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    )


def add_derived_features(frame):
    frame = frame.copy()
    for label, x_name, y_name in [
        ("raw", "xS2T_max", "yS2T_max"),
        ("cor", "xS2Tcor_max", "yS2Tcor_max"),
        ("cdf", "xS2max_cdfTM", "yS2max_cdfTM"),
        ("paf", "xS2max_cdfPAF", "yS2max_cdfPAF"),
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
            frame["phi_" + label] = np.arctan2(y, x)

    ratio_definitions = [
        ("ratio_s1_c_to_ub", "qS1C_max", "qS1ub_C"),
        ("ratio_s1_top_bottom_cor", "qS1Tub_C", "qS1Bub_C"),
        ("ratio_s1_top_bottom_raw", "qS1T_max", "qS1B_max"),
        ("ratio_s2_bc_to_desub", "qS2BC_max", "qS2Bdesub_C"),
        ("ratio_s2_total_bottom_cor", "qS2ub_C", "qS2Bub_C"),
        ("ratio_s2_des_total_bottom", "qS2desub_C", "qS2Bdesub_C"),
        ("ratio_s2_top_bottom_raw", "qS2T_max", "qS2B_max"),
        ("ratio_s2_stretch_to_base", "qS2Bdesub_C_stretch", "qS2Bdesub_C"),
        ("ratio_s2_ubstretch_to_base", "qS2Bub_C_stretch", "qS2Bdesub_C"),
    ]
    for name, numerator, denominator in ratio_definitions:
        if numerator in frame.columns and denominator in frame.columns:
            ratio = (
                frame[numerator].to_numpy(dtype=float)
                / frame[denominator].to_numpy(dtype=float)
            )
            frame[name] = np.clip(ratio, -10.0, 10.0)
    return frame


def metrics(values):
    fit = gaussian_core_fit(values)
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    q05, q16, q50, q84, q95 = np.quantile(
        valid, [0.05, 0.15865, 0.5, 0.84135, 0.95]
    )
    return {
        "sigma_over_mu": fit["sigma_over_mu"],
        "fit_p": fit["p_value"],
        "r68": float((q84 - q16) / (2.0 * q50)),
        "r90": float((q95 - q05) / (2.0 * q50)),
    }


def crossfit_once(frame, base_energy, features, config, repeat):
    fold = balanced_folds(len(frame), 5, SEED + 1009 * repeat)
    output = np.full(len(frame), np.nan)
    for outer in range(5):
        train_index = np.flatnonzero(fold != outer)
        test_index = np.flatnonzero(fold == outer)
        train_x, test_x = fill_from_training(
            frame.iloc[train_index][features].to_numpy(dtype=float),
            frame.iloc[test_index][features].to_numpy(dtype=float),
        )
        train_energy = base_energy[train_index]
        test_energy = base_energy[test_index]
        center = np.median(train_energy)
        target = np.log(np.clip(train_energy / center, 1.0e-6, None))
        model = HistGradientBoostingRegressor(
            loss=config["loss"],
            learning_rate=0.035,
            max_iter=config["iterations"],
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["min_leaf"],
            l2_regularization=config["l2"],
            random_state=SEED + repeat,
        )
        model.fit(train_x, target)
        train_prediction = model.predict(train_x)
        test_prediction = model.predict(test_x)
        train_correction = np.exp(
            np.clip(
                train_prediction - np.median(train_prediction),
                -config["clip"],
                config["clip"],
            )
        )
        test_correction = np.exp(
            np.clip(
                test_prediction - np.median(train_prediction),
                -config["clip"],
                config["clip"],
            )
        )
        corrected_train = train_energy / train_correction
        corrected_test = test_energy / test_correction
        train_fit = gaussian_core_fit(corrected_train)
        scale = train_fit["mu"]
        if not np.isfinite(scale) or scale <= 0:
            scale = np.median(corrected_train)
        output[test_index] = corrected_test / scale
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.input, sep="\t", nrows=0)
    requested = ["runNumber", "eventNumber", "qS1ub_C", "qS2Bdesub_C"]
    requested += RAW_FEATURES
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
    frame = add_derived_features(frame)

    excluded = {
        "sourceRow",
        "runNumber",
        "eventNumber",
        "qS1ub_C",
        "qS2Bdesub_C",
        "qS1C_max",
        "qS1Tub_C",
        "qS1Bub_C",
        "qS1T_max",
        "qS1B_max",
        "qS2BC_max",
        "qS2ub_C",
        "qS2Bub_C",
        "qS2desub_C",
        "qS2T_max",
        "qS2B_max",
        "qS2Bub_C_stretch",
        "qS2Bdesub_C_stretch",
    }
    features = []
    for name in frame.columns:
        if name in excluded:
            continue
        values = frame[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == len(frame) and np.std(finite) > 0:
            features.append(name)
    # Ratios are allowed; their absolute charge parents above are excluded.
    features += [
        name for name in frame.columns
        if name.startswith("ratio_") and name not in features
    ]

    summary_rows = []
    ensemble_outputs = {}
    repeat_rows = []
    for base_name, k_value in BASES.items():
        base = energy(frame, k_value)
        base_score = metrics(base)
        summary_rows.append(
            {
                "method": base_name,
                "k": k_value,
                "config": "none",
                "repeats": 0,
                **base_score
            }
        )
        for config in MODEL_CONFIGS:
            repeated = []
            for repeat in range(REPEATS):
                prediction = crossfit_once(
                    frame, base, features, config, repeat
                )
                repeated.append(prediction)
                score = metrics(prediction)
                repeat_rows.append(
                    {
                        "base": base_name,
                        "config": config["name"],
                        "repeat": repeat,
                        **score
                    }
                )
            repeated = np.asarray(repeated)
            ensemble = np.exp(np.mean(np.log(repeated), axis=0))
            key = base_name + "__" + config["name"]
            ensemble_outputs[key] = ensemble
            score = metrics(ensemble)
            summary_rows.append(
                {
                    "method": key,
                    "k": k_value,
                    "config": config["name"],
                    "repeats": REPEATS,
                    **score
                }
            )

    # Pre-registered equal-weight model-family blends.  No blend weight is
    # tuned on the final spectrum.
    blend_specs = {
        "current_k100__blend_lad_l2": [
            "current_k100__flexible_lad",
            "current_k100__flexible_l2",
        ],
        "current_k100__blend_flex_aggressive": [
            "current_k100__flexible_lad",
            "current_k100__aggressive_lad",
        ],
        "current_k100__blend_three_lad": [
            "current_k100__balanced_lad",
            "current_k100__flexible_lad",
            "current_k100__aggressive_lad",
        ],
        "candidate_k104__blend_lad_l2": [
            "candidate_k104__flexible_lad",
            "candidate_k104__flexible_l2",
        ],
    }
    for blend_name, members in blend_specs.items():
        if not all(member in ensemble_outputs for member in members):
            continue
        stacked = np.asarray([ensemble_outputs[member] for member in members])
        blend = np.exp(np.mean(np.log(stacked), axis=0))
        ensemble_outputs[blend_name] = blend
        score = metrics(blend)
        summary_rows.append(
            {
                "method": blend_name,
                "k": 1.04 if blend_name.startswith("candidate") else 1.00,
                "config": "equal_weight_blend",
                "repeats": REPEATS,
                **score
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["sigma_over_mu", "r68", "r90"]
    )
    events = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    for base_name, k_value in BASES.items():
        events[base_name] = energy(frame, k_value)
    for name, values in ensemble_outputs.items():
        events[name] = values
    events.to_csv(output_dir / "oof_ensemble_events.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(
        output_dir / "repeat_metrics.csv", index=False
    )
    metadata = {
        "events": len(frame),
        "rejected_source_rows": rejected,
        "repeats": REPEATS,
        "features": features,
        "model_configs": MODEL_CONFIGS,
        "validation": (
            "Every event prediction is out-of-fold in every repeat; "
            "repeated predictions are geometrically averaged."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
