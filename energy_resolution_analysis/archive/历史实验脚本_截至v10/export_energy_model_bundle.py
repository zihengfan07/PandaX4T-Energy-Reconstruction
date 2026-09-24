#!/usr/bin/env python3
"""Train and serialize the validated two-stage repeated-fold model bundle."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from full_feature_sigma_search import balanced_folds, gaussian_core_fit
from repeated_oof_ensemble_sigma import RAW_FEATURES, add_derived_features


SEED = 10972
REPEATS = 5
CLIP = 0.21


def base_energy(frame):
    return (
        frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    )


def prepare_frame(path):
    header = pd.read_csv(path, sep="\t", nrows=0)
    requested = ["runNumber", "eventNumber", "qS1ub_C", "qS2Bdesub_C"]
    requested += RAW_FEATURES
    usecols = [name for name in requested if name in header.columns]
    frame = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    validity = (
        np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    frame = frame.loc[validity].reset_index(drop=True)
    return add_derived_features(frame)


def choose_features(frame):
    absolute_charge_columns = {
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
    excluded = absolute_charge_columns | {
        "sourceRow",
        "runNumber",
        "eventNumber",
    }
    features = []
    for name in frame.columns:
        if name in excluded:
            continue
        values = frame[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == len(frame) and np.std(finite) > 0:
            features.append(name)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = prepare_frame(args.input)
    features = choose_features(frame)
    all_energy = base_energy(frame)
    members = []

    for repeat in range(REPEATS):
        fold = balanced_folds(len(frame), 5, SEED + 2027 * repeat)
        for outer in range(5):
            train_index = np.flatnonzero(fold != outer)
            train = frame.iloc[train_index]
            train_energy = all_energy[train_index]
            train_x = train[features].to_numpy(dtype=float)
            medians = np.nanmedian(train_x, axis=0)
            medians[~np.isfinite(medians)] = 0.0
            train_x = np.where(np.isfinite(train_x), train_x, medians)

            energy_center = np.median(train_energy)
            target = np.log(np.clip(train_energy / energy_center, 1.0e-6, None))
            first = HistGradientBoostingRegressor(
                loss="least_absolute_deviation",
                learning_rate=0.035,
                max_iter=170,
                max_leaf_nodes=9,
                min_samples_leaf=85,
                l2_regularization=16.0,
                random_state=SEED + repeat,
            )
            first.fit(train_x, target)
            first_train = first.predict(train_x)
            residual = target - first_train
            second = ExtraTreesRegressor(
                n_estimators=140,
                min_samples_leaf=30,
                max_features=0.75,
                bootstrap=False,
                n_jobs=4,
                random_state=SEED + 100 + repeat,
            )
            second.fit(train_x, residual)
            total_train_prediction = first_train + second.predict(train_x)
            prediction_center = np.median(total_train_prediction)
            correction = np.exp(
                np.clip(
                    total_train_prediction - prediction_center,
                    -CLIP,
                    CLIP,
                )
            )
            corrected_train = train_energy / correction
            train_fit = gaussian_core_fit(corrected_train)
            energy_scale = train_fit["mu"]
            if not np.isfinite(energy_scale) or energy_scale <= 0:
                energy_scale = np.median(corrected_train)
            members.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer,
                    "feature_medians": medians,
                    "first_model": first,
                    "second_model": second,
                    "prediction_center": float(prediction_center),
                    "energy_scale": float(energy_scale),
                }
            )
            print(
                "trained repeat={} fold={} members={}".format(
                    repeat, outer, len(members)
                )
            )

    bundle = {
        "format_version": 1,
        "model_name": "PandaX_run10972_two_stage_repeated_fold_v8",
        "training_run": 10972,
        "training_events": len(frame),
        "feature_names": features,
        "clip": CLIP,
        "base_formula": (
            "qS1ub_C/0.125 + qS2Bdesub_C/10.58; "
            "multiply final relative output by a separately validated energy scale"
        ),
        "members": members,
        "warning": (
            "The bundle was trained on run10972. Evaluate performance only on "
            "an independent run. Do not quote in-sample application as validation."
        ),
    }
    joblib.dump(bundle, str(output), compress=3)
    manifest = {
        "model_name": bundle["model_name"],
        "format_version": bundle["format_version"],
        "training_run": bundle["training_run"],
        "training_events": bundle["training_events"],
        "feature_count": len(features),
        "member_count": len(members),
        "clip": CLIP,
        "model_file": output.name,
        "warning": bundle["warning"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
