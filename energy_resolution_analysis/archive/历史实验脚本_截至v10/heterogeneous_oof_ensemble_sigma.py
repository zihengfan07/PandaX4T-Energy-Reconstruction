#!/usr/bin/env python3
"""Heterogeneous repeated-OOF ensemble for the energy response correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from full_feature_sigma_search import (
    balanced_folds,
    fill_from_training,
    gaussian_core_fit,
)
from repeated_oof_ensemble_sigma import RAW_FEATURES, add_derived_features, metrics


SEED = 10972
REPEATS = 5

CONFIGS = [
    {
        "name": "hgb_aggressive",
        "kind": "hgb",
        "leaves": 13,
        "min_leaf": 70,
        "l2": 12.0,
        "iterations": 210,
        "clip": 0.19,
    },
    {
        "name": "extra_leaf20",
        "kind": "extra",
        "trees": 180,
        "min_leaf": 20,
        "max_features": 0.75,
        "clip": 0.20,
    },
    {
        "name": "extra_leaf35",
        "kind": "extra",
        "trees": 180,
        "min_leaf": 35,
        "max_features": 1.0,
        "clip": 0.18,
    },
    {
        "name": "forest_leaf25",
        "kind": "forest",
        "trees": 180,
        "min_leaf": 25,
        "max_features": 0.75,
        "clip": 0.18,
    },
    {
        "name": "two_stage_hgb_extra",
        "kind": "two_stage",
        "leaves": 9,
        "min_leaf": 85,
        "l2": 16.0,
        "iterations": 170,
        "trees": 140,
        "extra_min_leaf": 30,
        "max_features": 0.75,
        "clip": 0.21,
    },
]


def base_energy(frame):
    return (
        frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    )


def build_model_predictions(train_x, test_x, target, config, repeat):
    kind = config["kind"]
    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            loss="least_absolute_deviation",
            learning_rate=0.035,
            max_iter=config["iterations"],
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["min_leaf"],
            l2_regularization=config["l2"],
            random_state=SEED + repeat,
        )
        model.fit(train_x, target)
        return model.predict(train_x), model.predict(test_x)

    if kind == "extra":
        model = ExtraTreesRegressor(
            n_estimators=config["trees"],
            min_samples_leaf=config["min_leaf"],
            max_features=config["max_features"],
            bootstrap=False,
            n_jobs=4,
            random_state=SEED + repeat,
        )
        model.fit(train_x, target)
        return model.predict(train_x), model.predict(test_x)

    if kind == "forest":
        model = RandomForestRegressor(
            n_estimators=config["trees"],
            min_samples_leaf=config["min_leaf"],
            max_features=config["max_features"],
            bootstrap=True,
            n_jobs=4,
            random_state=SEED + repeat,
        )
        model.fit(train_x, target)
        return model.predict(train_x), model.predict(test_x)

    if kind == "two_stage":
        first = HistGradientBoostingRegressor(
            loss="least_absolute_deviation",
            learning_rate=0.035,
            max_iter=config["iterations"],
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["min_leaf"],
            l2_regularization=config["l2"],
            random_state=SEED + repeat,
        )
        first.fit(train_x, target)
        first_train = first.predict(train_x)
        first_test = first.predict(test_x)
        residual = target - first_train
        second = ExtraTreesRegressor(
            n_estimators=config["trees"],
            min_samples_leaf=config["extra_min_leaf"],
            max_features=config["max_features"],
            bootstrap=False,
            n_jobs=4,
            random_state=SEED + 100 + repeat,
        )
        second.fit(train_x, residual)
        return (
            first_train + second.predict(train_x),
            first_test + second.predict(test_x),
        )
    raise ValueError("Unknown model kind: " + kind)


def crossfit(frame, features, config, repeat):
    energy = base_energy(frame)
    fold = balanced_folds(len(frame), 5, SEED + 2027 * repeat)
    output = np.full(len(frame), np.nan)
    for outer in range(5):
        train_index = np.flatnonzero(fold != outer)
        test_index = np.flatnonzero(fold == outer)
        train_x, test_x = fill_from_training(
            frame.iloc[train_index][features].to_numpy(dtype=float),
            frame.iloc[test_index][features].to_numpy(dtype=float),
        )
        train_energy = energy[train_index]
        test_energy = energy[test_index]
        center = np.median(train_energy)
        target = np.log(np.clip(train_energy / center, 1.0e-6, None))
        train_prediction, test_prediction = build_model_predictions(
            train_x, test_x, target, config, repeat
        )
        clip_value = config["clip"]
        prediction_center = np.median(train_prediction)
        train_correction = np.exp(
            np.clip(
                train_prediction - prediction_center,
                -clip_value,
                clip_value,
            )
        )
        test_correction = np.exp(
            np.clip(
                test_prediction - prediction_center,
                -clip_value,
                clip_value,
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

    outputs = {}
    repeat_rows = []
    summary_rows = []
    raw = base_energy(frame)
    raw_score = metrics(raw)
    summary_rows.append({"method": "current_formula", **raw_score})

    for config in CONFIGS:
        repeated = []
        for repeat in range(REPEATS):
            prediction = crossfit(frame, features, config, repeat)
            repeated.append(prediction)
            repeat_rows.append(
                {
                    "config": config["name"],
                    "repeat": repeat,
                    **metrics(prediction)
                }
            )
        ensemble = np.exp(np.mean(np.log(np.asarray(repeated)), axis=0))
        outputs[config["name"]] = ensemble
        summary_rows.append(
            {"method": config["name"], **metrics(ensemble)}
        )

    blend_specs = {
        "blend_hgb_extra20": ["hgb_aggressive", "extra_leaf20"],
        "blend_hgb_two_stage": ["hgb_aggressive", "two_stage_hgb_extra"],
        "blend_extra20_two_stage": ["extra_leaf20", "two_stage_hgb_extra"],
        "blend_three": [
            "hgb_aggressive",
            "extra_leaf20",
            "two_stage_hgb_extra",
        ],
    }
    for name, members in blend_specs.items():
        blend = np.exp(
            np.mean(np.log(np.asarray([outputs[x] for x in members])), axis=0)
        )
        outputs[name] = blend
        summary_rows.append({"method": name, **metrics(blend)})

    summary = pd.DataFrame(summary_rows).sort_values(
        ["sigma_over_mu", "r68", "r90"]
    )
    events = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    events["current_formula"] = raw
    for name, values in outputs.items():
        events[name] = values
    events.to_csv(output_dir / "oof_events.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(
        output_dir / "repeat_metrics.csv", index=False
    )
    metadata = {
        "events": len(frame),
        "rejected_source_rows": rejected,
        "repeats": REPEATS,
        "features": features,
        "configs": CONFIGS,
        "validation": (
            "Repeated five-fold OOF; no event is predicted by a model that "
            "trained on that event."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
