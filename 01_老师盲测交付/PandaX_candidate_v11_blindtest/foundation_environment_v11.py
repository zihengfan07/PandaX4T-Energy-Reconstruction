#!/usr/bin/env python3
"""Nested complete-file validation for a stronger PandaX energy model.

The outer test block is never used to choose the S1/S2 foundation, feature
layer, or regularization. fileNumber is grouping metadata only.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from grouped_nested_physics_v9 import (
    BLOCK_WIDTH,
    NOMINAL_ENERGY_KEV,
    OUTER_BLOCKS,
    RAW_FEATURES,
    SEED,
    calibrated_baseline,
    fit_predictor,
    fit_protocol,
    predict_with_state,
    protocol_metrics,
    tail_metrics,
)


BASE_PATHS = {
    "ub_current": ("qS1ub_C", "qS2Bdesub_C"),
    "legacy_lifetime": ("qS1C_max", "qS2Belife_max"),
    "image_des": (
        "qS1C_desImageMCPAF_maxS2",
        "qS2BdesC_desImageMCPAF_maxS2",
    ),
    "image_raw": (
        "qS1C_desImageMCPAF_maxS2",
        "qS2BC_desImageMCPAF_maxS2",
    ),
    "image_stretch": (
        "qS1C_ImageMCPAF_stretch_maxS2forMS",
        "qS2BdesC_ImageMCPAF_total_stretch_maxS2forMS",
    ),
}

POSITION_FEATURES = [
    "dt",
    "xS2T_max", "yS2T_max", "xS2Tcor_max", "yS2Tcor_max",
    "xS2B_max", "yS2B_max", "xS2Bcor_max", "yS2Bcor_max",
    "xS2max_cdfTM", "yS2max_cdfTM",
    "xS2max_cdfTMs", "yS2max_cdfTMs",
    "xS2max_cdfPAF", "yS2max_cdfPAF",
    "xS2max_desCorCog_maxS2", "yS2max_desCorCog_maxS2",
    "xS2max_desImageMCPAF_maxS2", "yS2max_desImageMCPAF_maxS2",
    "xS2max_desCorCog_firstS2", "yS2max_desCorCog_firstS2",
    "xS2max_desImageMCPAF_firstS2", "yS2max_desImageMCPAF_firstS2",
    "lhfS2max_cdfPAF", "lhfS2max_desImageMCPAF_maxS2",
    "lhfS2max_desImageMCPAF_firstS2",
]

SHAPE_FEATURES = [
    "wS1_max", "wS1CDF_max", "tDiffBottomTopS1_max",
    "pS1_max", "hS1_max",
    "wS2_max", "wS2CDF_max", "wS2FWHM_max", "widthTenS2_max",
    "tDiffBottomTopS2_max", "pS2_max", "hS2_max",
]

PATTERN_FEATURES = [
    "nPMTS1_max", "nPMTS2_max",
    "ratioqS1PrePeak_max", "ratioqS1PrePeakSmr_max",
    "ratioqS2PrePeak_max", "ratioqS2PrePeakSmr_max",
    "qS1hitStdev_max", "qS1channelStdev_max",
    "qS2hitStdev_max", "qS2channelStdev_max",
    "qS1hitStdevTo1_max", "qS1channelStdevTo1_max",
    "qS2hitStdevTo1_max", "qS2channelStdevTo1_max",
    "rmsCogS1T_max", "rmsCogS1B_max",
]

ENVIRONMENT_FEATURES = [
    "ratioTSignal", "duration", "nRealDesS2",
    "qNearS1max", "qElseBeforeS1max", "qElseBetweenS1S2max",
    "qNearS2max", "qElseAfterS2max",
    "nSignalBeforeS1max", "nSignalBetweenS1S2max",
    "nSignalAfterS2max", "qS2maxPreEvent", "tDiffPreEvent",
    "nS1", "nPostS1", "nGoodS1", "nCandidateS1", "nPeakS1_max",
    "nS2", "nPostS2", "nRealPostS2", "nPeakS2_max",
]

MODEL_CONFIGS = [
    {
        "name": "spline_a0001_c100",
        "kind": "spline_ridge",
        "alpha": 0.0001,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_a001_c100",
        "kind": "spline_ridge",
        "alpha": 0.01,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_a1_c100",
        "kind": "spline_ridge",
        "alpha": 1.0,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_a30_c100",
        "kind": "spline_ridge",
        "alpha": 30.0,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
    {
        "name": "spline_a100_c100",
        "kind": "spline_ridge",
        "alpha": 100.0,
        "clip": 0.10,
        "cap_transform": "tanh",
    },
]


def add_geometry(frame):
    frame = frame.copy()
    pairs = [
        ("raw", "xS2T_max", "yS2T_max"),
        ("cor", "xS2Tcor_max", "yS2Tcor_max"),
        ("bottom", "xS2B_max", "yS2B_max"),
        ("bottom_cor", "xS2Bcor_max", "yS2Bcor_max"),
        ("cdf", "xS2max_cdfTM", "yS2max_cdfTM"),
        ("cdfs", "xS2max_cdfTMs", "yS2max_cdfTMs"),
        ("paf", "xS2max_cdfPAF", "yS2max_cdfPAF"),
        (
            "mcpaf",
            "xS2max_desImageMCPAF_maxS2",
            "yS2max_desImageMCPAF_maxS2",
        ),
    ]
    for label, x_name, y_name in pairs:
        if x_name in frame and y_name in frame:
            x = frame[x_name].to_numpy(dtype=float)
            y = frame[y_name].to_numpy(dtype=float)
            frame["r2_" + label] = x * x + y * y
    return frame


def prepare_frame(path):
    header = pd.read_csv(path, sep="\t", nrows=0)
    required = [
        "runNumber", "fileNumber", "eventNumber",
        "qS1ub_C", "qS2Bdesub_C",
    ]
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    requested = set(required)
    for pair in BASE_PATHS.values():
        requested.update(pair)
    requested.update(POSITION_FEATURES)
    requested.update(SHAPE_FEATURES)
    requested.update(PATTERN_FEATURES)
    requested.update(ENVIRONMENT_FEATURES)
    usecols = [name for name in header.columns if name in requested]
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


def usable_features(frame, requested):
    result = []
    for name in requested:
        if name not in frame or name in result:
            continue
        values = frame[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) >= 0.90 * len(values) and len(np.unique(finite)) >= 8:
            result.append(name)
    return result


def feature_sets(frame):
    radius = [name for name in frame.columns if name.startswith("r2_")]
    core = usable_features(frame, list(RAW_FEATURES) + radius)
    detector = usable_features(
        frame, POSITION_FEATURES + SHAPE_FEATURES + PATTERN_FEATURES + radius
    )
    environment = usable_features(
        frame,
        POSITION_FEATURES + SHAPE_FEATURES + PATTERN_FEATURES
        + ENVIRONMENT_FEATURES + radius,
    )
    return {
        "core": core,
        "detector": detector,
        "environment": environment,
    }


def fit_base_state(frame, indices, path_name):
    s1_name, s2_name = BASE_PATHS[path_name]
    s1 = frame[s1_name].to_numpy(dtype=float)[indices]
    s2 = frame[s2_name].to_numpy(dtype=float)[indices]
    good1 = s1[np.isfinite(s1) & (s1 > 0)]
    good2 = s2[np.isfinite(s2) & (s2 > 0)]
    med1 = float(np.median(good1))
    med2 = float(np.median(good2))
    s1 = np.where(np.isfinite(s1) & (s1 > 0), s1, med1) / med1
    s2 = np.where(np.isfinite(s2) & (s2 > 0), s2, med2) / med2
    best = None
    for weight in np.linspace(0.05, 0.95, 37):
        energy = weight * s1 + (1.0 - weight) * s2
        score = tail_metrics(energy)["r68"]
        item = (float(score), float(weight))
        if best is None or item < best:
            best = item
    return {
        "path": path_name,
        "s1_name": s1_name,
        "s2_name": s2_name,
        "s1_median": med1,
        "s2_median": med2,
        "s1_weight": best[1],
        "training_r68": best[0],
    }


def apply_base(frame, indices, state):
    s1 = frame[state["s1_name"]].to_numpy(dtype=float)[indices]
    s2 = frame[state["s2_name"]].to_numpy(dtype=float)[indices]
    s1 = np.where(
        np.isfinite(s1) & (s1 > 0), s1, state["s1_median"]
    ) / state["s1_median"]
    s2 = np.where(
        np.isfinite(s2) & (s2 > 0), s2, state["s2_median"]
    ) / state["s2_median"]
    weight = state["s1_weight"]
    return weight * s1 + (1.0 - weight) * s2


def current_energy(frame):
    return (
        frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    )


def legacy_energy(frame):
    energy = 0.0137 * current_energy(frame)
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


def quick_metrics(values):
    tail = tail_metrics(values)
    fit = fit_protocol(values, "linear", 0.18)
    center_bias = (
        abs(fit["mu"] / NOMINAL_ENERGY_KEV - 1.0)
        if fit.get("success") else 1.0
    )
    return {
        "r68": tail["r68"],
        "r90": tail["r90"],
        "center_bias": float(center_bias),
    }


def inner_splits(frame, allowed_blocks):
    for validation_block in allowed_blocks:
        train_blocks = [b for b in allowed_blocks if b != validation_block]
        train_index = np.flatnonzero(
            frame["outerBlock"].isin(train_blocks).to_numpy()
        )
        validation_index = np.flatnonzero(
            (frame["outerBlock"] == validation_block).to_numpy()
        )
        yield validation_block, train_index, validation_index


def choose_base_path(frame, allowed_blocks):
    records = []
    for path_name in BASE_PATHS:
        folds = []
        for block, train_index, validation_index in inner_splits(
            frame, allowed_blocks
        ):
            state = fit_base_state(frame, train_index, path_name)
            train_energy = apply_base(frame, train_index, state)
            validation_energy = apply_base(frame, validation_index, state)
            prediction = calibrated_baseline(train_energy, validation_energy)
            metrics = quick_metrics(prediction)
            folds.append({
                "validation_block": int(block),
                "r68": metrics["r68"],
                "r90": metrics["r90"],
                "center_bias": metrics["center_bias"],
                "weight": state["s1_weight"],
            })
        records.append({
            "path": path_name,
            "median_r68": float(np.median([x["r68"] for x in folds])),
            "median_r90": float(np.median([x["r90"] for x in folds])),
            "folds": folds,
        })
    winner = min(
        records,
        key=lambda row: (row["median_r68"] + 0.20 * row["median_r90"], row["path"]),
    )
    return winner["path"], records


def choose_model(frame, allowed_blocks, path_name, sets, seed_offset):
    records = []
    for set_name, features in sets.items():
        for config in MODEL_CONFIGS:
            folds = []
            for block, train_index, validation_index in inner_splits(
                frame, allowed_blocks
            ):
                base_state = fit_base_state(frame, train_index, path_name)
                train_energy = apply_base(frame, train_index, base_state)
                validation_energy = apply_base(
                    frame, validation_index, base_state
                )
                train_x = frame.iloc[train_index][features].to_numpy(float)
                validation_x = frame.iloc[validation_index][features].to_numpy(float)
                state = fit_predictor(
                    train_x,
                    train_energy,
                    config,
                    SEED + seed_offset + block,
                )
                prediction, correction, touched = predict_with_state(
                    state, validation_x, validation_energy
                )
                metrics = quick_metrics(prediction)
                score = metrics["r68"] + 0.20 * metrics["r90"]
                score += 10.0 * max(metrics["center_bias"] - 0.003, 0.0)
                score += 1.5 * max(float(np.mean(touched)) - 0.03, 0.0)
                folds.append({
                    "validation_block": int(block),
                    "score": float(score),
                    "r68": metrics["r68"],
                    "r90": metrics["r90"],
                    "center_bias": metrics["center_bias"],
                    "near_cap_fraction": float(np.mean(touched)),
                })
            records.append({
                "feature_set": set_name,
                "feature_count": len(features),
                "config": config,
                "median_score": float(np.median([x["score"] for x in folds])),
                "worst_score": float(np.max([x["score"] for x in folds])),
                "folds": folds,
            })
    winner = min(
        records,
        key=lambda row: (
            row["median_score"],
            row["feature_count"],
            row["config"]["alpha"],
        ),
    )
    return winner, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frame, rejected = prepare_frame(args.input)
    sets = feature_sets(frame)
    n = len(frame)
    current_all = current_energy(frame)
    legacy_all = legacy_energy(frame)
    current_oof = np.full(n, np.nan)
    legacy_oof = np.full(n, np.nan)
    foundation_oof = np.full(n, np.nan)
    candidate_oof = np.full(n, np.nan)
    correction_oof = np.full(n, np.nan)
    touched_oof = np.zeros(n, dtype=bool)
    outer_rows = []
    selection_records = []

    for outer_block in range(OUTER_BLOCKS):
        allowed_blocks = [b for b in range(OUTER_BLOCKS) if b != outer_block]
        train_index = np.flatnonzero(
            frame["outerBlock"].isin(allowed_blocks).to_numpy()
        )
        test_index = np.flatnonzero(
            (frame["outerBlock"] == outer_block).to_numpy()
        )
        path_name, path_records = choose_base_path(frame, allowed_blocks)
        model_winner, model_records = choose_model(
            frame, allowed_blocks, path_name, sets, 10000 * outer_block
        )
        features = sets[model_winner["feature_set"]]
        config = model_winner["config"]

        base_state = fit_base_state(frame, train_index, path_name)
        train_energy = apply_base(frame, train_index, base_state)
        test_energy = apply_base(frame, test_index, base_state)
        predictor_state = fit_predictor(
            frame.iloc[train_index][features].to_numpy(float),
            train_energy,
            config,
            SEED + 900000 + outer_block,
        )
        prediction, correction, touched = predict_with_state(
            predictor_state,
            frame.iloc[test_index][features].to_numpy(float),
            test_energy,
        )

        current_oof[test_index] = calibrated_baseline(
            current_all[train_index], current_all[test_index]
        )
        legacy_oof[test_index] = calibrated_baseline(
            legacy_all[train_index], legacy_all[test_index]
        )
        foundation_oof[test_index] = calibrated_baseline(
            train_energy, test_energy
        )
        candidate_oof[test_index] = prediction
        correction_oof[test_index] = correction
        touched_oof[test_index] = touched

        current_metrics = protocol_metrics(current_oof[test_index])
        legacy_metrics = protocol_metrics(legacy_oof[test_index])
        foundation_metrics = protocol_metrics(foundation_oof[test_index])
        candidate_metrics = protocol_metrics(candidate_oof[test_index])
        outer_rows.append({
            "outer_block": int(outer_block),
            "file_start": int(outer_block * BLOCK_WIDTH),
            "file_stop": int((outer_block + 1) * BLOCK_WIDTH),
            "events": int(len(test_index)),
            "base_path": path_name,
            "s1_weight": base_state["s1_weight"],
            "feature_set": model_winner["feature_set"],
            "feature_count": len(features),
            "config": config["name"],
            "current_sigma": current_metrics["sigma_median"],
            "legacy_sigma": legacy_metrics["sigma_median"],
            "foundation_sigma": foundation_metrics["sigma_median"],
            "candidate_sigma": candidate_metrics["sigma_median"],
            "candidate_worst": candidate_metrics["sigma_worst"],
            "candidate_r68": candidate_metrics["r68"],
            "candidate_r90": candidate_metrics["r90"],
            "candidate_center_bias_max": candidate_metrics["center_bias_max"],
            "near_cap_fraction": float(np.mean(touched)),
        })
        selection_records.append({
            "outer_block": int(outer_block),
            "chosen_path": path_name,
            "chosen_model": model_winner,
            "path_records": path_records,
            "model_records": model_records,
        })
        print(
            "outer {} path={} set={} config={} current={:.5f} "
            "foundation={:.5f} candidate={:.5f}".format(
                outer_block,
                path_name,
                model_winner["feature_set"],
                config["name"],
                current_metrics["sigma_median"],
                foundation_metrics["sigma_median"],
                candidate_metrics["sigma_median"],
            ),
            flush=True,
        )

    outer_table = pd.DataFrame(outer_rows)
    event_table = frame[
        ["sourceRow", "runNumber", "fileNumber", "eventNumber", "outerBlock"]
    ].copy()
    event_table["current_formula_energy"] = current_oof
    event_table["legacy_energy_cor"] = legacy_oof
    event_table["selected_foundation_energy"] = foundation_oof
    event_table["candidate_v11_energy"] = candidate_oof
    event_table["log_correction"] = correction_oof
    event_table["near_cap"] = touched_oof
    event_table.to_csv(output / "oof_events.csv", index=False)
    outer_table.to_csv(output / "outer_block_metrics.csv", index=False)
    (output / "nested_selection.json").write_text(
        json.dumps(selection_records, indent=2), encoding="utf-8"
    )

    chosen_path = Counter(
        row["chosen_path"] for row in selection_records
    ).most_common(1)[0][0]
    chosen_identity = Counter(
        (
            row["chosen_model"]["feature_set"],
            row["chosen_model"]["config"]["name"],
        )
        for row in selection_records
    ).most_common(1)[0][0]
    chosen_set, chosen_config_name = chosen_identity
    chosen_config = next(
        config for config in MODEL_CONFIGS
        if config["name"] == chosen_config_name
    )
    final_base_state = fit_base_state(frame, np.arange(n), chosen_path)
    final_energy = apply_base(frame, np.arange(n), final_base_state)
    final_features = sets[chosen_set]
    final_predictor_state = fit_predictor(
        frame[final_features].to_numpy(float),
        final_energy,
        chosen_config,
        SEED + 999999,
    )
    bundle = {
        "model_name": "PandaX_foundation_environment_v11",
        "base_state": final_base_state,
        "feature_names": final_features,
        "feature_set": chosen_set,
        "predictor_state": final_predictor_state,
        "config": chosen_config,
        "current_energy_scale": training_scale(current_all),
        "legacy_energy_scale": training_scale(legacy_all),
        "foundation_energy_scale": training_scale(final_energy),
        "training_run": 10972,
        "fileNumber_is_predictor": False,
        "warning": (
            "Validated across complete file blocks inside Run 10972 only. "
            "Independent-run blind validation is required."
        ),
    }
    joblib.dump(bundle, output / "candidate_v11.joblib")

    summary = {
        "model_name": bundle["model_name"],
        "events": int(n),
        "files": int(frame["fileNumber"].nunique()),
        "rejected_source_rows": rejected,
        "feature_sets": {k: v for k, v in sets.items()},
        "outer_choices": [
            {
                "outer_block": row["outer_block"],
                "path": row["chosen_path"],
                "feature_set": row["chosen_model"]["feature_set"],
                "feature_count": row["chosen_model"]["feature_count"],
                "config": row["chosen_model"]["config"]["name"],
            }
            for row in selection_records
        ],
        "final_choice": {
            "path": chosen_path,
            "base_state": final_base_state,
            "feature_set": chosen_set,
            "feature_count": len(final_features),
            "features": final_features,
            "config": chosen_config,
        },
        "merged_oof": {
            "current_formula": protocol_metrics(current_oof),
            "legacy_energy_cor": protocol_metrics(legacy_oof),
            "selected_foundation": protocol_metrics(foundation_oof),
            "candidate_v11": protocol_metrics(candidate_oof),
        },
        "outer_medians": {
            "current_sigma": float(np.median(outer_table["current_sigma"])),
            "legacy_sigma": float(np.median(outer_table["legacy_sigma"])),
            "foundation_sigma": float(np.median(
                outer_table["foundation_sigma"]
            )),
            "candidate_sigma": float(np.median(
                outer_table["candidate_sigma"]
            )),
        },
        "all_outer_candidate_better_than_current": bool(np.all(
            outer_table["candidate_sigma"] < outer_table["current_sigma"]
        )),
        "all_outer_candidate_better_than_foundation": bool(np.all(
            outer_table["candidate_sigma"] < outer_table["foundation_sigma"]
        )),
        "near_cap_fraction": float(np.mean(touched_oof)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "final_choice": summary["final_choice"],
        "outer_medians": summary["outer_medians"],
        "merged_candidate": {
            key: value for key, value
            in summary["merged_oof"]["candidate_v11"].items()
            if key != "protocol_rows"
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
