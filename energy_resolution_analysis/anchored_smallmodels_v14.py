#!/usr/bin/env python3
"""Compare ten small, bounded Energy_cor-anchored correction models.

The model input never contains fileNumber or an absolute S1/S2 charge.  Model
selection uses time-contiguous OOF folds and background-only spectrum vetoes.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from safe_general_v12 import (
    energy_trend,
    load_table,
    local_spike_score,
    predict,
    prepare_transform,
    robust_metrics,
    transform,
)
from grouped_nested_physics_v9 import protocol_metrics


NOMINAL_KEV = 2614.5
OUTER_FOLDS = 5
ALPHA = 1000.0
CAP = 0.0025

F_DT = "dt"
F_S2_PATTERN = "qS2channelStdevTo1_max"
F_X = "xS2T_max"
F_Y = "yS2B_max"
F_S1_PATTERN = "qS1channelStdevTo1_max"
F_S2_WIDTH = "wS2CDF_max"

# Exactly ten small model structures.  M02--M06 isolate one added factor;
# M07--M10 test physically motivated combinations.
MODEL_SETS = {
    "M01_dt": [F_DT],
    "M02_dt_s2pattern": [F_DT, F_S2_PATTERN],
    "M03_dt_x": [F_DT, F_X],
    "M04_dt_y": [F_DT, F_Y],
    "M05_dt_s1pattern": [F_DT, F_S1_PATTERN],
    "M06_dt_s2width": [F_DT, F_S2_WIDTH],
    "M07_dt_xy": [F_DT, F_X, F_Y],
    "M08_dt_patterns": [F_DT, F_S2_PATTERN, F_S1_PATTERN],
    "M09_dt_xy_width": [F_DT, F_X, F_Y, F_S2_WIDTH],
    "M10_all_six": [
        F_DT, F_S2_PATTERN, F_X, F_Y, F_S1_PATTERN, F_S2_WIDTH,
    ],
}

HISTORICAL_IMPORTANCE = {
    F_S2_PATTERN: {"rank": 1, "normalized": 1.0000},
    F_X: {"rank": 2, "normalized": 0.9681},
    F_Y: {"rank": 3, "normalized": 0.8901},
    F_S1_PATTERN: {"rank": 4, "normalized": 0.8438},
    F_S2_WIDTH: {"rank": 12, "normalized": 0.5033},
    F_DT: {"rank": 38, "normalized": 0.2392},
}


def energy_cor(frame):
    s1 = frame.qS1ub_C.to_numpy(float)
    s2 = frame.qS2Bdesub_C.to_numpy(float)
    valid = np.isfinite(s1) & np.isfinite(s2) & (s1 > 0) & (s2 > 0)
    raw = np.full(len(frame), np.nan)
    raw[valid] = (s1[valid] / 0.125 + s2[valid] / 10.58) * 0.0137
    output = np.full(len(frame), np.nan)
    x = raw[valid]
    output[valid] = (
        -1.73706e-09 * x ** 3 + 7.98193e-06 * x ** 2
        + 1.07904 * x - 9.22086
    )
    valid &= np.isfinite(output) & (output > 0)
    return output, valid


def fit_anchored_predictor(values, energy):
    state = prepare_transform(values)
    design, _ = transform(values, state)
    target = np.log(np.clip(energy / NOMINAL_KEV, 1.0e-12, None))
    weights = np.exp(-0.5 * ((energy - NOMINAL_KEV) / (0.10 * NOMINAL_KEV)) ** 2)
    model = Ridge(alpha=ALPHA, fit_intercept=True)
    center = float(np.average(target, weights=weights))
    model.fit(design, target - center, sample_weight=weights)
    prediction_center = float(np.average(model.predict(design), weights=weights))
    return {
        "model": model,
        "transform": state,
        "prediction_center": prediction_center,
        "alpha": ALPHA,
        "cap": CAP,
        "energy_scale": 1.0,
        "strong_raw_limit": float(1.5 * CAP),
        "global_scale_policy": "fixed_to_one",
    }


def chronological_folds(timestamp, n_folds):
    values = np.asarray(timestamp, dtype=float)
    finite = np.isfinite(values)
    fill = np.nanmedian(values[finite]) if np.any(finite) else 0.0
    order = np.argsort(np.where(finite, values, fill), kind="mergesort")
    fold = np.empty(len(values), dtype=int)
    fold[order] = np.minimum(
        np.arange(len(values), dtype=int) * n_folds // max(len(values), 1),
        n_folds - 1,
    )
    return fold


def time_audit(timestamp, base, candidate):
    table = pd.DataFrame({
        "time": np.asarray(timestamp, dtype=float),
        "ratio": np.asarray(candidate, dtype=float) / np.asarray(base, dtype=float),
    })
    valid = np.isfinite(table.time) & np.isfinite(table.ratio)
    table = table.loc[valid].copy()
    table["block"] = pd.qcut(table.time, 10, labels=False, duplicates="drop")
    medians = []
    applied_fractions = []
    supported_blocks = 0
    for _, part in table.groupby("block"):
        ratio = part.ratio.to_numpy(float)
        applied = np.abs(ratio - 1.0) > 1.0e-12
        applied_fractions.append(float(np.mean(applied)))
        if np.sum(applied) >= 20:
            medians.append(float(np.median(ratio[applied])))
            supported_blocks += 1
    if not medians:
        return {"median_ratio_min": np.nan, "median_ratio_max": np.nan,
                "median_ratio_span": np.inf,
                "applied_fraction_min": 0.0, "applied_fraction_max": 0.0,
                "applied_fraction_span": np.inf, "supported_blocks": 0}
    return {
        "median_ratio_min": float(np.min(medians)),
        "median_ratio_max": float(np.max(medians)),
        "median_ratio_span": float(np.ptp(medians)),
        "applied_fraction_min": float(np.min(applied_fractions)),
        "applied_fraction_max": float(np.max(applied_fractions)),
        "applied_fraction_span": float(np.ptp(applied_fractions)),
        "supported_blocks": int(supported_blocks),
    }


def feature_quality(frame, features):
    rows = []
    for name in features:
        values = frame[name].to_numpy(float)
        finite = np.isfinite(values)
        unique = len(np.unique(values[finite])) if np.any(finite) else 0
        rows.append({
            "feature": name,
            "finite_fraction": float(np.mean(finite)),
            "missing_fraction": float(np.mean(~finite)),
            "unique_finite": int(unique),
        })
    return rows


def safety_flags(row, base_protocol):
    flags = {
        "sigma_improves": row["relative_sigma_improvement"] > 0,
        "r68_not_worse": row["relative_r68_improvement"] >= -0.001,
        "fit_protocol_ok": row["calibration_protocol_successes"] >= 10,
        "center_stable": abs(
            row["calibration_center_bias_max"] - base_protocol["center_bias_max"]
        ) <= 0.002,
        "background_coverage": row["background_applied"] >= 0.10,
        "background_ood": row["background_ood"] <= 0.20,
        "no_cap_pileup": row["background_near_cap"] == 0,
        "ranking_preserved": row["background_rank_corr"] >= 0.9999,
        "no_spike_growth": row["background_spike_growth"] <= 1.10,
        "scale_stable": abs(row["background_median_scale_shift"]) <= 0.001,
        "time_stable": row["time_ratio_span"] <= 0.002,
        "time_coverage_stable": row["time_applied_fraction_span"] <= 0.25,
        "time_blocks_supported": row["time_supported_blocks"] >= 8,
    }
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = sorted(set(sum(MODEL_SETS.values(), [])))
    columns = [
        "runNumber", "eventNumber", "t", "qS1ub_C", "qS2Bdesub_C"
    ] + all_features
    calibration = load_table(args.calibration, columns)
    background = load_table(args.background, columns)
    cal_base, cal_valid = energy_cor(calibration)
    bg_base, bg_valid = energy_cor(background)
    base_metrics = robust_metrics(cal_base[cal_valid])
    base_protocol = protocol_metrics(cal_base[cal_valid])
    bg_base_spike = local_spike_score(bg_base)
    folds = chronological_folds(calibration.t.to_numpy(float), OUTER_FOLDS)

    quality = {
        "historical_importance": HISTORICAL_IMPORTANCE,
        "calibration": feature_quality(calibration, all_features),
        "background": feature_quality(background, all_features),
    }
    (output_dir / "selected_feature_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True), encoding="utf-8"
    )

    candidates = []
    states = {}
    for model_name, features in MODEL_SETS.items():
        cal_values = calibration[features].to_numpy(float)
        bg_values = background[features].to_numpy(float)
        oof = np.full(len(calibration), np.nan)
        for block in range(OUTER_FOLDS):
            test = folds == block
            train = (~test) & cal_valid
            state = fit_anchored_predictor(cal_values[train], cal_base[train])
            pred = predict(state, cal_values[test], cal_base[test], cal_valid[test])
            oof[test] = pred["energy"]

        final_state = fit_anchored_predictor(
            cal_values[cal_valid], cal_base[cal_valid]
        )
        bg_pred = predict(final_state, bg_values, bg_base, bg_valid)
        cal_metric = robust_metrics(oof[cal_valid])
        cal_protocol = protocol_metrics(oof[cal_valid])
        bg_spike = local_spike_score(bg_pred["energy"])
        spike_growth = bg_spike["ratio"] / max(bg_base_spike["ratio"], 1.0)
        ratio = bg_pred["energy"][bg_valid] / bg_base[bg_valid]
        rank = pd.Series(bg_base[bg_valid]).corr(
            pd.Series(bg_pred["energy"][bg_valid]), method="spearman"
        )
        time = time_audit(
            background.t.to_numpy(float), bg_base, bg_pred["energy"]
        )
        row = {
            "name": model_name,
            "features": "|".join(features),
            "feature_count": len(features),
            "alpha": ALPHA,
            "cap": CAP,
            "calibration_base_r68": base_metrics["r68"],
            "calibration_oof_r68": cal_metric["r68"],
            "calibration_oof_r90": cal_metric["r90"],
            "relative_r68_improvement": 1.0 - cal_metric["r68"] / base_metrics["r68"],
            "calibration_base_sigma": base_protocol["sigma_median"],
            "calibration_oof_sigma": cal_protocol["sigma_median"],
            "relative_sigma_improvement": 1.0 - cal_protocol["sigma_median"] / base_protocol["sigma_median"],
            "calibration_protocol_successes": cal_protocol["protocol_successes"],
            "calibration_center_bias_max": cal_protocol["center_bias_max"],
            "background_applied": float(np.mean(bg_pred["applied"])),
            "background_ood": float(np.mean(bg_pred["ood"])),
            "background_strong": float(np.mean(bg_pred["strong"])),
            "background_near_cap": float(np.mean(bg_pred["near_cap"])),
            "background_energy_trend": energy_trend(
                bg_base, bg_pred["delta"], bg_pred["applied"]
            ),
            "background_rank_corr": float(rank),
            "background_spike_growth": float(spike_growth),
            "background_median_scale_shift": float(np.median(ratio) - 1.0),
            "background_ratio_q01": float(np.quantile(ratio, 0.01)),
            "background_ratio_q99": float(np.quantile(ratio, 0.99)),
            "time_ratio_span": time["median_ratio_span"],
            "time_applied_fraction_span": time["applied_fraction_span"],
            "time_supported_blocks": time["supported_blocks"],
        }
        flags = safety_flags(row, base_protocol)
        row["safety_pass_count"] = int(sum(bool(value) for value in flags.values()))
        row["safe"] = bool(all(flags.values()))
        candidates.append(row)
        states[model_name] = (final_state, features, bg_pred, oof, flags)
        print(json.dumps(dict(row, safety_flags=flags), sort_keys=True), flush=True)

    table = pd.DataFrame(candidates).sort_values(
        ["safe", "safety_pass_count", "calibration_oof_sigma"],
        ascending=[False, False, True],
    )
    table.to_csv(output_dir / "ten_small_models.csv", index=False)

    safe_rows = [row for row in candidates if row["safe"]]
    if safe_rows:
        winner = sorted(safe_rows, key=lambda row: row["calibration_oof_sigma"])[0]
        selection_status = "safe_candidate_selected"
    else:
        winner = sorted(
            candidates,
            key=lambda row: (-row["safety_pass_count"], row["calibration_oof_sigma"]),
        )[0]
        selection_status = "no_model_passed_all_gates_best_diagnostic_only"

    state, features, bg_pred, oof, flags = states[winner["name"]]
    bundle = {
        "model_name": "PandaX_EnergyCor_smallmodels_v14",
        "version": "v14_smallmodels_20260814",
        "status": selection_status,
        "base_formula": "given_Energy_cor",
        "feature_names": features,
        "predictor_state": state,
        "selection": winner,
        "safety_flags": flags,
        "warning": (
            "Single-run mixed calibration sample. Do not promote without "
            "multi-line and cross-run validation."
        ),
    }
    joblib.dump(bundle, output_dir / "candidate_v14_smallmodels.joblib")

    cal_out = calibration[["runNumber", "eventNumber", "t"]].copy()
    cal_out["energy_cor_kev"] = cal_base
    cal_out["candidate_v14_oof_energy_kev"] = oof
    cal_out.to_csv(output_dir / "calibration_oof_events.csv", index=False)
    bg_out = background[["runNumber", "eventNumber", "t"]].copy()
    bg_out["energy_cor_kev"] = bg_base
    bg_out["candidate_v14_energy_kev"] = bg_pred["energy"]
    bg_out["log_correction"] = bg_pred["delta"]
    bg_out["correction_applied"] = bg_pred["applied"]
    bg_out["out_of_distribution"] = bg_pred["ood"]
    bg_out["strong_raw_fallback"] = bg_pred["strong"]
    bg_out.to_csv(output_dir / "background_events.csv", index=False)

    summary = {
        "selection_status": selection_status,
        "winner": winner,
        "winner_safety_flags": flags,
        "safe_models": len(safe_rows),
        "total_models": len(candidates),
        "model_sets": MODEL_SETS,
        "historical_importance": HISTORICAL_IMPORTANCE,
        "calibration_base_metrics": base_metrics,
        "calibration_base_protocols": {
            key: value for key, value in base_protocol.items()
            if key != "protocol_rows"
        },
        "calibration_candidate_metrics": robust_metrics(oof[cal_valid]),
        "calibration_candidate_protocols": {
            key: value for key, value in protocol_metrics(oof[cal_valid]).items()
            if key != "protocol_rows"
        },
        "background_base_spike": bg_base_spike,
        "background_candidate_spike": local_spike_score(bg_pred["energy"]),
        "warning": bundle["warning"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("OUTPUT", str(output_dir))
    print("SELECTION", selection_status, winner["name"])


if __name__ == "__main__":
    main()
