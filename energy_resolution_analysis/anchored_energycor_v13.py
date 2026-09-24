#!/usr/bin/env python3
"""Train v13 as a bounded geometry correction anchored to the given Energy_cor."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from safe_general_v12 import (
    add_geometry,
    energy_trend,
    expanded_features,
    fit_predictor,
    load_table,
    local_spike_score,
    predict,
    prepare_transform,
    robust_metrics,
    transform,
)
from grouped_nested_physics_v9 import protocol_metrics


NOMINAL_KEV = 2614.5
BLOCK_WIDTH = 200
OUTER_BLOCKS = 5
FEATURE_SETS = {
    "drift_only": ["dt"],
    "geometry": ["dt", "xS2T_max", "yS2T_max", "xS2B_max", "yS2B_max"],
    "geometry_ratio": [
        "dt", "xS2T_max", "yS2T_max", "xS2B_max", "yS2B_max",
        "qS1hitStdevTo1_max", "qS1channelStdevTo1_max",
        "qS2hitStdevTo1_max", "qS2channelStdevTo1_max",
    ],
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


def fit_anchored_predictor(values, energy, alpha, cap):
    """Learn only geometry-correlated residuals; never change global energy scale."""
    state = prepare_transform(values)
    design, _ = transform(values, state)
    target = np.log(np.clip(energy / NOMINAL_KEV, 1.0e-12, None))
    # The input contains continuum.  Give weight only by proximity to the known
    # line, while keeping a broad width to avoid a hard energy-window edge.
    weights = np.exp(-0.5 * ((energy - NOMINAL_KEV) / (0.10 * NOMINAL_KEV)) ** 2)
    model = Ridge(alpha=alpha, fit_intercept=True)
    center = float(np.average(target, weights=weights))
    model.fit(design, target - center, sample_weight=weights)
    prediction_center = float(np.average(model.predict(design), weights=weights))
    return {
        "model": model,
        "transform": state,
        "prediction_center": prediction_center,
        "alpha": float(alpha),
        "cap": float(cap),
        "energy_scale": 1.0,
        "strong_raw_limit": float(1.5 * cap),
        "global_scale_policy": "fixed_to_one",
    }


def time_audit(frame, base, candidate):
    table = pd.DataFrame({
        "fileNumber": frame.fileNumber.to_numpy(int),
        "ratio": candidate / base,
    })
    table["block"] = pd.qcut(table.fileNumber, 10, labels=False, duplicates="drop")
    medians = []
    for _, part in table.groupby("block"):
        values = part.ratio.to_numpy(float)
        values = values[np.isfinite(values)]
        medians.append(np.median(values))
    return {
        "median_ratio_min": float(np.min(medians)),
        "median_ratio_max": float(np.max(medians)),
        "median_ratio_span": float(np.ptp(medians)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = sorted(set(sum(FEATURE_SETS.values(), [])))
    columns = [
        "runNumber", "fileNumber", "eventNumber", "qS1ub_C", "qS2Bdesub_C"
    ] + all_features
    calibration = add_geometry(load_table(args.calibration, columns))
    background = add_geometry(load_table(args.background, columns))
    cal_base, cal_valid = energy_cor(calibration)
    bg_base, bg_valid = energy_cor(background)
    cal_base_metrics = robust_metrics(cal_base[cal_valid])
    cal_base_protocol = protocol_metrics(cal_base[cal_valid])
    bg_base_spike = local_spike_score(bg_base)

    candidates = []
    states = {}
    for set_name, requested in FEATURE_SETS.items():
        features = expanded_features(requested)
        # Radius terms are only meaningful when their source coordinates exist.
        if set_name == "drift_only":
            features = ["dt"]
        cal_values = calibration[features].to_numpy(float)
        bg_values = background[features].to_numpy(float)
        for alpha in [10.0, 100.0, 1000.0]:
            for cap in [0.0025, 0.005, 0.0075, 0.01]:
                oof = np.full(len(calibration), np.nan)
                for block in range(OUTER_BLOCKS):
                    test = (calibration.fileNumber.to_numpy(int) // BLOCK_WIDTH) == block
                    train = (~test) & cal_valid
                    state = fit_anchored_predictor(
                        cal_values[train], cal_base[train], alpha, cap
                    )
                    pred = predict(state, cal_values[test], cal_base[test], cal_valid[test])
                    oof[test] = pred["energy"]

                final_state = fit_anchored_predictor(
                    cal_values[cal_valid], cal_base[cal_valid], alpha, cap
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
                time = time_audit(background, bg_base, bg_pred["energy"])
                row = {
                    "name": "{}_a{}_c{}".format(set_name, int(alpha), int(cap * 10000)),
                    "feature_set": set_name,
                    "feature_count": len(features),
                    "alpha": alpha,
                    "cap": cap,
                    "calibration_base_r68": cal_base_metrics["r68"],
                    "calibration_oof_r68": cal_metric["r68"],
                    "calibration_oof_r90": cal_metric["r90"],
                    "relative_r68_improvement": 1.0 - cal_metric["r68"] / cal_base_metrics["r68"],
                    "calibration_base_sigma": cal_base_protocol["sigma_median"],
                    "calibration_oof_sigma": cal_protocol["sigma_median"],
                    "relative_sigma_improvement": 1.0 - cal_protocol["sigma_median"] / cal_base_protocol["sigma_median"],
                    "calibration_protocol_successes": cal_protocol["protocol_successes"],
                    "calibration_center_bias_max": cal_protocol["center_bias_max"],
                    "background_applied": float(np.mean(bg_pred["applied"])),
                    "background_ood": float(np.mean(bg_pred["ood"])),
                    "background_strong": float(np.mean(bg_pred["strong"])),
                    "background_near_cap": float(np.mean(bg_pred["near_cap"])),
                    "background_energy_trend": energy_trend(bg_base, bg_pred["delta"], bg_pred["applied"]),
                    "background_rank_corr": float(rank),
                    "background_spike_growth": float(spike_growth),
                    "background_median_scale_shift": float(np.median(ratio) - 1.0),
                    "background_ratio_q01": float(np.quantile(ratio, 0.01)),
                    "background_ratio_q99": float(np.quantile(ratio, 0.99)),
                    "time_ratio_span": time["median_ratio_span"],
                }
                row["safe"] = bool(
                    row["relative_sigma_improvement"] > 0
                    and row["calibration_protocol_successes"] >= 10
                    and abs(
                        row["calibration_center_bias_max"]
                        - cal_base_protocol["center_bias_max"]
                    ) <= 0.002
                    and row["background_applied"] >= 0.10
                    and row["background_near_cap"] == 0
                    and row["background_rank_corr"] >= 0.9999
                    and row["background_spike_growth"] <= 1.10
                    and abs(row["background_median_scale_shift"]) <= 0.01
                    and row["time_ratio_span"] <= 0.002
                )
                candidates.append(row)
                states[row["name"]] = (final_state, features, bg_pred, oof)
                print(json.dumps(row, sort_keys=True), flush=True)

    safe = [row for row in candidates if row["safe"]]
    pd.DataFrame(candidates).sort_values(
        ["safe", "calibration_oof_sigma"], ascending=[False, True]
    ).to_csv(output_dir / "candidate_search.csv", index=False)
    if not safe:
        raise RuntimeError("No v13 candidate passed all safety gates")
    winner = sorted(safe, key=lambda row: row["calibration_oof_sigma"])[0]
    state, features, bg_pred, oof = states[winner["name"]]
    bundle = {
        "model_name": "PandaX_EnergyCor_anchored_v13",
        "version": "v13_anchored_20260807",
        "nominal_energy_kev": NOMINAL_KEV,
        "base_formula": "given_Energy_cor",
        "feature_names": features,
        "predictor_state": state,
        "selection": winner,
        "warning": (
            "Geometry-only bounded correction anchored to the supplied Energy_cor. "
            "Single-line training still requires multi-line external validation."
        ),
    }
    joblib.dump(bundle, output_dir / "candidate_v13_anchored.joblib")

    cal_out = calibration[["runNumber", "fileNumber", "eventNumber"]].copy()
    cal_out["energy_cor_kev"] = cal_base
    cal_out["candidate_v13_oof_energy_kev"] = oof
    cal_out.to_csv(output_dir / "calibration_oof_events.csv", index=False)
    bg_out = background[["runNumber", "fileNumber", "eventNumber"]].copy()
    bg_out["energy_cor_kev"] = bg_base
    bg_out["candidate_v13_energy_kev"] = bg_pred["energy"]
    bg_out["log_correction"] = bg_pred["delta"]
    bg_out["correction_applied"] = bg_pred["applied"]
    bg_out["out_of_distribution"] = bg_pred["ood"]
    bg_out["strong_raw_fallback"] = bg_pred["strong"]
    bg_out["correction_near_cap"] = bg_pred["near_cap"]
    bg_out.to_csv(output_dir / "background_events.csv", index=False)
    bg_out.to_csv(output_dir / "background_events.txt", sep=" ", index=False, na_rep="NaN")

    summary = {
        "winner": winner,
        "safe_candidates": len(safe),
        "total_candidates": len(candidates),
        "calibration_base_metrics": cal_base_metrics,
        "calibration_base_protocols": {
            key: value for key, value in cal_base_protocol.items()
            if key != "protocol_rows"
        },
        "calibration_candidate_metrics": robust_metrics(oof[cal_valid]),
        "calibration_candidate_protocols": {
            key: value for key, value in protocol_metrics(oof[cal_valid]).items()
            if key != "protocol_rows"
        },
        "background_base_spike": bg_base_spike,
        "background_candidate_spike": local_spike_score(bg_pred["energy"]),
        "background_time_audit": time_audit(background, bg_base, bg_pred["energy"]),
        "warning": bundle["warning"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("WINNER=" + winner["name"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
