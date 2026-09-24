#!/usr/bin/env python3
"""Develop a spectrum-safe v12 correction using calibration plus background vetoes.

The background sample is never assigned a target energy.  It is used only to
reject candidates that saturate, extrapolate, or create narrow spectral piles.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from grouped_nested_physics_v9 import protocol_metrics


NOMINAL_KEV = 2614.5
BLOCK_WIDTH = 200
OUTER_BLOCKS = 5
SEED = 20260803

FOUNDATION = ("qS1ub_C", "qS2Bdesub_C")
FEATURE_SETS = {
    "geometry": [
        "dt", "xS2T_max", "yS2T_max", "xS2B_max", "yS2B_max",
    ],
    "geometry_ratio": [
        "dt", "xS2T_max", "yS2T_max", "xS2B_max", "yS2B_max",
        "qS1hitStdevTo1_max", "qS1channelStdevTo1_max",
        "qS2hitStdevTo1_max", "qS2channelStdevTo1_max",
    ],
}


def finite_positive(values):
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values > 0)


def load_table(path, columns):
    header = pd.read_csv(path, sep="\t", nrows=0)
    missing = [name for name in columns if name not in header.columns]
    if missing:
        raise ValueError("Missing columns in {}: {}".format(path, ", ".join(missing)))
    frame = pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)
    return frame.replace([np.inf, -np.inf], np.nan)


def fit_base_state(frame, indices):
    s1 = frame[FOUNDATION[0]].to_numpy(float)[indices]
    s2 = frame[FOUNDATION[1]].to_numpy(float)[indices]
    valid = finite_positive(s1) & finite_positive(s2)
    s1, s2 = s1[valid], s2[valid]
    med1, med2 = float(np.median(s1)), float(np.median(s2))
    z1, z2 = s1 / med1, s2 / med2
    best = None
    for weight in np.linspace(0.05, 0.95, 37):
        energy = weight * z1 + (1.0 - weight) * z2
        q16, q50, q84 = np.quantile(energy, [0.16, 0.50, 0.84])
        score = (q84 - q16) / (2.0 * q50)
        item = (float(score), float(weight))
        if best is None or item < best:
            best = item
    return {
        "s1_name": FOUNDATION[0], "s2_name": FOUNDATION[1],
        "s1_median": med1, "s2_median": med2,
        "s1_weight": best[1], "training_r68": best[0],
        "invalid_policy": "no_imputation",
    }


def apply_base(frame, state):
    s1 = frame[state["s1_name"]].to_numpy(float)
    s2 = frame[state["s2_name"]].to_numpy(float)
    valid = finite_positive(s1) & finite_positive(s2)
    energy = np.full(len(frame), np.nan)
    weight = state["s1_weight"]
    energy[valid] = (
        weight * s1[valid] / state["s1_median"]
        + (1.0 - weight) * s2[valid] / state["s2_median"]
    )
    return energy, valid


def add_geometry(frame):
    result = frame.copy()
    result["r2S2T_max"] = result["xS2T_max"] ** 2 + result["yS2T_max"] ** 2
    result["r2S2B_max"] = result["xS2B_max"] ** 2 + result["yS2B_max"] ** 2
    return result


def expanded_features(names):
    return list(names) + ["r2S2T_max", "r2S2B_max"]


def prepare_transform(values):
    values = np.asarray(values, dtype=float)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    filled = np.where(np.isfinite(values), values, medians)
    centers = np.median(filled, axis=0)
    q25, q75 = np.quantile(filled, [0.25, 0.75], axis=0)
    scales = q75 - q25
    scales[~np.isfinite(scales) | (scales < 1.0e-9)] = 1.0
    z = np.clip((filled - centers) / scales, -5.0, 5.0)
    design = np.concatenate([z, z ** 2], axis=1)
    design_center = np.mean(design, axis=0)
    design_scale = np.std(design, axis=0)
    design_scale[~np.isfinite(design_scale) | (design_scale < 1.0e-9)] = 1.0
    lower = np.quantile(filled, 0.001, axis=0) - 0.5 * scales
    upper = np.quantile(filled, 0.999, axis=0) + 0.5 * scales
    return {
        "medians": medians, "centers": centers, "scales": scales,
        "design_center": design_center, "design_scale": design_scale,
        "lower": lower, "upper": upper,
    }


def transform(values, state):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, state["medians"])
    outside = (~finite) | (filled < state["lower"]) | (filled > state["upper"])
    ood_count = np.sum(outside, axis=1)
    z = np.clip((filled - state["centers"]) / state["scales"], -5.0, 5.0)
    design = np.concatenate([z, z ** 2], axis=1)
    design = (design - state["design_center"]) / state["design_scale"]
    return design, ood_count


def peak_weights(energy):
    center = np.nanmedian(energy)
    rel = energy / center - 1.0
    return np.exp(-0.5 * (rel / 0.08) ** 2)


def fit_predictor(values, energy, alpha, cap):
    state = prepare_transform(values)
    design, _ = transform(values, state)
    target = np.log(np.clip(energy, 1.0e-12, None))
    center = np.average(target, weights=peak_weights(energy))
    target = target - center
    model = Ridge(alpha=alpha, fit_intercept=True)
    weights = peak_weights(energy)
    model.fit(design, target, sample_weight=weights)
    prediction_center = float(np.average(model.predict(design), weights=weights))
    raw = model.predict(design) - prediction_center
    delta = cap * np.tanh(raw / cap)
    corrected = energy * np.exp(-delta)
    scale = NOMINAL_KEV / np.median(corrected)
    return {
        "model": model, "transform": state,
        "prediction_center": prediction_center,
        "alpha": float(alpha), "cap": float(cap),
        "energy_scale": float(scale),
        # Never evaluate the saturated part of tanh.  This also guarantees
        # that accepted corrections remain below the near-cap definition.
        "strong_raw_limit": float(1.5 * cap),
    }


def predict(state, values, base_energy, base_valid):
    design, ood_count = transform(values, state["transform"])
    raw = state["model"].predict(design) - state["prediction_center"]
    ood = ood_count > 0
    strong = np.abs(raw) > state["strong_raw_limit"]
    correction_applied = base_valid & (~ood) & (~strong)
    delta = np.zeros(len(base_energy), dtype=float)
    delta[correction_applied] = state["cap"] * np.tanh(
        raw[correction_applied] / state["cap"]
    )
    output = np.full(len(base_energy), np.nan)
    output[base_valid] = (
        base_energy[base_valid] * np.exp(-delta[base_valid])
        * state["energy_scale"]
    )
    near_cap = correction_applied & (np.abs(delta) >= 0.95 * state["cap"])
    return {
        "energy": output, "delta": delta, "raw_delta": raw,
        "ood": ood, "ood_count": ood_count, "strong": strong,
        "applied": correction_applied, "near_cap": near_cap,
    }


def robust_metrics(energy):
    energy = np.asarray(energy, dtype=float)
    energy = energy[np.isfinite(energy) & (energy > 0)]
    q05, q10, q16, q50, q84, q90, q95 = np.quantile(
        energy, [0.05, 0.10, 0.16, 0.50, 0.84, 0.90, 0.95]
    )
    return {
        "n": int(len(energy)), "median": float(q50),
        "r68": float((q84 - q16) / (2.0 * q50)),
        "r90": float((q95 - q05) / (2.0 * q50)),
        "q10": float(q10), "q90": float(q90),
    }


def local_spike_score(energy, width=5.0, lo=1000.0, hi=3800.0):
    values = np.asarray(energy, dtype=float)
    values = values[np.isfinite(values)]
    edges = np.arange(lo, hi + width, width)
    counts, _ = np.histogram(values, bins=edges)
    best = (0.0, np.nan, 0)
    for index in range(3, len(counts) - 3):
        neighbors = np.r_[counts[index-3:index], counts[index+1:index+4]]
        baseline = float(np.median(neighbors))
        if baseline < 20:
            continue
        ratio = float(counts[index] / baseline)
        if ratio > best[0]:
            center = float((edges[index] + edges[index+1]) / 2.0)
            best = (ratio, center, int(counts[index]))
    return {"ratio": best[0], "center": best[1], "count": best[2]}


def energy_trend(base_energy, delta, applied):
    mask = np.isfinite(base_energy) & applied
    if np.sum(mask) < 100:
        return np.inf
    values = base_energy[mask]
    corr = delta[mask]
    edges = np.quantile(values, np.linspace(0, 1, 21))
    medians = []
    for left, right in zip(edges[:-1], edges[1:]):
        q = corr[(values >= left) & (values <= right)]
        if len(q) >= 20:
            medians.append(np.median(q))
    return float(np.ptp(medians)) if medians else np.inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = sorted(set(sum(FEATURE_SETS.values(), [])))
    columns = ["runNumber", "fileNumber", "eventNumber"] + list(FOUNDATION) + all_features
    calibration = add_geometry(load_table(args.calibration, columns))
    background = add_geometry(load_table(args.background, columns))

    base_state = fit_base_state(calibration, np.arange(len(calibration)))
    cal_base, cal_valid = apply_base(calibration, base_state)
    bg_base, bg_valid = apply_base(background, base_state)
    base_scale = NOMINAL_KEV / np.median(cal_base[cal_valid])
    bg_base_kev = bg_base * base_scale
    base_spike = local_spike_score(bg_base_kev)

    candidates = []
    states = {}
    for set_name, requested in FEATURE_SETS.items():
        features = expanded_features(requested)
        train_values_all = calibration[features].to_numpy(float)
        bg_values = background[features].to_numpy(float)
        for alpha in [10.0, 100.0, 1000.0]:
            for cap in [0.003, 0.005, 0.0075, 0.01]:
                oof = np.full(len(calibration), np.nan)
                fold_near = []
                fold_applied = []
                for block in range(OUTER_BLOCKS):
                    test = (calibration.fileNumber.to_numpy(int) // BLOCK_WIDTH) == block
                    train = (~test) & cal_valid
                    test_valid = test & cal_valid
                    state = fit_predictor(
                        train_values_all[train], cal_base[train], alpha, cap
                    )
                    pred = predict(state, train_values_all[test], cal_base[test], cal_valid[test])
                    oof[test] = pred["energy"]
                    fold_near.append(float(np.mean(pred["near_cap"])))
                    fold_applied.append(float(np.mean(pred["applied"])))

                final_state = fit_predictor(
                    train_values_all[cal_valid], cal_base[cal_valid], alpha, cap
                )
                bg_pred = predict(final_state, bg_values, bg_base, bg_valid)
                cal_metric = robust_metrics(oof[cal_valid])
                bg_metric = robust_metrics(bg_pred["energy"][bg_valid])
                bg_spike = local_spike_score(bg_pred["energy"])
                spike_growth = bg_spike["ratio"] / max(base_spike["ratio"], 1.0)
                trend = energy_trend(bg_base, bg_pred["delta"], bg_pred["applied"])
                rank = pd.Series(bg_base[bg_valid]).corr(
                    pd.Series(bg_pred["energy"][bg_valid]), method="spearman"
                )
                row = {
                    "name": "{}_a{}_c{}".format(set_name, int(alpha), int(cap*1000)),
                    "feature_set": set_name, "feature_count": len(features),
                    "alpha": alpha, "cap": cap,
                    "calibration_r68": cal_metric["r68"],
                    "calibration_r90": cal_metric["r90"],
                    "calibration_applied_median": float(np.median(fold_applied)),
                    "calibration_near_cap_max": float(np.max(fold_near)),
                    "background_applied": float(np.mean(bg_pred["applied"])),
                    "background_ood": float(np.mean(bg_pred["ood"])),
                    "background_strong": float(np.mean(bg_pred["strong"])),
                    "background_near_cap": float(np.mean(bg_pred["near_cap"])),
                    "background_energy_trend": trend,
                    "background_rank_corr": float(rank),
                    "background_spike_ratio": bg_spike["ratio"],
                    "background_spike_center": bg_spike["center"],
                    "background_spike_growth": spike_growth,
                    "background_r68": bg_metric["r68"],
                }
                row["safe"] = bool(
                    row["background_near_cap"] <= 0.01
                    and row["background_energy_trend"] <= 0.01
                    and row["background_rank_corr"] >= 0.999
                    and row["background_spike_growth"] <= 1.25
                )
                row["score"] = float(
                    row["calibration_r68"]
                    + 2.0 * max(row["background_spike_growth"] - 1.0, 0.0)
                    + 5.0 * row["background_near_cap"]
                    + 2.0 * row["background_energy_trend"]
                )
                candidates.append(row)
                states[row["name"]] = (final_state, features, bg_pred, oof)
                print(json.dumps(row, sort_keys=True), flush=True)

    safe_rows = [row for row in candidates if row["safe"]]
    if not safe_rows:
        pd.DataFrame(candidates).sort_values("score").to_csv(
            output_dir / "candidate_search.csv", index=False
        )
        raise RuntimeError(
            "No candidate passed all background safety gates; no model exported"
        )
    winner = sorted(
        safe_rows, key=lambda row: (row["calibration_r68"], row["score"])
    )[0]
    final_state, final_features, bg_pred, oof = states[winner["name"]]
    bundle = {
        "model_name": "PandaX_safe_general_v12",
        "version": "v12_safe_20260803",
        "training_run": 10972,
        "nominal_energy_kev": NOMINAL_KEV,
        "base_state": base_state,
        "base_energy_scale": float(base_scale),
        "feature_names": final_features,
        "predictor_state": final_state,
        "safety_policy": {
            "foundation_imputation": False,
            "ood_fallback": True,
            "strong_raw_fallback": True,
            "background_used_as_label": False,
            "time_is_predictor": False,
        },
        "selection": winner,
        "warning": (
            "Trained on a single 2614.5 keV calibration line. The background "
            "sample is used only for anti-sculpting vetoes. Multi-line validation "
            "is required before claiming full-energy-range reconstruction."
        ),
    }
    joblib.dump(bundle, output_dir / "candidate_v12_safe.joblib")

    pd.DataFrame(candidates).sort_values(["safe", "score"], ascending=[False, True]).to_csv(
        output_dir / "candidate_search.csv", index=False
    )
    cal_out = calibration[["runNumber", "fileNumber", "eventNumber"]].copy()
    cal_out["base_valid"] = cal_valid
    cal_out["foundation_energy_kev"] = cal_base * base_scale
    cal_out["candidate_v12_oof_energy_kev"] = oof
    cal_out.to_csv(output_dir / "calibration_oof_events.csv", index=False)

    bg_out = background[["runNumber", "fileNumber", "eventNumber"]].copy()
    bg_out["base_valid"] = bg_valid
    bg_out["foundation_energy_kev"] = bg_base_kev
    bg_out["candidate_v12_energy_kev"] = bg_pred["energy"]
    bg_out["log_correction"] = bg_pred["delta"]
    bg_out["correction_applied"] = bg_pred["applied"]
    bg_out["out_of_distribution"] = bg_pred["ood"]
    bg_out["strong_raw_fallback"] = bg_pred["strong"]
    bg_out["correction_near_cap"] = bg_pred["near_cap"]
    bg_out.to_csv(output_dir / "background_events.csv", index=False)
    bg_out.to_csv(output_dir / "background_events.txt", sep=" ", index=False, na_rep="NaN")

    summary = {
        "winner": winner,
        "safe_candidates": len(safe_rows),
        "total_candidates": len(candidates),
        "base_state": base_state,
        "calibration_base_metrics": robust_metrics(cal_base[cal_valid] * base_scale),
        "calibration_candidate_metrics": robust_metrics(oof[cal_valid]),
        "calibration_candidate_protocols": {
            k: v for k, v in protocol_metrics(oof[cal_valid]).items()
            if k != "protocol_rows"
        },
        "background_base_valid_fraction": float(np.mean(bg_valid)),
        "background_base_spike": base_spike,
        "background_candidate_spike": local_spike_score(bg_pred["energy"]),
        "background_candidate_metrics": robust_metrics(bg_pred["energy"][bg_valid]),
        "warning": bundle["warning"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("WINNER=" + winner["name"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
