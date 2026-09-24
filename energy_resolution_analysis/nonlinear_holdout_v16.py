#!/usr/bin/env python3
"""Constrained nonlinear follow-up to the v15 multivariable search.

Candidate selection uses only the first 80% of the calibration timeline.  The
last 20% is kept untouched until one candidate has been selected.  Background
events have no target label and are used only for spectrum-shape safety vetoes.
"""

from __future__ import print_function

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from anchored_smallmodels_v14 import chronological_folds, energy_cor, time_audit
from grouped_nested_physics_v9 import protocol_metrics
from safe_general_v12 import energy_trend, load_table, local_spike_score, robust_metrics


NOMINAL_KEV = 2614.5
CAPS = (0.0025, 0.0050, 0.0075)

FEATURE_SETS = {
    "F01_v15_corrected_position": [
        "dt", "wS2CDF_max", "yS2Tcor_max", "xS2Bcor_max",
    ],
    "F02_top_position": [
        "dt", "wS2CDF_max", "xS2T_max", "yS2T_max",
    ],
    "F03_both_position_systems": [
        "dt", "wS2CDF_max", "xS2T_max", "yS2T_max",
        "yS2Tcor_max", "xS2Bcor_max",
    ],
    "F04_position_and_shape": [
        "dt", "wS2CDF_max", "yS2Tcor_max", "xS2Bcor_max",
        "wS2FWHM_max", "tDiffBottomTopS2_max",
    ],
}

FAMILIES = {
    "ridge_additive": (300.0, 1000.0, 3000.0),
    "ridge_interactions": (300.0, 1000.0, 3000.0),
    "ridge_spline": (300.0, 1000.0, 3000.0),
    "shallow_gbr": (80.0, 160.0, 320.0),
}


def prepare(values):
    x = np.asarray(values, dtype=float)
    medians = np.nanmedian(x, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    filled = np.where(np.isfinite(x), x, medians)
    centers = np.median(filled, axis=0)
    q25, q75 = np.quantile(filled, [0.25, 0.75], axis=0)
    scales = q75 - q25
    scales[~np.isfinite(scales) | (scales < 1.0e-9)] = 1.0
    lower = np.quantile(filled, 0.0005, axis=0) - scales
    upper = np.quantile(filled, 0.9995, axis=0) + scales
    return {
        "medians": medians, "centers": centers, "scales": scales,
        "lower": lower, "upper": upper,
    }


def standardized(values, state):
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    filled = np.where(finite, x, state["medians"])
    outside = (~finite) | (filled < state["lower"]) | (filled > state["upper"])
    z = np.clip((filled - state["centers"]) / state["scales"], -5.0, 5.0)
    return z, np.sum(outside, axis=1)


def raw_design(z, family):
    if family == "ridge_additive":
        return np.concatenate([z, z ** 2], axis=1)
    if family == "ridge_interactions":
        parts = [z, z ** 2]
        for i in range(z.shape[1]):
            for j in range(i + 1, z.shape[1]):
                parts.append((z[:, i] * z[:, j])[:, None])
        return np.concatenate(parts, axis=1)
    if family == "ridge_spline":
        parts = [z, z ** 2, z ** 3]
        for knot in (-1.5, -0.5, 0.5, 1.5):
            parts.append(np.maximum(z - knot, 0.0) ** 3)
        return np.concatenate(parts, axis=1)
    if family == "shallow_gbr":
        return z
    raise ValueError(family)


def fit_state(values, energy, family, hyper, cap):
    prep = prepare(values)
    z, _ = standardized(values, prep)
    design = raw_design(z, family)
    dcenter = np.mean(design, axis=0)
    dscale = np.std(design, axis=0)
    dscale[~np.isfinite(dscale) | (dscale < 1.0e-9)] = 1.0
    if family != "shallow_gbr":
        design = (design - dcenter) / dscale
    target = np.log(np.clip(energy / NOMINAL_KEV, 1.0e-12, None))
    weights = np.exp(-0.5 * ((energy - NOMINAL_KEV) / (0.10 * NOMINAL_KEV)) ** 2)
    target_center = float(np.average(target, weights=weights))
    if family == "shallow_gbr":
        model = GradientBoostingRegressor(
            loss="huber", learning_rate=0.03, n_estimators=120,
            max_depth=2, min_samples_leaf=int(hyper), subsample=0.8,
            random_state=20260814,
        )
    else:
        model = Ridge(alpha=float(hyper), fit_intercept=True)
    model.fit(design, target - target_center, sample_weight=weights)
    prediction_center = float(np.average(model.predict(design), weights=weights))
    return {
        "preprocess": prep, "family": family, "hyper": float(hyper),
        "design_center": dcenter, "design_scale": dscale,
        "model": model, "prediction_center": prediction_center,
        "cap": float(cap), "strong_raw_limit": float(1.5 * cap),
    }


def apply_state(state, values, base, valid):
    z, ood_count = standardized(values, state["preprocess"])
    design = raw_design(z, state["family"])
    if state["family"] != "shallow_gbr":
        design = (design - state["design_center"]) / state["design_scale"]
    raw = state["model"].predict(design) - state["prediction_center"]
    allowed = max(1, int(np.floor(0.20 * values.shape[1])))
    ood = ood_count > allowed
    strong = np.abs(raw) > state["strong_raw_limit"]
    applied = valid & (~ood) & (~strong)
    delta = np.zeros(len(base), dtype=float)
    delta[applied] = state["cap"] * np.tanh(raw[applied] / state["cap"])
    output = np.full(len(base), np.nan)
    output[valid] = base[valid] * np.exp(-delta[valid])
    near_cap = applied & (np.abs(delta) >= 0.95 * state["cap"])
    return {
        "energy": output, "delta": delta, "applied": applied,
        "ood": ood, "strong": strong, "near_cap": near_cap,
    }


def safety(row, base_protocol):
    flags = {
        "sigma_improves": row["dev_sigma_gain"] > 0,
        "r68_not_worse": row["dev_r68_gain"] >= -0.001,
        "r90_not_worse": row["dev_r90_gain"] >= -0.002,
        "fit_protocol_ok": row["dev_protocol_successes"] >= 10,
        "center_stable": abs(row["dev_center_bias_max"] - base_protocol["center_bias_max"]) <= 0.003,
        "background_coverage": row["background_applied"] >= 0.10,
        "background_ood": row["background_ood"] <= 0.20,
        "background_fallback": row["background_strong"] <= 0.85,
        "no_cap_pileup": row["background_near_cap"] <= 0.001,
        "ranking_preserved": row["background_rank_corr"] >= 0.9995,
        "no_global_spike_growth": row["background_spike_growth"] <= 1.10,
        "no_2600_spike_growth": row["background_2600_spike_growth"] <= 1.10,
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
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    all_features = sorted(set(itertools.chain.from_iterable(FEATURE_SETS.values())))
    columns = ["runNumber", "eventNumber", "t", "qS1ub_C", "qS2Bdesub_C"] + all_features
    cal = load_table(args.calibration, columns)
    bg = load_table(args.background, columns)
    cal_base, cal_valid = energy_cor(cal)
    bg_base, bg_valid = energy_cor(bg)

    order = np.argsort(cal.t.to_numpy(float), kind="mergesort")
    split = int(0.80 * len(cal))
    dev_mask = np.zeros(len(cal), dtype=bool)
    dev_mask[order[:split]] = True
    hold_mask = ~dev_mask
    dev_indices = np.flatnonzero(dev_mask)
    hold_indices = np.flatnonzero(hold_mask)
    dev_folds = chronological_folds(cal.t.to_numpy(float)[dev_indices], 4)

    dev_base_robust = robust_metrics(cal_base[dev_mask & cal_valid])
    dev_base_protocol = protocol_metrics(cal_base[dev_mask & cal_valid])
    bg_spike = local_spike_score(bg_base)
    bg_2600 = local_spike_score(bg_base, width=5.0, lo=2450.0, hi=2750.0)

    rows, saved = [], {}
    for feature_set, names in FEATURE_SETS.items():
        cal_x = cal[names].to_numpy(float)
        bg_x = bg[names].to_numpy(float)
        for family, hypers in FAMILIES.items():
            for hyper, cap in itertools.product(hypers, CAPS):
                oof = np.full(len(cal), np.nan)
                for fold in range(4):
                    test_idx = dev_indices[dev_folds == fold]
                    train_idx = dev_indices[(dev_folds != fold) & cal_valid[dev_indices]]
                    fold_state = fit_state(cal_x[train_idx], cal_base[train_idx], family, hyper, cap)
                    oof[test_idx] = apply_state(fold_state, cal_x[test_idx], cal_base[test_idx], cal_valid[test_idx])["energy"]
                train_idx = dev_indices[cal_valid[dev_indices]]
                state = fit_state(cal_x[train_idx], cal_base[train_idx], family, hyper, cap)
                bg_pred = apply_state(state, bg_x, bg_base, bg_valid)
                dev_values = oof[dev_mask & cal_valid]
                robust = robust_metrics(dev_values)
                protocol = protocol_metrics(dev_values)
                spike = local_spike_score(bg_pred["energy"])
                spike2600 = local_spike_score(bg_pred["energy"], width=5.0, lo=2450.0, hi=2750.0)
                ratio = bg_pred["energy"][bg_valid] / bg_base[bg_valid]
                rank = pd.Series(bg_base[bg_valid]).corr(pd.Series(bg_pred["energy"][bg_valid]), method="spearman")
                taudit = time_audit(bg.t.to_numpy(float), bg_base, bg_pred["energy"])
                model_id = "{}_{}_h{}_c{}".format(feature_set, family, int(hyper), int(round(cap * 10000)))
                row = {
                    "model_id": model_id, "feature_set": feature_set,
                    "features": "|".join(names), "feature_count": len(names),
                    "family": family, "hyper": hyper, "cap": cap,
                    "dev_base_sigma": dev_base_protocol["sigma_median"],
                    "dev_oof_sigma": protocol["sigma_median"],
                    "dev_sigma_gain": 1.0 - protocol["sigma_median"] / dev_base_protocol["sigma_median"],
                    "dev_base_r68": dev_base_robust["r68"], "dev_oof_r68": robust["r68"],
                    "dev_r68_gain": 1.0 - robust["r68"] / dev_base_robust["r68"],
                    "dev_base_r90": dev_base_robust["r90"], "dev_oof_r90": robust["r90"],
                    "dev_r90_gain": 1.0 - robust["r90"] / dev_base_robust["r90"],
                    "dev_protocol_successes": protocol["protocol_successes"],
                    "dev_center_bias_max": protocol["center_bias_max"],
                    "background_applied": float(np.mean(bg_pred["applied"])),
                    "background_ood": float(np.mean(bg_pred["ood"])),
                    "background_strong": float(np.mean(bg_pred["strong"])),
                    "background_near_cap": float(np.mean(bg_pred["near_cap"])),
                    "background_energy_trend": energy_trend(bg_base, bg_pred["delta"], bg_pred["applied"]),
                    "background_rank_corr": float(rank),
                    "background_spike_growth": float(spike["ratio"] / max(bg_spike["ratio"], 1.0)),
                    "background_2600_spike_growth": float(spike2600["ratio"] / max(bg_2600["ratio"], 1.0)),
                    "background_median_scale_shift": float(np.median(ratio) - 1.0),
                    "time_ratio_span": taudit["median_ratio_span"],
                    "time_applied_fraction_span": taudit["applied_fraction_span"],
                    "time_supported_blocks": taudit["supported_blocks"],
                }
                flags = safety(row, dev_base_protocol)
                row["safety_pass_count"] = int(sum(bool(v) for v in flags.values()))
                row["safe"] = bool(all(flags.values()))
                rows.append(row)
                saved[model_id] = (state, names, oof, bg_pred, flags)
                print(json.dumps({"id": model_id, "safe": row["safe"], "gain": row["dev_sigma_gain"]}), flush=True)

    table = pd.DataFrame(rows).sort_values(["safe", "safety_pass_count", "dev_sigma_gain"], ascending=[False, False, False])
    table.to_csv(outdir / "nonlinear_candidates.csv", index=False)
    safe_rows = [row for row in rows if row["safe"]]
    winner = max(safe_rows, key=lambda r: r["dev_sigma_gain"]) if safe_rows else max(rows, key=lambda r: (r["safety_pass_count"], r["dev_sigma_gain"]))
    state, names, oof, bg_pred, flags = saved[winner["model_id"]]

    hold_pred = apply_state(state, cal[names].to_numpy(float)[hold_indices], cal_base[hold_indices], cal_valid[hold_indices])
    hold_valid = cal_valid[hold_indices]
    hold_base_robust = robust_metrics(cal_base[hold_indices][hold_valid])
    hold_new_robust = robust_metrics(hold_pred["energy"][hold_valid])
    hold_base_protocol = protocol_metrics(cal_base[hold_indices][hold_valid])
    hold_new_protocol = protocol_metrics(hold_pred["energy"][hold_valid])
    holdout = {
        "n": int(np.sum(hold_valid)),
        "base_sigma": hold_base_protocol["sigma_median"],
        "candidate_sigma": hold_new_protocol["sigma_median"],
        "sigma_gain": 1.0 - hold_new_protocol["sigma_median"] / hold_base_protocol["sigma_median"],
        "base_r68": hold_base_robust["r68"], "candidate_r68": hold_new_robust["r68"],
        "r68_gain": 1.0 - hold_new_robust["r68"] / hold_base_robust["r68"],
        "base_r90": hold_base_robust["r90"], "candidate_r90": hold_new_robust["r90"],
        "r90_gain": 1.0 - hold_new_robust["r90"] / hold_base_robust["r90"],
        "applied_fraction": float(np.mean(hold_pred["applied"])),
        "protocol_successes": hold_new_protocol["protocol_successes"],
    }
    holdout_pass = bool(holdout["sigma_gain"] > 0 and holdout["r68_gain"] >= -0.002 and holdout["r90_gain"] >= -0.003)
    status = "holdout_passed_research_candidate" if safe_rows and holdout_pass else "holdout_not_passed_diagnostic_only"
    bundle = {
        "model_name": "PandaX_EnergyCor_nonlinear_holdout_v16",
        "version": "v16_nonlinear_holdout_20260814", "status": status,
        "feature_names": names, "predictor_state": state,
        "dev_selection": winner, "dev_safety_flags": flags,
        "untouched_holdout": holdout,
        "warning": "Do not deploy before cross-run and multi-line blind validation.",
    }
    joblib.dump(bundle, outdir / "candidate_v16.joblib")
    event_out = cal[["runNumber", "eventNumber", "t"]].copy()
    event_out["split"] = np.where(dev_mask, "development", "holdout")
    event_out["energy_cor_kev"] = cal_base
    event_out["candidate_energy_kev"] = oof
    event_out.loc[hold_indices, "candidate_energy_kev"] = hold_pred["energy"]
    event_out.to_csv(outdir / "calibration_events.csv", index=False)
    summary = {
        "status": status, "winner": winner, "winner_safety_flags": flags,
        "untouched_holdout": holdout, "holdout_pass": holdout_pass,
        "safe_candidates": len(safe_rows), "total_candidates": len(rows),
        "feature_sets": FEATURE_SETS, "families": FAMILIES, "caps": CAPS,
        "background_base_spike": bg_spike, "background_base_2600_spike": bg_2600,
        "warning": bundle["warning"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OUTPUT", str(outdir))
    print("SELECTION", status, winner["model_id"])
    print("HOLDOUT", json.dumps(holdout, sort_keys=True))


if __name__ == "__main__":
    main()
