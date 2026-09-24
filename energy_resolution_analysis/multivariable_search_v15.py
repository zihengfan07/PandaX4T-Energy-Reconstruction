#!/usr/bin/env python3
"""Search bounded 3--8 variable Energy_cor-anchored correction models.

Calibration labels are used only in chronological out-of-fold fits.  The
background sample has no energy label and is used only for portability and
spectrum-shape vetoes.  fileNumber and absolute S1/S2 charge variables are
never predictors.
"""

from __future__ import print_function

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from anchored_smallmodels_v14 import (
    chronological_folds,
    energy_cor,
    feature_quality,
    time_audit,
)
from grouped_nested_physics_v9 import protocol_metrics
from safe_general_v12 import energy_trend, load_table, local_spike_score, robust_metrics


NOMINAL_KEV = 2614.5
OUTER_FOLDS = 5

DT = "dt"
WIDTH = "wS2CDF_max"
XT, YT = "xS2T_max", "yS2T_max"
XB, YB = "xS2B_max", "yS2B_max"
YTC, XBC = "yS2Tcor_max", "xS2Bcor_max"
S2H, S2C = "qS2hitStdevTo1_max", "qS2channelStdevTo1_max"
S1H, S1C = "qS1hitStdevTo1_max", "qS1channelStdevTo1_max"
FWHM, TBA = "wS2FWHM_max", "tDiffBottomTopS2_max"
NPMT, RMSS1 = "nPMTS2_max", "rmsCogS1T_max"

# The first entry reproduces M06.  The remaining entries deliberately use
# 4--8 inputs, grouped by detector meaning instead of arbitrary combinations.
FEATURE_SETS = {
    "S01_M06_reference": [DT, WIDTH],
    "S02_top_xy": [DT, WIDTH, XT, YT],
    "S03_bottom_xy": [DT, WIDTH, XB, YB],
    "S04_top_bottom_xy": [DT, WIDTH, XT, YT, XB, YB],
    "S05_s2_pattern": [DT, WIDTH, S2H, S2C],
    "S06_s1_pattern": [DT, WIDTH, S1H, S1C],
    "S07_s1_s2_pattern": [DT, WIDTH, S2H, S2C, S1H, S1C],
    "S08_s2_shape": [DT, WIDTH, FWHM, TBA],
    "S09_hit_topology": [DT, WIDTH, NPMT, RMSS1],
    "S10_corrected_position": [DT, WIDTH, YTC, XBC],
    "S11_historical_core": [DT, WIDTH, XT, YB],
    "S12_core_channel_pattern": [DT, WIDTH, XT, YB, S2C, S1C],
    "S13_core_s2_shape": [DT, WIDTH, XT, YB, FWHM, TBA],
    "S14_core_hit_topology": [DT, WIDTH, XT, YB, NPMT, RMSS1],
    "S15_core_hit_pattern": [DT, WIDTH, XT, YB, S2H, S1H],
    "S16_geometry_s2_pattern": [DT, WIDTH, XT, YT, XB, YB, S2H, S2C],
    "S17_geometry_shape": [DT, WIDTH, XT, YT, XB, YB, FWHM, TBA],
}

ALPHAS = (100.0, 300.0, 1000.0, 3000.0)
CAPS = (0.0025, 0.0050, 0.0075)


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
    dcenter = np.mean(design, axis=0)
    dscale = np.std(design, axis=0)
    dscale[~np.isfinite(dscale) | (dscale < 1.0e-9)] = 1.0
    # A wider, robust envelope is needed for multi-variable portability.
    lower = np.quantile(filled, 0.0005, axis=0) - 1.0 * scales
    upper = np.quantile(filled, 0.9995, axis=0) + 1.0 * scales
    return {
        "medians": medians, "centers": centers, "scales": scales,
        "design_center": dcenter, "design_scale": dscale,
        "lower": lower, "upper": upper,
    }


def transform(values, state):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, state["medians"])
    outside = (~finite) | (filled < state["lower"]) | (filled > state["upper"])
    count = np.sum(outside, axis=1)
    z = np.clip((filled - state["centers"]) / state["scales"], -5.0, 5.0)
    design = np.concatenate([z, z ** 2], axis=1)
    design = (design - state["design_center"]) / state["design_scale"]
    return design, count


def fit_state(values, energy, alpha, cap):
    state = prepare_transform(values)
    design, _ = transform(values, state)
    target = np.log(np.clip(energy / NOMINAL_KEV, 1.0e-12, None))
    weights = np.exp(-0.5 * ((energy - NOMINAL_KEV) / (0.10 * NOMINAL_KEV)) ** 2)
    center = float(np.average(target, weights=weights))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, target - center, sample_weight=weights)
    prediction_center = float(np.average(model.predict(design), weights=weights))
    return {
        "model": model, "transform": state,
        "prediction_center": prediction_center,
        "alpha": float(alpha), "cap": float(cap),
        "energy_scale": 1.0,
        # Keep accepted corrections below tanh(1.5)=0.905 of the cap, so an
        # accepted event can never enter the >=95% near-cap pile-up region.
        "strong_raw_limit": float(1.5 * cap),
        "ood_policy": "more_than_20pct_and_at_least_two_features_outside",
    }


def apply_state(state, values, base, valid):
    design, ood_count = transform(values, state["transform"])
    raw = state["model"].predict(design) - state["prediction_center"]
    n_features = values.shape[1]
    allowed = max(1, int(np.floor(0.20 * n_features)))
    ood = ood_count > allowed
    strong = np.abs(raw) > state["strong_raw_limit"]
    applied = valid & (~ood) & (~strong)
    delta = np.zeros(len(base), dtype=float)
    delta[applied] = state["cap"] * np.tanh(raw[applied] / state["cap"])
    output = np.full(len(base), np.nan)
    output[valid] = base[valid] * np.exp(-delta[valid])
    near_cap = applied & (np.abs(delta) >= 0.95 * state["cap"])
    return {
        "energy": output, "delta": delta, "raw_delta": raw,
        "ood": ood, "ood_count": ood_count, "strong": strong,
        "applied": applied, "near_cap": near_cap,
    }


def safety_flags(row, base_protocol):
    return {
        "sigma_improves": row["relative_sigma_improvement"] > 0,
        "r68_not_worse": row["relative_r68_improvement"] >= -0.001,
        "r90_not_worse": row["relative_r90_improvement"] >= -0.002,
        "fit_protocol_ok": row["calibration_protocol_successes"] >= 10,
        "center_stable": abs(row["calibration_center_bias_max"] - base_protocol["center_bias_max"]) <= 0.003,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    features = sorted(set(itertools.chain.from_iterable(FEATURE_SETS.values())))
    columns = ["runNumber", "eventNumber", "t", "qS1ub_C", "qS2Bdesub_C"] + features
    cal = load_table(args.calibration, columns)
    bg = load_table(args.background, columns)
    cal_base, cal_valid = energy_cor(cal)
    bg_base, bg_valid = energy_cor(bg)
    folds = chronological_folds(cal.t.to_numpy(float), OUTER_FOLDS)
    base_robust = robust_metrics(cal_base[cal_valid])
    base_protocol = protocol_metrics(cal_base[cal_valid])
    bg_spike = local_spike_score(bg_base)
    bg_2600 = local_spike_score(bg_base, width=5.0, lo=2450.0, hi=2750.0)

    quality = {
        "calibration": feature_quality(cal, features),
        "background": feature_quality(bg, features),
        "excluded_predictors": ["fileNumber", "qS1ub_C", "qS2Bdesub_C"],
    }
    (outdir / "feature_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    rows, states = [], {}
    for set_name, names in FEATURE_SETS.items():
        cal_x = cal[names].to_numpy(float)
        bg_x = bg[names].to_numpy(float)
        for alpha, cap in itertools.product(ALPHAS, CAPS):
            oof = np.full(len(cal), np.nan)
            for fold in range(OUTER_FOLDS):
                test = folds == fold
                train = (~test) & cal_valid
                fold_state = fit_state(cal_x[train], cal_base[train], alpha, cap)
                oof[test] = apply_state(fold_state, cal_x[test], cal_base[test], cal_valid[test])["energy"]
            final_state = fit_state(cal_x[cal_valid], cal_base[cal_valid], alpha, cap)
            bg_pred = apply_state(final_state, bg_x, bg_base, bg_valid)
            rmetric = robust_metrics(oof[cal_valid])
            pmetric = protocol_metrics(oof[cal_valid])
            spike = local_spike_score(bg_pred["energy"])
            spike2600 = local_spike_score(bg_pred["energy"], width=5.0, lo=2450.0, hi=2750.0)
            valid_ratio = bg_pred["energy"][bg_valid] / bg_base[bg_valid]
            rank = pd.Series(bg_base[bg_valid]).corr(pd.Series(bg_pred["energy"][bg_valid]), method="spearman")
            taudit = time_audit(bg.t.to_numpy(float), bg_base, bg_pred["energy"])
            model_id = "{}_a{}_c{}".format(set_name, int(alpha), int(round(cap * 10000)))
            row = {
                "model_id": model_id, "feature_set": set_name,
                "features": "|".join(names), "feature_count": len(names),
                "alpha": alpha, "cap": cap,
                "calibration_base_sigma": base_protocol["sigma_median"],
                "calibration_oof_sigma": pmetric["sigma_median"],
                "relative_sigma_improvement": 1.0 - pmetric["sigma_median"] / base_protocol["sigma_median"],
                "calibration_base_r68": base_robust["r68"],
                "calibration_oof_r68": rmetric["r68"],
                "relative_r68_improvement": 1.0 - rmetric["r68"] / base_robust["r68"],
                "calibration_base_r90": base_robust["r90"],
                "calibration_oof_r90": rmetric["r90"],
                "relative_r90_improvement": 1.0 - rmetric["r90"] / base_robust["r90"],
                "calibration_protocol_successes": pmetric["protocol_successes"],
                "calibration_center_bias_max": pmetric["center_bias_max"],
                "background_applied": float(np.mean(bg_pred["applied"])),
                "background_ood": float(np.mean(bg_pred["ood"])),
                "background_strong": float(np.mean(bg_pred["strong"])),
                "background_near_cap": float(np.mean(bg_pred["near_cap"])),
                "background_energy_trend": energy_trend(bg_base, bg_pred["delta"], bg_pred["applied"]),
                "background_rank_corr": float(rank),
                "background_spike_growth": float(spike["ratio"] / max(bg_spike["ratio"], 1.0)),
                "background_2600_spike_growth": float(spike2600["ratio"] / max(bg_2600["ratio"], 1.0)),
                "background_median_scale_shift": float(np.median(valid_ratio) - 1.0),
                "background_ratio_q01": float(np.quantile(valid_ratio, 0.01)),
                "background_ratio_q99": float(np.quantile(valid_ratio, 0.99)),
                "time_ratio_span": taudit["median_ratio_span"],
                "time_applied_fraction_span": taudit["applied_fraction_span"],
                "time_supported_blocks": taudit["supported_blocks"],
            }
            flags = safety_flags(row, base_protocol)
            row["safety_pass_count"] = int(sum(bool(v) for v in flags.values()))
            row["safe"] = bool(all(flags.values()))
            rows.append(row)
            states[model_id] = (final_state, names, bg_pred, oof, flags)
            print(json.dumps({"id": model_id, "safe": row["safe"], "sigma_gain": row["relative_sigma_improvement"], "applied": row["background_applied"]}), flush=True)

    table = pd.DataFrame(rows).sort_values(
        ["safe", "safety_pass_count", "relative_sigma_improvement"],
        ascending=[False, False, False],
    )
    table.to_csv(outdir / "multivariable_candidates.csv", index=False)
    safe_rows = [row for row in rows if row["safe"]]
    if safe_rows:
        winner = max(safe_rows, key=lambda r: r["relative_sigma_improvement"])
        status = "safe_candidate_selected"
    else:
        winner = max(rows, key=lambda r: (r["safety_pass_count"], r["relative_sigma_improvement"]))
        status = "no_candidate_passed_all_gates"
    state, names, bg_pred, oof, flags = states[winner["model_id"]]
    bundle = {
        "model_name": "PandaX_EnergyCor_multivariable_v15",
        "version": "v15_multivariable_20260814",
        "status": status, "base_formula": "given_Energy_cor",
        "feature_names": names, "predictor_state": state,
        "selection": winner, "safety_flags": flags,
        "warning": "Single calibration run: require new runs and multiple monoenergetic lines before deployment.",
    }
    joblib.dump(bundle, outdir / "candidate_v15_multivariable.joblib")
    bgout = bg[["runNumber", "eventNumber", "t"]].copy()
    bgout["energy_cor_kev"] = bg_base
    bgout["candidate_v15_energy_kev"] = bg_pred["energy"]
    bgout["correction_applied"] = bg_pred["applied"]
    bgout["out_of_distribution"] = bg_pred["ood"]
    bgout.to_csv(outdir / "winner_background_events.csv", index=False)
    calout = cal[["runNumber", "eventNumber", "t"]].copy()
    calout["energy_cor_kev"] = cal_base
    calout["candidate_v15_oof_energy_kev"] = oof
    calout.to_csv(outdir / "winner_calibration_oof_events.csv", index=False)
    summary = {
        "status": status, "winner": winner, "winner_safety_flags": flags,
        "safe_candidates": len(safe_rows), "total_candidates": len(rows),
        "feature_sets": FEATURE_SETS, "alphas": ALPHAS, "caps": CAPS,
        "background_base_spike": bg_spike, "background_base_2600_spike": bg_2600,
        "warning": bundle["warning"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OUTPUT", str(outdir))
    print("SELECTION", status, winner["model_id"])


if __name__ == "__main__":
    main()
