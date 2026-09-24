#!/usr/bin/env python3
"""Blend three pre-specified bounded additive corrections on a weight simplex."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anchored_smallmodels_v14 import chronological_folds, energy_cor, time_audit
from grouped_nested_physics_v9 import protocol_metrics
from multivariable_search_v15 import apply_state, fit_state
from safe_general_v12 import load_table, local_spike_score, robust_metrics


COMPONENTS = {
    "M06": (["dt", "wS2CDF_max"], 1000.0, 0.0025),
    "S10_corrected_xy": (["dt", "wS2CDF_max", "yS2Tcor_max", "xS2Bcor_max"], 3000.0, 0.0075),
    "S02_top_xy": (["dt", "wS2CDF_max", "xS2T_max", "yS2T_max"], 3000.0, 0.0075),
}


def metrics_row(name, weights, base, energy, bg_base, bg_energy, timestamp, bg_spike, bg_2600):
    base_r = robust_metrics(base)
    new_r = robust_metrics(energy)
    base_p = protocol_metrics(base)
    new_p = protocol_metrics(energy)
    valid = np.isfinite(bg_base) & np.isfinite(bg_energy) & (bg_base > 0)
    ratio = bg_energy[valid] / bg_base[valid]
    applied = valid & (np.abs(bg_energy / bg_base - 1.0) > 1.0e-12)
    spike = local_spike_score(bg_energy)
    spike2600 = local_spike_score(bg_energy, width=5.0, lo=2450.0, hi=2750.0)
    rank = pd.Series(bg_base[valid]).corr(pd.Series(bg_energy[valid]), method="spearman")
    audit = time_audit(timestamp, bg_base, bg_energy)
    row = {
        "name": name, "w_m06": weights[0], "w_s10": weights[1], "w_s02": weights[2],
        "sigma": new_p["sigma_median"], "sigma_gain": 1.0 - new_p["sigma_median"] / base_p["sigma_median"],
        "r68": new_r["r68"], "r68_gain": 1.0 - new_r["r68"] / base_r["r68"],
        "r90": new_r["r90"], "r90_gain": 1.0 - new_r["r90"] / base_r["r90"],
        "protocol_successes": new_p["protocol_successes"],
        "background_applied": float(np.mean(applied)),
        "background_rank_corr": float(rank),
        "background_spike_growth": float(spike["ratio"] / max(bg_spike["ratio"], 1.0)),
        "background_2600_spike_growth": float(spike2600["ratio"] / max(bg_2600["ratio"], 1.0)),
        "background_ratio_q01": float(np.quantile(ratio, 0.01)),
        "background_ratio_q99": float(np.quantile(ratio, 0.99)),
        "time_ratio_span": audit["median_ratio_span"],
        "time_applied_fraction_span": audit["applied_fraction_span"],
        "time_supported_blocks": audit["supported_blocks"],
    }
    flags = {
        "sigma": row["sigma_gain"] > 0,
        "r68": row["r68_gain"] >= -0.001,
        "r90": row["r90_gain"] >= -0.002,
        "protocol": row["protocol_successes"] >= 10,
        "coverage": row["background_applied"] >= 0.10,
        "rank": row["background_rank_corr"] >= 0.9995,
        "spike": row["background_spike_growth"] <= 1.10,
        "spike2600": row["background_2600_spike_growth"] <= 1.10,
        "scale": row["background_ratio_q01"] >= 0.992 and row["background_ratio_q99"] <= 1.008,
        "time": row["time_ratio_span"] <= 0.002,
        "time_coverage": row["time_applied_fraction_span"] <= 0.25,
        "blocks": row["time_supported_blocks"] >= 8,
    }
    row["safe"] = bool(all(flags.values()))
    row["safety_pass_count"] = int(sum(flags.values()))
    return row, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    all_features = sorted(set(sum([v[0] for v in COMPONENTS.values()], [])))
    columns = ["runNumber", "eventNumber", "t", "qS1ub_C", "qS2Bdesub_C"] + all_features
    cal = load_table(args.calibration, columns); bg = load_table(args.background, columns)
    cal_base, cal_valid = energy_cor(cal); bg_base, bg_valid = energy_cor(bg)
    folds = chronological_folds(cal.t.to_numpy(float), 5)
    cal_deltas, bg_deltas = [], []
    final_states = {}
    for component, (features, alpha, cap) in COMPONENTS.items():
        x = cal[features].to_numpy(float); bx = bg[features].to_numpy(float)
        oof = np.zeros(len(cal), dtype=float)
        for fold in range(5):
            test = folds == fold; train = (~test) & cal_valid
            state = fit_state(x[train], cal_base[train], alpha, cap)
            pred = apply_state(state, x[test], cal_base[test], cal_valid[test])
            good = cal_valid[test] & np.isfinite(pred["energy"])
            target = np.flatnonzero(test)[good]
            oof[target] = -np.log(pred["energy"][good] / cal_base[target])
        state = fit_state(x[cal_valid], cal_base[cal_valid], alpha, cap)
        pred = apply_state(state, bx, bg_base, bg_valid)
        delta = np.zeros(len(bg)); good = bg_valid & np.isfinite(pred["energy"])
        delta[good] = -np.log(pred["energy"][good] / bg_base[good])
        cal_deltas.append(oof); bg_deltas.append(delta); final_states[component] = state
    cal_deltas = np.asarray(cal_deltas); bg_deltas = np.asarray(bg_deltas)
    bg_spike = local_spike_score(bg_base); bg_2600 = local_spike_score(bg_base, width=5.0, lo=2450.0, hi=2750.0)
    rows, flag_map = [], {}
    grid = np.arange(0.0, 1.0001, 0.05)
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < -1.0e-9: continue
            weights = np.asarray([w0, w1, max(0.0, w2)])
            cdelta = np.sum(weights[:, None] * cal_deltas, axis=0)
            bdelta = np.sum(weights[:, None] * bg_deltas, axis=0)
            cenergy = cal_base * np.exp(-cdelta); benergy = bg_base * np.exp(-bdelta)
            name = "W_{:.2f}_{:.2f}_{:.2f}".format(*weights)
            row, flags = metrics_row(name, weights, cal_base[cal_valid], cenergy[cal_valid], bg_base, benergy, bg.t.to_numpy(float), bg_spike, bg_2600)
            rows.append(row); flag_map[name] = flags
    table = pd.DataFrame(rows).sort_values(["safe", "safety_pass_count", "sigma_gain"], ascending=[False, False, False])
    table.to_csv(outdir / "ensemble_weights.csv", index=False)
    safe = table.loc[table.safe]
    winner = (safe.iloc[0] if len(safe) else table.iloc[0]).to_dict()
    weights = np.asarray([winner["w_m06"], winner["w_s10"], winner["w_s02"]])
    bundle = {"model_name": "PandaX_bounded_ensemble_v17", "components": COMPONENTS,
              "component_states": final_states, "weights": weights, "selection": winner,
              "safety_flags": flag_map[winner["name"]],
              "warning": "Research candidate; requires cross-run and multi-line blind validation."}
    joblib_path = outdir / "candidate_v17.joblib"
    import joblib; joblib.dump(bundle, joblib_path)
    summary = {"winner": winner, "safe_weights": int(len(safe)), "total_weights": int(len(table)),
               "components": COMPONENTS, "safety_flags": flag_map[winner["name"]], "warning": bundle["warning"]}
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.item()),
        encoding="utf-8",
    )
    print(table.head(15).to_string(index=False)); print("OUTPUT", outdir); print("WINNER", winner["name"])


if __name__ == "__main__":
    main()
