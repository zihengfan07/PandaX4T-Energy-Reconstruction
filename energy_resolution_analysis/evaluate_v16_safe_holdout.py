#!/usr/bin/env python3
"""Diagnostic holdout audit for every v16 candidate that passed dev safety."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from anchored_smallmodels_v14 import energy_cor
from grouped_nested_physics_v9 import protocol_metrics
from nonlinear_holdout_v16 import apply_state, fit_state
from safe_general_v12 import load_table, robust_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.candidates)
    safe = table.loc[table.safe.astype(bool)].copy()
    feature_names = sorted(set("|".join(safe.features).split("|")))
    columns = ["runNumber", "eventNumber", "t", "qS1ub_C", "qS2Bdesub_C"] + feature_names
    cal = load_table(args.calibration, columns)
    base, valid = energy_cor(cal)
    order = np.argsort(cal.t.to_numpy(float), kind="mergesort")
    split = int(0.80 * len(cal))
    dev = order[:split]
    hold = order[split:]
    dev = dev[valid[dev]]
    hold_valid = valid[hold]
    base_robust = robust_metrics(base[hold][hold_valid])
    base_protocol = protocol_metrics(base[hold][hold_valid])
    rows = []
    for _, candidate in safe.iterrows():
        names = candidate.features.split("|")
        x = cal[names].to_numpy(float)
        state = fit_state(x[dev], base[dev], candidate.family, candidate.hyper, candidate.cap)
        pred = apply_state(state, x[hold], base[hold], hold_valid)
        robust = robust_metrics(pred["energy"][hold_valid])
        protocol = protocol_metrics(pred["energy"][hold_valid])
        rows.append({
            "model_id": candidate.model_id,
            "family": candidate.family,
            "features": candidate.features,
            "dev_sigma_gain": candidate.dev_sigma_gain,
            "holdout_base_sigma": base_protocol["sigma_median"],
            "holdout_sigma": protocol["sigma_median"],
            "holdout_sigma_gain": 1.0 - protocol["sigma_median"] / base_protocol["sigma_median"],
            "holdout_r68_gain": 1.0 - robust["r68"] / base_robust["r68"],
            "holdout_r90_gain": 1.0 - robust["r90"] / base_robust["r90"],
            "holdout_applied": float(np.mean(pred["applied"])),
            "holdout_pass": bool(
                protocol["sigma_median"] < base_protocol["sigma_median"]
                and robust["r68"] <= 1.002 * base_robust["r68"]
                and robust["r90"] <= 1.003 * base_robust["r90"]
            ),
        })
    result = pd.DataFrame(rows).sort_values("holdout_sigma_gain", ascending=False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print("OUTPUT", str(path))


if __name__ == "__main__":
    main()
