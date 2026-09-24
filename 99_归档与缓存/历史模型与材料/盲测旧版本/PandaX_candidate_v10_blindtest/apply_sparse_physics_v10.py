#!/usr/bin/env python3
"""Apply the frozen sparse candidate_v10 bundle to an unseen scalar TXT file."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from grouped_nested_physics_v9 import (
    add_geometry,
    base_energy,
    predict_with_state,
    protocol_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    header = pd.read_csv(args.input, sep="\t", nrows=0)
    required = (
        ["qS1ub_C", "qS2Bdesub_C"]
        + bundle["feature_names"]
    )
    # Derived radius features are generated below and are not raw input fields.
    derived = {"r2_raw", "r2_cor", "r2_mcpaf"}
    raw_required = [name for name in required if name not in derived]
    missing = [name for name in raw_required if name not in header.columns]
    if missing:
        raise ValueError("Schema mismatch; missing columns: " + ", ".join(missing))
    identity = [
        name
        for name in ["runNumber", "fileNumber", "eventNumber"]
        if name in header.columns
    ]
    usecols = list(dict.fromkeys(identity + raw_required))
    frame = pd.read_csv(
        args.input, sep="\t", usecols=usecols, low_memory=False
    ).replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    frame = add_geometry(frame)
    validity = (
        np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    valid_frame = frame.loc[validity].copy()
    energy = base_energy(valid_frame)
    legacy_input_energy = 0.0137 * energy
    legacy_energy_cor = (
        -1.73706e-09 * legacy_input_energy**3
        + 7.98193e-06 * legacy_input_energy**2
        + 1.07904 * legacy_input_energy
        - 9.22086
    ) * bundle["legacy_energy_scale"]
    x = valid_frame[bundle["feature_names"]].to_numpy(dtype=float)
    prediction, correction, touched = predict_with_state(
        bundle["predictor_state"], x, energy
    )
    baseline_prediction = energy * bundle["baseline_energy_scale"]
    output = frame[["sourceRow"] + identity].copy()
    output["current_formula_energy_kev"] = np.nan
    output["legacy_energy_cor_kev"] = np.nan
    output["candidate_energy_kev"] = np.nan
    output["log_correction"] = np.nan
    output["correction_near_cap"] = False
    output.loc[validity, "current_formula_energy_kev"] = baseline_prediction
    output.loc[validity, "legacy_energy_cor_kev"] = legacy_energy_cor
    output.loc[validity, "candidate_energy_kev"] = prediction
    output.loc[validity, "log_correction"] = correction
    output.loc[validity, "correction_near_cap"] = touched
    output.to_csv(output_dir / "validation_events.csv", index=False)

    baseline_metrics = protocol_metrics(baseline_prediction)
    legacy_metrics = protocol_metrics(legacy_energy_cor)
    candidate_metrics = protocol_metrics(prediction)
    summary = {
        "model_name": bundle["model_name"],
        "input_rows": int(len(frame)),
        "valid_rows": int(np.sum(validity)),
        "invalid_rows": int(np.sum(~validity)),
        "finite_positive_predictions": bool(
            np.all(np.isfinite(prediction)) and np.all(prediction > 0)
        ),
        "near_cap_fraction": float(np.mean(touched)),
        "current_formula_metrics": {
            key: value for key, value in baseline_metrics.items()
            if key != "protocol_rows"
        },
        "legacy_energy_cor_metrics": {
            key: value for key, value in legacy_metrics.items()
            if key != "protocol_rows"
        },
        "candidate_metrics": {
            key: value for key, value in candidate_metrics.items()
            if key != "protocol_rows"
        },
        "relative_sigma_improvement": float(
            1.0
            - candidate_metrics["sigma_median"]
            / baseline_metrics["sigma_median"]
        ),
        "warning": bundle["warning"],
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    protocol_rows = []
    for method, metrics in [
        ("current_formula", baseline_metrics),
        ("legacy_energy_cor", legacy_metrics),
        ("candidate_v10", candidate_metrics),
    ]:
        for row in metrics["protocol_rows"]:
            item = dict(row)
            item["method"] = method
            protocol_rows.append(item)
    pd.DataFrame(protocol_rows).to_csv(
        output_dir / "validation_protocols.csv", index=False
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
