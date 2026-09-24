#!/usr/bin/env python3
"""Apply the frozen candidate_v11 model to an unseen scalar TXT file."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from foundation_environment_v11 import (
    add_geometry,
    apply_base,
    current_energy,
    legacy_energy,
)
from grouped_nested_physics_v9 import predict_with_state, protocol_metrics


def clean_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if key != "protocol_rows"
    }


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
    derived = {name for name in bundle["feature_names"] if name.startswith("r2_")}
    required = [
        "qS1ub_C", "qS2Bdesub_C",
        bundle["base_state"]["s1_name"],
        bundle["base_state"]["s2_name"],
    ] + [name for name in bundle["feature_names"] if name not in derived]
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError("Schema mismatch; missing columns: " + ", ".join(missing))
    identity = [
        name for name in ["runNumber", "fileNumber", "eventNumber"]
        if name in header.columns
    ]
    usecols = list(dict.fromkeys(identity + required))
    frame = pd.read_csv(
        args.input, sep="\t", usecols=usecols, low_memory=False
    ).replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    validity = (
        np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    valid = add_geometry(frame.loc[validity].copy())
    indices = np.arange(len(valid))
    foundation = apply_base(valid, indices, bundle["base_state"])
    x = valid[bundle["feature_names"]].to_numpy(dtype=float)
    candidate, correction, touched = predict_with_state(
        bundle["predictor_state"], x, foundation
    )
    current = current_energy(valid) * bundle["current_energy_scale"]
    legacy = legacy_energy(valid) * bundle["legacy_energy_scale"]
    foundation_calibrated = foundation * bundle["foundation_energy_scale"]

    output = frame[["sourceRow"] + identity].copy()
    for column in [
        "current_formula_energy_kev",
        "legacy_energy_cor_kev",
        "selected_foundation_energy_kev",
        "candidate_v11_energy_kev",
        "log_correction",
    ]:
        output[column] = np.nan
    output["correction_near_cap"] = False
    output.loc[validity, "current_formula_energy_kev"] = current
    output.loc[validity, "legacy_energy_cor_kev"] = legacy
    output.loc[validity, "selected_foundation_energy_kev"] = foundation_calibrated
    output.loc[validity, "candidate_v11_energy_kev"] = candidate
    output.loc[validity, "log_correction"] = correction
    output.loc[validity, "correction_near_cap"] = touched
    output.to_csv(output_dir / "validation_events.csv", index=False)
    output.to_csv(
        output_dir / "validation_events.txt", sep=" ", index=False
    )

    methods = [
        ("current_formula", current),
        ("legacy_energy_cor", legacy),
        ("selected_foundation", foundation_calibrated),
        ("candidate_v11", candidate),
    ]
    metrics = {name: protocol_metrics(values) for name, values in methods}
    summary = {
        "model_name": bundle["model_name"],
        "input_rows": int(len(frame)),
        "valid_rows": int(np.sum(validity)),
        "invalid_rows": int(np.sum(~validity)),
        "finite_positive_predictions": bool(
            np.all(np.isfinite(candidate)) and np.all(candidate > 0)
        ),
        "near_cap_fraction": float(np.mean(touched)),
        "metrics": {
            name: clean_metrics(result) for name, result in metrics.items()
        },
        "relative_sigma_improvement_vs_current": float(
            1.0
            - metrics["candidate_v11"]["sigma_median"]
            / metrics["current_formula"]["sigma_median"]
        ),
        "warning": bundle["warning"],
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    protocol_rows = []
    for method, result in metrics.items():
        for row in result["protocol_rows"]:
            item = dict(row)
            item["method"] = method
            protocol_rows.append(item)
    protocol_frame = pd.DataFrame(protocol_rows)
    protocol_frame.to_csv(
        output_dir / "validation_protocols.csv", index=False
    )
    protocol_frame.to_csv(
        output_dir / "validation_protocols.txt", sep=" ", index=False
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
