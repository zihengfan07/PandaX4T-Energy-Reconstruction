#!/usr/bin/env python3
"""Blindly evaluate candidate_v8 on a new PandaX scalar TXT file."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from full_feature_sigma_search import gaussian_core_fit
from repeated_oof_ensemble_sigma import RAW_FEATURES, add_derived_features


def distribution_metrics(values):
    fit = gaussian_core_fit(values)
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    q05, q16, q50, q84, q95 = np.quantile(
        valid, [0.05, 0.15865, 0.5, 0.84135, 0.95]
    )
    return {
        "events": int(len(valid)),
        "sigma_over_mu": fit["sigma_over_mu"],
        "fit_p_value": fit["p_value"],
        "fit_success": fit["success"],
        "r68": float((q84 - q16) / (2.0 * q50)),
        "r90": float((q95 - q05) / (2.0 * q50)),
        "median": float(q50),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="New scalar TXT")
    parser.add_argument(
        "--model",
        default="model/candidate_v8.joblib",
        help="Serialized candidate model bundle",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_output",
        help="Directory for blind-test outputs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)

    header = pd.read_csv(args.input, sep="\t", nrows=0)
    requested = ["runNumber", "eventNumber", "qS1ub_C", "qS2Bdesub_C"]
    requested += RAW_FEATURES
    usecols = [name for name in requested if name in header.columns]
    frame = pd.read_csv(args.input, sep="\t", usecols=usecols, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    frame = add_derived_features(frame)

    missing = [
        name for name in bundle["feature_names"] if name not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            "The new data are missing required features: " + ", ".join(missing)
        )

    validity = (
        np.isfinite(frame["qS1ub_C"])
        & np.isfinite(frame["qS2Bdesub_C"])
        & (frame["qS1ub_C"] > 0)
        & (frame["qS2Bdesub_C"] > 0)
    )
    base = (
        frame["qS1ub_C"].to_numpy(dtype=float) / 0.125
        + frame["qS2Bdesub_C"].to_numpy(dtype=float) / 10.58
    )
    raw_x = frame[bundle["feature_names"]].to_numpy(dtype=float)
    member_outputs = []
    for member in bundle["members"]:
        medians = member["feature_medians"]
        x = np.where(np.isfinite(raw_x), raw_x, medians)
        prediction = (
            member["first_model"].predict(x)
            + member["second_model"].predict(x)
        )
        correction = np.exp(
            np.clip(
                prediction - member["prediction_center"],
                -bundle["clip"],
                bundle["clip"],
            )
        )
        corrected = base / correction / member["energy_scale"]
        member_outputs.append(np.log(np.clip(corrected, 1.0e-12, None)))

    model_energy = np.exp(np.mean(np.asarray(member_outputs), axis=0))
    model_energy[~validity.to_numpy()] = np.nan
    base_relative = base / np.nanmedian(base[validity])
    base_relative[~validity.to_numpy()] = np.nan

    baseline_metrics = distribution_metrics(base_relative)
    model_metrics = distribution_metrics(model_energy)
    summary = pd.DataFrame(
        [
            {"method": "current_formula", **baseline_metrics},
            {"method": "candidate_v8", **model_metrics},
        ]
    )
    summary["sigma_over_mu_percent"] = 100.0 * summary["sigma_over_mu"]
    summary["r68_percent"] = 100.0 * summary["r68"]
    summary["r90_percent"] = 100.0 * summary["r90"]
    summary.to_csv(output_dir / "validation_summary.csv", index=False)

    events = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    events["input_valid"] = validity
    events["current_energy_relative"] = base_relative
    events["candidate_v8_energy_relative"] = model_energy
    events.to_csv(output_dir / "validation_events.csv", index=False)

    delta = (
        model_metrics["sigma_over_mu"] - baseline_metrics["sigma_over_mu"]
    )
    relative = delta / baseline_metrics["sigma_over_mu"]
    report = [
        "PandaX candidate_v8 independent-run validation",
        "input={}".format(Path(args.input).resolve()),
        "model={}".format(Path(args.model).resolve()),
        "rows={}".format(len(frame)),
        "valid_rows={}".format(int(validity.sum())),
        "",
        "current sigma/mu = {:.6f}%".format(
            100.0 * baseline_metrics["sigma_over_mu"]
        ),
        "model   sigma/mu = {:.6f}%".format(
            100.0 * model_metrics["sigma_over_mu"]
        ),
        "absolute delta   = {:.6f} percentage points".format(100.0 * delta),
        "relative change  = {:.3f}%".format(100.0 * relative),
        "",
        "current R68 = {:.6f}%".format(100.0 * baseline_metrics["r68"]),
        "model   R68 = {:.6f}%".format(100.0 * model_metrics["r68"]),
        "current R90 = {:.6f}%".format(100.0 * baseline_metrics["r90"]),
        "model   R90 = {:.6f}%".format(100.0 * model_metrics["r90"]),
        "",
        "current GOF p = {:.6g}".format(baseline_metrics["fit_p_value"]),
        "model   GOF p = {:.6g}".format(model_metrics["fit_p_value"]),
        "",
        "Interpretation: a negative delta means the candidate is narrower.",
        "Do not retrain or tune the model using this new run before reporting it.",
    ]
    (output_dir / "validation_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output_dir / "model_manifest_used.json").write_text(
        json.dumps(
            {
                "model_name": bundle["model_name"],
                "format_version": bundle["format_version"],
                "training_run": bundle["training_run"],
                "training_events": bundle["training_events"],
                "feature_count": len(bundle["feature_names"]),
                "member_count": len(bundle["members"]),
                "warning": bundle["warning"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
