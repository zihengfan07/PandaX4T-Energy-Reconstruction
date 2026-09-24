#!/usr/bin/env python3
"""Apply a serialized PandaX energy-response model bundle to a scalar TXT."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from repeated_oof_ensemble_sigma import RAW_FEATURES, add_derived_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
        raise RuntimeError("Missing required model features: " + ", ".join(missing))

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
    log_predictions = []
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
        log_predictions.append(np.log(np.clip(corrected, 1.0e-12, None)))

    ensemble = np.exp(np.mean(np.asarray(log_predictions), axis=0))
    ensemble[~validity.to_numpy()] = np.nan
    result = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    result["input_valid"] = validity
    result["energy_base_relative"] = base / np.nanmedian(base[validity])
    result["energy_model_relative"] = ensemble
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        "wrote {} events ({} valid) to {}".format(
            len(result), int(validity.sum()), args.output
        )
    )


if __name__ == "__main__":
    main()
