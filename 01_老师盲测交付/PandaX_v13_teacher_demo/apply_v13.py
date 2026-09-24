#!/usr/bin/env python3
"""Apply portable v13 and output only energy before/after correction."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def transform(values, state):
    medians = np.asarray(state["medians"], dtype=float)
    centers = np.asarray(state["centers"], dtype=float)
    scales = np.asarray(state["scales"], dtype=float)
    lower = np.asarray(state["lower"], dtype=float)
    upper = np.asarray(state["upper"], dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, medians)
    outside = (~finite) | (filled < lower) | (filled > upper)
    z = np.clip((filled - centers) / scales, -5.0, 5.0)
    design = np.concatenate([z, z ** 2], axis=1)
    design = (
        design - np.asarray(state["design_center"], dtype=float)
    ) / np.asarray(state["design_scale"], dtype=float)
    return design, np.sum(outside, axis=1)


def energy_cor(frame):
    s1 = frame.qS1ub_C.to_numpy(dtype=float)
    s2 = frame.qS2Bdesub_C.to_numpy(dtype=float)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.model, "r", encoding="utf-8") as stream:
        bundle = json.load(stream)

    required = ["qS1ub_C", "qS2Bdesub_C"] + bundle["feature_names"]
    header = pd.read_csv(args.input, sep="\t", nrows=0)
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError("Schema mismatch; missing columns: " + ", ".join(missing))
    frame = pd.read_csv(
        args.input,
        sep="\t",
        usecols=list(dict.fromkeys(required)),
        low_memory=False,
    ).replace([np.inf, -np.inf], np.nan)
    before, valid = energy_cor(frame)

    state = bundle["predictor_state"]
    values = frame[bundle["feature_names"]].to_numpy(dtype=float)
    design, ood_count = transform(values, state["transform"])
    raw_delta = (
        design.dot(np.asarray(state["coef"], dtype=float))
        + state["intercept"] - state["prediction_center"]
    )
    ood = ood_count > 0
    strong = np.abs(raw_delta) > state["strong_raw_limit"]
    applied = valid & (~ood) & (~strong)
    delta = np.zeros(len(frame), dtype=float)
    delta[applied] = state["cap"] * np.tanh(raw_delta[applied] / state["cap"])
    after = np.full(len(frame), np.nan)
    after[valid] = before[valid] * np.exp(-delta[valid]) * state["energy_scale"]

    output = pd.DataFrame({
        "energy_before_kev": before,
        "energy_after_kev": after,
    })
    output.to_csv(
        output_dir / "energy_before_after.txt",
        sep=" ", index=False, na_rep="NaN",
    )
    summary = {
        "input_rows": int(len(frame)),
        "valid_rows": int(np.sum(valid)),
        "invalid_rows": int(np.sum(~valid)),
        "correction_applied_fraction": float(np.mean(applied)),
        "out_of_distribution_fraction": float(np.mean(ood)),
        "strong_correction_fallback_fraction": float(np.mean(strong)),
        "model_status": bundle["status"],
        "warning": bundle["warning"],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
