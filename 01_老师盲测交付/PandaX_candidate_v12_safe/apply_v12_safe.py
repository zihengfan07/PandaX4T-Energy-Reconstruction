#!/usr/bin/env python3
"""Apply the portable PandaX v12_safe model to a scalar TXT table."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DERIVED = {"r2S2T_max", "r2S2B_max"}


def add_geometry(frame):
    frame["r2S2T_max"] = frame["xS2T_max"] ** 2 + frame["yS2T_max"] ** 2
    frame["r2S2B_max"] = frame["xS2B_max"] ** 2 + frame["yS2B_max"] ** 2
    return frame


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

    base = bundle["base_state"]
    features = bundle["feature_names"]
    required = [base["s1_name"], base["s2_name"]] + [
        name for name in features if name not in DERIVED
    ]
    header = pd.read_csv(args.input, sep="\t", nrows=0)
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError("Schema mismatch; missing columns: " + ", ".join(missing))
    identity = [
        name for name in ["runNumber", "fileNumber", "eventNumber"]
        if name in header.columns
    ]
    frame = pd.read_csv(
        args.input,
        sep="\t",
        usecols=list(dict.fromkeys(identity + required)),
        low_memory=False,
    ).replace([np.inf, -np.inf], np.nan)
    frame = add_geometry(frame)
    s1 = frame[base["s1_name"]].to_numpy(dtype=float)
    s2 = frame[base["s2_name"]].to_numpy(dtype=float)
    valid = np.isfinite(s1) & np.isfinite(s2) & (s1 > 0) & (s2 > 0)
    foundation = np.full(len(frame), np.nan)
    weight = base["s1_weight"]
    foundation[valid] = (
        weight * s1[valid] / base["s1_median"]
        + (1.0 - weight) * s2[valid] / base["s2_median"]
    )

    state = bundle["predictor_state"]
    design, ood_count = transform(frame[features].to_numpy(dtype=float), state["transform"])
    raw = (
        design.dot(np.asarray(state["coef"], dtype=float))
        + state["intercept"] - state["prediction_center"]
    )
    ood = ood_count > 0
    strong = np.abs(raw) > state["strong_raw_limit"]
    applied = valid & (~ood) & (~strong)
    delta = np.zeros(len(frame), dtype=float)
    delta[applied] = state["cap"] * np.tanh(raw[applied] / state["cap"])
    energy = np.full(len(frame), np.nan)
    energy[valid] = foundation[valid] * np.exp(-delta[valid]) * state["energy_scale"]
    near_cap = applied & (np.abs(delta) >= 0.95 * state["cap"])

    output = frame[identity].copy()
    output.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    output["foundation_energy_kev"] = foundation * bundle["base_energy_scale"]
    output["candidate_v12_energy_kev"] = energy
    output["log_correction"] = delta
    output["correction_applied"] = applied
    output["out_of_distribution"] = ood
    output["strong_raw_fallback"] = strong
    output["correction_near_cap"] = near_cap
    output.to_csv(output_dir / "validation_events.txt", sep=" ", index=False, na_rep="NaN")

    finite_energy = np.isfinite(energy) & (energy > 0)
    summary = {
        "model_name": bundle["model_name"],
        "model_version": bundle["version"],
        "input_rows": int(len(frame)),
        "valid_prediction_rows": int(np.sum(finite_energy)),
        "invalid_foundation_rows": int(np.sum(~valid)),
        "correction_applied_fraction": float(np.mean(applied)),
        "out_of_distribution_fraction": float(np.mean(ood)),
        "strong_raw_fallback_fraction": float(np.mean(strong)),
        "correction_near_cap_fraction": float(np.mean(near_cap)),
        "warning": bundle["warning"],
    }
    with open(output_dir / "validation_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
