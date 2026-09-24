#!/usr/bin/env python3
"""One-time migration adding the frozen baseline scale to a v9 bundle."""

import argparse

import joblib
import numpy as np

from grouped_nested_physics_v9 import (
    NOMINAL_ENERGY_KEV,
    base_energy,
    fit_protocol,
    prepare_frame,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame, _ = prepare_frame(args.input_data)
    energy = base_energy(frame)
    fit = fit_protocol(energy, "linear", 0.18)
    training_mu = (
        fit["mu"] if fit.get("success") else float(np.median(energy))
    )
    bundle["baseline_energy_scale"] = NOMINAL_ENERGY_KEV / training_mu
    joblib.dump(bundle, args.model, compress=3)
    print("baseline_energy_scale={:.12g}".format(
        bundle["baseline_energy_scale"]
    ))


if __name__ == "__main__":
    main()
