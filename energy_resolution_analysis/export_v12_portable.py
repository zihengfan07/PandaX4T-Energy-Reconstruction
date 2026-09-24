#!/usr/bin/env python3
"""Export the fitted v12 bundle to a portable JSON inference model."""

import argparse
import json

import joblib
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.input)
    state = bundle["predictor_state"]
    model = state["model"]
    portable = {
        "format": "PandaX-v12-portable-json-1",
        "model_name": bundle["model_name"],
        "version": bundle["version"],
        "nominal_energy_kev": bundle["nominal_energy_kev"],
        "base_state": plain(bundle["base_state"]),
        "base_energy_scale": bundle["base_energy_scale"],
        "feature_names": bundle["feature_names"],
        "predictor_state": {
            "transform": plain(state["transform"]),
            "coef": plain(model.coef_),
            "intercept": float(model.intercept_),
            "prediction_center": state["prediction_center"],
            "cap": state["cap"],
            "energy_scale": state["energy_scale"],
            "strong_raw_limit": state["strong_raw_limit"],
        },
        "safety_policy": bundle["safety_policy"],
        "selection": plain(bundle["selection"]),
        "warning": bundle["warning"],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(portable, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
