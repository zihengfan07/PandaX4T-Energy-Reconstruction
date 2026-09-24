#!/usr/bin/env python3
"""Export the anchored v13 joblib bundle as portable inference JSON."""

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
        "format": "PandaX-v13-portable-json-1",
        "model_name": bundle["model_name"],
        "version": bundle["version"],
        "base_formula": bundle["base_formula"],
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
        "selection": plain(bundle["selection"]),
        "warning": bundle["warning"],
        "status": "conservative_research_candidate_not_final_physics_model",
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(portable, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
