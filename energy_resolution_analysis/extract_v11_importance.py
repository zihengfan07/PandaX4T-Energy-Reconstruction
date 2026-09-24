#!/usr/bin/env python3
"""Extract grouped spline coefficient norms from a frozen v11 bundle."""

import argparse
import json

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    features = bundle["feature_names"]
    coefficients = np.asarray(
        bundle["predictor_state"]["model"].coef_, dtype=float
    )
    if len(coefficients) % len(features):
        raise RuntimeError("Unexpected coefficient layout")
    blocks = coefficients.reshape((-1, len(features)))
    importance = np.sqrt(np.sum(blocks ** 2, axis=0))
    order = np.argsort(-importance)
    rows = [
        {
            "rank": int(rank + 1),
            "feature": features[int(index)],
            "importance": float(importance[int(index)]),
            "normalized_importance": float(
                importance[int(index)] / np.max(importance)
            ),
        }
        for rank, index in enumerate(order)
    ]
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
