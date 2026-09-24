#!/usr/bin/env python3
"""Paired bootstrap uncertainty for the time-matched Kr 2615-keV audit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kr_residual_correction_v21 import fit_2615


def bootstrap_group(base, new, iterations, rng):
    n = len(base)
    gains, base_r, new_r = [], [], []
    for _ in range(iterations):
        take = rng.integers(0, n, n)
        fb, fn = fit_2615(base[take]), fit_2615(new[take])
        if not (fb.get("success") and fn.get("success")):
            continue
        rb, rn = fb["sigma_over_mu"], fn["sigma_over_mu"]
        if 0.005 < rb < 0.06 and 0.005 < rn < 0.06:
            base_r.append(rb); new_r.append(rn); gains.append(1 - rn / rb)
    def summary(values):
        q = np.quantile(values, [.025, .5, .975])
        return {"p025": float(q[0]), "median": float(q[1]), "p975": float(q[2])}
    return {
        "successful_iterations": len(gains),
        "base_sigma_over_mu": summary(base_r),
        "corrected_sigma_over_mu": summary(new_r),
        "relative_gain": summary(gains),
        "probability_gain_positive": float(np.mean(np.asarray(gains) > 0)),
        "gain_samples": gains,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iterations", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = np.genfromtxt(args.input, names=True)
    rng = np.random.default_rng(20260822)
    groups = [("all", np.ones(len(data), bool))] + [
        (f"run_{int(run)}", data["runNumber"] == run) for run in np.unique(data["runNumber"])
    ]
    results = {}
    for name, take in groups:
        results[name] = bootstrap_group(
            data["Erec_base_kev"][take], data["Erec_timeMatchedKr_kev"][take],
            args.iterations, rng,
        )
    serializable = {name: {k: v for k, v in row.items() if k != "gain_samples"} for name, row in results.items()}
    (out / "time_matched_bootstrap.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, row) in zip(axes.ravel(), results.items()):
        values = 100 * np.asarray(row["gain_samples"])
        ax.hist(values, bins=30, histtype="stepfilled", alpha=.45)
        ax.axvline(0, color="black", lw=.8)
        q = row["relative_gain"]
        ax.set(title=name, xlabel="relative sigma/mu improvement [%]", ylabel="bootstrap samples")
        ax.text(.03, .95, f"median={100*q['median']:.2f}%\n95% CI=[{100*q['p025']:.2f}, {100*q['p975']:.2f}]%\nP(gain>0)={row['probability_gain_positive']:.3f}",
                transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out / "10_time_matched_bootstrap.png", dpi=190)
    plt.close(fig)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
