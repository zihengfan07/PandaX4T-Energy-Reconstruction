#!/usr/bin/env python3
"""Apply every Kr residual candidate to 2615 data for a diagnostic-only audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kr_residual_correction_v21 import (
    VARS, SIGS, read, kr_peak_mask, fit_maps, apply_maps, erec, fit_2615
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kr", required=True); ap.add_argument("--th2615", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cols = SIGS + VARS + ["runNumber"]
    kr, th = read(args.kr, cols), read(args.th2615, cols)
    candidates = [
        ("M00_no_residual", [], []),
        ("M01_s2_dt", [], ["dt"]),
        ("M02_s2_dt_width", [], ["dt", "wS2CDF_max"]),
        ("M03_s2_all4", [], VARS),
        ("M04_both_dt_width", ["dt", "wS2CDF_max"], ["dt", "wS2CDF_max"]),
        ("M05_both_all4", VARS, VARS),
    ]
    peak = kr_peak_mask(kr); all_train = np.ones(len(kr["runNumber"]), bool)
    records = []
    for name, s1_features, s2_features in candidates:
        maps = fit_maps(kr, all_train, peak, s1_features, s2_features)
        s1, s2, factors = apply_maps(th, maps)
        e = erec(s1, s2); fit = fit_2615(e)
        per_run = {}
        for run in np.unique(th["runNumber"]):
            take = th["runNumber"] == run
            per_run[str(int(run))] = fit_2615(e[take])
        records.append({
            "name": name, "s1_features": s1_features, "s2_features": s2_features,
            "fit_all": fit, "fit_per_run": per_run,
            "s1_factor_p01_p50_p99": [float(v) for v in np.quantile(factors["S1"], [.01,.5,.99])],
            "s2_factor_p01_p50_p99": [float(v) for v in np.quantile(factors["S2"], [.01,.5,.99])],
        })
    base = records[0]["fit_all"]["sigma_over_mu"]
    for rec in records:
        rec["2615_relative_sigma_gain"] = float(1 - rec["fit_all"]["sigma_over_mu"] / base)
    payload = {"warning": "Diagnostic only. 2615 results were not used for Kr cross-validation model selection.", "candidates": records}
    (out / "2615_candidate_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    names = [r["name"] for r in records]
    gains = [100*r["2615_relative_sigma_gain"] for r in records]
    centers = [r["fit_all"]["mu_kev"] for r in records]
    widths = [100*r["fit_all"]["sigma_over_mu"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    x = np.arange(len(names))
    axes[0].bar(x, gains); axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(ylabel="2615 fitted sigma/mu improvement [%]", title="Diagnostic peak-width change")
    axes[1].bar(x, widths); axes[1].set(ylabel="fitted sigma/mu [%]", title="Absolute fitted resolution")
    axes[2].bar(x, centers); axes[2].axhline(2614.5, color="gray", ls=":", lw=.8)
    axes[2].set(ylabel="fitted peak center [keV]", title="Peak-center movement")
    for ax in axes:
        ax.set_xticks(x, names, rotation=30, ha="right")
    fig.suptitle("All Kr candidates applied to 2615 data (diagnostic only)")
    fig.tight_layout(); fig.savefig(out / "05_all_candidates_2615_audit.png", dpi=190); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
