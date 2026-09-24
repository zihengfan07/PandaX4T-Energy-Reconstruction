#!/usr/bin/env python3
"""Kr-derived residual S1/S2 correction, evaluated on independent 2615-keV data.

The inputs qS1_PCs and qS2Bdes_PCs are already position-corrected quantities.
This code therefore learns only small, smooth *residual* factors from the Kr
single-energy control sample.  Candidate maps are selected with held-out Kr
runs before any 2615-keV result is inspected.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import uproot
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


G1, G2B, W = 0.128755, 8.6125, 0.0137
VARS = ["dt", "wS2CDF_max", "xS2max_desImageMCPAF_firstS2", "yS2max_desImageMCPAF_firstS2"]
SIGS = ["qS1_PCs", "qS2Bdes_PCs"]
MAX_FACTOR_DEVIATION = 0.03


def read(path, cols):
    tree = uproot.open(path)["out_tree"]
    return tree.arrays(cols, library="np")


def erec(s1, s2):
    return W * (s1 / G1 + s2 / G2B)


def kr_peak_mask(data):
    e = erec(data["qS1_PCs"], data["qS2Bdes_PCs"])
    return np.isfinite(e) & (e >= 37.75) & (e <= 44.75)


def normalize_by_run(values, runs, mask):
    result = np.full(len(values), np.nan)
    for run in np.unique(runs[mask]):
        take = mask & (runs == run) & np.isfinite(values) & (values > 0)
        if take.sum() >= 100:
            result[take] = values[take] / np.median(values[take])
    return result


def make_curve(x, y, bins=48):
    good = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = np.asarray(x[good], float), np.asarray(y[good], float)
    edges = np.unique(np.quantile(x, np.linspace(.01, .99, bins + 1)))
    centers, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (x >= lo) & (x < hi if hi != edges[-1] else x <= hi)
        if take.sum() < 200:
            continue
        centers.append(float(np.median(x[take])))
        med.append(float(np.median(y[take])))
    centers, med = np.asarray(centers), np.asarray(med)
    med /= np.median(med)
    return {"x": centers.tolist(), "response": med.tolist()}


def apply_curve(x, curve):
    xx, yy = np.asarray(curve["x"], float), np.asarray(curve["response"], float)
    response = np.interp(np.asarray(x, float), xx, yy, left=yy[0], right=yy[-1])
    return np.clip(1.0 / response, 1.0 - MAX_FACTOR_DEVIATION, 1.0 + MAX_FACTOR_DEVIATION)


def fit_maps(data, train_mask, peak_mask, s1_features, s2_features):
    runs = data["runNumber"]
    n1 = normalize_by_run(data["qS1_PCs"], runs, peak_mask)
    n2 = normalize_by_run(data["qS2Bdes_PCs"], runs, peak_mask)
    maps = {"S1": [], "S2": []}
    for label, norm, features in (("S1", n1, s1_features), ("S2", n2, s2_features)):
        corrected = norm.copy()
        for feature in features:
            take = train_mask & peak_mask & np.isfinite(corrected) & np.isfinite(data[feature])
            curve = make_curve(data[feature][take], corrected[take])
            factor = apply_curve(data[feature], curve)
            corrected *= factor
            maps[label].append({"feature": feature, "curve": curve})
    return maps


def apply_maps(data, maps):
    factors = {"S1": np.ones(len(data["runNumber"])), "S2": np.ones(len(data["runNumber"]))}
    for label in ("S1", "S2"):
        for item in maps[label]:
            factors[label] *= apply_curve(data[item["feature"]], item["curve"])
        factors[label] = np.clip(factors[label], 1.0 - MAX_FACTOR_DEVIATION, 1.0 + MAX_FACTOR_DEVIATION)
    s1 = data["qS1_PCs"] * factors["S1"]
    s2 = data["qS2Bdes_PCs"] * factors["S2"]
    return s1, s2, factors


def r68_by_run(e, runs, base_peak):
    pieces = []
    for run in np.unique(runs[base_peak]):
        take = base_peak & (runs == run) & np.isfinite(e) & (e > 0)
        if take.sum() >= 100:
            pieces.append(e[take] / np.median(e[take]))
    x = np.concatenate(pieces)
    q16, q84 = np.quantile(x, [.16, .84])
    return float((q84 - q16) / 2.0), int(len(x))


def gaussian_linear(x, amplitude, mu, sigma, offset, slope):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset + slope * (x - mu)


def fit_2615(e):
    x = np.asarray(e, float)
    x = x[np.isfinite(x) & (x > 0)]
    bins = np.arange(2450, 2852.5, 2.5)
    counts, edges = np.histogram(x, bins=bins)
    centers = .5 * (edges[:-1] + edges[1:])
    p0 = [max(float(counts.max() - np.median(counts)), 1), float(centers[counts.argmax()]), 70., float(np.median(counts)), 0.]
    try:
        pars, _ = curve_fit(
            gaussian_linear, centers, counts, p0=p0,
            bounds=([0, 2550, 5, 0, -10], [np.inf, 2750, 200, np.inf, 10]),
            maxfev=50000,
        )
        return {"success": True, "amplitude": float(pars[0]), "mu_kev": float(pars[1]),
                "sigma_kev": float(pars[2]), "sigma_over_mu": float(pars[2] / pars[1]),
                "offset": float(pars[3]), "slope": float(pars[4])}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def cross_validate(kr, candidates):
    peak = kr_peak_mask(kr)
    run_ids = np.unique(kr["runNumber"])
    folds = [np.asarray(v) for v in np.array_split(run_ids, 4)]
    result = []
    for name, s1_features, s2_features in candidates:
        base_values, corrected_values = [], []
        for held_runs in folds:
            hold = np.isin(kr["runNumber"], held_runs)
            maps = fit_maps(kr, ~hold, peak, s1_features, s2_features)
            s1, s2, _ = apply_maps(kr, maps)
            base = erec(kr["qS1_PCs"], kr["qS2Bdes_PCs"])
            new = erec(s1, s2)
            base_r, n = r68_by_run(base[hold], kr["runNumber"][hold], peak[hold])
            new_r, _ = r68_by_run(new[hold], kr["runNumber"][hold], peak[hold])
            base_values.append(base_r); corrected_values.append(new_r)
        result.append({"name": name, "s1_features": s1_features, "s2_features": s2_features,
                       "cv_r68_base": float(np.mean(base_values)), "cv_r68_corrected": float(np.mean(corrected_values)),
                       "cv_relative_gain": float(1.0 - np.mean(corrected_values) / np.mean(base_values))})
    return result


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
    cv = cross_validate(kr, candidates)
    safe = [row for row in cv if row["cv_relative_gain"] > 0.001]
    chosen = max(safe or [cv[0]], key=lambda row: row["cv_relative_gain"])
    peak = kr_peak_mask(kr)
    maps = fit_maps(kr, np.ones(len(kr["runNumber"]), bool), peak, chosen["s1_features"], chosen["s2_features"])
    s1_new, s2_new, factors = apply_maps(th, maps)
    e_base, e_new = erec(th["qS1_PCs"], th["qS2Bdes_PCs"]), erec(s1_new, s2_new)
    fits = {"all_base": fit_2615(e_base), "all_corrected": fit_2615(e_new), "per_run": {}}
    for run in np.unique(th["runNumber"]):
        take = th["runNumber"] == run
        fits["per_run"][str(int(run))] = {"base": fit_2615(e_base[take]), "corrected": fit_2615(e_new[take])}
    summary = {"formula_base": "0.0137 * (qS1_PCs / 0.128755 + qS2Bdes_PCs / 8.6125)",
               "method": "small residual maps learned only from Kr single-energy events; model chosen by held-out Kr runs",
               "candidate_cross_validation": cv, "selected_model": chosen, "2615_peak_fits": fits,
               "factor_clip": [1-MAX_FACTOR_DEVIATION, 1+MAX_FACTOR_DEVIATION]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "residual_maps.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")
    np.savetxt(out / "2615_energy_output.txt", np.column_stack([th["runNumber"], e_base, e_new, factors["S1"], factors["S2"]]),
               header="runNumber Erec_base_kev Erec_krResidual_kev S1_factor S2_factor", comments="")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["name"] for r in cv]; gains = [100*r["cv_relative_gain"] for r in cv]
    colors = ["tab:green" if r["name"] == chosen["name"] else "tab:blue" for r in cv]
    ax.bar(np.arange(len(names)), gains, color=colors)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    ax.set(ylabel="Held-out Kr relative R68 improvement [%]", title="Residual-map candidates selected without 2615 data")
    fig.tight_layout(); fig.savefig(out / "01_kr_cross_validation.png", dpi=190); plt.close(fig)

    ncols = max(len(maps["S1"]), len(maps["S2"]), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(4*ncols, 6), squeeze=False)
    for i, label in enumerate(("S1", "S2")):
        for j in range(ncols):
            ax = axes[i, j]
            if j < len(maps[label]):
                item = maps[label][j]; xx = np.asarray(item["curve"]["x"]); yy = np.asarray(item["curve"]["response"])
                ax.plot(xx, 1/yy, lw=2); ax.axhline(1, color="black", ls="--", lw=.7)
                ax.set(title=label + " factor: " + item["feature"], xlabel=item["feature"], ylabel="multiplicative factor")
            else: ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "02_selected_residual_maps.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = np.arange(2450, 2852.5, 2.5)
    for ax, title, take in ((axes[0,0], "All 2615 events", np.ones(len(e_base), bool)),
                            (axes[0,1], "run 9563", th["runNumber"]==9563),
                            (axes[1,0], "run 9566", th["runNumber"]==9566),
                            (axes[1,1], "run 9698", th["runNumber"]==9698)):
        ax.hist(e_base[take], bins=bins, histtype="step", lw=1.2, label="given PCs")
        ax.hist(e_new[take], bins=bins, histtype="step", lw=1.2, label="Kr residual corrected")
        ax.axvline(2614.5, color="gray", ls=":", lw=.8); ax.set(title=title, xlabel="Erec [keV]", ylabel="events / 2.5 keV")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "03_2615_spectrum_comparison.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(factors["S1"], bins=80, histtype="step", label="S1 factor")
    axes[0].hist(factors["S2"], bins=80, histtype="step", label="S2 factor")
    axes[0].set(title="Applied residual factors on 2615 data", xlabel="factor", ylabel="events"); axes[0].legend()
    good = np.isfinite(e_base) & np.isfinite(e_new) & (e_base > 0)
    axes[1].hist(e_new[good]/e_base[good], bins=100, histtype="step")
    axes[1].axvline(1, color="black", ls="--", lw=.7)
    axes[1].set(title="Final energy ratio", xlabel="E corrected / E base", ylabel="events")
    fig.tight_layout(); fig.savefig(out / "04_2615_correction_size.png", dpi=190); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
