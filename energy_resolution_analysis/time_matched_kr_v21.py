#!/usr/bin/env python3
"""Time-matched Kr residual correction audit for the 2615-keV sample.

The purpose is diagnostic: determine whether a Kr response map must be learned
locally in run/time instead of globally.  Candidate maps are selected using
held-out Kr events only.  The 2615-keV data are inspected only after selection.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kr_residual_correction_v21 import (
    G1, G2B, W, VARS, SIGS, read, erec, kr_peak_mask, normalize_by_run,
    apply_curve, apply_maps, fit_2615,
)


def make_local_curve(x, y, bins=16, min_count=80):
    good = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = np.asarray(x[good], float), np.asarray(y[good], float)
    edges = np.unique(np.quantile(x, np.linspace(0.02, 0.98, bins + 1)))
    centers, response = [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        take = (x >= lo) & (x < hi if i < len(edges) - 2 else x <= hi)
        if take.sum() >= min_count:
            centers.append(float(np.median(x[take])))
            response.append(float(np.median(y[take])))
    response = np.asarray(response, float)
    response /= np.median(response)
    return {"x": centers, "response": response.tolist()}


def fit_local_maps(data, train, peak, s1_features, s2_features):
    runs = data["runNumber"]
    maps = {"S1": [], "S2": []}
    for label, signal, features in (
        ("S1", "qS1_PCs", s1_features), ("S2", "qS2Bdes_PCs", s2_features)
    ):
        corrected = normalize_by_run(data[signal], runs, peak)
        for feature in features:
            take = train & peak & np.isfinite(corrected) & np.isfinite(data[feature])
            curve = make_local_curve(data[feature][take], corrected[take])
            factor = apply_curve(data[feature], curve)
            corrected *= factor
            maps[label].append({"feature": feature, "curve": curve})
    return maps


def normalized_r68(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values > 0)]
    values = values / np.median(values)
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return float((q84 - q16) / 2)


def select_on_kr(data, candidates, seed=20260822):
    peak = kr_peak_mask(data)
    idx = np.flatnonzero(peak)
    rng = np.random.default_rng(seed)
    fold_id = rng.integers(0, 4, len(idx))
    records = []
    for name, s1_features, s2_features in candidates:
        base_scores, new_scores = [], []
        for fold in range(4):
            hold = np.zeros(len(data["runNumber"]), dtype=bool)
            hold[idx[fold_id == fold]] = True
            maps = fit_local_maps(data, ~hold, peak, s1_features, s2_features)
            s1, s2, _ = apply_maps(data, maps)
            base_scores.append(normalized_r68(erec(data["qS1_PCs"], data["qS2Bdes_PCs"])[hold]))
            new_scores.append(normalized_r68(erec(s1, s2)[hold]))
        base, new = float(np.mean(base_scores)), float(np.mean(new_scores))
        records.append({
            "name": name, "s1_features": s1_features, "s2_features": s2_features,
            "kr_cv_r68_base": base, "kr_cv_r68_corrected": new,
            "kr_cv_relative_gain": float(1 - new / base),
        })
    safe = [r for r in records if r["kr_cv_relative_gain"] > 0.001]
    chosen = max(safe or [records[0]], key=lambda r: r["kr_cv_relative_gain"])
    maps = fit_local_maps(
        data, np.ones(len(data["runNumber"]), dtype=bool), peak,
        chosen["s1_features"], chosen["s2_features"],
    )
    return records, chosen, maps


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kr", required=True)
    ap.add_argument("--th2615", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cols = SIGS + VARS + ["runNumber"]
    kr, th = read(args.kr, cols), read(args.th2615, cols)
    periods = [
        {"name": "P1", "kr_runs": [9568], "th_runs": [9563, 9566]},
        {"name": "P2", "kr_runs": [9685, 9695], "th_runs": [9698]},
    ]
    candidates = [
        ("M00_none", [], []),
        ("M01_s2_dt", [], ["dt"]),
        ("M02_s2_dt_width", [], ["dt", "wS2CDF_max"]),
        ("M03_both_dt", ["dt"], ["dt"]),
        ("M04_both_dt_width", ["dt", "wS2CDF_max"], ["dt", "wS2CDF_max"]),
    ]

    e_base = erec(th["qS1_PCs"], th["qS2Bdes_PCs"])
    e_new = e_base.copy()
    factors_s1 = np.ones(len(e_base))
    factors_s2 = np.ones(len(e_base))
    result_periods = []

    for period in periods:
        kr_mask = np.isin(kr["runNumber"], period["kr_runs"])
        th_mask = np.isin(th["runNumber"], period["th_runs"])
        local_kr = subset(kr, kr_mask)
        records, chosen, maps = select_on_kr(local_kr, candidates)
        local_th = subset(th, th_mask)
        s1, s2, factors = apply_maps(local_th, maps)
        e_new[th_mask] = erec(s1, s2)
        factors_s1[th_mask] = factors["S1"]
        factors_s2[th_mask] = factors["S2"]
        support = {
            "kr_dt_p001_p01_p50_p99_p999": [float(v) for v in np.quantile(local_kr["dt"], [.001, .01, .5, .99, .999])],
            "th2615_dt_p001_p01_p50_p99_p999": [float(v) for v in np.quantile(local_th["dt"], [.001, .01, .5, .99, .999])],
            "s1_factor_p001_p01_p50_p99_p999": [float(v) for v in np.quantile(factors["S1"], [.001, .01, .5, .99, .999])],
            "s2_factor_p001_p01_p50_p99_p999": [float(v) for v in np.quantile(factors["S2"], [.001, .01, .5, .99, .999])],
            "s1_clip_fraction": float(np.mean((factors["S1"] <= .970001) | (factors["S1"] >= 1.029999))),
            "s2_clip_fraction": float(np.mean((factors["S2"] <= .970001) | (factors["S2"] >= 1.029999))),
        }
        result_periods.append({
            **period,
            "n_kr": int(kr_mask.sum()), "n_2615": int(th_mask.sum()),
            "candidates": records, "selected": chosen, "maps": maps, "support": support,
        })

    fits = {"all": {"base": fit_2615(e_base), "corrected": fit_2615(e_new)}, "per_run": {}}
    for run in np.unique(th["runNumber"]):
        take = th["runNumber"] == run
        fits["per_run"][str(int(run))] = {
            "base": fit_2615(e_base[take]), "corrected": fit_2615(e_new[take])
        }

    payload = {
        "formula": f"{W} * (qS1_PCs / {G1} + qS2Bdes_PCs / {G2B})",
        "status": "diagnostic; maps selected with held-out time-matched Kr events only",
        "periods": result_periods, "2615_fits": fits,
    }
    (out / "time_matched_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savetxt(
        out / "2615_time_matched_output.txt",
        np.column_stack([th["runNumber"], e_base, e_new, factors_s1, factors_s2]),
        header="runNumber Erec_base_kev Erec_timeMatchedKr_kev S1_factor S2_factor", comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, period in zip(axes, result_periods):
        names = [r["name"] for r in period["candidates"]]
        gains = [100 * r["kr_cv_relative_gain"] for r in period["candidates"]]
        colors = ["tab:green" if n == period["selected"]["name"] else "tab:blue" for n in names]
        ax.bar(np.arange(len(names)), gains, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
        ax.set_title(f"{period['name']}: Kr {period['kr_runs']} -> 2615 {period['th_runs']}")
        ax.set_ylabel("held-out Kr R68 improvement [%]")
    fig.tight_layout()
    fig.savefig(out / "06_time_matched_kr_selection.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = np.arange(2450, 2852.5, 2.5)
    panels = [("All", np.ones(len(e_base), bool))] + [
        (f"run {int(run)}", th["runNumber"] == run) for run in np.unique(th["runNumber"])
    ]
    for ax, (title, take) in zip(axes.ravel(), panels):
        ax.hist(e_base[take], bins=bins, histtype="step", lw=1.2, label="given PCs")
        ax.hist(e_new[take], bins=bins, histtype="step", lw=1.2, label="time-matched Kr")
        ax.axvline(2614.5, color="gray", ls=":", lw=0.8)
        ax.set(title=title, xlabel="Erec [keV]", ylabel="events / 2.5 keV")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "07_time_matched_2615_comparison.png", dpi=190)
    plt.close(fig)

    runs = np.unique(th["runNumber"])
    centers_base = [fits["per_run"][str(int(r))]["base"]["mu_kev"] for r in runs]
    centers_new = [fits["per_run"][str(int(r))]["corrected"]["mu_kev"] for r in runs]
    widths_base = [100 * fits["per_run"][str(int(r))]["base"]["sigma_over_mu"] for r in runs]
    widths_new = [100 * fits["per_run"][str(int(r))]["corrected"]["sigma_over_mu"] for r in runs]
    x = np.arange(len(runs)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x-width/2, centers_base, width, label="base")
    axes[0].bar(x+width/2, centers_new, width, label="time-matched Kr")
    axes[0].axhline(2614.5, color="gray", ls=":", lw=0.8)
    axes[0].set(ylabel="fitted center [keV]", title="Run-dependent peak centers")
    axes[1].bar(x-width/2, widths_base, width, label="base")
    axes[1].bar(x+width/2, widths_new, width, label="time-matched Kr")
    axes[1].set(ylabel="fitted sigma / mu [%]", title="Within-run resolution")
    for ax in axes:
        ax.set_xticks(x, [str(int(r)) for r in runs]); ax.set_xlabel("runNumber"); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "08_run_center_and_resolution.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(result_periods), figsize=(12, 7), squeeze=False)
    for col, period in enumerate(result_periods):
        kr_take = np.isin(kr["runNumber"], period["kr_runs"])
        th_take = np.isin(th["runNumber"], period["th_runs"])
        bins = np.linspace(0, 860000, 55)
        axes[0, col].hist(kr["dt"][kr_take], bins=bins, density=True, histtype="step", lw=1.4, label="Kr")
        axes[0, col].hist(th["dt"][th_take], bins=bins, density=True, histtype="step", lw=1.4, label="2615")
        axes[0, col].set(title=f"{period['name']} dt coverage", xlabel="dt", ylabel="normalized density")
        axes[0, col].legend()
        grid = np.linspace(0, 860000, 500)
        for label, color in (("S1", "tab:orange"), ("S2", "tab:blue")):
            factor = np.ones_like(grid)
            for item in period["maps"][label]:
                if item["feature"] == "dt":
                    factor *= apply_curve(grid, item["curve"])
            axes[1, col].plot(grid, factor, color=color, label=f"{label} factor")
        axes[1, col].axhline(1, color="black", ls="--", lw=.7)
        axes[1, col].axhline(.97, color="gray", ls=":", lw=.7)
        axes[1, col].axhline(1.03, color="gray", ls=":", lw=.7)
        clip = 100 * period["support"]["s2_clip_fraction"]
        axes[1, col].set(title=f"selected factors; S2 clip = {clip:.1f}%", xlabel="dt", ylabel="multiplicative factor")
        axes[1, col].legend()
    fig.tight_layout()
    fig.savefig(out / "09_dt_coverage_and_factor_limits.png", dpi=190)
    plt.close(fig)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
