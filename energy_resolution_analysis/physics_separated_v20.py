#!/usr/bin/env python3
"""v20 physics-first audit: correct S1 and S2 separately before combining energy.

This script intentionally does not train an event-level regressor toward a peak.  It learns only
smooth drift-time response functions on the first 80% of the calibration run, applies them to the
untouched last 20%, and checks the result on a background sample when one is supplied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    plt = None
    HAVE_MATPLOTLIB = False


NOMINAL_KEV = 2614.5
NEEDED = [
    "runNumber", "eventNumber", "t", "dt",
    "qS1_max", "qS2Bdes_max", "qS1ub_C", "qS2Bdesub_C",
    "xS2Tcor_max", "yS2Tcor_max",
]


def load_table(path: str) -> pd.DataFrame:
    header = Path(path).open("r", encoding="utf-8", errors="replace").readline()
    sep = "\t" if "\t" in header else r"\s+"
    names = header.strip().split("\t") if sep == "\t" else header.strip().split()
    missing = [c for c in NEEDED if c not in names]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))
    return pd.read_csv(path, sep=sep, usecols=NEEDED, engine="python")


def given_energy(frame):
    s1 = frame["qS1ub_C"].to_numpy(float)
    s2 = frame["qS2Bdesub_C"].to_numpy(float)
    linear = 0.0137 * (s1 / 0.125 + s2 / 10.58)
    cubic = (
        -1.73706e-9 * linear ** 3
        + 7.98193e-6 * linear ** 2
        + 1.07904 * linear
        - 9.22086
    )
    valid = np.isfinite(linear) & np.isfinite(cubic) & (s1 > 0) & (s2 > 0)
    return cubic, valid


def binned_medians(x: np.ndarray, y: np.ndarray, bins: int = 24):
    good = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[good], y[good]
    edges = np.unique(np.quantile(x, np.linspace(0.01, 0.99, bins + 1)))
    bx, by, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (x >= lo) & (x < hi if hi != edges[-1] else x <= hi)
        if take.sum() < 20:
            continue
        bx.append(float(np.median(x[take])))
        by.append(float(np.median(y[take])))
        bn.append(int(take.sum()))
    return np.asarray(bx), np.asarray(by), np.asarray(bn)


def fit_state(frame: pd.DataFrame) -> dict:
    dt_us = frame["dt"].to_numpy(float) / 1000.0
    q1 = frame["qS1_max"].to_numpy(float)
    q2 = frame["qS2Bdes_max"].to_numpy(float)
    good = np.isfinite(dt_us) & np.isfinite(q1) & np.isfinite(q2) & (q1 > 0) & (q2 > 0)
    dt_us, q1, q2 = dt_us[good], q1[good], q2[good]
    dt_lo, dt_hi = np.quantile(dt_us, [0.01, 0.99])
    ref = float(np.median(dt_us))
    scale = max(float((np.quantile(dt_us, 0.84) - np.quantile(dt_us, 0.16)) / 2), 1.0)

    bx2, by2, _ = binned_medians(dt_us, q2)
    slope, intercept = np.polyfit(bx2, np.log(by2), 1)
    # Electron attachment requires a non-positive slope.  Bound only catastrophic fits.
    slope = float(np.clip(slope, -1.0 / 250.0, 0.0))
    tau_us = float(-1.0 / slope) if slope < 0 else float("inf")

    bx1, by1, _ = binned_medians(dt_us, q1)
    u1 = (bx1 - ref) / scale
    q1_poly = np.polyfit(u1, by1, 3).tolist()
    q1_ref = float(np.polyval(q1_poly, 0.0))
    return {
        "dt_reference_us": ref,
        "dt_scale_us": scale,
        "dt_fit_range_us": [float(dt_lo), float(dt_hi)],
        "q2_log_slope_per_us": slope,
        "q2_log_intercept": float(intercept),
        "electron_lifetime_us": tau_us,
        "q1_polynomial_descending": [float(v) for v in q1_poly],
        "q1_reference_response": q1_ref,
        "correction_factor_bounds": [0.65, 1.35],
    }


def apply_state(frame, state):
    dt = frame["dt"].to_numpy(float) / 1000.0
    q1 = frame["qS1_max"].to_numpy(float)
    q2 = frame["qS2Bdes_max"].to_numpy(float)
    lo, hi = state["dt_fit_range_us"]
    dt_eval = np.clip(dt, lo, hi)
    ref, scale = state["dt_reference_us"], state["dt_scale_us"]
    q2_factor = np.exp(-state["q2_log_slope_per_us"] * (dt_eval - ref))
    response = np.polyval(state["q1_polynomial_descending"], (dt_eval - ref) / scale)
    q1_factor = state["q1_reference_response"] / response
    factor_lo, factor_hi = state["correction_factor_bounds"]
    q1_factor = np.clip(q1_factor, factor_lo, factor_hi)
    q2_factor = np.clip(q2_factor, factor_lo, factor_hi)
    q1c, q2c = q1 * q1_factor, q2 * q2_factor
    # Keep the currently used g1/g2 weights fixed in this first audit.
    e_before = 0.0137 * (q1 / 0.125 + q2 / 10.58)
    e_after = 0.0137 * (q1c / 0.125 + q2c / 10.58)
    return {
        "q1_corrected": q1c, "q2_corrected": q2c,
        "q1_factor": q1_factor, "q2_factor": q2_factor,
        "energy_before": e_before, "energy_after": e_after,
        "dt_ood": (dt < lo) | (dt > hi),
    }


def robust_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 30:
        return {"n": int(len(x))}
    q05, q16, q50, q84, q95 = np.quantile(x, [0.05, 0.16, 0.50, 0.84, 0.95])
    return {
        "n": int(len(x)), "median_kev": float(q50),
        "r68": float((q84 - q16) / (2 * q50)),
        "r90": float((q95 - q05) / (2 * q50)),
    }


def weak_line_count(values: np.ndarray) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    signal = int(np.sum((x >= 2595) & (x < 2635)))
    side = int(np.sum(((x >= 2555) & (x < 2595)) | ((x >= 2635) & (x < 2675))))
    expected = side / 2.0
    return {
        "events_2595_2635": signal,
        "sideband_expected": float(expected),
        "simple_excess": float(signal - expected),
    }


def rescale_from_dev(dev_given: np.ndarray, dev_physics: np.ndarray) -> float:
    good = np.isfinite(dev_given) & np.isfinite(dev_physics) & (dev_physics > 0)
    return float(np.median(dev_given[good]) / np.median(dev_physics[good]))


def relation(ax, x, y, label, color):
    bx, by, _ = binned_medians(np.asarray(x, float), np.asarray(y, float), bins=22)
    ax.scatter(np.asarray(x)[::max(len(x)//3500, 1)], np.asarray(y)[::max(len(y)//3500, 1)],
               s=3, alpha=0.08, color=color)
    ax.plot(bx, by, lw=2.2, color=color, label=label)


def make_plots(out: Path, cal: pd.DataFrame, split: int, given: np.ndarray,
               applied: dict, scale_factor: float, bg_pack=None):
    dt = cal["dt"].to_numpy(float) / 1000.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    relation(axes[0, 0], dt, cal["qS1_max"], "raw S1", "tab:blue")
    relation(axes[0, 1], dt, applied["q1_corrected"], "corrected S1", "tab:orange")
    relation(axes[1, 0], dt, cal["qS2Bdes_max"], "raw S2", "tab:blue")
    relation(axes[1, 1], dt, applied["q2_corrected"], "corrected S2", "tab:orange")
    for ax in axes.ravel():
        ax.set_xlabel("drift time [us]"); ax.legend(); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("S1 charge [PE]"); axes[0, 1].set_ylabel("S1 charge [PE]")
    axes[1, 0].set_ylabel("S2 bottom charge [PE]"); axes[1, 1].set_ylabel("S2 bottom charge [PE]")
    fig.suptitle("v20 physics audit: drift-time response before and after separate correction")
    fig.tight_layout(); fig.savefig(out / "01_s1_s2_vs_dt.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(cal["qS1_max"], cal["qS2Bdes_max"], gridsize=65, bins="log", mincnt=1)
    axes[1].hexbin(applied["q1_corrected"], applied["q2_corrected"], gridsize=65, bins="log", mincnt=1)
    axes[0].set_title("Before separate correction"); axes[1].set_title("After separate correction")
    for ax in axes:
        ax.set_xlabel("S1 charge [PE]"); ax.set_ylabel("S2 bottom charge [PE]")
    fig.tight_layout(); fig.savefig(out / "02_s1_s2_anticorrelation.png", dpi=190); plt.close(fig)

    physical = scale_factor * applied["energy_after"]
    bins = np.arange(1900, 3100 + 5, 5)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, idx, title in ((axes[0], slice(0, split), "Development first 80%"),
                           (axes[1], slice(split, None), "Untouched last 20%")):
        ax.hist(given[idx], bins=bins, histtype="step", lw=1.25, label="given cubic energy")
        ax.hist(physical[idx], bins=bins, histtype="step", lw=1.25, label="v20 separate S1/S2")
        ax.axvline(NOMINAL_KEV, color="gray", ls=":", lw=1)
        ax.set_ylabel("events / 5 keV"); ax.set_title(title); ax.legend()
    axes[1].set_xlabel("energy [keV]")
    fig.tight_layout(); fig.savefig(out / "03_calibration_energy_comparison.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ratio = physical / given
    relation(axes[0], dt, ratio, "median ratio", "tab:red")
    axes[0].axhline(1, color="black", ls="--", lw=.8)
    axes[0].set(xlabel="drift time [us]", ylabel="v20 / given energy")
    axes[1].hist(ratio[np.isfinite(ratio)], bins=np.linspace(.7, 1.3, 121), histtype="step", lw=1.5)
    axes[1].set(xlabel="v20 / given energy", ylabel="events")
    fig.tight_layout(); fig.savefig(out / "04_energy_ratio_audit.png", dpi=190); plt.close(fig)

    if bg_pack is not None:
        bg, bg_given, bg_applied = bg_pack
        bg_physical = scale_factor * bg_applied["energy_after"]
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        bins = np.arange(500, 3800 + 5, 5)
        axes[0].hist(bg_given, bins=bins, histtype="step", lw=1.1, label="given cubic energy")
        axes[0].hist(bg_physical, bins=bins, histtype="step", lw=1.1, label="v20 separate S1/S2")
        axes[0].set(xlabel="energy [keV]", ylabel="events / 5 keV", title="Background spectrum safety check")
        axes[0].legend()
        bins = np.arange(2450, 3000 + 2.5, 2.5)
        axes[1].hist(bg_given, bins=bins, histtype="step", lw=1.1, label="given cubic energy")
        axes[1].hist(bg_physical, bins=bins, histtype="step", lw=1.1, label="v20 separate S1/S2")
        axes[1].axvline(2600, color="gray", ls=":", lw=1)
        axes[1].set(xlabel="energy [keV]", ylabel="events / 2.5 keV", title="2450--3000 keV zoom")
        axes[1].legend()
        fig.tight_layout(); fig.savefig(out / "05_background_safety.png", dpi=190); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--background")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    cal = load_table(args.calibration).sort_values("t", kind="mergesort").reset_index(drop=True)
    given, valid = given_energy(cal)
    cal = cal.loc[valid].reset_index(drop=True); given = given[valid]
    split = int(0.80 * len(cal))
    state = fit_state(cal.iloc[:split])
    applied = apply_state(cal, state)
    factor = rescale_from_dev(given[:split], applied["energy_after"][:split])
    physical = factor * applied["energy_after"]

    summary = {
        "status": "physics_audit_not_final_model",
        "calibration_events": int(len(cal)),
        "development_events": int(split),
        "untouched_holdout_events": int(len(cal) - split),
        "state": state,
        "energy_scale_factor_from_development_median": factor,
        "development": {
            "given": robust_stats(given[:split]),
            "v20": robust_stats(physical[:split]),
            "given_2615_line_count": weak_line_count(given[:split]),
            "v20_2615_line_count": weak_line_count(physical[:split]),
        },
        "holdout": {
            "given": robust_stats(given[split:]),
            "v20": robust_stats(physical[split:]),
            "given_2615_line_count": weak_line_count(given[split:]),
            "v20_2615_line_count": weak_line_count(physical[split:]),
        },
        "warnings": [
            "The calibration file is continuum dominated; the 2614.5 keV line is weak.",
            "R68/R90 over all events are spectrum-shape diagnostics, not peak resolution.",
            "The v20 scale is anchored to the development median and is not a new absolute calibration.",
            "Cross-run and multi-line calibration are required before deployment.",
        ],
    }

    bg_pack = None
    if args.background:
        bg = load_table(args.background)
        bg_given, bg_valid = given_energy(bg)
        bg = bg.loc[bg_valid].reset_index(drop=True); bg_given = bg_given[bg_valid]
        bg_applied = apply_state(bg, state)
        bg_physical = factor * bg_applied["energy_after"]
        summary["background"] = {
            "events": int(len(bg)),
            "given": robust_stats(bg_given), "v20": robust_stats(bg_physical),
            "dt_ood_fraction": float(np.mean(bg_applied["dt_ood"])),
            "given_2615_line_count": weak_line_count(bg_given),
            "v20_2615_line_count": weak_line_count(bg_physical),
        }
        bg_pack = (bg, bg_given, bg_applied)
        pd.DataFrame({
            "given_energy_kev": bg_given,
            "v20_physics_energy_kev": bg_physical,
        }).to_csv(out / "background_energies.txt", sep=" ", index=False)

    pd.DataFrame({
        "runNumber": cal.runNumber, "eventNumber": cal.eventNumber, "t": cal.t,
        "given_energy_kev": given, "v20_physics_energy_kev": physical,
        "qS1_factor": applied["q1_factor"], "qS2_factor": applied["q2_factor"],
    }).to_csv(out / "calibration_events.csv", index=False)
    (out / "v20_physics_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if HAVE_MATPLOTLIB:
        make_plots(out, cal, split, given, applied, factor, bg_pack)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
