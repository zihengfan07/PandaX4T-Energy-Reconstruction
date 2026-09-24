#!/usr/bin/env python3
"""Create compact diagnostic plots for the v12 safe model."""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def window_fraction(values, center, half_width):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    return float(np.mean(np.abs(values[valid] - center) <= half_width))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent / "results" / "v12_safe_20260803"),
    )
    args = parser.parse_args()
    root = Path(args.root)
    bg = pd.read_csv(root / "background_events.csv")
    cal = pd.read_csv(root / "calibration_oof_events.csv")

    before = bg.foundation_energy_kev.to_numpy(float)
    after = bg.candidate_v12_energy_kev.to_numpy(float)
    edges = np.arange(1000.0, 3805.0, 5.0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(before, bins=edges, histtype="step", linewidth=1.2, color="#2455c3", label="robust foundation")
    ax.hist(after, bins=edges, histtype="step", linewidth=1.2, color="#d62728", label="v12 safe")
    for value, label in [(1307.5, "old spike A"), (2600.0, "signal-region check"), (2887.5, "old spike B")]:
        ax.axvline(value, color="0.45", linestyle="--", linewidth=0.8)
        ax.text(value + 8, ax.get_ylim()[1] * 0.82, label, rotation=90, va="top", fontsize=8)
    ax.set(xlabel="Energy [keV]", ylabel="Events / 5 keV", title="Background spectrum: before and after v12_safe")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "background_spectrum_v12_comparison.png", dpi=180)
    plt.close(fig)

    c_before = cal.foundation_energy_kev.to_numpy(float)
    c_after = cal.candidate_v12_oof_energy_kev.to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    edges = np.arange(2200.0, 3050.0, 10.0)
    ax.hist(c_before, bins=edges, histtype="step", linewidth=1.5, color="#2455c3", label="robust foundation")
    ax.hist(c_after, bins=edges, histtype="step", linewidth=1.5, color="#d62728", label="v12 grouped OOF")
    ax.axvline(2614.5, color="0.3", linestyle="--", linewidth=1.0)
    ax.set(xlabel="Energy [keV]", ylabel="Events / 10 keV", title="2614.5 keV calibration line (out-of-fold)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "calibration_oof_v12_comparison.png", dpi=180)
    plt.close(fig)

    bg["time_block"] = pd.qcut(bg.fileNumber, 10, labels=False, duplicates="drop")
    rows = []
    for block, part in bg.groupby("time_block"):
        valid = np.isfinite(part.foundation_energy_kev) & np.isfinite(part.candidate_v12_energy_kev)
        ratio = part.loc[valid, "candidate_v12_energy_kev"] / part.loc[valid, "foundation_energy_kev"]
        rows.append({
            "block": int(block), "file_min": int(part.fileNumber.min()),
            "file_max": int(part.fileNumber.max()), "events": int(len(part)),
            "median_energy_ratio": float(np.median(ratio)),
            "applied_fraction": float(part.correction_applied.mean()),
            "near_cap_fraction": float(part.correction_near_cap.mean()),
            "ood_fraction": float(part.out_of_distribution.mean()),
        })
    blocks = pd.DataFrame(rows)
    blocks.to_csv(root / "background_time_block_audit.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(blocks.block, 100.0 * (blocks.median_energy_ratio - 1.0), "o-", color="#6a3d9a")
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set(xlabel="Chronological fileNumber decile", ylabel="Median energy change [%]", title="v12_safe time stability on background")
    fig.tight_layout()
    fig.savefig(root / "background_time_stability_v12.png", dpi=180)
    plt.close(fig)

    windows = []
    for center, width in [(1307.5, 5.0), (2600.0, 20.0), (2887.5, 5.0)]:
        a = window_fraction(before, center, width)
        b = window_fraction(after, center, width)
        windows.append({
            "center_kev": center, "half_width_kev": width,
            "foundation_fraction": a, "v12_fraction": b,
            "fraction_ratio": b / a if a else None,
        })
    audit = {
        "problem_windows": windows,
        "max_abs_time_block_median_energy_change": float(
            np.max(np.abs(blocks.median_energy_ratio - 1.0))
        ),
        "max_time_block_near_cap_fraction": float(blocks.near_cap_fraction.max()),
    }
    (root / "background_detailed_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
