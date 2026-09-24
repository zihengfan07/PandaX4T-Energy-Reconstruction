#!/usr/bin/env python3
"""Create the v19 summary and publication-ready figures from server outputs."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v19_incremental_features_20260815"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def binned(x, y, n=18):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    edges = np.unique(np.quantile(x, np.linspace(0.01, 0.99, n + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        take = (x >= lo) & (x <= hi if i == len(edges) - 2 else x < hi)
        if take.sum() >= 10:
            q = np.quantile(y[take], [0.16, 0.50, 0.84])
            rows.append((np.median(x[take]), q[0], q[1], q[2]))
    return np.asarray(rows)


def main():
    bundle = joblib.load(OUT / "candidate_v19.joblib")
    candidates = pd.read_csv(OUT / "v19_candidates.csv")
    best = pd.read_csv(OUT / "v19_best_by_feature_set.csv")
    cal = pd.read_csv(OUT / "calibration_events.csv")
    bg = pd.read_csv(OUT / "background_events.txt", sep=r"\s+")

    summary = {
        "status": bundle["status"],
        "recommended_model": bundle["recommended"],
        "recommended_features": bundle["feature_names"],
        "four_variable_baseline": bundle["four_variable_baseline"],
        "dev_selected_extra_candidate": bundle["dev_selected_extra_candidate"],
        "extra_candidate_holdout_pass": bundle["extra_candidate_holdout_pass"],
        "total_candidates": int(len(candidates)),
        "safe_candidates": int(candidates["safe"].sum()),
        "warning": bundle["warning"],
    }
    (OUT / "summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # Best hyperparameter selected on development data for each feature set.
    comp = best.sort_values("dev_sigma_gain", ascending=False).head(16)
    labels = [x.replace("_plus_", "+\n").replace("_", " ") for x in comp.feature_set]
    xx = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.bar(xx - 0.2, 100 * comp.dev_sigma_gain, .4, label="Development time-OOF")
    ax.bar(xx + 0.2, 100 * comp.hold_sigma_gain, .4, label="Untouched last 20%")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel(r"Relative improvement in $\sigma/\mu$ [%]")
    ax.set_title("v19 incremental-variable search: development versus time holdout")
    ax.set_xticks(xx, labels, rotation=48, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_feature_set_comparison.png", dpi=190)
    plt.close(fig)

    names = ["B04_S10"] + ["A{:02d}".format(i) for i in range(1, 10)]
    ablation = []
    for prefix in names:
        rows = best[best.feature_set.str.startswith(prefix)]
        if len(rows):
            ablation.append(rows.iloc[0])
    abl = pd.DataFrame(ablation)
    labels = ["S10\n4 variables"] + ["+" + x.split("_plus_", 1)[1] for x in abl.feature_set.iloc[1:]]
    xx = np.arange(len(abl))
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.bar(xx - .2, 100 * abl.dev_sigma_gain, .4, label="Development time-OOF")
    ax.bar(xx + .2, 100 * abl.hold_sigma_gain, .4, label="Untouched last 20%")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel(r"Relative improvement in $\sigma/\mu$ [%]")
    ax.set_title("Add-one-variable ablation from the four-variable S10 baseline")
    ax.set_xticks(xx, labels, rotation=42, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "02_single_variable_ablation.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].hist(cal.energy_cor_kev, bins=np.arange(2100, 2905, 5), histtype="step", lw=1.5, label="Given cubic Energy_cor")
    axes[0].hist(cal.v19_validation_energy_kev, bins=np.arange(2100, 2905, 5), histtype="step", lw=1.5, label="v19 recommended (S10 retained)")
    axes[0].set(xlabel="Energy [keV]", ylabel="Events / 5 keV", title="Calibration sample: validation spectrum")
    axes[0].legend()
    axes[1].hist(bg.energy_cor_kev, bins=np.arange(500, 3805, 5), histtype="step", lw=1.2, label="Given cubic Energy_cor")
    axes[1].hist(bg.v19_energy_kev, bins=np.arange(500, 3805, 5), histtype="step", lw=1.2, label="v19 recommended (S10 retained)")
    axes[1].set(xlabel="Energy [keV]", ylabel="Events / 5 keV", title="Background sample: spectrum-shape safety check")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "03_energy_spectra.png", dpi=190)
    plt.close(fig)

    features = bundle["feature_names"]
    cal_ratio = cal.v19_validation_energy_kev / cal.energy_cor_kev
    bg_ratio = bg.v19_energy_kev / bg.energy_cor_kev
    fig, axes = plt.subplots(2, len(features), figsize=(4.0 * len(features), 7.2), squeeze=False)
    for col, feature in enumerate(features):
        for row, (frame, ratio, title) in enumerate(((cal, cal_ratio, "Calibration"), (bg, bg_ratio, "Background"))):
            ax = axes[row, col]
            x = frame[feature].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(ratio)
            ids = np.flatnonzero(keep)
            if len(ids) > 5000:
                ids = ids[np.linspace(0, len(ids) - 1, 5000).astype(int)]
            ax.scatter(x[ids], np.asarray(ratio)[ids], s=3, alpha=.13, color="#4C9BE8")
            curve = binned(x, ratio)
            if len(curve):
                ax.fill_between(curve[:, 0], curve[:, 1], curve[:, 3], color="#F4A259", alpha=.22)
                ax.plot(curve[:, 0], curve[:, 2], color="#E76F51", lw=2)
            ax.axhline(1, color="black", ls="--", lw=.8)
            ax.set_xlabel(feature)
            if col == 0:
                ax.set_ylabel(r"$E_{new}/E_{cor}$")
            ax.set_title(title + ": " + feature)
    fig.suptitle("Recommended correction ratio versus the four retained inputs", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "04_ratio_vs_model_inputs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    bins = np.arange(2450, 3002.5, 2.5)
    ax.hist(bg.energy_cor_kev, bins=bins, histtype="step", lw=1.25, label="Given cubic Energy_cor")
    ax.hist(bg.v19_energy_kev, bins=bins, histtype="step", lw=1.25, label="v19 recommended (S10 retained)")
    ax.axvline(2600, color="gray", ls=":", lw=1)
    ax.set(xlabel="Energy [keV]", ylabel="Events / 2.5 keV", title="Background high-energy zoom: fake-peak check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_background_2450_3000_zoom.png", dpi=190)
    plt.close(fig)

    # Separate, larger ratio panels for the report.
    for sample_name, frame, ratio, filename in (
        ("Calibration", cal, cal_ratio, "06_calibration_ratio_vs_inputs.png"),
        ("Background", bg, bg_ratio, "07_background_ratio_vs_inputs.png"),
    ):
        fig, axes = plt.subplots(1, len(features), figsize=(16, 4.3), squeeze=False)
        for col, feature in enumerate(features):
            ax = axes[0, col]
            x = frame[feature].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(ratio)
            ids = np.flatnonzero(keep)
            if len(ids) > 6000:
                ids = ids[np.linspace(0, len(ids) - 1, 6000).astype(int)]
            ax.scatter(x[ids], np.asarray(ratio)[ids], s=3, alpha=.13, color="#4C9BE8")
            curve = binned(x, ratio)
            if len(curve):
                ax.fill_between(curve[:, 0], curve[:, 1], curve[:, 3], color="#F4A259", alpha=.22)
                ax.plot(curve[:, 0], curve[:, 2], color="#E76F51", lw=2)
            ax.axhline(1, color="black", ls="--", lw=.8)
            ax.set_xlabel(feature)
            if col == 0:
                ax.set_ylabel(r"$E_{new}/E_{cor}$")
            ax.set_title(feature)
        fig.suptitle(sample_name + " sample: correction ratio versus retained inputs", fontsize=15)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=190)
        plt.close(fig)

    # Direct metric comparison for the retained baseline and the dev-selected extra model.
    baseline = bundle["four_variable_baseline"]
    extra = bundle["dev_selected_extra_candidate"]
    metrics = ["sigma_gain", "r68_gain", "r90_gain"]
    labels = [r"$\sigma/\mu$", r"$R_{68}$", r"$R_{90}$"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, model, title in zip(axes, (baseline, extra), ("Four-variable S10", "Five-variable candidate")):
        x = np.arange(3)
        dev_values = [100 * float(model["dev_" + m]) for m in metrics]
        hold_values = [100 * float(model["hold_" + m]) for m in metrics]
        ax.bar(x - .2, dev_values, .4, label="Development time-OOF")
        ax.bar(x + .2, hold_values, .4, label="Untouched last 20%")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel("Relative improvement [%]")
    axes[1].legend(loc="best")
    fig.suptitle("Resolution and tail metrics: development versus independent time holdout", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "08_metric_summary.png", dpi=190)
    plt.close(fig)

    # Time stability of the actual recommended output.
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    for frame, ratio, label, marker in ((cal, cal_ratio, "Calibration", "o"), (bg, bg_ratio, "Background", "s")):
        order = np.argsort(frame.t.to_numpy(float), kind="mergesort")
        blocks = np.array_split(order, 10)
        medians, applied = [], []
        for block in blocks:
            values = np.asarray(ratio)[block]
            values = values[np.isfinite(values)]
            medians.append(np.median(values))
            applied.append(np.mean(np.abs(values - 1.0) > 1.0e-12))
        axes[0].plot(np.arange(1, 11), medians, marker=marker, label=label)
        axes[1].plot(np.arange(1, 11), 100 * np.asarray(applied), marker=marker, label=label)
    axes[0].axhline(1, color="black", ls="--", lw=.8)
    axes[0].set_ylabel(r"Median $E_{new}/E_{cor}$")
    axes[0].set_title("Time-block stability of the retained four-variable output")
    axes[0].legend()
    axes[1].set(xlabel="Chronological block", ylabel="Correction applied [%]")
    axes[1].set_xticks(np.arange(1, 11))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "09_time_block_stability.png", dpi=190)
    plt.close(fig)

    readme = (
        "# v19 增量变量实验结论\n\n"
        "共测试 288 个候选，11 个通过开发集与本底安全门槛。\n\n"
        "开发集选出的五变量模型在最后 20% 时间留出集上未通过："
        "sigma/mu 与 R90 变差。\n\n"
        "因此本轮不接受新增变量，正式推荐仍保留四变量 S10。"
        "新增变量结果作为诊断证据保存。\n"
    )
    (OUT / "README_结果说明.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
