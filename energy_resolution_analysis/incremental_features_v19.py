#!/usr/bin/env python3
"""v19: add variables to the four-variable S10 model with time holdout checks."""

from __future__ import print_function

import argparse
import itertools
import json
from pathlib import Path

import joblib
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

from anchored_smallmodels_v14 import chronological_folds, energy_cor, time_audit
from grouped_nested_physics_v9 import protocol_metrics
from safe_general_v12 import energy_trend, load_table, local_spike_score, robust_metrics
from multivariable_search_v15 import fit_state, apply_state


BASE = ["dt", "wS2CDF_max", "yS2Tcor_max", "xS2Bcor_max"]
EXTRAS = [
    "qS2channelStdevTo1_max",
    "xS2T_max",
    "yS2B_max",
    "qS1channelStdevTo1_max",
    "qS2hitStdevTo1_max",
    "rmsCogS1T_max",
    "wS2FWHM_max",
    "tDiffBottomTopS2_max",
    "nPMTS2_max",
]


def build_feature_sets():
    sets = {"B04_S10": BASE}
    for i, extra in enumerate(EXTRAS, 1):
        sets["A{:02d}_plus_{}".format(i, extra)] = BASE + [extra]
    for i, (a, b) in enumerate(itertools.combinations(EXTRAS[:5], 2), 1):
        sets["P{:02d}_plus_pair".format(i)] = BASE + [a, b]
    sets.update({
        "G01_plus_s2_pattern": BASE + ["qS2channelStdevTo1_max", "qS2hitStdevTo1_max"],
        "G02_plus_s2_shape": BASE + ["wS2FWHM_max", "tDiffBottomTopS2_max"],
        "G03_plus_topology": BASE + ["nPMTS2_max", "rmsCogS1T_max"],
        "G04_plus_raw_xy": BASE + ["xS2T_max", "yS2B_max"],
        "G05_plus_top5": BASE + EXTRAS[:5],
    })
    # Deduplicate feature lists while keeping informative names.
    unique = {}
    seen = set()
    for name, values in sets.items():
        key = tuple(values)
        if key not in seen:
            unique[name] = values
            seen.add(key)
    return unique


FEATURE_SETS = build_feature_sets()
ALPHAS = (300.0, 1000.0, 3000.0, 10000.0)
CAPS = (0.0025, 0.0050, 0.0075)


def gains(base_values, new_values):
    b_r = robust_metrics(base_values)
    n_r = robust_metrics(new_values)
    b_p = protocol_metrics(base_values)
    n_p = protocol_metrics(new_values)
    return {
        "base_sigma": b_p["sigma_median"],
        "new_sigma": n_p["sigma_median"],
        "sigma_gain": 1.0 - n_p["sigma_median"] / b_p["sigma_median"],
        "base_r68": b_r["r68"], "new_r68": n_r["r68"],
        "r68_gain": 1.0 - n_r["r68"] / b_r["r68"],
        "base_r90": b_r["r90"], "new_r90": n_r["r90"],
        "r90_gain": 1.0 - n_r["r90"] / b_r["r90"],
        "protocol_successes": n_p["protocol_successes"],
        "center_bias_max": n_p["center_bias_max"],
    }


def safety_flags(row, base_center):
    return {
        "sigma_improves": row["dev_sigma_gain"] > 0,
        "r68_not_worse": row["dev_r68_gain"] >= -0.001,
        "r90_not_worse": row["dev_r90_gain"] >= -0.002,
        "fit_protocol_ok": row["dev_protocol_successes"] >= 8,
        "center_stable": abs(row["dev_center_bias_max"] - base_center) <= 0.003,
        "background_coverage": row["background_applied"] >= 0.10,
        "background_ood": row["background_ood"] <= 0.20,
        "background_fallback": row["background_strong"] <= 0.85,
        "no_cap_pileup": row["background_near_cap"] <= 0.001,
        "ranking_preserved": row["background_rank_corr"] >= 0.9995,
        "no_global_spike_growth": row["background_spike_growth"] <= 1.10,
        "no_2600_spike_growth": row["background_2600_spike_growth"] <= 1.10,
        "scale_stable": abs(row["background_median_scale_shift"]) <= 0.001,
        "time_stable": row["time_ratio_span"] <= 0.002,
        "time_coverage_stable": row["time_applied_fraction_span"] <= 0.25,
        "time_blocks_supported": row["time_supported_blocks"] >= 8,
    }


def binned_relation(x, ratio, bins=18):
    mask = np.isfinite(x) & np.isfinite(ratio)
    x, ratio = np.asarray(x)[mask], np.asarray(ratio)[mask]
    if len(x) < 30:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.unique(np.quantile(x, np.linspace(0.01, 0.99, bins + 1)))
    centers, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        take = (x >= a) & (x <= b if b == edges[-1] else x < b)
        if np.sum(take) < 10:
            continue
        centers.append(np.median(x[take]))
        q = np.quantile(ratio[take], [0.16, 0.50, 0.84])
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    return map(np.asarray, (centers, med, lo, hi))


def plot_results(outdir, table, best_by_set, recommended, dev_winner,
                 cal, bg, cal_base, bg_base, cal_pred, bg_pred, feature_names):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # 1. Feature-set comparison, one dev-selected hyperparameter per set.
    comp = best_by_set.sort_values("dev_sigma_gain", ascending=False).head(16).copy()
    labels = [s.replace("_plus_", "+\n").replace("_", " ") for s in comp.feature_set]
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.bar(x - 0.2, 100 * comp.dev_sigma_gain, 0.4, label="Development time-OOF")
    ax.bar(x + 0.2, 100 * comp.hold_sigma_gain, 0.4, label="Untouched last 20%")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(r"Relative improvement in $\sigma/\mu$ [%]")
    ax.set_title("v19 incremental-variable search: development versus time holdout")
    ax.set_xticks(x, labels, rotation=48, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_feature_set_comparison.png", dpi=190)
    plt.close(fig)

    # 2. Single-variable ablation relative to four-variable baseline.
    single_names = ["B04_S10"] + [k for k in FEATURE_SETS if k.startswith("A")]
    abl = best_by_set[best_by_set.feature_set.isin(single_names)].copy()
    abl["order"] = abl.feature_set.map({n: i for i, n in enumerate(single_names)})
    abl = abl.sort_values("order")
    labels = ["S10\n4 variables"] + ["+" + n.split("_plus_", 1)[1] for n in abl.feature_set.iloc[1:]]
    x = np.arange(len(abl))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(x - 0.2, 100 * abl.dev_sigma_gain, 0.4, label="Development time-OOF")
    ax.bar(x + 0.2, 100 * abl.hold_sigma_gain, 0.4, label="Untouched last 20%")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(r"Relative improvement in $\sigma/\mu$ [%]")
    ax.set_title("Add-one-variable ablation from the four-variable S10 baseline")
    ax.set_xticks(x, labels, rotation=42, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_single_variable_ablation.png", dpi=190)
    plt.close(fig)

    # 3. Calibration and background spectra.
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    bins = np.arange(2100, 2900 + 5, 5)
    axes[0].hist(cal_base[np.isfinite(cal_base)], bins=bins, histtype="step", lw=1.5, label="Given cubic Energy_cor")
    axes[0].hist(cal_pred[np.isfinite(cal_pred)], bins=bins, histtype="step", lw=1.5, label="v19 recommended")
    axes[0].set(xlabel="Energy [keV]", ylabel="Events / 5 keV", title="Calibration sample: peak-region spectrum")
    axes[0].legend()
    bins = np.arange(500, 3800 + 5, 5)
    axes[1].hist(bg_base[np.isfinite(bg_base)], bins=bins, histtype="step", lw=1.2, label="Given cubic Energy_cor")
    axes[1].hist(bg_pred["energy"][np.isfinite(bg_pred["energy"])], bins=bins, histtype="step", lw=1.2, label="v19 recommended")
    axes[1].set(xlabel="Energy [keV]", ylabel="Events / 5 keV", title="Background sample: spectrum-shape safety check")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_energy_spectra.png", dpi=190)
    plt.close(fig)

    # 4. Ratio versus selected inputs for calibration and background.
    cal_ratio = cal_pred / cal_base
    bg_ratio = bg_pred["energy"] / bg_base
    show = feature_names[:8]
    fig, axes = plt.subplots(2, len(show), figsize=(3.4 * len(show), 7.0), squeeze=False)
    for j, name in enumerate(show):
        for i, (frame, ratio, title) in enumerate(((cal, cal_ratio, "Calibration"), (bg, bg_ratio, "Background"))):
            ax = axes[i, j]
            xx = frame[name].to_numpy(float)
            mask = np.isfinite(xx) & np.isfinite(ratio)
            if np.sum(mask) > 5000:
                idx = np.linspace(0, np.sum(mask) - 1, 5000).astype(int)
                ax.scatter(xx[mask][idx], ratio[mask][idx], s=2, alpha=0.12)
            else:
                ax.scatter(xx[mask], ratio[mask], s=3, alpha=0.18)
            c, m, lo, hi = binned_relation(xx, ratio)
            if len(c):
                ax.fill_between(c, lo, hi, alpha=0.18)
                ax.plot(c, m, lw=2)
            ax.axhline(1.0, color="black", ls="--", lw=0.8)
            ax.set_xlabel(name)
            if j == 0:
                ax.set_ylabel(r"$E_{new}/E_{cor}$")
            ax.set_title(title + ": " + name)
    fig.suptitle("v19 correction ratio versus model inputs", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "04_ratio_vs_model_inputs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. High-energy background zoom.
    fig, ax = plt.subplots(figsize=(13, 5.8))
    bins = np.arange(2450, 3000 + 2.5, 2.5)
    ax.hist(bg_base[np.isfinite(bg_base)], bins=bins, histtype="step", lw=1.25, label="Given cubic Energy_cor")
    ax.hist(bg_pred["energy"][np.isfinite(bg_pred["energy"])], bins=bins, histtype="step", lw=1.25, label="v19 recommended")
    ax.axvline(2600, color="gray", ls=":", lw=1)
    ax.set(xlabel="Energy [keV]", ylabel="Events / 2.5 keV", title="Background high-energy zoom: fake-peak check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "05_background_2450_3000_zoom.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    all_features = sorted(set(itertools.chain.from_iterable(FEATURE_SETS.values())))
    columns = ["runNumber", "eventNumber", "t", "qS1ub_C", "qS2Bdesub_C"] + all_features
    cal = load_table(args.calibration, columns)
    bg = load_table(args.background, columns)
    cal_base, cal_valid = energy_cor(cal)
    bg_base, bg_valid = energy_cor(bg)

    order = np.argsort(cal.t.to_numpy(float), kind="mergesort")
    split = int(0.80 * len(cal))
    dev_idx, hold_idx = order[:split], order[split:]
    dev_folds = chronological_folds(cal.t.to_numpy(float)[dev_idx], 4)
    dev_valid = cal_valid[dev_idx]
    hold_valid = cal_valid[hold_idx]
    dev_base = cal_base[dev_idx][dev_valid]
    hold_base = cal_base[hold_idx][hold_valid]
    base_center = protocol_metrics(dev_base)["center_bias_max"]
    bg_spike = local_spike_score(bg_base)
    bg_2600 = local_spike_score(bg_base, width=5.0, lo=2450.0, hi=2750.0)

    rows, saved = [], {}
    for set_name, names in FEATURE_SETS.items():
        cal_x = cal[names].to_numpy(float)
        bg_x = bg[names].to_numpy(float)
        for alpha, cap in itertools.product(ALPHAS, CAPS):
            oof_dev = np.full(len(dev_idx), np.nan)
            for fold in range(4):
                test_local = np.flatnonzero(dev_folds == fold)
                train_local = np.flatnonzero((dev_folds != fold) & dev_valid)
                state_fold = fit_state(cal_x[dev_idx[train_local]], cal_base[dev_idx[train_local]], alpha, cap)
                oof_dev[test_local] = apply_state(
                    state_fold, cal_x[dev_idx[test_local]], cal_base[dev_idx[test_local]], cal_valid[dev_idx[test_local]]
                )["energy"]
            state = fit_state(cal_x[dev_idx[dev_valid]], cal_base[dev_idx][dev_valid], alpha, cap)
            hold_pred = apply_state(state, cal_x[hold_idx], cal_base[hold_idx], cal_valid[hold_idx])
            bg_pred = apply_state(state, bg_x, bg_base, bg_valid)
            dev = gains(dev_base, oof_dev[dev_valid])
            hold = gains(hold_base, hold_pred["energy"][hold_valid])
            spike = local_spike_score(bg_pred["energy"])
            spike2600 = local_spike_score(bg_pred["energy"], width=5.0, lo=2450.0, hi=2750.0)
            ratio = bg_pred["energy"][bg_valid] / bg_base[bg_valid]
            rank = pd.Series(bg_base[bg_valid]).corr(pd.Series(bg_pred["energy"][bg_valid]), method="spearman")
            taudit = time_audit(bg.t.to_numpy(float), bg_base, bg_pred["energy"])
            model_id = "{}_a{}_c{}".format(set_name, int(alpha), int(round(cap * 10000)))
            row = {
                "model_id": model_id, "feature_set": set_name,
                "features": "|".join(names), "feature_count": len(names),
                "alpha": alpha, "cap": cap,
                **{"dev_" + k: v for k, v in dev.items()},
                **{"hold_" + k: v for k, v in hold.items()},
                "background_applied": float(np.mean(bg_pred["applied"])),
                "background_ood": float(np.mean(bg_pred["ood"])),
                "background_strong": float(np.mean(bg_pred["strong"])),
                "background_near_cap": float(np.mean(bg_pred["near_cap"])),
                "background_energy_trend": energy_trend(bg_base, bg_pred["delta"], bg_pred["applied"]),
                "background_rank_corr": float(rank),
                "background_spike_growth": float(spike["ratio"] / max(bg_spike["ratio"], 1.0)),
                "background_2600_spike_growth": float(spike2600["ratio"] / max(bg_2600["ratio"], 1.0)),
                "background_median_scale_shift": float(np.median(ratio) - 1.0),
                "time_ratio_span": taudit["median_ratio_span"],
                "time_applied_fraction_span": taudit["applied_fraction_span"],
                "time_supported_blocks": taudit["supported_blocks"],
            }
            flags = safety_flags(row, base_center)
            row["safety_pass_count"] = sum(bool(v) for v in flags.values())
            row["safe"] = all(flags.values())
            rows.append(row)
            saved[model_id] = {"state": state, "features": names, "oof_dev": oof_dev,
                               "hold_pred": hold_pred, "bg_pred": bg_pred, "flags": flags}
            print(json.dumps({"id": model_id, "safe": row["safe"],
                              "dev_gain": row["dev_sigma_gain"], "hold_gain": row["hold_sigma_gain"]}), flush=True)

    table = pd.DataFrame(rows)
    table = table.sort_values(["safe", "safety_pass_count", "dev_sigma_gain"], ascending=[False, False, False])
    table.to_csv(outdir / "v19_candidates.csv", index=False)
    best_by_set = table.sort_values(["safe", "safety_pass_count", "dev_sigma_gain"], ascending=[False, False, False]).groupby("feature_set", as_index=False).first()
    best_by_set.to_csv(outdir / "v19_best_by_feature_set.csv", index=False)

    base_rows = table[(table.feature_set == "B04_S10") & table.safe]
    baseline = base_rows.iloc[0].to_dict() if len(base_rows) else table[table.feature_set == "B04_S10"].iloc[0].to_dict()
    extra_safe = table[(table.feature_count > 4) & table.safe]
    dev_winner = extra_safe.iloc[0].to_dict() if len(extra_safe) else table[table.feature_count > 4].iloc[0].to_dict()
    holdout_pass = bool(
        dev_winner["hold_sigma_gain"] > 0
        and dev_winner["hold_r68_gain"] >= -0.002
        and dev_winner["hold_r90_gain"] >= -0.003
        and dev_winner["hold_new_sigma"] < baseline["hold_new_sigma"]
    )
    recommended = dev_winner if holdout_pass else baseline
    status = "extra_variables_accepted" if holdout_pass else "extra_variables_not_confirmed_keep_S10"

    rec_saved = saved[recommended["model_id"]]
    rec_names = rec_saved["features"]
    # Refit the recommended specification on all calibration events after validation.
    full_state = fit_state(cal[rec_names].to_numpy(float)[cal_valid], cal_base[cal_valid], recommended["alpha"], recommended["cap"])
    full_bg_pred = apply_state(full_state, bg[rec_names].to_numpy(float), bg_base, bg_valid)
    cal_combined = np.full(len(cal), np.nan)
    cal_combined[dev_idx] = rec_saved["oof_dev"]
    cal_combined[hold_idx] = rec_saved["hold_pred"]["energy"]

    bundle = {
        "model_name": "PandaX_incremental_features_v19",
        "version": "v19_incremental_features_20260815",
        "status": status,
        "feature_names": rec_names,
        "predictor_state": full_state,
        "recommended": recommended,
        "four_variable_baseline": baseline,
        "dev_selected_extra_candidate": dev_winner,
        "extra_candidate_holdout_pass": holdout_pass,
        "warning": "One calibration run only; cross-run and multi-line validation remain mandatory.",
    }
    joblib.dump(bundle, outdir / "candidate_v19.joblib")

    calout = cal[["runNumber", "eventNumber", "t"] + rec_names].copy()
    calout["energy_cor_kev"] = cal_base
    calout["v19_validation_energy_kev"] = cal_combined
    calout.to_csv(outdir / "calibration_events.csv", index=False)
    bgout = bg[["runNumber", "eventNumber", "t"] + rec_names].copy()
    bgout["energy_cor_kev"] = bg_base
    bgout["v19_energy_kev"] = full_bg_pred["energy"]
    bgout.to_csv(outdir / "background_events.txt", sep=" ", index=False)

    summary = {
        "status": status,
        "four_variable_baseline": baseline,
        "dev_selected_extra_candidate": dev_winner,
        "extra_candidate_holdout_pass": holdout_pass,
        "recommended": recommended,
        "feature_sets": FEATURE_SETS,
        "alphas": ALPHAS, "caps": CAPS,
        "total_candidates": len(table), "safe_candidates": int(table.safe.sum()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if HAVE_MATPLOTLIB:
        plot_results(outdir, table, best_by_set, recommended, dev_winner,
                     cal, bg, cal_base, bg_base, cal_combined, full_bg_pred, rec_names)
    else:
        print("PLOTS_SKIPPED matplotlib_not_available")
    print("OUTPUT", outdir)
    print("STATUS", status)
    print("RECOMMENDED", recommended["model_id"], "FEATURES", rec_names)
    print("EXTRA_CANDIDATE", dev_winner["model_id"], "HOLDOUT_PASS", holdout_pass)


if __name__ == "__main__":
    main()
