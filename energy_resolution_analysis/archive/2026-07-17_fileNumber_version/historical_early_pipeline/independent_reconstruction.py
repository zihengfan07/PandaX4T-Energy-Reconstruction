from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_analysis as core


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "independent_outputs"
FIG = OUT / "figures"
DATA = ROOT.parent / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
S1 = "qS1_max"
S2 = "qS2B_max"
X = "xS2T_max"
Y = "yS2T_max"
SEQUENCES = [[], ["dt"], ["r2"], ["dt", "r2"]]
ALPHAS = np.linspace(0.05, 0.95, 31)
TOLERANCE = 0.0005


def features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    x = frame[X].to_numpy(float)
    y = frame[Y].to_numpy(float)
    return {
        "dt": frame.dt.to_numpy(float) / 1000.0,
        "r2": x * x + y * y,
        "phi": np.arctan2(y, x),
    }


def learn_channel(frame: pd.DataFrame, values: np.ndarray, sequence: list[str]):
    current = np.asarray(values, float).copy()
    feats = features(frame)
    maps = []
    for name in sequence:
        response = core.learn_response_map(current, feats[name], bins=6, periodic=(name == "phi"))
        current = core.apply_response_map(current, feats[name], response)
        maps.append((name, response))
    return current, maps


def apply_channel(frame: pd.DataFrame, values: np.ndarray, maps):
    current = np.asarray(values, float).copy()
    feats = features(frame)
    for name, response in maps:
        current = core.apply_response_map(current, feats[name], response)
    return current


def prepare_channels(train: pd.DataFrame, target: pd.DataFrame, s1_sequence: list[str], s2_sequence: list[str]):
    s1_scale = core.estimate_mode(train[S1].to_numpy(float))
    s2_scale = core.estimate_mode(train[S2].to_numpy(float))
    train_s1 = train[S1].to_numpy(float) / s1_scale
    train_s2 = train[S2].to_numpy(float) / s2_scale
    target_s1 = target[S1].to_numpy(float) / s1_scale
    target_s2 = target[S2].to_numpy(float) / s2_scale
    corrected_train_s1, s1_maps = learn_channel(train, train_s1, s1_sequence)
    corrected_train_s2, s2_maps = learn_channel(train, train_s2, s2_sequence)
    corrected_target_s1 = apply_channel(target, target_s1, s1_maps)
    corrected_target_s2 = apply_channel(target, target_s2, s2_maps)
    return corrected_train_s1, corrected_train_s2, corrected_target_s1, corrected_target_s2, s1_maps, s2_maps


def scan_alpha(train_s1: np.ndarray, train_s2: np.ndarray, val_s1: np.ndarray, val_s2: np.ndarray):
    rows = []
    for alpha in ALPHAS:
        train_energy = alpha * train_s1 + (1 - alpha) * train_s2
        scale = core.estimate_mode(train_energy)
        val_energy = (alpha * val_s1 + (1 - alpha) * val_s2) / scale
        fit = core.fit_peak(val_energy)
        rows.append({"alpha": float(alpha), "resolution": fit.resolution, "mu": fit.mu, "signal_fraction": fit.signal_fraction, "success": fit.success})
    table = pd.DataFrame(rows)
    valid = table[table.success.astype(bool) & np.isfinite(table.resolution)]
    best = valid.loc[valid.resolution.idxmin()]
    return float(best.alpha), table


def select_configuration(table: pd.DataFrame):
    best = float(table.validation_resolution.min())
    eligible = table[table.validation_resolution <= best + TOLERANCE].copy()
    eligible["complexity"] = eligible.s1_sequence.map(lambda x: 0 if x == "none" else len(x.split("+"))) + eligible.s2_sequence.map(lambda x: 0 if x == "none" else len(x.split("+")))
    return eligible.sort_values(["complexity", "validation_resolution", "s1_sequence", "s2_sequence"]).iloc[0]


def tune(inner_train: pd.DataFrame, inner_validation: pd.DataFrame):
    rows = []
    alpha_tables = []
    for s1_sequence in SEQUENCES:
        for s2_sequence in SEQUENCES:
            tr1, tr2, va1, va2, _, _ = prepare_channels(inner_train, inner_validation, s1_sequence, s2_sequence)
            alpha, scan = scan_alpha(tr1, tr2, va1, va2)
            best = scan.loc[scan.resolution.idxmin()]
            s1_label = "+".join(s1_sequence) if s1_sequence else "none"
            s2_label = "+".join(s2_sequence) if s2_sequence else "none"
            scan.insert(0, "s1_sequence", s1_label)
            scan.insert(1, "s2_sequence", s2_label)
            alpha_tables.append(scan)
            rows.append({"s1_sequence": s1_label, "s2_sequence": s2_label, "alpha": alpha, "validation_resolution": float(best.resolution), "validation_mu": float(best.mu), "validation_signal_fraction": float(best.signal_fraction)})
    table = pd.DataFrame(rows)
    selected = select_configuration(table)
    return selected, table, pd.concat(alpha_tables, ignore_index=True)


def sequence_from_label(label: str) -> list[str]:
    return [] if label == "none" else label.split("+")


def nested_crossfit(frame: pd.DataFrame):
    # 固定随机种子的平衡事件级五折；身份编号不参与划分。
    rng = np.random.default_rng(10972)
    permutation = rng.permutation(len(frame))
    fold = np.empty(len(frame), dtype=int)
    fold[permutation] = np.arange(len(frame), dtype=int) % 5
    oof_raw = np.full(len(frame), np.nan)
    oof_independent = np.full(len(frame), np.nan)
    fold_rows = []
    tuning_tables = []
    for outer in range(5):
        outer_mask = fold == outer
        inner_val_fold = (outer + 1) % 5
        inner_val_mask = fold == inner_val_fold
        inner_train_mask = (~outer_mask) & (~inner_val_mask)
        dev_mask = ~outer_mask
        inner_train = frame.iloc[np.flatnonzero(inner_train_mask)].copy()
        inner_val = frame.iloc[np.flatnonzero(inner_val_mask)].copy()
        outer_dev = frame.iloc[np.flatnonzero(dev_mask)].copy()
        outer_test = frame.iloc[np.flatnonzero(outer_mask)].copy()

        selected, tuning, _ = tune(inner_train, inner_val)
        tuning.insert(0, "outer_fold", outer)
        tuning_tables.append(tuning)
        s1_sequence = sequence_from_label(str(selected.s1_sequence))
        s2_sequence = sequence_from_label(str(selected.s2_sequence))
        alpha = float(selected.alpha)

        # Independently optimize the raw-combination control on the same inner split.
        raw_tr1, raw_tr2, raw_va1, raw_va2, _, _ = prepare_channels(inner_train, inner_val, [], [])
        raw_alpha, _ = scan_alpha(raw_tr1, raw_tr2, raw_va1, raw_va2)

        dev1, dev2, test1, test2, _, _ = prepare_channels(outer_dev, outer_test, s1_sequence, s2_sequence)
        dev_energy = alpha * dev1 + (1 - alpha) * dev2
        scale = core.estimate_mode(dev_energy)
        test_energy = (alpha * test1 + (1 - alpha) * test2) / scale

        raw_dev1, raw_dev2, raw_test1, raw_test2, _, _ = prepare_channels(outer_dev, outer_test, [], [])
        raw_dev_energy = raw_alpha * raw_dev1 + (1 - raw_alpha) * raw_dev2
        raw_scale = core.estimate_mode(raw_dev_energy)
        raw_test_energy = (raw_alpha * raw_test1 + (1 - raw_alpha) * raw_test2) / raw_scale

        positions = np.flatnonzero(outer_mask)
        oof_raw[positions] = raw_test_energy
        oof_independent[positions] = test_energy
        raw_fit = core.fit_peak(raw_test_energy)
        fit = core.fit_peak(test_energy)
        fold_rows.append({
            "outer_fold": outer,
            "events": len(outer_test),
            "raw_alpha": raw_alpha,
            "independent_alpha": alpha,
            "s1_sequence": str(selected.s1_sequence),
            "s2_sequence": str(selected.s2_sequence),
            "raw_resolution": raw_fit.resolution,
            "independent_resolution": fit.resolution,
            "delta": fit.resolution - raw_fit.resolution,
            "raw_mu": raw_fit.mu,
            "independent_mu": fit.mu,
        })
    return oof_raw, oof_independent, pd.DataFrame(fold_rows), pd.concat(tuning_tables, ignore_index=True)


def paired_event_bootstrap(
    raw_values: np.ndarray,
    corrected_values: np.ndarray,
    n_boot: int = 300,
) -> pd.DataFrame:
    """对冻结 OOF 事件做配对重采样；每次 raw/corrected 使用同一索引。"""
    rng = np.random.default_rng(10972)
    rows = []
    n_events = len(raw_values)
    for replicate in range(n_boot):
        sampled = rng.integers(0, n_events, size=n_events)
        raw_fit = core.fit_peak(raw_values[sampled])
        corrected_fit = core.fit_peak(corrected_values[sampled])
        if not (np.isfinite(raw_fit.resolution) and np.isfinite(corrected_fit.resolution)):
            continue
        delta = corrected_fit.resolution - raw_fit.resolution
        rows.append(
            {
                "replicate": replicate,
                "baseline_resolution": raw_fit.resolution,
                "improved_resolution": corrected_fit.resolution,
                "delta": delta,
                "ratio": corrected_fit.resolution / raw_fit.resolution,
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    required = ["runNumber", "eventNumber", S1, S2, "dt", X, Y]
    raw = pd.read_csv(DATA, sep="\t", usecols=required).replace([np.inf, -np.inf], np.nan)
    raw.insert(0, "sourceRow", np.arange(len(raw), dtype=int))
    frame = raw.loc[raw[required].notna().all(axis=1), required].copy().reset_index(drop=True)
    frame.insert(0, "sourceRow", raw.loc[raw[required].notna().all(axis=1), "sourceRow"].to_numpy(int))

    oof_raw, oof_independent, folds, tuning = nested_crossfit(frame)
    raw_fit = core.fit_peak(oof_raw)
    independent_fit = core.fit_peak(oof_independent)
    boot = paired_event_bootstrap(oof_raw, oof_independent, n_boot=300)
    quantiles = []
    for metric in ["baseline_resolution", "improved_resolution", "delta", "ratio"]:
        q = boot[metric].quantile([0.025, 0.5, 0.975])
        quantiles.append({"metric": metric, "q025": float(q.iloc[0]), "median": float(q.iloc[1]), "q975": float(q.iloc[2]), "successful_replicates": len(boot)})

    rng = np.random.default_rng(10972)
    permutation = rng.permutation(len(frame))
    outer_fold = np.empty(len(frame), dtype=int)
    outer_fold[permutation] = np.arange(len(frame), dtype=int) % 5
    events = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
    events["outer_fold"] = outer_fold
    events["oof_raw_energy"] = oof_raw
    events["oof_independent_energy"] = oof_independent
    events.to_csv(OUT / "oof_events.tsv", sep="\t", index=False)
    folds.to_csv(OUT / "fold_results.csv", index=False)
    tuning.to_csv(OUT / "inner_tuning.csv", index=False)
    boot.to_csv(OUT / "paired_event_bootstrap.csv", index=False)
    pd.DataFrame(quantiles).to_csv(OUT / "bootstrap_summary.csv", index=False)

    summary = {
        "protocol": "basic-only independent reconstruction",
        "allowed_variables": [S1, S2, "dt", X, Y],
        "identity_metadata": ["sourceRow", "runNumber", "eventNumber"],
        "outer_split": "seeded balanced event-level five-fold OOF (seed=10972)",
        "events": len(frame),
        "raw_oof_fit": asdict(raw_fit),
        "independent_oof_fit": asdict(independent_fit),
        "raw_robust68": core.robust_central68(oof_raw),
        "independent_robust68": core.robust_central68(oof_independent),
        "folds": folds.to_dict("records"),
        "bootstrap": quantiles,
        "warning": "Intervals condition on the frozen OOF predictions; they do not rerun the full nested tuning inside each bootstrap replicate.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    core.plot_peak(ax, oof_raw, raw_fit, f"raw basic variables: {100*raw_fit.resolution:.2f}%", "#E45756")
    core.plot_peak(ax, oof_independent, independent_fit, f"independent channel corrections: {100*independent_fit.resolution:.2f}%", "#4C78A8")
    ax.set_title("Independent five-fold out-of-fold reconstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_independent_oof.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(folds))
    ax.plot(x, 100 * folds.raw_resolution, "o-", label="raw optimized combination")
    ax.plot(x, 100 * folds.independent_resolution, "o-", label="independent channel corrections")
    ax.set_xticks(x, [str(i) for i in folds.outer_fold])
    ax.set_xlabel("Outer event fold")
    ax.set_ylabel("Core resolution sigma/mu [%]")
    ax.set_title("Independent reconstruction by outer fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_fold_stability.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
