from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import ndtr


SEED = 10972
FIT_WINDOW = (0.78, 1.22)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
OUT = ROOT / "outputs"
FIG = OUT / "figures"


@dataclass
class PeakFit:
    success: bool
    n: int
    mu: float
    sigma: float
    resolution: float
    signal_fraction: float
    background_slope: float
    nll: float
    message: str = ""


def finite_array(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def robust_central68(values: np.ndarray, window=FIT_WINDOW) -> float:
    x = finite_array(values)
    x = x[(x >= window[0]) & (x <= window[1])]
    if x.size < 20:
        return float("nan")
    qlo, qmed, qhi = np.quantile(x, [0.15865, 0.5, 0.84135])
    return float((qhi - qlo) / (2.0 * qmed))


def estimate_mode(values: np.ndarray, bins: int = 140) -> float:
    x = finite_array(values)
    if x.size < 20:
        return float("nan")
    lo, hi = np.quantile(x, [0.01, 0.99])
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    smooth = gaussian_filter1d(hist.astype(float), 1.4)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[int(np.argmax(smooth))])


def _truncated_exp_pdf(x: np.ndarray, slope: float, lo: float, hi: float) -> np.ndarray:
    width = hi - lo
    # Keep the exactly-zero limit explicit, but do not create a flat numerical
    # plateau around zero: finite-difference optimizers must see the slope.
    if abs(slope) < 1e-12:
        return np.full_like(x, 1.0 / width)
    denom = np.expm1(slope * width)
    return slope * np.exp(slope * (x - lo)) / denom


def fit_peak(values: np.ndarray, window=FIT_WINDOW) -> PeakFit:
    x = finite_array(values)
    lo, hi = window
    x = x[(x >= lo) & (x <= hi)]
    if x.size < 80:
        return PeakFit(False, int(x.size), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "too few events")

    mode = estimate_mode(x, bins=80)
    if not np.isfinite(mode):
        mode = float(np.median(x))
    near = x[np.abs(x - mode) < 0.08]
    if near.size < 30:
        near = x
    mad = 1.4826 * np.median(np.abs(near - np.median(near)))
    sigma0 = float(np.clip(mad, 0.01, 0.08))

    def nll(par: np.ndarray) -> float:
        mu, log_sigma, logit_f, slope = par
        sigma = math.exp(log_sigma)
        frac = 1.0 / (1.0 + math.exp(-logit_f))
        z = (x - mu) / sigma
        norm = max(float(ndtr((hi - mu) / sigma) - ndtr((lo - mu) / sigma)), 1e-12)
        gauss = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma * norm)
        expo = _truncated_exp_pdf(x, slope, lo, hi)
        pdf = frac * gauss + (1.0 - frac) * expo
        if np.any(~np.isfinite(pdf)) or np.any(pdf <= 0):
            return 1e100
        return float(-np.log(pdf).sum())

    start = np.array([np.clip(mode, 0.88, 1.12), math.log(sigma0), math.log(0.7 / 0.3), -2.0])
    bounds = [(0.86, 1.14), (math.log(0.003), math.log(0.14)), (-4.0, 7.0), (-12.0, 12.0)]
    result = minimize(nll, start, method="L-BFGS-B", bounds=bounds)
    mu, log_sigma, logit_f, slope = result.x
    sigma = math.exp(log_sigma)
    frac = 1.0 / (1.0 + math.exp(-logit_f))
    ok = bool(result.success and 0.86 < mu < 1.14 and 0.003 < sigma < 0.14 and frac > 0.08)
    return PeakFit(ok, int(x.size), float(mu), float(sigma), float(sigma / mu), float(frac), float(slope), float(result.fun), str(result.message))


def normalized_projection(frame: pd.DataFrame, s1: str, s2: str, alpha: float, medians: tuple[float, float]) -> np.ndarray:
    m1, m2 = medians
    return alpha * frame[s1].to_numpy(float) / m1 + (1.0 - alpha) * frame[s2].to_numpy(float) / m2


def split_label(file_number: pd.Series) -> pd.Series:
    mod = file_number.astype(int) % 5
    return pd.Series(np.where(mod <= 2, "train", np.where(mod == 3, "validation", "test")), index=file_number.index)


def scan_pair(train: pd.DataFrame, validation: pd.DataFrame, s1: str, s2: str, alphas: np.ndarray):
    medians = (float(train[s1].median()), float(train[s2].median()))
    rows = []
    for alpha in alphas:
        train_e = normalized_projection(train, s1, s2, float(alpha), medians)
        scale = estimate_mode(train_e)
        val_e = normalized_projection(validation, s1, s2, float(alpha), medians) / scale
        fit = fit_peak(val_e)
        rows.append({"alpha": float(alpha), "scale_train": scale, "validation_resolution": fit.resolution, "validation_mu": fit.mu, "validation_sigma": fit.sigma, "validation_signal_fraction": fit.signal_fraction, "fit_success": fit.success})
    scan = pd.DataFrame(rows)
    valid = scan[scan.fit_success & np.isfinite(scan.validation_resolution)]
    if valid.empty:
        raise RuntimeError(f"No successful alpha fit for {s1} + {s2}")
    best = valid.loc[valid.validation_resolution.idxmin()]
    return float(best.alpha), medians, scan


def evaluate_method(dev: pd.DataFrame, test: pd.DataFrame, s1: str, s2: str, alpha: float):
    medians = (float(dev[s1].median()), float(dev[s2].median()))
    dev_e = normalized_projection(dev, s1, s2, alpha, medians)
    scale = estimate_mode(dev_e)
    test_e = normalized_projection(test, s1, s2, alpha, medians) / scale
    return test_e, fit_peak(test_e), medians, scale


def learn_response_map(energy: np.ndarray, feature: np.ndarray, bins: int, periodic: bool = False):
    mask = np.isfinite(energy) & np.isfinite(feature)
    e, f = energy[mask], feature[mask]
    edges = np.unique(np.quantile(f, np.linspace(0, 1, bins + 1)))
    global_fit = fit_peak(e)
    rows = []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        sel = (f >= edges[i]) & ((f <= edges[i + 1]) if last else (f < edges[i + 1]))
        local = fit_peak(e[sel])
        rows.append({"bin": i, "low": float(edges[i]), "high": float(edges[i + 1]), "center": float(np.median(f[sel])) if sel.any() else np.nan, "n": int(sel.sum()), "mu": local.mu, "sigma": local.sigma, "resolution": local.resolution, "fit_success": local.success})
    table = pd.DataFrame(rows)
    good = table.fit_success & np.isfinite(table.mu)
    centers = table.loc[good, "center"].to_numpy(float)
    log_response = np.log(table.loc[good, "mu"].to_numpy(float) / global_fit.mu)
    if centers.size < 3:
        raise RuntimeError("Too few successful local peak fits")
    order = np.argsort(centers)
    centers, log_response = centers[order], log_response[order]
    if periodic:
        period = 2 * math.pi
        centers = np.concatenate([centers - period, centers, centers + period])
        log_response = np.tile(log_response, 3)
    return {"centers": centers, "log_response": log_response, "periodic": periodic, "table": table, "global_mu": global_fit.mu}


def apply_response_map(energy: np.ndarray, feature: np.ndarray, response) -> np.ndarray:
    f = np.asarray(feature, float)
    if response["periodic"]:
        f = (f + math.pi) % (2 * math.pi) - math.pi
    log_r = np.interp(f, response["centers"], response["log_response"])
    return np.asarray(energy, float) * np.exp(-log_r)


def feature_arrays(frame: pd.DataFrame):
    x = frame["xS2Tcor_max"].to_numpy(float)
    y = frame["yS2Tcor_max"].to_numpy(float)
    return {
        "dt": frame["dt"].to_numpy(float) / 1000.0,
        "r2": x * x + y * y,
        "phi": np.arctan2(y, x),
        "fileNumber": frame["fileNumber"].to_numpy(float),
    }


def learn_sequence(dev: pd.DataFrame, energy: np.ndarray, sequence: list[str], bins=6):
    features = feature_arrays(dev)
    current = np.asarray(energy, float).copy()
    maps = []
    for name in sequence:
        response = learn_response_map(current, features[name], bins=bins, periodic=(name == "phi"))
        current = apply_response_map(current, features[name], response)
        maps.append((name, response))
    return current, maps


def apply_sequence(frame: pd.DataFrame, energy: np.ndarray, maps):
    features = feature_arrays(frame)
    current = np.asarray(energy, float).copy()
    for name, response in maps:
        current = apply_response_map(current, features[name], response)
    return current


def bootstrap_by_file(frame: pd.DataFrame, baseline: np.ndarray, improved: np.ndarray, n_boot=250):
    rng = np.random.default_rng(SEED)
    groups = frame["fileNumber"].to_numpy(int)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices[g] for g in sampled])
        f0, f1 = fit_peak(baseline[idx]), fit_peak(improved[idx])
        if f0.success and f1.success:
            rows.append({"bootstrap": b, "baseline_resolution": f0.resolution, "improved_resolution": f1.resolution, "delta": f1.resolution - f0.resolution, "ratio": f1.resolution / f0.resolution})
    return pd.DataFrame(rows)


def choose_parsimonious_sequence(table: pd.DataFrame, tolerance: float = 0.0005) -> str:
    """Choose the simplest sequence within 0.05 percentage point of best R."""
    valid = table[table.validation_success.astype(bool) & np.isfinite(table.validation_resolution)].copy()
    best = float(valid.validation_resolution.min())
    eligible = valid[valid.validation_resolution <= best + tolerance].copy()
    eligible["complexity"] = eligible.sequence.map(lambda s: 0 if s == "none" else len(str(s).split("+")))
    eligible = eligible.sort_values(["complexity", "validation_resolution", "sequence"])
    return str(eligible.iloc[0].sequence)


def nested_crossfit(frame: pd.DataFrame, s1: str, s2: str, method: str, sequences: list[list[str]]):
    fold = (frame.fileNumber.astype(int) % 5).to_numpy()
    oof_base = np.full(len(frame), np.nan)
    oof_corrected = np.full(len(frame), np.nan)
    fold_rows = []
    alpha_grid = np.linspace(0.03, 0.97, 48)
    for outer in range(5):
        outer_test = frame.iloc[np.flatnonzero(fold == outer)].copy()
        inner_validation_fold = (outer + 1) % 5
        inner_validation = frame.iloc[np.flatnonzero(fold == inner_validation_fold)].copy()
        inner_train = frame.iloc[np.flatnonzero((fold != outer) & (fold != inner_validation_fold))].copy()
        outer_dev = frame.iloc[np.flatnonzero(fold != outer)].copy()

        alpha, _, _ = scan_pair(inner_train, inner_validation, s1, s2, alpha_grid)
        med = (float(inner_train[s1].median()), float(inner_train[s2].median()))
        inner_train_raw = normalized_projection(inner_train, s1, s2, alpha, med)
        inner_scale = estimate_mode(inner_train_raw)
        inner_train_energy = inner_train_raw / inner_scale
        inner_val_energy = normalized_projection(inner_validation, s1, s2, alpha, med) / inner_scale
        candidate_rows = []
        for sequence in sequences:
            label = "+".join(sequence) if sequence else "none"
            if sequence:
                _, maps = learn_sequence(inner_train, inner_train_energy, sequence)
                candidate = apply_sequence(inner_validation, inner_val_energy, maps)
            else:
                candidate = inner_val_energy
            local_fit = fit_peak(candidate)
            candidate_rows.append({"sequence": label, **{f"validation_{k}": v for k, v in asdict(local_fit).items()}})
        selected_label = choose_parsimonious_sequence(pd.DataFrame(candidate_rows))
        selected_sequence = [] if selected_label == "none" else selected_label.split("+")

        dev_med = (float(outer_dev[s1].median()), float(outer_dev[s2].median()))
        dev_raw = normalized_projection(outer_dev, s1, s2, alpha, dev_med)
        dev_scale = estimate_mode(dev_raw)
        dev_energy = dev_raw / dev_scale
        test_energy = normalized_projection(outer_test, s1, s2, alpha, dev_med) / dev_scale
        if selected_sequence:
            _, maps = learn_sequence(outer_dev, dev_energy, selected_sequence)
            corrected = apply_sequence(outer_test, test_energy, maps)
        else:
            corrected = test_energy.copy()
        positions = np.flatnonzero(fold == outer)
        oof_base[positions] = test_energy
        oof_corrected[positions] = corrected
        base_fit, corrected_fit = fit_peak(test_energy), fit_peak(corrected)
        fold_rows.append({"method": method, "outer_fold": outer, "events": len(outer_test), "alpha": alpha, "selected_sequence": selected_label, "baseline_resolution": base_fit.resolution, "corrected_resolution": corrected_fit.resolution, "baseline_mu": base_fit.mu, "corrected_mu": corrected_fit.mu})
    return oof_base, oof_corrected, pd.DataFrame(fold_rows)


def plot_peak(ax, values: np.ndarray, fit: PeakFit, label: str, color: str):
    x = finite_array(values)
    x = x[(x >= FIT_WINDOW[0]) & (x <= FIT_WINDOW[1])]
    counts, edges, _ = ax.hist(x, bins=65, range=FIT_WINDOW, histtype="step", lw=1.8, color=color, label=label)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]
    if fit.success:
        norm = ndtr((FIT_WINDOW[1] - fit.mu) / fit.sigma) - ndtr((FIT_WINDOW[0] - fit.mu) / fit.sigma)
        gauss = np.exp(-0.5 * ((centers - fit.mu) / fit.sigma) ** 2) / (math.sqrt(2 * math.pi) * fit.sigma * norm)
        expo = _truncated_exp_pdf(centers, fit.background_slope, *FIT_WINDOW)
        model = x.size * bw * (fit.signal_fraction * gauss + (1 - fit.signal_fraction) * expo)
        ax.plot(centers, model, color=color, ls="--", alpha=0.9)
    ax.set_xlabel("Relative energy")
    ax.set_ylabel("Events / bin")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH, sep="\t")
    numeric = raw.select_dtypes(include=[np.number])
    inf_counts = pd.Series(np.isinf(numeric).sum(), index=numeric.columns)
    df = raw.replace([np.inf, -np.inf], np.nan)

    audit = pd.DataFrame({
        "variable": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_fraction": [float(df[c].isna().mean()) for c in df.columns],
        "inf_count": [int(inf_counts.get(c, 0)) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "zero_fraction": [float((df[c] == 0).mean()) if pd.api.types.is_numeric_dtype(df[c]) else np.nan for c in df.columns],
    })
    audit["constant_or_empty"] = audit.n_unique <= 1
    audit.to_csv(OUT / "variable_audit.csv", index=False)

    cut_columns = [
        "basicCut", "deadtime_enhanceCut", "fv_extendCut", "S1PerPmtCut",
        "S1PatternCut", "S1AsyCut", "S2ShapeCut", "gas_s2Cut", "S2AsyCut",
        "S2TBACut", "drCut", "wallCut", "diffusion_enhanceCut", "ssCut",
    ]
    cut_rows = []
    for column in cut_columns:
        if column in df:
            counts = df[column].value_counts(dropna=False)
            cut_rows.append({
                "cut_variable": column,
                "count_0": int(counts.get(0, 0)),
                "count_1": int(counts.get(1, 0)),
                "count_other_or_missing": int(len(df) - counts.get(0, 0) - counts.get(1, 0)),
                "semantics_confirmed": False,
            })
    pd.DataFrame(cut_rows).to_csv(OUT / "cut_flag_audit.csv", index=False)

    df["split"] = split_label(df.fileNumber)
    split_summary = df.groupby("split").agg(events=("eventNumber", "size"), files=("fileNumber", "nunique")).reset_index()
    split_summary.to_csv(OUT / "split_summary.csv", index=False)

    methods = [
        ("s1_only_corrected", "qS1C_max", "qS1C_max", "S1-only control"),
        ("s2b_only_lifetime", "qS2Belife_max", "qS2Belife_max", "lifetime-corrected bottom-S2-only control"),
        ("raw_bottom", "qS1_max", "qS2B_max", "documented raw-like branches"),
        ("corrected_bottom", "qS1C_max", "qS2BC_max", "corrected bottom S2 before the named lifetime branch"),
        ("corrected_lifetime", "qS1C_max", "qS2Belife_max", "documented corrected branches; primary baseline"),
        ("corrected_total_s2", "qS1C_max", "qS2C_max", "uses total corrected S2"),
        ("ub_bottom_stretch", "qS1ub_C", "qS2Bub_C_stretch", "undocumented derived branches"),
        ("ub_desat_bottom_stretch", "qS1ub_C", "qS2Bdesub_C_stretch", "undocumented derived branches"),
        ("mcpaf_desat", "qS1C_desImageMCPAF_maxS2", "qS2BdesC_desImageMCPAF_maxS2", "undocumented MCPAF branches"),
    ]
    required = sorted({c for _, a, b, _ in methods for c in (a, b)} | {"dt", "xS2Tcor_max", "yS2Tcor_max", "fileNumber"})
    common = df[required].notna().all(axis=1)
    analysis = df.loc[common].copy()
    train = analysis[analysis.split == "train"].copy()
    validation = analysis[analysis.split == "validation"].copy()
    test = analysis[analysis.split == "test"].copy()
    dev = analysis[analysis.split != "test"].copy()

    alphas = np.linspace(0.03, 0.97, 48)
    method_rows, scan_frames, test_energy = [], [], {}
    for name, s1, s2, note in methods:
        alpha, _, scan = scan_pair(train, validation, s1, s2, alphas)
        scan.insert(0, "method", name)
        scan_frames.append(scan)
        energy, fit, medians, scale = evaluate_method(dev, test, s1, s2, alpha)
        test_energy[name] = energy
        beta = np.nan if s1 == s2 else (alpha / (1 - alpha)) * (medians[1] / medians[0])
        valid_scan = scan[scan.fit_success.astype(bool) & np.isfinite(scan.validation_resolution)]
        if valid_scan.empty:
            raise RuntimeError(f"No successful validation fit for method {name}")
        best_validation = valid_scan.loc[valid_scan.validation_resolution.idxmin()]
        method_rows.append({"method": name, "s1": s1, "s2": s2, "note": note, "alpha_normalized_s1": alpha, "equivalent_s1_coefficient_if_s2_is_1": beta, "best_validation_resolution": float(best_validation.validation_resolution), "test_events": len(test), **{f"test_{k}": v for k, v in asdict(fit).items()}})
    alpha_scans = pd.concat(scan_frames, ignore_index=True)
    alpha_scans.to_csv(OUT / "alpha_scans.csv", index=False)
    method_results = pd.DataFrame(method_rows).sort_values("test_resolution")
    method_results.to_csv(OUT / "method_results.csv", index=False)

    primary = method_results.loc[method_results.method == "corrected_lifetime"].iloc[0]
    primary_alpha = float(primary.alpha_normalized_s1)
    s1, s2 = "qS1C_max", "qS2Belife_max"
    dev_medians = (float(dev[s1].median()), float(dev[s2].median()))
    dev_raw = normalized_projection(dev, s1, s2, primary_alpha, dev_medians)
    dev_scale = estimate_mode(dev_raw)
    dev_energy = dev_raw / dev_scale
    test_base = normalized_projection(test, s1, s2, primary_alpha, dev_medians) / dev_scale
    baseline_fit = fit_peak(test_base)

    sequences = [[], ["dt"], ["r2"], ["dt", "r2"], ["dt", "r2", "phi"], ["dt", "r2", "phi", "fileNumber"]]
    train_medians = (float(train[s1].median()), float(train[s2].median()))
    train_raw = normalized_projection(train, s1, s2, primary_alpha, train_medians)
    train_scale = estimate_mode(train_raw)
    train_energy = train_raw / train_scale
    val_energy = normalized_projection(validation, s1, s2, primary_alpha, train_medians) / train_scale
    nuisance_rows = []
    for sequence in sequences:
        label = "+".join(sequence) if sequence else "none"
        if sequence:
            _, maps = learn_sequence(train, train_energy, sequence)
            corrected = apply_sequence(validation, val_energy, maps)
        else:
            corrected = val_energy
        fit = fit_peak(corrected)
        nuisance_rows.append({"sequence": label, **{f"validation_{k}": v for k, v in asdict(fit).items()}})
    nuisance = pd.DataFrame(nuisance_rows).sort_values("validation_resolution")
    nuisance.to_csv(OUT / "nuisance_validation.csv", index=False)
    selected_label = choose_parsimonious_sequence(nuisance)
    selected_sequence = [] if selected_label == "none" else selected_label.split("+")
    if selected_sequence:
        dev_corrected, final_maps = learn_sequence(dev, dev_energy, selected_sequence)
        test_corrected = apply_sequence(test, test_base, final_maps)
    else:
        dev_corrected, final_maps, test_corrected = dev_energy, [], test_base.copy()
    corrected_fit = fit_peak(test_corrected)

    systematic_rows = []
    for lo, hi in [(0.76, 1.24), (0.78, 1.22), (0.80, 1.20), (0.82, 1.18)]:
        for label, values in [("baseline", test_base), ("selected_correction", test_corrected)]:
            local_fit = fit_peak(values, window=(lo, hi))
            systematic_rows.append({"method": label, "window_low": lo, "window_high": hi, **asdict(local_fit)})
    pd.DataFrame(systematic_rows).to_csv(OUT / "fit_window_systematics.csv", index=False)

    # A deliberately separate exploratory pipeline uses the simpler of the two
    # nearly identical ub/stretch candidates. Its branch semantics are not in the
    # supplied TechNote, so it is never promoted to the documented baseline.
    exploratory_method = "ub_bottom_stretch"
    exploratory_row = method_results.loc[method_results.method == exploratory_method].iloc[0]
    ex_s1, ex_s2 = str(exploratory_row.s1), str(exploratory_row.s2)
    ex_alpha = float(exploratory_row.alpha_normalized_s1)
    ex_train_medians = (float(train[ex_s1].median()), float(train[ex_s2].median()))
    ex_train_raw = normalized_projection(train, ex_s1, ex_s2, ex_alpha, ex_train_medians)
    ex_train_scale = estimate_mode(ex_train_raw)
    ex_train_energy = ex_train_raw / ex_train_scale
    ex_val_energy = normalized_projection(validation, ex_s1, ex_s2, ex_alpha, ex_train_medians) / ex_train_scale
    ex_nuisance_rows = []
    for sequence in sequences:
        label = "+".join(sequence) if sequence else "none"
        if sequence:
            _, maps = learn_sequence(train, ex_train_energy, sequence)
            corrected = apply_sequence(validation, ex_val_energy, maps)
        else:
            corrected = ex_val_energy
        local_fit = fit_peak(corrected)
        ex_nuisance_rows.append({"sequence": label, **{f"validation_{k}": v for k, v in asdict(local_fit).items()}})
    ex_nuisance = pd.DataFrame(ex_nuisance_rows).sort_values("validation_resolution")
    ex_nuisance.to_csv(OUT / "exploratory_nuisance_validation.csv", index=False)
    ex_selected_label = choose_parsimonious_sequence(ex_nuisance)
    ex_selected_sequence = [] if ex_selected_label == "none" else ex_selected_label.split("+")
    ex_dev_medians = (float(dev[ex_s1].median()), float(dev[ex_s2].median()))
    ex_dev_raw = normalized_projection(dev, ex_s1, ex_s2, ex_alpha, ex_dev_medians)
    ex_dev_scale = estimate_mode(ex_dev_raw)
    ex_dev_energy = ex_dev_raw / ex_dev_scale
    ex_test_base = normalized_projection(test, ex_s1, ex_s2, ex_alpha, ex_dev_medians) / ex_dev_scale
    ex_base_fit = fit_peak(ex_test_base)
    if ex_selected_sequence:
        _, ex_final_maps = learn_sequence(dev, ex_dev_energy, ex_selected_sequence)
        ex_test_corrected = apply_sequence(test, ex_test_base, ex_final_maps)
    else:
        ex_final_maps, ex_test_corrected = [], ex_test_base.copy()
    ex_corrected_fit = fit_peak(ex_test_corrected)
    ex_boot = bootstrap_by_file(test, ex_test_base, ex_test_corrected, n_boot=250)
    ex_boot.to_csv(OUT / "paired_file_bootstrap_exploratory.csv", index=False)

    width_rows = []
    for label, values in [
        ("documented_baseline", test_base),
        ("documented_plus_selected_nuisance", test_corrected),
        ("ub_stretch_uncorrected", ex_test_base),
        ("ub_stretch_plus_selected_nuisance", ex_test_corrected),
    ]:
        width_rows.append({"method": label, "fit_core_resolution": fit_peak(values).resolution, "robust_central68_halfwidth_over_median": robust_central68(values), "events_in_window": int(((values >= FIT_WINDOW[0]) & (values <= FIT_WINDOW[1]) & np.isfinite(values)).sum())})
    pd.DataFrame(width_rows).to_csv(OUT / "resolution_crosscheck.csv", index=False)

    ratio_rows = []
    for numerator, denominator in [
        ("qS2Belife_max", "qS2BC_max"),
        ("qS2Bdesub_C_stretch", "qS2Bub_C_stretch"),
    ]:
        ratio = (df[numerator] / df[denominator]).replace([np.inf, -np.inf], np.nan).dropna()
        ratio_rows.append({"numerator": numerator, "denominator": denominator, "n": int(len(ratio)), "mean_ratio": float(ratio.mean()), "std_ratio": float(ratio.std()), "min_ratio": float(ratio.min()), "max_ratio": float(ratio.max())})
    pd.DataFrame(ratio_rows).to_csv(OUT / "branch_equivalence_checks.csv", index=False)

    cv_specs = [
        ("documented_corrected_lifetime", "qS1C_max", "qS2Belife_max"),
        ("exploratory_ub_bottom_stretch", "qS1ub_C", "qS2Bub_C_stretch"),
    ]
    cv_fold_tables, cv_event_tables, cv_summary_rows, cv_bootstrap_summary_rows = [], [], [], []
    cv_arrays = {}
    for cv_name, cv_s1, cv_s2 in cv_specs:
        cv_base, cv_corr, cv_folds = nested_crossfit(analysis, cv_s1, cv_s2, cv_name, sequences)
        cv_fold_tables.append(cv_folds)
        cv_arrays[cv_name] = (cv_base, cv_corr)
        event_table = analysis[["runNumber", "fileNumber", "eventNumber"]].copy()
        event_table["method"] = cv_name
        event_table["outer_fold"] = analysis.fileNumber.astype(int) % 5
        event_table["oof_baseline_energy"] = cv_base
        event_table["oof_corrected_energy"] = cv_corr
        cv_event_tables.append(event_table)
        for stage, values in [("baseline", cv_base), ("corrected", cv_corr)]:
            local_fit = fit_peak(values)
            cv_summary_rows.append({"method": cv_name, "stage": stage, **asdict(local_fit), "robust_central68": robust_central68(values)})
        cv_boot = bootstrap_by_file(analysis, cv_base, cv_corr, n_boot=250)
        cv_boot.insert(0, "method", cv_name)
        cv_boot.to_csv(OUT / f"nested_cv_bootstrap_{cv_name}.csv", index=False)
        for metric in ["baseline_resolution", "improved_resolution", "delta", "ratio"]:
            quantiles = cv_boot[metric].quantile([0.025, 0.5, 0.975])
            cv_bootstrap_summary_rows.append({"method": cv_name, "metric": metric, "q025": float(quantiles.iloc[0]), "median": float(quantiles.iloc[1]), "q975": float(quantiles.iloc[2]), "successful_replicates": int(len(cv_boot))})
    cv_folds_all = pd.concat(cv_fold_tables, ignore_index=True)
    cv_folds_all.to_csv(OUT / "nested_cv_folds.csv", index=False)
    pd.concat(cv_event_tables, ignore_index=True).to_csv(OUT / "nested_cv_events.tsv", sep="\t", index=False)
    cv_summary = pd.DataFrame(cv_summary_rows)
    cv_summary.to_csv(OUT / "nested_cv_summary.csv", index=False)
    cv_bootstrap_summary = pd.DataFrame(cv_bootstrap_summary_rows)
    cv_bootstrap_summary.to_csv(OUT / "nested_cv_bootstrap_summary.csv", index=False)
    cv_window_rows = []
    for cv_name, (cv_base, cv_corr) in cv_arrays.items():
        for lo, hi in [(0.76, 1.24), (0.78, 1.22), (0.80, 1.20), (0.82, 1.18)]:
            for stage, values in [("baseline", cv_base), ("corrected", cv_corr)]:
                local_fit = fit_peak(values, window=(lo, hi))
                cv_window_rows.append({"method": cv_name, "stage": stage, "window_low": lo, "window_high": hi, **asdict(local_fit)})
    pd.DataFrame(cv_window_rows).to_csv(OUT / "nested_cv_window_systematics.csv", index=False)

    map_tables = []
    for order, (name, response) in enumerate(final_maps):
        table = response["table"].copy()
        table.insert(0, "sequence_order", order)
        table.insert(1, "feature", name)
        map_tables.append(table)
    if map_tables:
        pd.concat(map_tables, ignore_index=True).to_csv(OUT / "nuisance_response_maps.csv", index=False)

    boot = bootstrap_by_file(test, test_base, test_corrected, n_boot=250)
    boot.to_csv(OUT / "paired_file_bootstrap.csv", index=False)
    boot_summary = {}
    for col in ["baseline_resolution", "improved_resolution", "delta", "ratio"]:
        if len(boot):
            q = boot[col].quantile([0.025, 0.5, 0.975])
            boot_summary[col] = {"q025": float(q.iloc[0]), "median": float(q.iloc[1]), "q975": float(q.iloc[2])}

    block_rows = []
    file_quantiles = np.unique(np.quantile(test.fileNumber, np.linspace(0, 1, 6)))
    for i in range(len(file_quantiles) - 1):
        sel = (test.fileNumber >= file_quantiles[i]) & (test.fileNumber <= file_quantiles[i + 1] if i == len(file_quantiles) - 2 else test.fileNumber < file_quantiles[i + 1])
        fb, fi = fit_peak(test_base[sel.to_numpy()]), fit_peak(test_corrected[sel.to_numpy()])
        block_rows.append({"block": i, "file_low": float(file_quantiles[i]), "file_high": float(file_quantiles[i + 1]), "events": int(sel.sum()), "baseline_resolution": fb.resolution, "improved_resolution": fi.resolution, "baseline_mu": fb.mu, "improved_mu": fi.mu})
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(OUT / "time_block_stability.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(analysis.qS1C_max, analysis.qS2Belife_max, c=analysis.dt / 1000, s=8, alpha=0.45, cmap="viridis", rasterized=True)
    ax.set_xlabel("qS1C_max [p.e.]")
    ax.set_ylabel("qS2Belife_max [p.e.]")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("dt [us]")
    ax.set_title("Corrected S1 vs lifetime-corrected bottom S2")
    fig.tight_layout()
    fig.savefig(FIG / "01_s1_s2_dt.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name, group in alpha_scans.groupby("method"):
        ax.plot(group.alpha, 100 * group.validation_resolution, marker=".", ms=3, lw=1, label=name)
    ax.set_xlabel("Normalized S1 weight alpha")
    ax.set_ylabel("Validation core resolution sigma/mu [%]")
    ax.set_title("Positive S1-S2 weight scan (validation only)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "02_alpha_scan.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ordered = method_results.sort_values("test_resolution")
    ax.barh(ordered.method, 100 * ordered.test_resolution, color="#4C78A8")
    ax.set_xlabel("Test core resolution sigma/mu [%]")
    ax.set_title("Pre-specified branch comparison (exploratory)")
    fig.tight_layout()
    fig.savefig(FIG / "03_method_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    plot_peak(ax, test_base, baseline_fit, f"baseline: {100*baseline_fit.resolution:.2f}%", "#E45756")
    plot_peak(ax, test_corrected, corrected_fit, f"selected nuisance correction: {100*corrected_fit.resolution:.2f}%", "#54A24B")
    ax.legend()
    ax.set_title("Frozen test sample peak fit")
    fig.tight_layout()
    fig.savefig(FIG / "04_test_peak_before_after.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(blocks))
    ax.plot(x, 100 * blocks.baseline_resolution, "o-", label="baseline")
    ax.plot(x, 100 * blocks.improved_resolution, "o-", label="selected correction")
    ax.set_xticks(x, [f"{int(a)}-{int(b)}" for a, b in zip(blocks.file_low, blocks.file_high)], rotation=30)
    ax.set_xlabel("Test fileNumber block")
    ax.set_ylabel("Core resolution sigma/mu [%]")
    ax.set_title("Time-block stress test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_time_block_stability.png", dpi=180)
    plt.close(fig)

    if final_maps:
        fig, axes = plt.subplots(1, len(final_maps), figsize=(5.0 * len(final_maps), 4.2), squeeze=False)
        for ax, (name, response) in zip(axes[0], final_maps):
            table = response["table"]
            good = table.fit_success.astype(bool)
            ax.plot(table.loc[good, "center"], table.loc[good, "mu"], "o-", color="#4C78A8")
            ax.axhline(response["global_mu"], color="black", ls="--", lw=1)
            ax.set_xlabel(name)
            ax.set_ylabel("Local fitted peak center")
            ax.set_title(f"Response map: {name}")
        fig.tight_layout()
        fig.savefig(FIG / "06_selected_response_maps.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    plot_peak(ax, test_base, baseline_fit, f"documented baseline: {100*baseline_fit.resolution:.2f}%", "#E45756")
    plot_peak(ax, ex_test_corrected, ex_corrected_fit, f"exploratory ub/stretch pipeline: {100*ex_corrected_fit.resolution:.2f}%", "#4C78A8")
    ax.legend()
    ax.set_title("Documented baseline vs exploratory branch pipeline")
    fig.tight_layout()
    fig.savefig(FIG / "07_exploratory_pipeline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, (cv_name, (cv_base, cv_corr)) in zip(axes, cv_arrays.items()):
        base_fit_cv, corr_fit_cv = fit_peak(cv_base), fit_peak(cv_corr)
        plot_peak(ax, cv_base, base_fit_cv, f"OOF baseline: {100*base_fit_cv.resolution:.2f}%", "#E45756")
        plot_peak(ax, cv_corr, corr_fit_cv, f"OOF corrected: {100*corr_fit_cv.resolution:.2f}%", "#54A24B")
        ax.set_title(cv_name)
        ax.legend(fontsize=8)
    fig.suptitle("Five-fold nested cross-fitted results")
    fig.tight_layout()
    fig.savefig(FIG / "08_nested_crossfit.png", dpi=180)
    plt.close(fig)

    summary = {
        "data": {"path": str(DATA_PATH), "rows": int(len(raw)), "columns": int(raw.shape[1]), "duplicate_event_keys": int(raw.duplicated(["runNumber", "fileNumber", "eventNumber"]).sum()), "common_analysis_rows": int(len(analysis)), "all_nan_columns": audit.loc[audit.missing_fraction == 1, "variable"].tolist(), "columns_with_inf": audit.loc[audit.inf_count > 0, ["variable", "inf_count"]].to_dict("records")},
        "split": split_summary.to_dict("records"),
        "primary_baseline": {"method": "corrected_lifetime", "alpha": primary_alpha, "fit": asdict(baseline_fit)},
        "nuisance_selection": {"selected_on_validation": selected_label, "test_fit": asdict(corrected_fit)},
        "exploratory_pipeline": {"method": exploratory_method, "alpha": ex_alpha, "selected_nuisance_on_validation": ex_selected_label, "uncorrected_test_fit": asdict(ex_base_fit), "corrected_test_fit": asdict(ex_corrected_fit), "warning": "ub/stretch branch definitions are absent from the supplied TechNote"},
        "nested_crossfit": cv_summary.to_dict("records"),
        "nested_crossfit_bootstrap": cv_bootstrap_summary.to_dict("records"),
        "bootstrap": {"successful_replicates": int(len(boot)), "summary": boot_summary},
        "limitations": ["Source identity and absolute peak energy are not established from the provided files.", "The input was preselected with an undocumented E > 2 MeV estimator; results are conditional on that selection.", "Cut-flag semantics are undocumented, so no additional cut is applied.", "Only run 10972 is available; external cross-run validation is still required.", "Methods using ub/stretch/MCPAF branches are exploratory because those branch definitions are not in the provided TechNote."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
