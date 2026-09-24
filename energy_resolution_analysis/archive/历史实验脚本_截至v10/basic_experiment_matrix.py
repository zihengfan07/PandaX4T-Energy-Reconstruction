#!/usr/bin/env python3
"""Independent basic-variable experiment matrix for PandaX energy reconstruction.

This script intentionally uses only the variables permitted by
INDEPENDENT_PROTOCOL.md.  It evaluates every valid input event with fixed,
non-parametric central-width metrics; it does not import or inspect any earlier
reconstruction implementation.

The experiment uses a seeded, balanced event-level five-fold OOF view.  Input
rows receive a synthetic ``sourceRow`` key before any sorting or filtering;
the acquisition-file identifier is neither read nor used.

For every outer fold and correction family, the positive S1/S2 weight is
selected by an inner cross-validation loop.  An overall family winner is also
selected inside the outer training set.  Thus the outer test events never
choose a weight, response model, or model family.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
OUT = HERE / "basic_matrix_outputs"

ALLOWED_COLUMNS = [
    "runNumber",
    "eventNumber",
    "qS1_max",
    "qS2B_max",
    "dt",
    "xS2T_max",
    "yS2T_max",
]

# Positive dimensionless S1 weights after each channel is divided by its
# training-set median.  The grid includes equal normalized weighting exactly.
ALPHAS = np.unique(
    np.r_[np.geomspace(0.05, 20.0, 15), np.array([0.75, 1.0, 1.5])]
).astype(float)

METRIC_TOLERANCE = 0.0005  # 0.05 percentage point in a fractional resolution.
MIN_CELL_COUNT = 12
MAX_LOG_CORRECTION = 0.30
BOOTSTRAP_REPLICATES = 1000
SEED = 10972


@dataclass(frozen=True)
class MethodSpec:
    name: str
    steps: Tuple[str, ...]
    complexity: int
    description: str


METHODS: Tuple[MethodSpec, ...] = (
    MethodSpec("raw", (), 0, "positive-weight raw qS1+qS2B combination"),
    MethodSpec("dt_quad", ("dt",), 3, "quadratic dt response from binned medians"),
    MethodSpec("r2_quad", ("r2",), 3, "quadratic r2 response from binned medians"),
    MethodSpec(
        "dt_r2_joint",
        ("dt_r2",),
        6,
        "joint quadratic dt/r2 response from 2-D cell medians",
    ),
    MethodSpec(
        "dt_then_r2",
        ("dt", "r2"),
        6,
        "sequential binned-median dt then r2 correction",
    ),
    MethodSpec(
        "r2_then_dt",
        ("r2", "dt"),
        6,
        "sequential binned-median r2 then dt correction",
    ),
    MethodSpec(
        "xy_quad",
        ("xy",),
        6,
        "quadratic x/y response from 2-D cell medians",
    ),
    MethodSpec(
        "dt_then_xy",
        ("dt", "xy"),
        9,
        "sequential binned-median dt then x/y correction",
    ),
    MethodSpec(
        "dt_r2_then_xy",
        ("dt_r2", "xy"),
        12,
        "joint dt/r2 followed by a low-order x/y residual correction",
    ),
)
METHOD_BY_NAME = {m.name: m for m in METHODS}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def central_widths(values: np.ndarray) -> Dict[str, float]:
    """Full-sample, model-free widths relative to the sample median."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("central_widths requires a non-empty finite array")
    q05, q16, q50, q84, q95 = np.quantile(
        x, [0.05, 0.15865, 0.50, 0.84135, 0.95], method="linear"
    )
    if q50 <= 0:
        raise ValueError("energy median must be positive")
    return {
        "n": int(len(x)),
        "median": float(q50),
        "q05": float(q05),
        "q16": float(q16),
        "q84": float(q84),
        "q95": float(q95),
        "r68": float((q84 - q16) / (2.0 * q50)),
        "r90": float((q95 - q05) / (2.0 * q50)),
    }


def weighted_mean(rows: Sequence[Mapping[str, float]], key: str) -> float:
    w = np.asarray([float(r["n"]) for r in rows])
    v = np.asarray([float(r[key]) for r in rows])
    return float(np.average(v, weights=w))


def safe_scale(values: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(values))
    lo, hi = np.quantile(values, [0.05, 0.95])
    scale = float((hi - lo) / 2.0)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return med, scale


def coordinate_matrix(df: pd.DataFrame, kind: str) -> np.ndarray:
    dt = df["dt"].to_numpy(float)
    x = df["xS2T_max"].to_numpy(float)
    y = df["yS2T_max"].to_numpy(float)
    r2 = x * x + y * y
    if kind == "dt":
        return dt[:, None]
    if kind == "r2":
        return r2[:, None]
    if kind == "dt_r2":
        return np.column_stack([dt, r2])
    if kind == "xy":
        return np.column_stack([x, y])
    raise KeyError(kind)


def design_matrix(z: np.ndarray, kind: str) -> np.ndarray:
    if kind in {"dt", "r2"}:
        a = z[:, 0]
        return np.column_stack([np.ones(len(z)), a, a * a])
    if kind == "dt_r2":
        a, b = z[:, 0], z[:, 1]
        return np.column_stack(
            [np.ones(len(z)), a, b, a * a, b * b, a * b]
        )
    if kind == "xy":
        x, y = z[:, 0], z[:, 1]
        return np.column_stack(
            [np.ones(len(z)), x, y, x * x, x * y, y * y]
        )
    raise KeyError(kind)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi <= lo:
            return np.array([-np.inf, np.inf])
        edges = np.linspace(lo, hi, min(n_bins, 2) + 1)
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def cell_summary(
    z: np.ndarray, log_energy: np.ndarray, bins_per_dimension: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Aggregate events first; only cell medians are fitted by the response model."""
    edges: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for j in range(z.shape[1]):
        e = quantile_edges(z[:, j], bins_per_dimension)
        edges.append(e)
        labels.append(np.digitize(z[:, j], e[1:-1], right=False))
    codes = np.column_stack(labels)
    frame = pd.DataFrame(codes, columns=[f"b{j}" for j in range(z.shape[1])])
    for j in range(z.shape[1]):
        frame[f"z{j}"] = z[:, j]
    frame["log_e"] = log_energy
    group_cols = [f"b{j}" for j in range(z.shape[1])]
    grouped = frame.groupby(group_cols, sort=True, observed=True)
    centers = grouped[[f"z{j}" for j in range(z.shape[1])]].median()
    targets = grouped["log_e"].median()
    counts = grouped.size()
    keep = counts >= MIN_CELL_COUNT
    return (
        centers.loc[keep].to_numpy(float),
        targets.loc[keep].to_numpy(float),
        counts.loc[keep].to_numpy(float),
        edges,
    )


def fit_response(df: pd.DataFrame, energy: np.ndarray, kind: str) -> Dict[str, object]:
    coords = coordinate_matrix(df, kind)
    centers = np.empty(coords.shape[1], dtype=float)
    scales = np.empty(coords.shape[1], dtype=float)
    z = np.empty_like(coords, dtype=float)
    for j in range(coords.shape[1]):
        centers[j], scales[j] = safe_scale(coords[:, j])
        z[:, j] = (coords[:, j] - centers[j]) / scales[j]

    bins = 10 if z.shape[1] == 1 else 5
    cell_z, cell_y, cell_n, _ = cell_summary(z, np.log(energy), bins)
    p = design_matrix(cell_z, kind).shape[1]
    if len(cell_z) < p + 1:
        raise RuntimeError(f"too few populated cells for {kind}: {len(cell_z)}")

    X = design_matrix(cell_z, kind)
    # Count weighting reflects the precision of a cell median, while the model
    # still sees one aggregate response per cell rather than per-event targets.
    sw = np.sqrt(cell_n)
    Xw = X * sw[:, None]
    yw = cell_y * sw
    ridge = np.eye(X.shape[1]) * 1e-8
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(Xw.T @ Xw + ridge, Xw.T @ yw)
    reference = float(np.median(np.log(energy)))
    train_pred = X @ beta
    pred_lo = float(np.quantile(train_pred - reference, 0.01))
    pred_hi = float(np.quantile(train_pred - reference, 0.99))
    pred_lo = max(pred_lo, -MAX_LOG_CORRECTION)
    pred_hi = min(pred_hi, MAX_LOG_CORRECTION)
    if pred_lo > pred_hi:
        pred_lo, pred_hi = -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION
    return {
        "kind": kind,
        "centers": centers,
        "scales": scales,
        "beta": beta,
        "reference": reference,
        "pred_lo": pred_lo,
        "pred_hi": pred_hi,
        "n_cells": int(len(cell_z)),
    }


def apply_response(
    df: pd.DataFrame, energy: np.ndarray, model: Mapping[str, object]
) -> np.ndarray:
    kind = str(model["kind"])
    coords = coordinate_matrix(df, kind)
    centers = np.asarray(model["centers"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    z = (coords - centers[None, :]) / scales[None, :]
    pred = design_matrix(z, kind) @ np.asarray(model["beta"], dtype=float)
    delta = pred - float(model["reference"])
    delta = np.clip(delta, float(model["pred_lo"]), float(model["pred_hi"]))
    return energy * np.exp(-delta)


def raw_combination(
    df: pd.DataFrame, alpha: float, med_s1: float, med_s2: float
) -> np.ndarray:
    return (
        df["qS2B_max"].to_numpy(float) / med_s2
        + alpha * df["qS1_max"].to_numpy(float) / med_s1
    )


def fit_method(
    train: pd.DataFrame, alpha: float, spec: MethodSpec
) -> Dict[str, object]:
    med_s1 = float(np.median(train["qS1_max"]))
    med_s2 = float(np.median(train["qS2B_max"]))
    energy = raw_combination(train, alpha, med_s1, med_s2)
    response_steps: List[Dict[str, object]] = []
    for kind in spec.steps:
        response = fit_response(train, energy, kind)
        response_steps.append(response)
        energy = apply_response(train, energy, response)
    scale = float(np.median(energy))
    return {
        "alpha": float(alpha),
        "effective_raw_beta": float(alpha * med_s2 / med_s1),
        "med_s1": med_s1,
        "med_s2": med_s2,
        "scale": scale,
        "steps": response_steps,
    }


def apply_method(
    frame: pd.DataFrame, spec: MethodSpec, model: Mapping[str, object]
) -> np.ndarray:
    energy = raw_combination(
        frame,
        float(model["alpha"]),
        float(model["med_s1"]),
        float(model["med_s2"]),
    )
    for response in model["steps"]:  # type: ignore[index]
        energy = apply_response(frame, energy, response)
    return energy / float(model["scale"])


def make_outer_folds(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(df))
    folds = np.full(len(df), -1, dtype=int)
    folds[permutation] = np.arange(len(df), dtype=int) % 5
    return {"event_random5": folds}


def score_candidate_inner(
    df: pd.DataFrame,
    outer_train_mask: np.ndarray,
    fold_labels: np.ndarray,
    spec: MethodSpec,
    alpha: float,
) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    for inner_fold in sorted(np.unique(fold_labels[outer_train_mask])):
        inner_val = outer_train_mask & (fold_labels == inner_fold)
        inner_train = outer_train_mask & (fold_labels != inner_fold)
        model = fit_method(df.loc[inner_train], alpha, spec)
        pred = apply_method(df.loc[inner_val], spec, model)
        metrics.append(central_widths(pred))
    return {
        "inner_r68": weighted_mean(metrics, "r68"),
        "inner_r90": weighted_mean(metrics, "r90"),
        "inner_n": int(sum(int(m["n"]) for m in metrics)),
    }


def choose_alpha(candidate_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return min(
        candidate_rows,
        key=lambda r: (float(r["inner_r68"]), float(r["inner_r90"]), abs(math.log(float(r["alpha"]))),),
    )


def choose_family(selection_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    best_r68 = min(float(r["inner_r68"]) for r in selection_rows)
    eligible = [
        r
        for r in selection_rows
        if float(r["inner_r68"]) <= best_r68 + METRIC_TOLERANCE
    ]
    return min(
        eligible,
        key=lambda r: (
            METHOD_BY_NAME[str(r["method"])].complexity,
            float(r["inner_r68"]),
            float(r["inner_r90"]),
        ),
    )


def run_nested_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    fold_sets = make_outer_folds(df)
    candidate_records: List[Dict[str, object]] = []
    selection_records: List[Dict[str, object]] = []
    fold_records: List[Dict[str, object]] = []
    prediction_records: List[pd.DataFrame] = []

    for scheme, folds in fold_sets.items():
        print(f"\n[{scheme}]", flush=True)
        for outer_fold in range(5):
            print(f"  outer fold {outer_fold + 1}/5", flush=True)
            test_mask = folds == outer_fold
            train_mask = ~test_mask
            fold_selection: List[Dict[str, object]] = []
            fold_predictions: Dict[str, np.ndarray] = {}
            fold_models: Dict[str, Dict[str, object]] = {}

            for spec in METHODS:
                method_candidates: List[Dict[str, object]] = []
                for alpha in ALPHAS:
                    score = score_candidate_inner(
                        df, train_mask, folds, spec, float(alpha)
                    )
                    row: Dict[str, object] = {
                        "scheme": scheme,
                        "outer_fold": outer_fold,
                        "method": spec.name,
                        "complexity": spec.complexity,
                        "alpha": float(alpha),
                        **score,
                    }
                    method_candidates.append(row)
                    candidate_records.append(row)
                chosen = dict(choose_alpha(method_candidates))
                selection_records.append(chosen)
                fold_selection.append(chosen)

                alpha = float(chosen["alpha"])
                model = fit_method(df.loc[train_mask], alpha, spec)
                pred = apply_method(df.loc[test_mask], spec, model)
                fold_predictions[spec.name] = pred
                fold_models[spec.name] = model
                metric = central_widths(pred)
                fold_records.append(
                    {
                        "scheme": scheme,
                        "outer_fold": outer_fold,
                        "method": spec.name,
                        "selected_alpha": alpha,
                        "effective_raw_beta": model["effective_raw_beta"],
                        "inner_r68": chosen["inner_r68"],
                        "inner_r90": chosen["inner_r90"],
                        **metric,
                    }
                )

                keys = df.loc[test_mask, ["sourceRow", "runNumber", "eventNumber"]].copy()
                keys["scheme"] = scheme
                keys["outer_fold"] = outer_fold
                keys["method"] = spec.name
                keys["selected_alpha"] = alpha
                keys["effective_raw_beta"] = model["effective_raw_beta"]
                keys["energy"] = pred
                prediction_records.append(keys)

            winner = dict(choose_family(fold_selection))
            winner_name = str(winner["method"])
            pred = fold_predictions[winner_name]
            model = fold_models[winner_name]
            metric = central_widths(pred)
            fold_records.append(
                {
                    "scheme": scheme,
                    "outer_fold": outer_fold,
                    "method": "nested_winner",
                    "selected_family": winner_name,
                    "selected_alpha": winner["alpha"],
                    "effective_raw_beta": model["effective_raw_beta"],
                    "inner_r68": winner["inner_r68"],
                    "inner_r90": winner["inner_r90"],
                    **metric,
                }
            )
            keys = df.loc[test_mask, ["sourceRow", "runNumber", "eventNumber"]].copy()
            keys["scheme"] = scheme
            keys["outer_fold"] = outer_fold
            keys["method"] = "nested_winner"
            keys["selected_family"] = winner_name
            keys["selected_alpha"] = winner["alpha"]
            keys["effective_raw_beta"] = model["effective_raw_beta"]
            keys["energy"] = pred
            prediction_records.append(keys)

    candidates = pd.DataFrame(candidate_records)
    selections = pd.DataFrame(selection_records)
    folds = pd.DataFrame(fold_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    return candidates, selections, folds, predictions


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (scheme, method), group in predictions.groupby(["scheme", "method"], sort=True):
        metric = central_widths(group["energy"].to_numpy(float))
        rows.append({"scheme": scheme, "method": method, **metric})
    result = pd.DataFrame(rows)
    complexity = {m.name: m.complexity for m in METHODS}
    complexity["nested_winner"] = -1
    result["complexity"] = result["method"].map(complexity)
    result["r68_percent"] = 100.0 * result["r68"]
    result["r90_percent"] = 100.0 * result["r90"]
    return result.sort_values(["scheme", "r68", "r90"]).reset_index(drop=True)


def paired_event_bootstrap(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Conditional paired event uncertainty for frozen OOF predictions."""

    rng = np.random.default_rng(SEED)
    records: List[Dict[str, object]] = []
    selected = predictions[predictions["method"].isin(["raw", "nested_winner"])]
    for scheme, local in selected.groupby("scheme", sort=True):
        wide = local.pivot(
            index=["sourceRow"],
            columns="method",
            values="energy",
        ).reset_index()
        raw = wide["raw"].to_numpy(float)
        winner = wide["nested_winner"].to_numpy(float)
        n_events = len(wide)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sample = rng.integers(0, n_events, size=n_events)
            raw_metric = central_widths(raw[sample])
            winner_metric = central_widths(winner[sample])
            records.append(
                {
                    "scheme": scheme,
                    "replicate": replicate,
                    "raw_r68": raw_metric["r68"],
                    "winner_r68": winner_metric["r68"],
                    "delta_r68": winner_metric["r68"] - raw_metric["r68"],
                    "ratio_r68": winner_metric["r68"] / raw_metric["r68"],
                    "raw_r90": raw_metric["r90"],
                    "winner_r90": winner_metric["r90"],
                    "delta_r90": winner_metric["r90"] - raw_metric["r90"],
                    "ratio_r90": winner_metric["r90"] / raw_metric["r90"],
                }
            )
    replicates = pd.DataFrame(records)
    summary_rows: List[Dict[str, object]] = []
    for scheme, local in replicates.groupby("scheme", sort=True):
        for metric in ["delta_r68", "ratio_r68", "delta_r90", "ratio_r90"]:
            q025, median, q975 = local[metric].quantile([0.025, 0.5, 0.975])
            summary_rows.append(
                {
                    "scheme": scheme,
                    "metric": metric,
                    "q025": q025,
                    "median": median,
                    "q975": q975,
                    "replicates": len(local),
                    "interpretation": "paired event resampling conditional on frozen OOF predictions; pipeline is not retrained",
                }
            )
    return replicates, pd.DataFrame(summary_rows)


def plot_method_matrix(summary: pd.DataFrame) -> None:
    methods = [m.name for m in METHODS] + ["nested_winner"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    scheme = "event_random5"
    color = "#277da1"
    for ax, metric, title in zip(
        axes,
        ["r68_percent", "r90_percent"],
        ["Full-event R68 / median", "Full-event R90 / median"],
    ):
        x = np.arange(len(methods))
        sub = summary[summary["scheme"] == scheme].set_index("method")
        vals = [sub.loc[m, metric] for m in methods]
        ax.scatter(x, vals, s=55, color=color, label="event-level OOF")
        ax.plot(x, vals, lw=1, alpha=0.55, color=color)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("percent")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Nested OOF basic-variable experiment matrix (all events)")
    fig.tight_layout()
    fig.savefig(OUT / "method_matrix.png", dpi=180)
    plt.close(fig)


def plot_fold_stability(folds: pd.DataFrame) -> None:
    methods = [m.name for m in METHODS] + ["nested_winner"]
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    sub = folds[folds["scheme"] == "event_random5"]
    matrix = (
        sub.pivot(index="method", columns="outer_fold", values="r68")
        .reindex(methods)
        .to_numpy(float)
        * 100.0
    )
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title("Seeded event-level outer-fold R68 (%)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > np.nanmedian(matrix) else "black")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    ax.set_xticks(range(5))
    ax.set_xticklabels([str(i) for i in range(5)])
    ax.set_xlabel("outer event fold")
    fig.tight_layout()
    fig.savefig(OUT / "fold_stability.png", dpi=180)
    plt.close(fig)


def plot_weight_selection(selections: pd.DataFrame) -> None:
    methods = [m.name for m in METHODS]
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    sub = selections[selections["scheme"] == "event_random5"]
    for j, method in enumerate(methods):
        vals = sub[sub["method"] == method].sort_values("outer_fold")
        ax.scatter(
            np.full(len(vals), j) + (vals["outer_fold"].to_numpy() - 2) * 0.035,
            vals["alpha"],
            s=32,
            alpha=0.8,
        )
    ax.set_title("Seeded event-level folds")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("inner-selected positive normalized S1 weight alpha")
    fig.tight_layout()
    fig.savefig(OUT / "weight_selection.png", dpi=180)
    plt.close(fig)


def write_readme(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    selections: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    lines = [
        "# Basic-variable independent experiment matrix",
        "",
        "This directory was generated by `basic_experiment_matrix.py` using only",
        "`qS1_max`, `qS2B_max`, `dt`, `xS2T_max`, `yS2T_max`,",
        "and event identity. No cut flag, prior reconstruction code, presumed source",
        "energy, or ub/stretch/MCPAF-derived branch enters training or selection.",
        "",
        "All 4500 input events are retained. The selection metric is model-free:",
        "`R68=(Q84.135-Q15.865)/(2*median)` with `R90=(Q95-Q05)/(2*median)` reported beside it.",
        "Response models fit aggregate bin medians, not individual event labels.",
        "",
        "For every outer fold, each method's positive weight is chosen using only",
        "inner event folds. `nested_winner` also chooses the correction family internally,",
        "preferring the simplest family within 0.05 percentage point of the best",
        "inner R68. The outer split is a seeded, balanced event-level five-fold split.",
        "",
        "## OOF summary",
        "",
        summary[["scheme", "method", "n", "r68_percent", "r90_percent"]]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "Negative results are intentionally retained; method rows are not filtered by",
        "whether they improve on `raw`. Acquisition-file identity is not read or used.",
        "",
        "## Files",
        "",
        "- `summary_metrics.csv`: all-event OOF R68/R90 by method and split scheme",
        "- `fold_metrics.csv`: honest outer-fold metrics and selected weights/families",
        "- `inner_selection.csv`: training-only selected alpha for each method/fold",
        "- `inner_candidate_scores.csv`: every positive-weight candidate, including losses",
        "- `oof_predictions.csv`: per-event OOF normalized energy",
        "- `paired_event_bootstrap_summary.csv`: paired event bootstrap, conditional on frozen OOF predictions",
        "- `metadata.json`: definitions, restrictions, fold counts, and data fingerprint",
        "- PNG figures: method matrix, fold stability, and selected weights",
        "",
        "The R68/R90 widths characterize the full observed distribution. They are not",
        "a fitted Gaussian peak resolution and should not be relabeled as sigma/mu.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Reading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, sep="\t", usecols=ALLOWED_COLUMNS)
    df.insert(0, "sourceRow", np.arange(len(df), dtype=int))
    finite = np.isfinite(df[ALLOWED_COLUMNS].to_numpy(float)).all(axis=1)
    positive = (df["qS1_max"] > 0) & (df["qS2B_max"] > 0)
    if not bool(np.all(finite & positive)):
        bad = int(np.sum(~(finite & positive)))
        raise RuntimeError(
            f"{bad} event(s) invalid in allowed fields; aborting rather than silently cutting"
        )
    if not df["sourceRow"].is_unique:
        raise RuntimeError("sourceRow must be unique")

    fold_sets = make_outer_folds(df)
    fold_counts = {
        str(fold): int(np.sum(fold_sets["event_random5"] == fold))
        for fold in range(5)
    }

    candidates, selections, folds, predictions = run_nested_matrix(df)
    summary = aggregate_predictions(predictions)
    bootstrap_replicates, bootstrap_summary = paired_event_bootstrap(predictions)

    candidates.to_csv(OUT / "inner_candidate_scores.csv", index=False)
    selections.to_csv(OUT / "inner_selection.csv", index=False)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "oof_predictions.csv", index=False)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    bootstrap_replicates.to_csv(OUT / "paired_event_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(OUT / "paired_event_bootstrap_summary.csv", index=False)

    metadata: Dict[str, object] = {
        "data_file": DATA_PATH.name,
        "data_sha256": sha256(DATA_PATH),
        "n_events": int(len(df)),
        "allowed_columns": ALLOWED_COLUMNS,
        "event_identity": "sourceRow is the zero-based original TXT row order and is used only for exact alignment",
        "all_events_retained": True,
        "forbidden_inputs_used": False,
        "alpha_definition": "E_raw=qS2B/median_train(qS2B)+alpha*qS1/median_train(qS1)",
        "positive_alpha_grid": [float(a) for a in ALPHAS],
        "metrics": {
            "R68": "(Q84.135-Q15.865)/(2*median), evaluated on all OOF events",
            "R90": "(Q95-Q05)/(2*median), evaluated on all OOF events",
        },
        "outer_schemes": {
            "event_random5": "seeded balanced random assignment of source rows to five outer folds",
        },
        "fold_counts": fold_counts,
        "inner_validation": "leave one of the remaining outer event folds out in turn",
        "family_selection_tolerance_fraction": METRIC_TOLERANCE,
        "response_learning": (
            "low-order surfaces fitted to quantile-cell median log energies"
        ),
        "bootstrap": (
            f"{BOOTSTRAP_REPLICATES} paired event-resampling replicates on frozen OOF predictions; "
            "the full method-selection and response-learning pipeline is not retrained"
        ),
        "methods": [
            {
                "name": m.name,
                "steps": list(m.steps),
                "complexity": m.complexity,
                "description": m.description,
            }
            for m in METHODS
        ],
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_method_matrix(summary)
    plot_fold_stability(folds)
    plot_weight_selection(selections)
    write_readme(summary, folds, selections, metadata)

    print("\nOOF summary (%):", flush=True)
    print(
        summary[["scheme", "method", "r68_percent", "r90_percent"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        ),
        flush=True,
    )
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
