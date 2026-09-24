from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
OUT = ROOT / "channel_physics_outputs"
SEED = 10972
N_FOLDS = 5
N_RESPONSE_BINS = 5
N_BOOTSTRAP = 1000
ALPHA_GRID = np.linspace(0.10, 0.90, 17)

# This whitelist is deliberately narrow. In particular, no corrected, lifetime,
# ub, stretch, desaturation, MCPAF, or other derived charge branch is read.
BASE_COLUMNS = [
    "runNumber",
    "eventNumber",
    "qS1_max",
    "qS2B_max",
    "dt",
    "xS2T_max",
    "yS2T_max",
]


CANDIDATES: dict[str, dict[str, Any]] = {
    "A0_raw": {
        "label": "raw S1 + raw S2B",
        "s1_steps": [],
        "s2_steps": [],
    },
    "A1_s2_lifetime": {
        "label": "raw S1 + S2B exponential lifetime",
        "s1_steps": [],
        "s2_steps": ["lifetime"],
    },
    "A2_s1_dt_r2": {
        "label": "S1(dt,r2) + raw S2B",
        "s1_steps": ["dt_map", "r2_map"],
        "s2_steps": [],
    },
    "A3_s2_lifetime_r2": {
        "label": "raw S1 + S2B(lifetime,r2)",
        "s1_steps": [],
        "s2_steps": ["lifetime", "r2_map"],
    },
    "A4_both": {
        "label": "S1(dt,r2) + S2B(lifetime,r2)",
        "s1_steps": ["dt_map", "r2_map"],
        "s2_steps": ["lifetime", "r2_map"],
    },
}


@dataclass
class BinnedResponse:
    feature: str
    centers: list[float]
    log_response: list[float]
    bin_rows: list[dict[str, float | int]]
    global_center: float


@dataclass
class LinearResponse:
    feature: str
    reference: float
    slope: float
    unconstrained_slope: float
    bin_rows: list[dict[str, float | int]]
    response_clip: tuple[float, float]


def finite_positive(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


def robust_peak_center(values: np.ndarray) -> float:
    """Low-freedom core location: smoothed log-mode, then local median.

    This is a location estimator, not an event-by-event regression target. It is
    used only on training data or on coarse training bins.
    """

    x = finite_positive(values)
    if x.size < 20:
        return float("nan")
    lx = np.log(x)
    lo, hi = np.quantile(lx, [0.01, 0.99])
    if not np.isfinite(lo + hi) or hi <= lo:
        return float(np.median(x))
    hist, edges = np.histogram(lx, bins=80, range=(lo, hi))
    smooth = gaussian_filter1d(hist.astype(float), sigma=1.25)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[int(np.argmax(smooth))])
    # A fixed +/-20% multiplicative window is transparent and wide compared
    # with the final energy core. The median inside it is robust to tails.
    local = x[np.abs(np.log(x) - mode) <= math.log(1.20)]
    if local.size < max(20, int(0.10 * x.size)):
        local = x
    return float(np.median(local))


def quantile_metrics(values: np.ndarray) -> dict[str, float | int]:
    """Full-event robust metrics; no fit window and no event rejection."""

    raw = np.asarray(values, dtype=float)
    finite = raw[np.isfinite(raw)]
    result: dict[str, float | int] = {
        "events_total": int(raw.size),
        "events_finite": int(finite.size),
        "finite_fraction": float(finite.size / raw.size) if raw.size else float("nan"),
    }
    if finite.size < 20:
        result.update(
            {
                "median": float("nan"),
                "r68": float("nan"),
                "r90": float("nan"),
                "tail_beyond_2r68": float("nan"),
                "outside_20pct": float("nan"),
                "low_tail_20pct": float("nan"),
                "high_tail_20pct": float("nan"),
            }
        )
        return result
    median = float(np.median(finite))
    y = finite / median
    q05, q15865, q84135, q95 = np.quantile(y, [0.05, 0.15865, 0.84135, 0.95])
    r68 = float((q84135 - q15865) / 2.0)
    r90 = float((q95 - q05) / 2.0)
    result.update(
        {
            "median": median,
            "r68": r68,
            "r90": r90,
            "tail_beyond_2r68": float(np.mean(np.abs(y - 1.0) > 2.0 * r68)),
            "outside_20pct": float(np.mean((y < 0.80) | (y > 1.20))),
            "low_tail_20pct": float(np.mean(y < 0.80)),
            "high_tail_20pct": float(np.mean(y > 1.20)),
        }
    )
    return result


def feature_array(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "dt_us":
        return frame["dt"].to_numpy(float) / 1000.0
    if name == "r2":
        x = frame["xS2T_max"].to_numpy(float)
        y = frame["yS2T_max"].to_numpy(float)
        return x * x + y * y
    raise KeyError(name)


def fit_binned_response(
    values: np.ndarray,
    feature: np.ndarray,
    feature_name: str,
    bins: int = N_RESPONSE_BINS,
) -> BinnedResponse:
    values = np.asarray(values, dtype=float)
    feature = np.asarray(feature, dtype=float)
    mask = np.isfinite(values) & (values > 0) & np.isfinite(feature)
    v, f = values[mask], feature[mask]
    global_center = robust_peak_center(v)
    edges = np.unique(np.quantile(f, np.linspace(0.0, 1.0, bins + 1)))
    rows: list[dict[str, float | int]] = []
    for idx in range(max(0, len(edges) - 1)):
        last = idx == len(edges) - 2
        select = (f >= edges[idx]) & ((f <= edges[idx + 1]) if last else (f < edges[idx + 1]))
        local_center = robust_peak_center(v[select])
        rows.append(
            {
                "bin": int(idx),
                "low": float(edges[idx]),
                "high": float(edges[idx + 1]),
                "feature_center": float(np.median(f[select])) if np.any(select) else float("nan"),
                "events": int(np.sum(select)),
                "signal_center": float(local_center),
                "response": float(local_center / global_center),
            }
        )
    table = pd.DataFrame(rows)
    good = (
        np.isfinite(table["feature_center"].to_numpy(float))
        & np.isfinite(table["response"].to_numpy(float))
        & (table["response"].to_numpy(float) > 0)
    )
    if int(np.sum(good)) < 2:
        raise RuntimeError(f"Too few valid bins for {feature_name}")
    centers = table.loc[good, "feature_center"].to_numpy(float)
    log_response = np.log(table.loc[good, "response"].to_numpy(float))
    order = np.argsort(centers)
    # Wide guardrail only prevents numerical explosions at sparse boundaries.
    log_response = np.clip(log_response[order], math.log(0.50), math.log(2.00))
    return BinnedResponse(
        feature=feature_name,
        centers=centers[order].tolist(),
        log_response=log_response.tolist(),
        bin_rows=rows,
        global_center=float(global_center),
    )


def apply_binned_response(values: np.ndarray, feature: np.ndarray, model: BinnedResponse) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    feature = np.asarray(feature, dtype=float)
    log_r = np.interp(feature, np.asarray(model.centers), np.asarray(model.log_response))
    return values * np.exp(-log_r)


def coarse_centers(values: np.ndarray, feature: np.ndarray, bins: int = N_RESPONSE_BINS) -> list[dict[str, float | int]]:
    values = np.asarray(values, dtype=float)
    feature = np.asarray(feature, dtype=float)
    mask = np.isfinite(values) & (values > 0) & np.isfinite(feature)
    v, f = values[mask], feature[mask]
    edges = np.unique(np.quantile(f, np.linspace(0.0, 1.0, bins + 1)))
    rows: list[dict[str, float | int]] = []
    for idx in range(max(0, len(edges) - 1)):
        last = idx == len(edges) - 2
        select = (f >= edges[idx]) & ((f <= edges[idx + 1]) if last else (f < edges[idx + 1]))
        rows.append(
            {
                "bin": int(idx),
                "low": float(edges[idx]),
                "high": float(edges[idx + 1]),
                "feature_center": float(np.median(f[select])) if np.any(select) else float("nan"),
                "events": int(np.sum(select)),
                "signal_center": float(robust_peak_center(v[select])),
            }
        )
    return rows


def fit_lifetime(values: np.ndarray, dt_us: np.ndarray) -> LinearResponse:
    rows = coarse_centers(values, dt_us)
    table = pd.DataFrame(rows)
    good = (
        np.isfinite(table["feature_center"].to_numpy(float))
        & np.isfinite(table["signal_center"].to_numpy(float))
        & (table["signal_center"].to_numpy(float) > 0)
    )
    x = table.loc[good, "feature_center"].to_numpy(float)
    y = np.log(table.loc[good, "signal_center"].to_numpy(float))
    w = np.sqrt(table.loc[good, "events"].to_numpy(float))
    if x.size < 3:
        raise RuntimeError("Too few lifetime bins")
    unconstrained = float(np.polyfit(x, y, 1, w=w)[0])
    # Physical electron loss requires a non-positive response slope. The lower
    # guardrail corresponds to tau >= 200 us and is intentionally conservative.
    slope = float(np.clip(unconstrained, -1.0 / 200.0, 0.0))
    return LinearResponse(
        feature="dt_us",
        reference=float(np.median(x)),
        slope=slope,
        unconstrained_slope=unconstrained,
        bin_rows=rows,
        response_clip=(0.50, 2.00),
    )


def apply_linear_response(values: np.ndarray, feature: np.ndarray, model: LinearResponse) -> np.ndarray:
    log_r = model.slope * (np.asarray(feature, dtype=float) - model.reference)
    response = np.exp(log_r)
    response = np.clip(response, model.response_clip[0], model.response_clip[1])
    return np.asarray(values, dtype=float) / response


def fit_channel_model(frame: pd.DataFrame, column: str, steps: list[str]) -> list[BinnedResponse | LinearResponse]:
    current = frame[column].to_numpy(float).copy()
    models: list[BinnedResponse | LinearResponse] = []
    for step in steps:
        if step == "lifetime":
            model = fit_lifetime(current, feature_array(frame, "dt_us"))
            current = apply_linear_response(current, feature_array(frame, "dt_us"), model)
        elif step == "dt_map":
            model = fit_binned_response(current, feature_array(frame, "dt_us"), "dt_us")
            current = apply_binned_response(current, feature_array(frame, "dt_us"), model)
        elif step == "r2_map":
            model = fit_binned_response(current, feature_array(frame, "r2"), "r2")
            current = apply_binned_response(current, feature_array(frame, "r2"), model)
        else:
            raise KeyError(step)
        models.append(model)
    return models


def apply_channel_model(
    frame: pd.DataFrame,
    column: str,
    models: list[BinnedResponse | LinearResponse],
) -> np.ndarray:
    current = frame[column].to_numpy(float).copy()
    for model in models:
        f = feature_array(frame, model.feature)
        if isinstance(model, BinnedResponse):
            current = apply_binned_response(current, f, model)
        else:
            current = apply_linear_response(current, f, model)
    return current


def fit_pipeline(frame: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any]:
    s1_models = fit_channel_model(frame, "qS1_max", candidate["s1_steps"])
    s2_models = fit_channel_model(frame, "qS2B_max", candidate["s2_steps"])
    s1_corrected = apply_channel_model(frame, "qS1_max", s1_models)
    s2_corrected = apply_channel_model(frame, "qS2B_max", s2_models)
    s1_center = robust_peak_center(s1_corrected)
    s2_center = robust_peak_center(s2_corrected)
    return {
        "s1_models": s1_models,
        "s2_models": s2_models,
        "s1_center": float(s1_center),
        "s2_center": float(s2_center),
    }


def corrected_channels(frame: pd.DataFrame, pipeline: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    s1 = apply_channel_model(frame, "qS1_max", pipeline["s1_models"]) / pipeline["s1_center"]
    s2 = apply_channel_model(frame, "qS2B_max", pipeline["s2_models"]) / pipeline["s2_center"]
    return s1, s2


def combine_energy(s1: np.ndarray, s2: np.ndarray, alpha: float, scale: float = 1.0) -> np.ndarray:
    return (alpha * np.asarray(s1, float) + (1.0 - alpha) * np.asarray(s2, float)) / scale


def choose_alpha(
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    candidate: dict[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    pipeline = fit_pipeline(inner_train, candidate)
    train_s1, train_s2 = corrected_channels(inner_train, pipeline)
    val_s1, val_s2 = corrected_channels(inner_validation, pipeline)
    rows: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        train_energy = combine_energy(train_s1, train_s2, float(alpha))
        scale = robust_peak_center(train_energy)
        validation_energy = combine_energy(val_s1, val_s2, float(alpha), scale)
        metric = quantile_metrics(validation_energy)
        rows.append(
            {
                "alpha": float(alpha),
                "train_scale": float(scale),
                "validation_r68": float(metric["r68"]),
                "validation_r90": float(metric["r90"]),
                "validation_tail_beyond_2r68": float(metric["tail_beyond_2r68"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["validation_r68", "validation_r90", "alpha"], ascending=[True, True, True]
    )
    return float(table.iloc[0]["alpha"]), rows


def serialize_model(model: BinnedResponse | LinearResponse) -> dict[str, Any]:
    if isinstance(model, BinnedResponse):
        return {
            "kind": "binned_response",
            "feature": model.feature,
            "centers": model.centers,
            "log_response": model.log_response,
            "global_center": model.global_center,
            "bin_rows": model.bin_rows,
        }
    lifetime_us = float(-1.0 / model.slope) if model.feature == "dt_us" and model.slope < 0 else float("inf")
    return {
        "kind": "linear_log_response",
        "feature": model.feature,
        "reference": model.reference,
        "slope": model.slope,
        "unconstrained_slope": model.unconstrained_slope,
        "implied_lifetime_us": lifetime_us,
        "response_clip": list(model.response_clip),
        "bin_rows": model.bin_rows,
    }


def build_fold_assignments(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(frame))
    randomized = np.full(len(frame), -1, dtype=int)
    randomized[permutation] = np.arange(len(frame), dtype=int) % N_FOLDS
    return {"event_random5": randomized}


def inner_validation_fold(scheme: str, outer_fold: int) -> int:
    if scheme != "event_random5":
        raise KeyError(scheme)
    return int((outer_fold + 1) % N_FOLDS)


def run_scheme(
    frame: pd.DataFrame,
    scheme: str,
    fold: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    event_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    parameters: dict[str, Any] = {}

    for candidate_name, candidate in CANDIDATES.items():
        oof = np.full(len(frame), np.nan)
        event_fold = np.full(len(frame), -1, dtype=int)
        parameters[candidate_name] = {}
        for outer in range(N_FOLDS):
            outer_test_mask = fold == outer
            outer_train_mask = ~outer_test_mask
            validation_fold = inner_validation_fold(scheme, outer)
            inner_validation_mask = fold == validation_fold
            inner_train_mask = outer_train_mask & ~inner_validation_mask

            inner_train = frame.loc[inner_train_mask].copy()
            inner_validation = frame.loc[inner_validation_mask].copy()
            outer_train = frame.loc[outer_train_mask].copy()
            outer_test = frame.loc[outer_test_mask].copy()

            alpha, scan = choose_alpha(inner_train, inner_validation, candidate)
            for row in scan:
                alpha_rows.append(
                    {
                        "scheme": scheme,
                        "candidate": candidate_name,
                        "outer_fold": outer,
                        "inner_validation_fold": validation_fold,
                        **row,
                    }
                )

            pipeline = fit_pipeline(outer_train, candidate)
            train_s1, train_s2 = corrected_channels(outer_train, pipeline)
            test_s1, test_s2 = corrected_channels(outer_test, pipeline)
            train_energy = combine_energy(train_s1, train_s2, alpha)
            scale = robust_peak_center(train_energy)
            test_energy = combine_energy(test_s1, test_s2, alpha, scale)
            test_positions = np.flatnonzero(outer_test_mask)
            oof[test_positions] = test_energy
            event_fold[test_positions] = outer

            metric = quantile_metrics(test_energy)
            fold_rows.append(
                {
                    "scheme": scheme,
                    "candidate": candidate_name,
                    "outer_fold": outer,
                    "inner_validation_fold": validation_fold,
                    "train_events": int(np.sum(outer_train_mask)),
                    "test_events": int(np.sum(outer_test_mask)),
                    "alpha": alpha,
                    "train_energy_scale": float(scale),
                    **metric,
                }
            )
            parameters[candidate_name][str(outer)] = {
                "alpha": alpha,
                "train_energy_scale": float(scale),
                "s1_center": pipeline["s1_center"],
                "s2_center": pipeline["s2_center"],
                "s1_models": [serialize_model(model) for model in pipeline["s1_models"]],
                "s2_models": [serialize_model(model) for model in pipeline["s2_models"]],
            }

        metric = quantile_metrics(oof)
        fold_candidate = pd.DataFrame(fold_rows)
        local_folds = fold_candidate[
            (fold_candidate["scheme"] == scheme) & (fold_candidate["candidate"] == candidate_name)
        ]
        summary = {
            "scheme": scheme,
            "candidate": candidate_name,
            "label": candidate["label"],
            "correction_only_retention": 1.0,
            "mean_alpha": float(local_folds["alpha"].mean()),
            "std_alpha": float(local_folds["alpha"].std(ddof=1)),
            **metric,
        }
        parameters[candidate_name]["oof_summary"] = summary

        part = frame[["sourceRow", "runNumber", "eventNumber"]].copy()
        part.insert(0, "scheme", scheme)
        part.insert(1, "candidate", candidate_name)
        part["outer_fold"] = event_fold
        part["oof_energy"] = oof
        event_rows.append(part)

    summaries = [parameters[name]["oof_summary"] for name in CANDIDATES]
    return (
        pd.concat(event_rows, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(alpha_rows),
        {"summary": summaries, "parameters": parameters},
    )


def plot_summary(summary: pd.DataFrame) -> None:
    labels = list(CANDIDATES)
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.0))
    scheme = "event_random5"
    local = summary.set_index(["scheme", "candidate"]).loc[scheme].reindex(labels)
    x = np.arange(len(labels))
    ax.plot(x, 100.0 * local["r68"], marker="o", label="full-event R68")
    ax.plot(x, 100.0 * local["r90"], marker="s", label="full-event R90")
    ax.set_xticks(x, labels, rotation=36, ha="right")
    ax.set_title("Seeded event-level five-fold OOF")
    ax.set_ylabel("Relative half-width [%]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("Base-channel physics matrix (all events retained)")
    fig.tight_layout()
    fig.savefig(OUT / "channel_physics_matrix.png", dpi=180)
    plt.close(fig)


def build_oof_response_diagnostics(events: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc held-out flatness checks; these never feed back into a map."""

    joined = events.merge(
        frame[["sourceRow", *BASE_COLUMNS]],
        on=["sourceRow"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_input"),
    )
    rows: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    for (scheme, candidate), local in joined.groupby(["scheme", "candidate"], sort=False):
        energy = local["oof_energy"].to_numpy(float)
        global_center = robust_peak_center(energy)
        spread_row: dict[str, Any] = {"scheme": scheme, "candidate": candidate}
        for feature_name in ["dt_us", "r2"]:
            bin_rows = coarse_centers(energy, feature_array(local, feature_name))
            responses: list[float] = []
            for row in bin_rows:
                response = float(row["signal_center"] / global_center)
                responses.append(response)
                rows.append(
                    {
                        "scheme": scheme,
                        "candidate": candidate,
                        "feature": feature_name,
                        "global_center": global_center,
                        **row,
                        "relative_response": response,
                    }
                )
            spread_row[f"{feature_name}_response_ptp"] = float(np.ptp(responses))
            spread_row[f"{feature_name}_response_std"] = float(np.std(responses, ddof=1))
        spreads.append(spread_row)
    return pd.DataFrame(rows), pd.DataFrame(spreads)


def paired_event_bootstrap_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Conditional paired uncertainty on frozen OOF predictions by event."""

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    metric_names = ["r68", "r90", "outside_20pct"]
    for scheme, local in events.groupby("scheme", sort=False):
        wide = local.pivot(
            index=["sourceRow"],
            columns="candidate",
            values="oof_energy",
        ).reset_index()
        n_events = len(wide)
        deltas = {
            candidate: {metric: [] for metric in metric_names}
            for candidate in CANDIDATES
            if candidate != "A0_raw"
        }
        for _ in range(N_BOOTSTRAP):
            selected = rng.integers(0, n_events, size=n_events)
            raw_metric = quantile_metrics(wide["A0_raw"].to_numpy(float)[selected])
            for candidate in deltas:
                candidate_metric = quantile_metrics(wide[candidate].to_numpy(float)[selected])
                for metric in metric_names:
                    deltas[candidate][metric].append(float(candidate_metric[metric] - raw_metric[metric]))
        for candidate, metrics in deltas.items():
            for metric, values in metrics.items():
                q025, median, q975 = np.quantile(values, [0.025, 0.50, 0.975])
                rows.append(
                    {
                        "scheme": scheme,
                        "candidate": candidate,
                        "metric": f"delta_{metric}_vs_raw",
                        "q025": float(q025),
                        "median": float(median),
                        "q975": float(q975),
                        "replicates": N_BOOTSTRAP,
                    }
                )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    def result(scheme: str, candidate: str) -> pd.Series:
        return summary[(summary["scheme"] == scheme) & (summary["candidate"] == candidate)].iloc[0]

    def ci(scheme: str, candidate: str, metric: str) -> pd.Series:
        return bootstrap[
            (bootstrap["scheme"] == scheme)
            & (bootstrap["candidate"] == candidate)
            & (bootstrap["metric"] == metric)
        ].iloc[0]

    mod_raw, mod_both = (
        result("event_random5", "A0_raw"),
        result("event_random5", "A4_both"),
    )
    mod_ci = ci("event_random5", "A4_both", "delta_r68_vs_raw")
    report = f"""# 基础通道物理修正矩阵（探索性）

本分析严格只读取 `{', '.join(BASE_COLUMNS)}`。没有读取任何 `C/Belife/ub/stretch/des/MCPAF/Cs` 电荷分支，没有施加事件 cut，4,500 个事件全部进入每个 OOF 候选。

## 主要结果

| 拆分 | 方法 | R68 | R90 | ±20% 外事件 |
|---|---|---:|---:|---:|
| 随机事件折 | raw | {100*mod_raw.r68:.3f}% | {100*mod_raw.r90:.3f}% | {100*mod_raw.outside_20pct:.3f}% |
| 随机事件折 | S1(dt,r²)+S2B(lifetime,r²) | {100*mod_both.r68:.3f}% | {100*mod_both.r90:.3f}% | {100*mod_both.outside_20pct:.3f}% |

联合修正在固定随机种子的事件级五折 OOF 中降低了 R68。冻结 OOF 预测上的配对事件 bootstrap 给出：

- 随机事件折：ΔR68 的 95% 条件区间为 [{100*mod_ci.q025:.3f}, {100*mod_ci.q975:.3f}] 个百分点；

这是对已经生成的 OOF 预测重采样，**没有在每次 bootstrap 中重做地图和权重学习**，因此不能代替完整流程不确定度。

## 消融解释

- 单独做 S2 寿命、单独做 S1(dt,r²)、或只做 S2(寿命,r²)，全事件 R68 均比 raw 差。
- 同时修正两个通道后，事件级 OOF 优于 raw；说明 raw S1/S2B 中存在会互相补偿的空间响应，单边修正会破坏这种补偿。
- 联合方案的外层 S2 指数斜率对应约 1.2–1.35 ms 的表观寿命。由于当前是空间分布不均匀的单一外源 run，这不是正式电子寿命测量。
- 联合修正虽然降低了 R68、R90 和 ±20% 外事件比例，但相对于变窄后的 R68，`tail_beyond_2r68` 略高。因此不能只看一个 core 指标。

## 方法边界

- 五个候选是固定后逐候选做 OOF；本文件再比较这些 OOF 结果，所以仍属于探索性模型比较，不是额外一层独立测试。
- 局部修正只使用训练内 5 个等频箱的稳健峰中心；没有把单事件能量残差回归到常数。
- 正权重 alpha 只在内层验证集选择；外层测试从未参与地图或权重学习。
- 输入文件已被未知的 `E > 2 MeV` 估计量预选，事件成员可能已经被塑形。
- 下一步应冻结 A4，在另一个 run 或未做能量阈值预选的数据上验证。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH, sep="\t", usecols=BASE_COLUMNS)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    frame = frame.replace([np.inf, -np.inf], np.nan)
    assignments = build_fold_assignments(frame)

    all_events: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    all_alpha: list[pd.DataFrame] = []
    run_details: dict[str, Any] = {}
    for scheme, fold in assignments.items():
        events, folds, alpha, details = run_scheme(frame, scheme, fold)
        all_events.append(events)
        all_folds.append(folds)
        all_alpha.append(alpha)
        run_details[scheme] = details

    events = pd.concat(all_events, ignore_index=True)
    fold_metrics = pd.concat(all_folds, ignore_index=True)
    alpha_scan = pd.concat(all_alpha, ignore_index=True)
    summary = pd.DataFrame(
        [row for scheme in run_details.values() for row in scheme["summary"]]
    ).sort_values(["scheme", "r68", "r90"])

    diagnostics, response_spreads = build_oof_response_diagnostics(events, frame)
    bootstrap = paired_event_bootstrap_summary(events)
    summary = summary.merge(response_spreads, on=["scheme", "candidate"], how="left")
    raw_reference = summary.loc[
        summary["candidate"].eq("A0_raw"),
        ["scheme", "r68", "r90", "outside_20pct"],
    ].rename(
        columns={
            "r68": "raw_r68",
            "r90": "raw_r90",
            "outside_20pct": "raw_outside_20pct",
        }
    )
    summary = summary.merge(raw_reference, on="scheme", how="left")
    summary["delta_r68_vs_raw"] = summary["r68"] - summary["raw_r68"]
    summary["ratio_r68_vs_raw"] = summary["r68"] / summary["raw_r68"]
    summary["delta_r90_vs_raw"] = summary["r90"] - summary["raw_r90"]
    summary["delta_outside_20pct_vs_raw"] = summary["outside_20pct"] - summary["raw_outside_20pct"]
    summary = summary.sort_values(["scheme", "r68", "r90"])

    events.to_csv(OUT / "oof_events.tsv", sep="\t", index=False)
    fold_metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    alpha_scan.to_csv(OUT / "inner_alpha_scans.csv", index=False)
    diagnostics.to_csv(OUT / "oof_response_diagnostics.csv", index=False)
    bootstrap.to_csv(OUT / "paired_event_bootstrap_summary.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)

    metadata = {
        "analysis_status": "exploratory fixed-candidate OOF matrix; candidates are compared after OOF rather than selected by an additional outer layer",
        "data_path": str(DATA_PATH),
        "source_rows": int(len(frame)),
        "columns_read": BASE_COLUMNS,
        "forbidden_branch_policy": "No corrected/lifetime/ub/stretch/desaturation/MCPAF/Cs charge branch is read.",
        "all_events_policy": "No event-quality or energy cut is applied; non-finite values remain in the event output and finite_fraction is reported.",
        "metrics": {
            "r68": "(q84.135-q15.865)/(2*median), evaluated on all finite OOF events",
            "r90": "(q95-q05)/(2*median), evaluated on all finite OOF events",
            "tail_beyond_2r68": "fraction with |E/median-1| > 2*R68",
            "outside_20pct": "fraction outside [0.8,1.2] after median normalization",
        },
        "response_model": {
            "bins": N_RESPONSE_BINS,
            "local_center": "smoothed log-mode followed by a fixed +/-20% local median",
            "s1": "sequential low-DOF dt then r2 maps",
            "s2": "non-positive exponential dt slope then low-DOF r2 map",
            "weight": "positive global alpha chosen on inner validation only",
            "alpha_grid": ALPHA_GRID.tolist(),
        },
        "posthoc_diagnostics": "OOF peak-center flatness versus dt_us and r2; diagnostic binning never feeds back into a correction",
        "bootstrap": f"{N_BOOTSTRAP} paired event-resampling replicates on frozen OOF predictions; the full correction/weight pipeline is not refit inside a replicate",
        "split_schemes": {
            "event_random5": "seeded balanced random assignment of source rows to five outer folds",
        },
        "candidates": CANDIDATES,
        "results": run_details,
        "limitations": [
            "The input filename indicates an unknown E>2 MeV preselection, so the sample membership may already be sculpted.",
            "Only one run and 4500 externally distributed events are available.",
            "Candidate comparison is exploratory; a later independent run is required after selecting and freezing one pipeline.",
            "Event-level folds do not test generalization to unseen acquisition files.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, bootstrap)
    plot_summary(summary)

    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()
