"""Standalone robust peak-model evaluation for existing OOF energy estimates.

This script intentionally does not import ``run_analysis.py`` and does not
assume that the peak has a known physical energy.  It compares a truncated
Gaussian signal with three positive, normalized background families inside one
fixed user-supplied window:

* exponential;
* first-degree Bernstein polynomial;
* second-degree Bernstein polynomial.

For each method/stage in ``outputs/nested_cv_events.tsv`` it reports the
all-finite-event non-parametric central-68% resolution, the fixed-window event
count, fitted signal yield, optimizer/boundary diagnostics, a 40-bin Poisson
deviance, and an optional refitted parametric-bootstrap goodness-of-fit p-value.

Examples
--------
Full default evaluation (200 bootstrap toys per fit)::

    python robust_peak_evaluation.py \
        --output outputs/robust_peak_evaluation.json

Fast smoke test of one OOF spectrum::

    python robust_peak_evaluation.py \
        --method exploratory_ub_bottom_stretch \
        --stage corrected --bootstrap 3 --n-starts 5

The default [0.80, 1.20] window is only a reproducible diagnostic choice for
the current normalized OOF arrays.  It is not a claim about the source or peak
energy.  A final analysis should supply a fixed physical window known to be on
the fully efficient plateau of the upstream E > 2 MeV selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betainc, expit, ndtr, ndtri, softmax
from scipy.stats import chi2


DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs" / "nested_cv_events.tsv"
DEFAULT_MODELS = ("gaussian_exp", "gaussian_bernstein1", "gaussian_bernstein2")
QUANTILE_PROBABILITIES = (0.15865, 0.5, 0.84135)
CENTRAL90_PROBABILITIES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    background: str
    degree: int = 0

    @property
    def parameter_names(self) -> tuple[str, ...]:
        common = ("mu", "log_sigma", "logit_signal_fraction")
        if self.background == "exponential":
            return common + ("background_exponent_slope",)
        return common + tuple(f"background_logit_{i}" for i in range(self.degree))


MODEL_SPECS = {
    "gaussian_exp": ModelSpec("gaussian_exp", "exponential"),
    "gaussian_bernstein1": ModelSpec("gaussian_bernstein1", "bernstein", 1),
    "gaussian_bernstein2": ModelSpec("gaussian_bernstein2", "bernstein", 2),
}


@dataclass
class FitResult:
    model: str
    success: bool
    optimizer_success: bool
    optimizer_message: str
    theta: np.ndarray
    nll: float
    mu: float
    sigma: float
    resolution: float
    signal_fraction: float
    signal_yield: float
    background_yield: float
    boundary_degenerate: bool
    boundary_parameters: list[str]
    background_weights: list[float]
    successful_starts: int
    starts_at_best: int
    n_starts: int


def finite_values(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def central68_resolution_all(values: Iterable[float]) -> tuple[float, list[float]]:
    """Return central-68% half-width / median using every finite event."""

    x = finite_values(values)
    if x.size < 3:
        return float("nan"), [float("nan")] * 3
    qlo, qmed, qhi = np.quantile(x, QUANTILE_PROBABILITIES)
    if not np.isfinite(qmed) or qmed == 0:
        return float("nan"), [float(qlo), float(qmed), float(qhi)]
    return float((qhi - qlo) / (2.0 * qmed)), [float(qlo), float(qmed), float(qhi)]


def central90_halfwidth_all(values: Iterable[float]) -> tuple[float, list[float]]:
    """Return central-90% half-width / median using every finite event."""

    x = finite_values(values)
    if x.size < 3:
        return float("nan"), [float("nan")] * 3
    qlo, qmed, qhi = np.quantile(x, CENTRAL90_PROBABILITIES)
    if not np.isfinite(qmed) or qmed == 0:
        return float("nan"), [float(qlo), float(qmed), float(qhi)]
    return float((qhi - qlo) / (2.0 * qmed)), [float(qlo), float(qmed), float(qhi)]


def estimate_mode_and_scale(x: np.ndarray, low: float, high: float) -> tuple[float, float]:
    width = high - low
    counts, edges = np.histogram(x, bins=80, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[int(np.argmax(counts))])
    near = x[np.abs(x - mode) <= 0.18 * width]
    if near.size < 30:
        near = x
    median = float(np.median(near))
    mad_sigma = float(1.4826 * np.median(np.abs(near - median)))
    sigma = float(np.clip(mad_sigma, 0.02 * width, 0.28 * width))
    return mode, sigma


def model_bounds(spec: ModelSpec, low: float, high: float) -> list[tuple[float, float]]:
    width = high - low
    bounds: list[tuple[float, float]] = [
        (low + 0.05 * width, high - 0.05 * width),
        (math.log(width / 500.0), math.log(0.35 * width)),
        (-6.0, 8.0),
    ]
    if spec.background == "exponential":
        # This slope acts on t=(x-low)/(high-low), so its meaning does not
        # change when the fixed window is expressed in different units.
        bounds.append((-25.0, 25.0))
    else:
        bounds.extend([(-12.0, 12.0)] * spec.degree)
    return bounds


def bernstein_weights(theta: np.ndarray, spec: ModelSpec) -> np.ndarray:
    if spec.background != "bernstein":
        return np.empty(0, dtype=float)
    # Fixing the final logit to zero removes the softmax shift degeneracy.
    return softmax(np.r_[theta[3 : 3 + spec.degree], 0.0])


def truncated_gaussian_logpdf(
    x: np.ndarray, mu: float, sigma: float, low: float, high: float
) -> np.ndarray:
    norm = float(ndtr((high - mu) / sigma) - ndtr((low - mu) / sigma))
    if not np.isfinite(norm) or norm <= 0:
        return np.full_like(x, -np.inf)
    z = (x - mu) / sigma
    return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - math.log(sigma) - math.log(norm)


def background_logpdf(
    x: np.ndarray, theta: np.ndarray, spec: ModelSpec, low: float, high: float
) -> np.ndarray:
    width = high - low
    t = (x - low) / width
    if spec.background == "exponential":
        slope = float(theta[3])
        if abs(slope) < 1e-9:
            return np.full_like(x, -math.log(width))
        # slope/expm1(slope) is positive for either sign of slope.
        return (
            math.log(abs(slope))
            + slope * t
            - math.log(width)
            - math.log(abs(float(np.expm1(slope))))
        )

    weights = bernstein_weights(theta, spec)
    if spec.degree == 1:
        basis = np.vstack((2.0 * (1.0 - t), 2.0 * t)) / width
    elif spec.degree == 2:
        basis = np.vstack(
            (3.0 * (1.0 - t) ** 2, 6.0 * t * (1.0 - t), 3.0 * t**2)
        ) / width
    else:  # guarded by the fixed model catalogue
        raise ValueError(f"Unsupported Bernstein degree: {spec.degree}")
    density = weights @ basis
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(density)


def negative_log_likelihood(
    theta: np.ndarray, x: np.ndarray, spec: ModelSpec, low: float, high: float
) -> float:
    mu = float(theta[0])
    sigma = math.exp(float(theta[1]))
    log_signal = -float(np.logaddexp(0.0, -theta[2]))
    log_background_fraction = -float(np.logaddexp(0.0, theta[2]))
    signal = truncated_gaussian_logpdf(x, mu, sigma, low, high)
    background = background_logpdf(x, theta, spec, low, high)
    log_density = np.logaddexp(log_signal + signal, log_background_fraction + background)
    if np.any(~np.isfinite(log_density)):
        return 1e100
    return float(-np.sum(log_density))


def make_starts(
    x: np.ndarray,
    spec: ModelSpec,
    low: float,
    high: float,
    n_starts: int,
    rng: np.random.Generator,
    warm_start: np.ndarray | None = None,
) -> list[np.ndarray]:
    mode, sigma0 = estimate_mode_and_scale(x, low, high)
    base = [mode, math.log(sigma0), math.log(0.75 / 0.25)]
    if spec.background == "exponential":
        base.append(0.0)
    else:
        base.extend([0.0] * spec.degree)
    base_array = np.asarray(base, dtype=float)
    bounds = model_bounds(spec, low, high)

    starts: list[np.ndarray] = []
    if warm_start is not None and len(warm_start) == len(base_array):
        starts.append(np.asarray(warm_start, dtype=float).copy())
    else:
        starts.append(base_array.copy())

    width = high - low
    while len(starts) < max(1, n_starts):
        candidate = base_array.copy()
        candidate[0] += rng.normal(0.0, 0.04 * width)
        candidate[1] += rng.normal(0.0, 0.35)
        candidate[2] += rng.normal(0.0, 0.9)
        if spec.background == "exponential":
            candidate[3] = rng.uniform(-8.0, 12.0)
        else:
            candidate[3:] = rng.normal(0.0, 2.0, spec.degree)
        for i, (lower, upper) in enumerate(bounds):
            padding = max(1e-10, 1e-7 * (upper - lower))
            candidate[i] = np.clip(candidate[i], lower + padding, upper - padding)
        starts.append(candidate)
    return starts


def boundary_diagnostics(
    theta: np.ndarray,
    spec: ModelSpec,
    low: float,
    high: float,
    boundary_fraction: float = 0.002,
) -> tuple[bool, list[str], list[float]]:
    names = spec.parameter_names
    bounds = model_bounds(spec, low, high)
    problems: list[str] = []
    for name, value, (lower, upper) in zip(names, theta, bounds):
        tolerance = boundary_fraction * (upper - lower)
        if value - lower <= tolerance:
            problems.append(f"{name}:lower")
        if upper - value <= tolerance:
            problems.append(f"{name}:upper")

    fraction = float(expit(theta[2]))
    if fraction <= 0.01:
        problems.append("signal_fraction:near_zero")
    if fraction >= 0.99:
        problems.append("signal_fraction:near_one")

    weights = bernstein_weights(theta, spec)
    if weights.size:
        for index, weight in enumerate(weights):
            if weight < 1e-3:
                problems.append(f"background_weight_{index}:near_zero")
    return bool(problems), problems, [float(value) for value in weights]


def fit_model(
    values: Iterable[float],
    spec: ModelSpec,
    low: float,
    high: float,
    n_starts: int,
    rng: np.random.Generator,
    warm_start: np.ndarray | None = None,
    maxiter: int = 1500,
) -> FitResult:
    x = finite_values(values)
    x = x[(x >= low) & (x <= high)]
    parameter_count = len(spec.parameter_names)
    if x.size <= parameter_count + 5:
        empty = np.full(parameter_count, np.nan)
        return FitResult(
            spec.name,
            False,
            False,
            "too few events",
            empty,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            True,
            ["too_few_events"],
            [],
            0,
            0,
            n_starts,
        )

    bounds = model_bounds(spec, low, high)
    starts = make_starts(x, spec, low, high, n_starts, rng, warm_start)
    optimizations = [
        minimize(
            negative_log_likelihood,
            start,
            args=(x, spec, low, high),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
        )
        for start in starts
    ]
    finite_fits = [result for result in optimizations if np.isfinite(result.fun)]
    if not finite_fits:
        empty = np.full(parameter_count, np.nan)
        return FitResult(
            spec.name,
            False,
            False,
            "all optimizer starts returned non-finite objectives",
            empty,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            True,
            ["optimizer_failure"],
            [],
            0,
            0,
            len(starts),
        )

    best = min(finite_fits, key=lambda result: float(result.fun))
    successful = [result for result in finite_fits if bool(result.success)]
    starts_at_best = sum(abs(float(result.fun) - float(best.fun)) <= 1e-4 for result in finite_fits)
    theta = np.asarray(best.x, dtype=float)
    mu = float(theta[0])
    sigma = float(math.exp(theta[1]))
    fraction = float(expit(theta[2]))
    degenerate, problem_parameters, weights = boundary_diagnostics(theta, spec, low, high)
    optimizer_success = bool(best.success)
    success = bool(optimizer_success and np.isfinite(best.fun) and sigma > 0 and mu != 0)
    return FitResult(
        model=spec.name,
        success=success,
        optimizer_success=optimizer_success,
        optimizer_message=str(best.message),
        theta=theta,
        nll=float(best.fun),
        mu=mu,
        sigma=sigma,
        resolution=float(sigma / mu),
        signal_fraction=fraction,
        signal_yield=float(fraction * x.size),
        background_yield=float((1.0 - fraction) * x.size),
        boundary_degenerate=degenerate,
        boundary_parameters=problem_parameters,
        background_weights=weights,
        successful_starts=len(successful),
        starts_at_best=int(starts_at_best),
        n_starts=len(starts),
    )


def component_cdfs(
    edges: np.ndarray, fit: FitResult, spec: ModelSpec, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma = fit.mu, fit.sigma
    gaussian_norm = float(ndtr((high - mu) / sigma) - ndtr((low - mu) / sigma))
    signal = (ndtr((edges - mu) / sigma) - ndtr((low - mu) / sigma)) / gaussian_norm

    t = np.clip((edges - low) / (high - low), 0.0, 1.0)
    if spec.background == "exponential":
        slope = float(fit.theta[3])
        if abs(slope) < 1e-9:
            background = t
        else:
            background = np.expm1(slope * t) / np.expm1(slope)
    else:
        weights = bernstein_weights(fit.theta, spec)
        background = np.zeros_like(t)
        for k, weight in enumerate(weights):
            background += weight * betainc(k + 1, spec.degree - k + 1, t)

    signal[0], signal[-1] = 0.0, 1.0
    background[0], background[-1] = 0.0, 1.0
    return signal, background


def poisson_deviance(
    values: Iterable[float],
    fit: FitResult,
    spec: ModelSpec,
    low: float,
    high: float,
    bins: int = 40,
) -> dict[str, float | int | None]:
    x = finite_values(values)
    x = x[(x >= low) & (x <= high)]
    if not fit.success or x.size == 0:
        return {
            "bins": bins,
            "deviance": None,
            "degrees_of_freedom": None,
            "asymptotic_p_value_diagnostic": None,
            "minimum_expected_bin_count": None,
            "maximum_absolute_pull": None,
        }
    observed, edges = np.histogram(x, bins=bins, range=(low, high))
    signal_cdf, background_cdf = component_cdfs(edges, fit, spec, low, high)
    probabilities = fit.signal_fraction * np.diff(signal_cdf) + (
        1.0 - fit.signal_fraction
    ) * np.diff(background_cdf)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    expected = np.maximum(x.size * probabilities, 1e-12)
    nonzero = observed > 0
    deviance_terms = expected.copy()
    deviance_terms[nonzero] = (
        observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
        - (observed[nonzero] - expected[nonzero])
    )
    deviance = float(2.0 * np.sum(deviance_terms))
    degrees_of_freedom = int(bins - 1 - len(spec.parameter_names))
    pulls = (observed - expected) / np.sqrt(expected)
    return {
        "bins": int(bins),
        "deviance": deviance,
        "degrees_of_freedom": degrees_of_freedom,
        # This is only a quick diagnostic.  The refitted parametric-bootstrap
        # p-value below is the appropriate final GOF result.
        "asymptotic_p_value_diagnostic": float(chi2.sf(deviance, degrees_of_freedom)),
        "minimum_expected_bin_count": float(np.min(expected)),
        "maximum_absolute_pull": float(np.max(np.abs(pulls))),
    }


def sample_truncated_model(
    fit: FitResult,
    spec: ModelSpec,
    size: int,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    is_signal = rng.random(size) < fit.signal_fraction
    sample = np.empty(size, dtype=float)

    n_signal = int(np.sum(is_signal))
    if n_signal:
        cdf_low = float(ndtr((low - fit.mu) / fit.sigma))
        cdf_high = float(ndtr((high - fit.mu) / fit.sigma))
        probability = cdf_low + rng.random(n_signal) * (cdf_high - cdf_low)
        probability = np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        sample[is_signal] = fit.mu + fit.sigma * ndtri(probability)

    n_background = size - n_signal
    if n_background:
        if spec.background == "exponential":
            slope = float(fit.theta[3])
            uniform = rng.random(n_background)
            if abs(slope) < 1e-9:
                t = uniform
            else:
                t = np.log1p(uniform * np.expm1(slope)) / slope
        else:
            weights = bernstein_weights(fit.theta, spec)
            components = rng.choice(spec.degree + 1, size=n_background, p=weights)
            t = np.empty(n_background, dtype=float)
            for k in range(spec.degree + 1):
                selected = components == k
                if np.any(selected):
                    t[selected] = rng.beta(k + 1, spec.degree - k + 1, int(np.sum(selected)))
        sample[~is_signal] = low + (high - low) * t
    return sample


def parametric_bootstrap_gof(
    fit: FitResult,
    spec: ModelSpec,
    observed_values: Iterable[float],
    observed_deviance: float | None,
    low: float,
    high: float,
    bins: int,
    n_bootstrap: int,
    bootstrap_starts: int,
    rng: np.random.Generator,
    progress_label: str,
    quiet: bool,
) -> dict[str, object]:
    x = finite_values(observed_values)
    x = x[(x >= low) & (x <= high)]
    if n_bootstrap <= 0 or not fit.success or observed_deviance is None:
        return {
            "requested": int(n_bootstrap),
            "successful": 0,
            "failed": 0,
            "p_value": None,
            "monte_carlo_standard_error": None,
            "deviance_quantiles_025_500_975": None,
        }

    simulated_deviances: list[float] = []
    failed = 0
    for index in range(n_bootstrap):
        toy = sample_truncated_model(fit, spec, x.size, low, high, rng)
        toy_fit = fit_model(
            toy,
            spec,
            low,
            high,
            n_starts=bootstrap_starts,
            rng=rng,
            warm_start=fit.theta,
            maxiter=800,
        )
        toy_gof = poisson_deviance(toy, toy_fit, spec, low, high, bins)
        toy_deviance = toy_gof["deviance"]
        if toy_fit.success and toy_deviance is not None and np.isfinite(toy_deviance):
            simulated_deviances.append(float(toy_deviance))
        else:
            failed += 1
        if not quiet and (index + 1) % 25 == 0:
            print(
                f"[{progress_label}] bootstrap {index + 1}/{n_bootstrap}",
                file=sys.stderr,
                flush=True,
            )

    if not simulated_deviances:
        return {
            "requested": int(n_bootstrap),
            "successful": 0,
            "failed": int(failed),
            "p_value": None,
            "monte_carlo_standard_error": None,
            "deviance_quantiles_025_500_975": None,
        }
    array = np.asarray(simulated_deviances)
    exceedances = int(np.sum(array >= observed_deviance))
    # The plus-one correction prevents a reported zero p-value.
    p_value = float((exceedances + 1) / (array.size + 1))
    mc_error = float(math.sqrt(p_value * (1.0 - p_value) / (array.size + 1)))
    return {
        "requested": int(n_bootstrap),
        "successful": int(array.size),
        "failed": int(failed),
        "p_value": p_value,
        "monte_carlo_standard_error": mc_error,
        "deviance_quantiles_025_500_975": [
            float(value) for value in np.quantile(array, [0.025, 0.5, 0.975])
        ],
    }


def fit_to_dict(fit: FitResult, spec: ModelSpec) -> dict[str, object]:
    parameter_values = {
        name: float(value) for name, value in zip(spec.parameter_names, fit.theta)
    }
    parameter_values["sigma"] = fit.sigma
    multistart_stable = fit.starts_at_best >= min(2, fit.n_starts)
    quality_accepted = fit.success and multistart_stable and not fit.boundary_degenerate
    event_count = fit.signal_yield + fit.background_yield
    parameter_count = len(spec.parameter_names)
    aic = 2.0 * fit.nll + 2.0 * parameter_count
    if event_count > parameter_count + 1:
        aicc = aic + 2.0 * parameter_count * (parameter_count + 1) / (
            event_count - parameter_count - 1
        )
    else:
        aicc = float("nan")
    bic = (
        2.0 * fit.nll + parameter_count * math.log(event_count)
        if event_count > 0
        else float("nan")
    )
    return {
        "success": fit.success,
        "quality_accepted_no_gof": quality_accepted,
        "multistart_stable": multistart_stable,
        "optimizer_success": fit.optimizer_success,
        "optimizer_message": fit.optimizer_message,
        "negative_log_likelihood": fit.nll,
        "parameter_count": parameter_count,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "parameters": parameter_values,
        "mu": fit.mu,
        "sigma": fit.sigma,
        "core_resolution_sigma_over_mu": fit.resolution,
        "signal_fraction": fit.signal_fraction,
        "fitted_signal_yield_in_window": fit.signal_yield,
        "fitted_background_yield_in_window": fit.background_yield,
        "background_weights": fit.background_weights,
        "boundary_degenerate": fit.boundary_degenerate,
        "boundary_parameters": fit.boundary_parameters,
        "successful_optimizer_starts": fit.successful_starts,
        "starts_reaching_best_nll_within_1e-4": fit.starts_at_best,
        "optimizer_starts": fit.n_starts,
    }


def load_oof_table(path: Path, delimiter: str | None) -> pd.DataFrame:
    if delimiter is not None:
        separator = delimiter
    elif path.suffix.lower() in {".tsv", ".txt"}:
        separator = "\t"
    else:
        separator = ","
    frame = pd.read_csv(path, sep=separator)
    # The strict basic-variable pipeline writes one row per event with these
    # names.  Adapt it in memory so the same evaluator can audit both the
    # legacy benchmark table and the independent reconstruction without
    # creating a second, subtly different peak-fitting implementation.
    independent_columns = {"oof_raw_energy", "oof_independent_energy"}
    if "method" not in frame.columns and independent_columns.issubset(frame.columns):
        frame = frame.rename(
            columns={
                "oof_raw_energy": "oof_baseline_energy",
                "oof_independent_energy": "oof_corrected_energy",
            }
        )
        frame["method"] = "independent_basic_variables"
    # The experiment matrix stores one long-format prediction per method and
    # split scheme.  Convert its raw/nested-winner pair to the same two-stage
    # interface, preserving the event identity and keeping the two validation
    # schemes as distinct datasets.
    long_columns = {"scheme", "method", "energy"}
    required_energy_columns = {"oof_baseline_energy", "oof_corrected_energy"}
    if required_energy_columns.isdisjoint(frame.columns) and long_columns.issubset(frame.columns):
        selected = frame[frame["method"].isin(["raw", "nested_winner"])].copy()
        identity = [
            column
            for column in ["scheme", "sourceRow", "runNumber", "eventNumber", "outer_fold"]
            if column in selected.columns
        ]
        wide = selected.pivot(index=identity, columns="method", values="energy").reset_index()
        if not {"raw", "nested_winner"}.issubset(wide.columns):
            raise ValueError("Long-format matrix input lacks a complete raw/nested_winner pair")
        wide = wide.rename(
            columns={
                "raw": "oof_baseline_energy",
                "nested_winner": "oof_corrected_energy",
            }
        )
        wide["method"] = "basic_matrix_" + wide["scheme"].astype(str)
        frame = wide
    channel_columns = {"scheme", "candidate", "oof_energy"}
    if required_energy_columns.isdisjoint(frame.columns) and channel_columns.issubset(frame.columns):
        selected = frame[frame["candidate"].isin(["A0_raw", "A4_both"])].copy()
        identity = [
            column
            for column in ["scheme", "sourceRow", "runNumber", "eventNumber", "outer_fold"]
            if column in selected.columns
        ]
        wide = selected.pivot(index=identity, columns="candidate", values="oof_energy").reset_index()
        if not {"A0_raw", "A4_both"}.issubset(wide.columns):
            raise ValueError("Channel-physics input lacks a complete A0_raw/A4_both pair")
        wide = wide.rename(
            columns={
                "A0_raw": "oof_baseline_energy",
                "A4_both": "oof_corrected_energy",
            }
        )
        wide["method"] = "channel_physics_" + wide["scheme"].astype(str)
        frame = wide
    return frame


def select_methods(frame: pd.DataFrame, requested: Sequence[str] | None) -> list[str]:
    available = [str(value) for value in frame["method"].dropna().unique()]
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested methods not found: {missing}; available={available}")
    return list(requested)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Optional explicit delimiter; defaults to tab for .tsv/.txt and comma otherwise.",
    )
    parser.add_argument("--window-low", type=float, default=0.80)
    parser.add_argument("--window-high", type=float, default=1.20)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument(
        "--stage", choices=("all", "baseline", "corrected"), default="all"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--n-starts", type=int, default=12)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Refitted parametric-bootstrap toys per spectrum/model; use 0 to disable.",
    )
    parser.add_argument("--bootstrap-starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10972)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.window_low < args.window_high:
        raise ValueError("--window-low must be smaller than --window-high")
    if args.bins <= len(max((MODEL_SPECS[name] for name in args.models), key=lambda s: len(s.parameter_names)).parameter_names) + 1:
        raise ValueError("--bins is too small for the requested model parameter count")
    if args.n_starts < 1 or args.bootstrap_starts < 1 or args.bootstrap < 0:
        raise ValueError("start counts must be positive and bootstrap must be non-negative")

    frame = load_oof_table(args.input, args.delimiter)
    required = {"method", "oof_baseline_energy", "oof_corrected_energy"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")
    methods = select_methods(frame, args.method)
    stages = ("baseline", "corrected") if args.stage == "all" else (args.stage,)
    rng = np.random.default_rng(args.seed)

    payload: dict[str, object] = {
        "metadata": {
            "input": str(args.input.resolve()),
            "fixed_window": [args.window_low, args.window_high],
            "poisson_deviance_bins": args.bins,
            "models": list(args.models),
            "optimizer_starts": args.n_starts,
            "parametric_bootstrap_requested_per_fit": args.bootstrap,
            "bootstrap_optimizer_starts": args.bootstrap_starts,
            "seed": args.seed,
            "interpretation": (
                "Conditional diagnostic for the supplied OOF sample; no source or absolute "
                "peak energy is assumed, and upstream E>2MeV selection bias is not removed."
            ),
        },
        "datasets": [],
    }

    for method in methods:
        method_frame = frame.loc[frame["method"].astype(str) == method]
        for stage in stages:
            column = f"oof_{stage}_energy"
            all_values = finite_values(method_frame[column].to_numpy(float))
            in_window = all_values[
                (all_values >= args.window_low) & (all_values <= args.window_high)
            ]
            r68, quantiles = central68_resolution_all(all_values)
            r90, quantiles90 = central90_halfwidth_all(all_values)
            dataset_result: dict[str, object] = {
                "method": method,
                "stage": stage,
                "energy_column": column,
                "all_finite_event_count": int(all_values.size),
                "all_event_central68_quantiles": quantiles,
                "all_event_nonparametric_r68": r68,
                "all_event_central90_quantiles": quantiles90,
                "all_event_nonparametric_r90_halfwidth": r90,
                "fixed_window_event_count": int(in_window.size),
                "fixed_window_fraction_of_finite_events": float(in_window.size / all_values.size)
                if all_values.size
                else float("nan"),
                "fits": [],
            }
            for model_name in args.models:
                spec = MODEL_SPECS[model_name]
                label = f"{method}/{stage}/{model_name}"
                if not args.quiet:
                    print(f"[{label}] fitting {in_window.size} events", file=sys.stderr, flush=True)
                fit = fit_model(
                    all_values,
                    spec,
                    args.window_low,
                    args.window_high,
                    n_starts=args.n_starts,
                    rng=rng,
                )
                gof = poisson_deviance(
                    all_values,
                    fit,
                    spec,
                    args.window_low,
                    args.window_high,
                    bins=args.bins,
                )
                bootstrap = parametric_bootstrap_gof(
                    fit,
                    spec,
                    all_values,
                    gof["deviance"],
                    args.window_low,
                    args.window_high,
                    args.bins,
                    args.bootstrap,
                    args.bootstrap_starts,
                    rng,
                    label,
                    args.quiet,
                )
                bootstrap_p_value = bootstrap["p_value"]
                # A handful of toys is useful for a smoke test, but is not
                # enough to make a p>0.05 acceptance decision.  One hundred
                # successful refits is the minimum for emitting that decision;
                # the CLI default requests 200.
                if int(bootstrap["successful"]) >= 100 and bootstrap_p_value is not None:
                    quality_accepted_with_gof: bool | None = (
                        fit.success
                        and fit.starts_at_best >= min(2, fit.n_starts)
                        and not fit.boundary_degenerate
                        and float(bootstrap_p_value) > 0.05
                    )
                else:
                    quality_accepted_with_gof = None
                model_result = {
                    "model": model_name,
                    **fit_to_dict(fit, spec),
                    "quality_accepted_with_bootstrap_gof_p_gt_0_05": quality_accepted_with_gof,
                    "poisson_deviance_gof": gof,
                    "parametric_bootstrap_gof": bootstrap,
                }
                dataset_result["fits"].append(model_result)
            payload["datasets"].append(dataset_result)

    serializable = json_ready(payload)
    rendered = json.dumps(serializable, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"Wrote {args.output.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
