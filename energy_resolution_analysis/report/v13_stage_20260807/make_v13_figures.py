"""Create report-ready diagnostic figures for the anchored v13 candidate."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "v13_anchored_20260807"
ASSETS = HERE / "assets"
CAL_INPUT = ROOT.parent / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"

BLUE = "#2455C3"
RED = "#D62728"
NAVY = "#17365D"
GOLD = "#E6A700"
GRAY = "#6B7280"


def setup() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_data():
    bg = pd.read_csv(RESULTS / "background_events.csv")
    cal = pd.read_csv(RESULTS / "calibration_oof_events.csv")
    search = pd.read_csv(RESULTS / "candidate_search.csv")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(
        CAL_INPUT,
        sep=r"\s+",
        usecols=["runNumber", "fileNumber", "eventNumber", "dt"],
    )
    cal = cal.merge(raw, on=["runNumber", "fileNumber", "eventNumber"], how="left")
    return bg, cal, search, summary


def spectrum_figure(bg: pd.DataFrame) -> None:
    before = bg["energy_cor_kev"].to_numpy(float)
    after = bg["candidate_v13_energy_kev"].to_numpy(float)
    valid = np.isfinite(before) & np.isfinite(after)
    before, after = before[valid], after[valid]

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.2), gridspec_kw={"height_ratios": [1.15, 1]})
    edges = np.arange(500, 3805, 5)
    axes[0].hist(before, bins=edges, histtype="step", lw=1.0, color=BLUE, label=r"Before: cubic $E_{cor}$")
    axes[0].hist(after, bins=edges, histtype="step", lw=1.0, color=RED, label=r"After: v13 $E_{v13}$")
    axes[0].set(title="v13 background-spectrum safety check", ylabel="Events / 5 keV", xlim=(500, 3800))
    axes[0].legend(loc="upper right")

    zoom = np.arange(2500, 3002.5, 2.5)
    axes[1].hist(before, bins=zoom, histtype="step", lw=1.0, color=BLUE, label=r"Before: cubic $E_{cor}$")
    axes[1].hist(after, bins=zoom, histtype="step", lw=1.0, color=RED, label=r"After: v13 $E_{v13}$")
    axes[1].axvline(2600, color=GRAY, ls="--", lw=0.8)
    axes[1].axvline(2888, color=GRAY, ls="--", lw=0.8)
    axes[1].set(title="High-energy zoom: no new narrow peak", xlabel="Energy [keV]", ylabel="Events / 2.5 keV", xlim=(2500, 3000))
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "v13_spectrum_full_zoom.png")


def event_correction_figure(bg: pd.DataFrame, summary: dict) -> None:
    before = bg["energy_cor_kev"].to_numpy(float)
    after = bg["candidate_v13_energy_kev"].to_numpy(float)
    valid = np.isfinite(before) & np.isfinite(after) & (before > 0)
    before, after = before[valid], after[valid]
    rel = 100.0 * (after / before - 1.0)
    rng = np.random.default_rng(130807)
    take = rng.choice(len(before), size=min(26000, len(before)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65))
    axes[0].scatter(before[take], after[take], s=2, alpha=0.12, color=BLUE, rasterized=True)
    lim = (500, 3800)
    axes[0].plot(lim, lim, color="black", ls="--", lw=0.9, label=r"$y=x$")
    axes[0].set(xlim=lim, ylim=lim, xlabel=r"Before: cubic $E_{cor}$ [keV]", ylabel=r"After: v13 $E_{v13}$ [keV]", title="Event-by-event comparison")
    axes[0].legend(loc="upper left")

    nz = np.abs(rel) > 1e-10
    axes[1].hist(rel[nz], bins=np.linspace(-0.25, 0.25, 101), histtype="stepfilled", alpha=0.34, color=RED, edgecolor=RED)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel=r"Relative correction $(E_{v13}/E_{cor}-1)$ [%]", ylabel="Corrected events", title="Applied correction amplitude")
    axes[1].text(
        0.04, 0.96,
        "Applied on background: {:.2f}%\nStrong fallback: {:.2f}%\nOOD: {:.2f}%".format(
            100 * summary["winner"]["background_applied"],
            100 * summary["winner"]["background_strong"],
            100 * summary["winner"]["background_ood"],
        ),
        transform=axes[1].transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#B7C4D3", "alpha": 0.92},
    )
    fig.tight_layout()
    save(fig, "v13_event_correction.png")


def drift_figure(cal: pd.DataFrame) -> None:
    valid = (
        np.isfinite(cal["energy_cor_kev"])
        & np.isfinite(cal["candidate_v13_oof_energy_kev"])
        & np.isfinite(cal["dt"])
        & (cal["energy_cor_kev"] > 0)
    )
    frame = cal.loc[valid].copy()
    frame["dt_us"] = frame["dt"] / 1000.0
    frame["rel_pct"] = 100.0 * (frame["candidate_v13_oof_energy_kev"] / frame["energy_cor_kev"] - 1.0)
    edges = np.linspace(frame["dt_us"].min(), frame["dt_us"].max(), 17)
    frame["bin"] = pd.cut(frame["dt_us"], edges, include_lowest=True)
    grouped = frame.groupby("bin", observed=True)["rel_pct"]
    x = np.array([interval.mid for interval in grouped.groups])
    med = grouped.median().to_numpy()
    q16 = grouped.quantile(0.16).to_numpy()
    q84 = grouped.quantile(0.84).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    sample = frame.sample(min(4499, len(frame)), random_state=13)
    axes[0].scatter(sample["dt_us"], sample["rel_pct"], s=5, alpha=0.16, color=BLUE, rasterized=True)
    axes[0].plot(x, med, color=RED, marker="o", ms=3, lw=1.3, label="Binned median")
    axes[0].fill_between(x, q16, q84, color=RED, alpha=0.12, label="16%-84%")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(xlabel=r"Drift time $t_d$ [$\mu$s]", ylabel="Relative correction [%]", title="v13 correction versus drift time")
    axes[0].legend(loc="upper right")

    before_ratio = cal.loc[valid, "energy_cor_kev"].to_numpy() / 2614.5
    after_ratio = cal.loc[valid, "candidate_v13_oof_energy_kev"].to_numpy() / 2614.5
    bins = np.linspace(0.75, 1.05, 90)
    axes[1].hist(before_ratio, bins=bins, histtype="step", lw=1.2, color=BLUE, label=r"Before: $E_{cor}$")
    axes[1].hist(after_ratio, bins=bins, histtype="step", lw=1.2, color=RED, label=r"After: $E_{v13}$")
    axes[1].axvline(1, color="black", ls="--", lw=0.8, label="2614.5 keV")
    axes[1].set(xlabel="Energy / 2614.5 keV", ylabel="Events", title="Calibration OOF distribution")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "v13_drift_calibration.png")


def metrics_and_search_figure(search: pd.DataFrame, summary: dict) -> None:
    base = summary["calibration_base_protocols"]
    cand = summary["calibration_candidate_protocols"]
    labels = [r"$\sigma/\mu$", r"$R_{68}$", r"$R_{90}$"]
    before = 100 * np.array([base["sigma_median"], base["r68"], base["r90"]])
    after = 100 * np.array([cand["sigma_median"], cand["r68"], cand["r90"]])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65))
    x = np.arange(3)
    w = 0.36
    axes[0].bar(x - w / 2, before, w, color=BLUE, alpha=0.82, label=r"Before: $E_{cor}$")
    axes[0].bar(x + w / 2, after, w, color=RED, alpha=0.82, label=r"After: $E_{v13}$")
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="Width metric [%]", title="Calibration OOF width metrics")
    axes[0].legend(loc="upper left")
    for i, (b, a) in enumerate(zip(before, after)):
        axes[0].text(i, max(b, a) + 0.18, f"{b:.4f} -> {a:.4f}", ha="center", va="bottom", fontsize=7.5)

    applied = 100 * search["background_applied"].to_numpy(float)
    improve = 100 * search["relative_sigma_improvement"].to_numpy(float)
    safe = search["safe"].astype(bool).to_numpy()
    axes[1].scatter(applied[~safe], improve[~safe], s=24, alpha=0.55, color=GRAY, label="Rejected candidates")
    axes[1].scatter(applied[safe], improve[safe], s=68, marker="*", color=GOLD, edgecolor=NAVY, linewidth=0.7, label="Selected safe candidate", zorder=3)
    axes[1].axvline(10, color=NAVY, ls="--", lw=0.9, label="10% coverage gate")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(xlabel="Background correction applied [%]", ylabel=r"Relative $\sigma/\mu$ improvement [%]", title="Search trade-off: improvement versus coverage")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save(fig, "v13_metrics_search.png")


def main() -> None:
    setup()
    bg, cal, search, summary = load_data()
    spectrum_figure(bg)
    event_correction_figure(bg, summary)
    drift_figure(cal)
    metrics_and_search_figure(search, summary)
    print(f"Wrote v13 figures to {ASSETS}")


if __name__ == "__main__":
    main()
