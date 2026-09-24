from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
OUT = Path(__file__).resolve().parent / "figures"

COLORS = {
    "raw": "#5B6F8A",
    "channel": "#2A9D8F",
    "challenger": "#E76F51",
    "accent": "#E9C46A",
}


def r68(values: np.ndarray) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    qlo, qmed, qhi = np.quantile(x, [0.15865, 0.5, 0.84135])
    return float((qhi - qlo) / (2.0 * qmed))


def normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, float)
    return x / np.nanmedian(x)


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def data_overview() -> None:
    cols = ["eventNumber", "qS1_max", "qS2B_max", "dt", "xS2T_max", "yS2T_max"]
    frame = pd.read_csv(DATA, sep="\t", usecols=cols)
    dt_us = frame["dt"].to_numpy(float) / 1000.0
    radius = np.hypot(frame["xS2T_max"], frame["yS2T_max"])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    scatter = axes[0, 0].scatter(
        frame["qS1_max"],
        frame["qS2B_max"],
        c=dt_us,
        s=9,
        alpha=0.42,
        cmap="viridis",
        linewidths=0,
    )
    axes[0, 0].set_xlabel(r"raw $S1$ [p.e.]")
    axes[0, 0].set_ylabel(r"raw $S2_{\mathrm{B}}$ [p.e.]")
    axes[0, 0].set_title(r"Raw $S1$--$S2_{\mathrm{B}}$ anti-correlation")
    colorbar = fig.colorbar(scatter, ax=axes[0, 0], pad=0.02)
    colorbar.set_label(r"drift time $t_{\mathrm{d}}$ [$\mu$s]")

    axes[0, 1].hist(dt_us, bins=35, color=COLORS["channel"], alpha=0.9)
    axes[0, 1].set_xlabel(r"$t_{\mathrm{d}}$ [$\mu$s]")
    axes[0, 1].set_ylabel("events")
    axes[0, 1].set_title("Drift-time coverage")

    axes[1, 0].hist(radius, bins=35, color=COLORS["accent"], alpha=0.95)
    axes[1, 0].set_xlabel(r"raw reconstructed radius $r$ [mm]")
    axes[1, 0].set_ylabel("events")
    axes[1, 0].set_title("Non-uniform radial coverage")

    axes[1, 1].hist(frame["eventNumber"], bins=35, color=COLORS["raw"], alpha=0.9)
    axes[1, 1].set_xlabel("eventNumber (identifier only)")
    axes[1, 1].set_ylabel("events")
    axes[1, 1].set_title("Event-identifier coverage")
    fig.suptitle("Run 10972: basic-variable data overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "data_overview.png")
    plt.close(fig)


def pivot_channel() -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(ROOT / "channel_physics_outputs" / "oof_events.tsv", sep="\t")
    result: dict[str, pd.DataFrame] = {}
    for scheme, local in frame.groupby("scheme"):
        wide = local.pivot(
            index=["sourceRow"],
            columns="candidate",
            values="oof_energy",
        ).reset_index()
        result[str(scheme)] = wide
    return result


def pivot_basic() -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(ROOT / "basic_matrix_outputs" / "oof_predictions.csv")
    frame = frame[frame["method"].isin(["raw", "nested_winner"])]
    result: dict[str, pd.DataFrame] = {}
    for scheme, local in frame.groupby("scheme"):
        wide = local.pivot(
            index=["sourceRow"],
            columns="method",
            values="energy",
        ).reset_index()
        result[str(scheme)] = wide
    return result


def oof_spectra() -> None:
    channel = pivot_channel()
    basic = pivot_basic()
    bins = np.linspace(0.60, 1.35, 76)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharex=True)
    c = channel["event_random5"]
    raw = normalize(c["A0_raw"].to_numpy(float))
    corrected = normalize(c["A4_both"].to_numpy(float))
    axes[0].hist(raw, bins=bins, histtype="step", lw=1.7, color=COLORS["raw"],
                 label=fr"raw: $R_{{68}}={100*r68(raw):.3f}\%$")
    axes[0].hist(corrected, bins=bins, histtype="step", lw=1.9, color=COLORS["channel"],
                 label=fr"A4: $R_{{68}}={100*r68(corrected):.3f}\%$")
    axes[0].set_title("Channel-physics A4")
    axes[0].set_ylabel("events / bin")
    axes[0].legend(frameon=False)

    b = basic["event_random5"]
    raw_b = normalize(b["raw"].to_numpy(float))
    winner = normalize(b["nested_winner"].to_numpy(float))
    axes[1].hist(raw_b, bins=bins, histtype="step", lw=1.7, color=COLORS["raw"],
                 label=fr"raw: $R_{{68}}={100*r68(raw_b):.3f}\%$")
    axes[1].hist(winner, bins=bins, histtype="step", lw=1.9, color=COLORS["challenger"],
                 label=fr"challenger: $R_{{68}}={100*r68(winner):.3f}\%$")
    axes[1].set_title("Combined-response challenger")
    axes[1].set_ylabel("events / bin")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel(r"OOF relative energy $E/Q_{50}$")

    fig.suptitle("Out-of-fold full-event energy spectra", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "oof_spectra.png")
    plt.close(fig)


def channel_flatness() -> None:
    frame = pd.read_csv(ROOT / "channel_physics_outputs" / "oof_response_diagnostics.csv")
    frame = frame[
        (frame["scheme"] == "event_random5")
        & frame["candidate"].isin(["A0_raw", "A4_both"])
        & frame["feature"].isin(["dt_us", "r2"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, feature in zip(axes, ["dt_us", "r2"]):
        for candidate, label, color in [
            ("A0_raw", "raw", COLORS["raw"]),
            ("A4_both", "A4 corrected", COLORS["channel"]),
        ]:
            local = frame[(frame["feature"] == feature) & (frame["candidate"] == candidate)]
            x = local["feature_center"].to_numpy(float)
            if feature == "r2":
                x = np.sqrt(np.maximum(x, 0.0))
            ax.plot(x, local["relative_response"], marker="o", lw=1.7, color=color, label=label)
        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel("held-out relative peak center")
        ax.legend(frameon=False)
        ax.set_title("OOF response versus " + (r"$t_{\mathrm{d}}$" if feature == "dt_us" else r"$r$"))
        ax.set_xlabel(r"$t_{\mathrm{d}}$ [$\mu$s]" if feature == "dt_us" else r"raw radius $r$ [mm]")
    fig.suptitle("A4 reduces held-out drift-time and radial response residuals", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "channel_flatness.png")
    plt.close(fig)


def result_summary() -> None:
    channel = pd.read_csv(ROOT / "channel_physics_outputs" / "summary.csv")
    basic = pd.read_csv(ROOT / "basic_matrix_outputs" / "summary_metrics.csv")
    rows = []
    scheme = "event_random5"
    raw = channel[(channel["scheme"] == scheme) & (channel["candidate"] == "A0_raw")].iloc[0]
    after = channel[(channel["scheme"] == scheme) & (channel["candidate"] == "A4_both")].iloc[0]
    rows.append(("A4", 100 * raw.r68, 100 * after.r68, 100 * raw.r90, 100 * after.r90))
    raw_b = basic[(basic["scheme"] == scheme) & (basic["method"] == "raw")].iloc[0]
    after_b = basic[(basic["scheme"] == scheme) & (basic["method"] == "nested_winner")].iloc[0]
    rows.append(("challenger", raw_b.r68_percent, after_b.r68_percent, raw_b.r90_percent, after_b.r90_percent))

    names = [row[0] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for ax, before_index, after_index, title in [
        (axes[0], 1, 2, r"Full-event $R_{68}$"),
        (axes[1], 3, 4, r"Full-event $R_{90}$"),
    ]:
        ax.bar(x - width / 2, [row[before_index] for row in rows], width, color=COLORS["raw"], label="raw")
        ax.bar(x + width / 2, [row[after_index] for row in rows], width, color=COLORS["channel"], label="corrected")
        ax.set_xticks(x, names, rotation=22, ha="right")
        ax.set_ylabel("relative half-width [%]")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("Independent basic-variable reconstruction results", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "result_summary.png")
    plt.close(fig)


def fold_stability_result() -> None:
    """Show whether the event-level result is driven by one outer fold."""
    channel = pd.read_csv(ROOT / "channel_physics_outputs" / "fold_metrics.csv")
    channel = channel[
        (channel["scheme"] == "event_random5")
        & channel["candidate"].isin(["A0_raw", "A4_both"])
    ]
    basic = pd.read_csv(ROOT / "basic_matrix_outputs" / "fold_metrics.csv")
    basic = basic[
        (basic["scheme"] == "event_random5")
        & basic["method"].isin(["raw", "nested_winner"])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    specifications = [
        (
            axes[0],
            channel,
            "candidate",
            "A0_raw",
            "A4_both",
            "A4 channel-physics pipeline",
            COLORS["channel"],
        ),
        (
            axes[1],
            basic,
            "method",
            "raw",
            "nested_winner",
            "Combined-response challenger",
            COLORS["challenger"],
        ),
    ]
    x = np.arange(5)
    width = 0.34
    for ax, frame, key, raw_name, corrected_name, title, corrected_color in specifications:
        raw = (
            frame[frame[key] == raw_name]
            .sort_values("outer_fold")["r68"]
            .to_numpy(float)
            * 100.0
        )
        corrected = (
            frame[frame[key] == corrected_name]
            .sort_values("outer_fold")["r68"]
            .to_numpy(float)
            * 100.0
        )
        ax.bar(x - width / 2, raw, width, color=COLORS["raw"], label="raw")
        ax.bar(x + width / 2, corrected, width, color=corrected_color, label="corrected")
        for index, (before, after) in enumerate(zip(raw, corrected)):
            ax.text(
                index,
                max(before, after) + 0.16,
                f"{after - before:+.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#334155",
            )
        ax.set_xticks(x, [f"fold {i}" for i in x])
        ax.set_xlabel("held-out event fold")
        ax.set_title(title)
        ax.legend(frameon=False, ncol=2, loc="lower left")
        ax.set_ylim(0.0, 9.6)
    axes[0].set_ylabel(r"OOF $R_{68}$ [\%]")
    fig.suptitle(r"Event-level outer-fold stability in $R_{68}$", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fold_stability_result.png")
    plt.close(fig)


def frozen_oof_bootstrap() -> None:
    """Forest plot for file-resampling intervals on frozen OOF predictions."""
    channel = pd.read_csv(
        ROOT / "channel_physics_outputs" / "paired_event_bootstrap_summary.csv"
    )
    channel = channel[
        (channel["candidate"] == "A4_both")
        & channel["metric"].isin(["delta_r68_vs_raw", "delta_r90_vs_raw"])
    ].copy()
    channel["pipeline"] = "A4"
    channel["metric_short"] = channel["metric"].map(
        {"delta_r68_vs_raw": "R68", "delta_r90_vs_raw": "R90"}
    )
    channel["scheme_short"] = channel["scheme"].map(
        {"event_random5": "event OOF"}
    )

    basic = pd.read_csv(
        ROOT / "basic_matrix_outputs" / "paired_event_bootstrap_summary.csv"
    )
    basic = basic[basic["metric"].isin(["delta_r68", "delta_r90"])].copy()
    basic["pipeline"] = "challenger"
    basic["metric_short"] = basic["metric"].map(
        {"delta_r68": "R68", "delta_r90": "R90"}
    )
    basic["scheme_short"] = basic["scheme"].map(
        {"event_random5": "event OOF"}
    )

    combined = pd.concat(
        [
            channel[
                [
                    "pipeline",
                    "metric_short",
                    "scheme_short",
                    "q025",
                    "median",
                    "q975",
                ]
            ],
            basic[
                [
                    "pipeline",
                    "metric_short",
                    "scheme_short",
                    "q025",
                    "median",
                    "q975",
                ]
            ],
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    ylabels = ["A4 / event OOF", "challenger / event OOF"]
    for ax, metric in zip(axes, ["R68", "R90"]):
        local = combined[combined["metric_short"] == metric].copy()
        points = []
        lows = []
        highs = []
        colors = []
        for label in ylabels:
            pipeline, scheme = [part.strip() for part in label.split("/")]
            row = local[
                (local["pipeline"] == pipeline)
                & (local["scheme_short"] == scheme)
            ].iloc[0]
            points.append(100.0 * float(row["median"]))
            lows.append(100.0 * float(row["q025"]))
            highs.append(100.0 * float(row["q975"]))
            colors.append(
                COLORS["channel"] if pipeline == "A4" else COLORS["challenger"]
            )
        y = np.arange(len(ylabels))[::-1]
        for yi, point, low, high, color in zip(y, points, lows, highs, colors):
            ax.errorbar(
                point,
                yi,
                xerr=[[point - low], [high - point]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3.5,
                lw=1.8,
                markersize=6,
            )
            ax.text(high + 0.08, yi, f"{point:.2f}", va="center", fontsize=8.5)
        ax.axvline(0.0, color="#111827", lw=0.9, ls="--")
        ax.set_yticks(y, ylabels)
        ax.set_xlabel("corrected minus raw [percentage points]")
        ax.set_title(rf"$\Delta {metric}$: median and 95\% interval")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-4.35, 0.25)
    fig.suptitle("Paired event bootstrap on frozen OOF predictions (1000 replicates)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "frozen_oof_bootstrap.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    data_overview()
    oof_spectra()
    channel_flatness()
    result_summary()
    fold_stability_result()
    frozen_oof_bootstrap()
    print(f"Wrote report figures to {OUT}")


if __name__ == "__main__":
    main()
