from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "light_ana_run10972_finalSS_Egt2MeV_scalar.txt"
OUT = HERE / "figures"

DEEP_BLUE = "#17324D"
TEAL = "#0B7A75"
GOLD = "#C58A19"
CORAL = "#C95F4A"
GRAY = "#667085"
LIGHT_BLUE = "#EAF2F6"


def setup_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        chinese_font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        chinese_font = "SimHei"

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": chinese_font,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#B8C2CC",
            "grid.color": "#D9E0E6",
            "grid.alpha": 0.65,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=DEEP_BLUE,
        va="top",
    )


def load_data() -> pd.DataFrame:
    # 只从表头统计源字段总数；实际分析严格读取七列白名单。
    header = pd.read_csv(DATA, sep="\t", nrows=0)
    selected_columns = [
        "runNumber",
        "eventNumber",
        "qS1_max",
        "qS2B_max",
        "dt",
        "xS2T_max",
        "yS2T_max",
    ]
    frame = pd.read_csv(DATA, sep="\t", usecols=selected_columns, low_memory=False)
    frame.insert(0, "sourceRow", np.arange(len(frame), dtype=int))
    frame.attrs["source_field_count"] = len(header.columns)
    return frame


def make_overview(frame: pd.DataFrame) -> None:
    dt_us = frame["dt"].to_numpy(float) / 1000.0
    radius = np.hypot(frame["xS2T_max"], frame["yS2T_max"])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1))

    scatter = axes[0, 0].scatter(
        frame["qS1_max"] / 1000.0,
        frame["qS2B_max"] / 1000.0,
        c=dt_us,
        s=10,
        alpha=0.46,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    axes[0, 0].set_xlabel(r"未在本项目中校正的 $S1$ [$10^3$ p.e.]")
    axes[0, 0].set_ylabel(r"未在本项目中校正的 $S2_{\mathrm{B}}$ [$10^3$ p.e.]")
    axes[0, 0].set_title(r"$S1$--$S2_{\mathrm{B}}$ 的联合分布")
    colorbar = fig.colorbar(scatter, ax=axes[0, 0], pad=0.02)
    colorbar.set_label(r"漂移时间 $t_{\mathrm{d}}$ [$\mu$s]")
    add_panel_label(axes[0, 0], "(a)")

    axes[0, 1].hist(dt_us, bins=35, color=TEAL, alpha=0.88, edgecolor="white", linewidth=0.35)
    axes[0, 1].axvline(np.median(dt_us), color=DEEP_BLUE, ls="--", lw=1.5, label=f"中位数 {np.median(dt_us):.1f} μs")
    axes[0, 1].set_xlabel(r"漂移时间 $t_{\mathrm{d}}$ [$\mu$s]")
    axes[0, 1].set_ylabel("事件数")
    axes[0, 1].set_title("漂移时间覆盖")
    axes[0, 1].legend(frameon=False)
    add_panel_label(axes[0, 1], "(b)")

    axes[1, 0].hist(radius, bins=38, color=GOLD, alpha=0.86, edgecolor="white", linewidth=0.35)
    axes[1, 0].axvline(np.median(radius), color=DEEP_BLUE, ls="--", lw=1.5, label=f"中位数 {np.median(radius):.1f} mm")
    axes[1, 0].set_xlabel(r"输入横向位置给出的半径 $r$ [mm]")
    axes[1, 0].set_ylabel("事件数")
    axes[1, 0].set_title("非均匀的径向覆盖")
    axes[1, 0].legend(frameon=False)
    add_panel_label(axes[1, 0], "(c)")

    xy_density = axes[1, 1].hexbin(
        frame["xS2T_max"],
        frame["yS2T_max"],
        gridsize=36,
        mincnt=1,
        cmap="magma",
        linewidths=0,
    )
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_xlabel(r"输入横向坐标 $x$ [mm]")
    axes[1, 1].set_ylabel(r"输入横向坐标 $y$ [mm]")
    axes[1, 1].set_title("横向事件密度")
    xy_bar = fig.colorbar(xy_density, ax=axes[1, 1], pad=0.02)
    xy_bar.set_label("每个六边形网格中的事件数")
    add_panel_label(axes[1, 1], "(d)")

    fig.suptitle("run10972：上游标量输入数据的四项基本观察", fontsize=15, fontweight="bold", color=DEEP_BLUE)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.90, wspace=0.30, hspace=0.34)
    save_figure(fig, "01_data_overview_cn")


def make_anticorrelation(frame: pd.DataFrame) -> tuple[float, float]:
    local = frame[["qS1_max", "qS2B_max", "dt"]].dropna().copy()
    pearson = local["qS1_max"].corr(local["qS2B_max"], method="pearson")
    spearman = local["qS1_max"].corr(local["qS2B_max"], method="spearman")
    local["s1_bin"] = pd.qcut(local["qS1_max"], q=18, duplicates="drop")
    trend = local.groupby("s1_bin", observed=True).agg(
        s1=("qS1_max", "median"),
        s2=("qS2B_max", "median"),
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    scatter = ax.scatter(
        local["qS1_max"] / 1000.0,
        local["qS2B_max"] / 1000.0,
        c=local["dt"] / 1000.0,
        s=13,
        alpha=0.38,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    ax.plot(
        trend["s1"] / 1000.0,
        trend["s2"] / 1000.0,
        color="white",
        lw=4.2,
        zorder=4,
    )
    ax.plot(
        trend["s1"] / 1000.0,
        trend["s2"] / 1000.0,
        color=CORAL,
        marker="o",
        ms=4.5,
        lw=2.0,
        label="等频分箱的中位数趋势",
        zorder=5,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"漂移时间 $t_{\mathrm{d}}$ [$\mu$s]")
    ax.set_xlabel(r"未在本项目中校正的 $S1$ [$10^3$ p.e.]")
    ax.set_ylabel(r"未在本项目中校正的 $S2_{\mathrm{B}}$ [$10^3$ p.e.]")
    ax.set_title(r"$S1$ 与 $S2_{\mathrm{B}}$ 的反相关及其漂移时间结构", color=DEEP_BLUE)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    ax.text(
        0.025,
        0.035,
        rf"Pearson $r={pearson:.3f}$" + "\n" + rf"Spearman $\rho={spearman:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#BCC8D2", "alpha": 0.93},
    )
    fig.tight_layout()
    save_figure(fig, "02_s1_s2_anticorrelation_cn")
    return float(pearson), float(spearman)


def make_spatial_coverage(frame: pd.DataFrame) -> None:
    dt_us = frame["dt"].to_numpy(float) / 1000.0
    x = frame["xS2T_max"].to_numpy(float)
    y = frame["yS2T_max"].to_numpy(float)
    radius = np.hypot(x, y)

    fig = plt.figure(figsize=(11.2, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], hspace=0.36, wspace=0.30)
    ax_dt = fig.add_subplot(grid[0, :])
    ax_xy = fig.add_subplot(grid[1, 0])
    ax_r = fig.add_subplot(grid[1, 1])

    ax_dt.hist(dt_us, bins=42, color=TEAL, alpha=0.88, edgecolor="white", linewidth=0.35)
    ax_dt.axvline(np.median(dt_us), color=DEEP_BLUE, ls="--", lw=1.5)
    ax_dt.set_xlabel(r"漂移时间 $t_{\mathrm{d}}$ [$\mu$s]")
    ax_dt.set_ylabel("事件数")
    ax_dt.set_title("(a) 深度方向：覆盖较宽，但两端统计量较少", loc="left")

    hb = ax_xy.hexbin(x, y, gridsize=42, mincnt=1, bins="log", cmap="magma", linewidths=0.15)
    ax_xy.add_patch(Circle((0, 0), np.median(radius), fill=False, ls="--", lw=1.2, ec="white", alpha=0.9))
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel(r"$x_{S2T}$ [mm]")
    ax_xy.set_ylabel(r"$y_{S2T}$ [mm]")
    ax_xy.set_title("(b) 横向位置：事件集中成环状/边缘结构", loc="left")
    cbar = fig.colorbar(hb, ax=ax_xy, pad=0.02)
    cbar.set_label("每个六边形内的事件数（对数色标）")

    ax_r.hist(radius, bins=42, color=GOLD, alpha=0.88, edgecolor="white", linewidth=0.35)
    ax_r.axvspan(340, 390, color=CORAL, alpha=0.12, label="主要集中区间（约 340--390 mm）")
    ax_r.axvline(np.median(radius), color=DEEP_BLUE, ls="--", lw=1.5, label=f"中位数 {np.median(radius):.1f} mm")
    ax_r.set_xlabel(r"半径 $r=\sqrt{x^2+y^2}$ [mm]")
    ax_r.set_ylabel("事件数")
    ax_r.set_title("(c) 径向位置：中心区域样本稀少", loc="left")
    ax_r.legend(frameon=False, fontsize=8.7)

    fig.suptitle("run10972 的空间覆盖并不均匀", fontsize=15, fontweight="bold", color=DEEP_BLUE)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.90)
    save_figure(fig, "03_spatial_coverage_cn")


def binned_median(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], float)
    y = np.asarray(y[finite], float)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    centers: list[float] = []
    medians: list[float] = []
    errors_low: list[float] = []
    errors_high: list[float] = []
    counts: list[int] = []
    global_median = np.median(y)

    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        select = (x >= low) & (x < high if index < len(edges) - 2 else x <= high)
        values = y[select]
        if values.size < 20:
            continue
        normalized = values / global_median
        center = float(np.median(x[select]))
        median = float(np.median(normalized))
        samples = rng.choice(normalized, size=(250, normalized.size), replace=True)
        boot = np.median(samples, axis=1)
        lo, hi = np.quantile(boot, [0.16, 0.84])
        centers.append(center)
        medians.append(median)
        errors_low.append(median - float(lo))
        errors_high.append(float(hi) - median)
        counts.append(int(values.size))

    return (
        np.asarray(centers),
        np.asarray(medians),
        np.asarray(errors_low),
        np.asarray(errors_high),
        np.asarray(counts),
    )


def make_conditional_trends(frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(10972)
    dt_us = frame["dt"].to_numpy(float) / 1000.0
    radius = np.hypot(frame["xS2T_max"], frame["yS2T_max"])
    s1 = frame["qS1_max"].to_numpy(float)
    s2 = frame["qS2B_max"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.9))
    for ax, x, xlabel, title in [
        (axes[0], dt_us, r"漂移时间 $t_{\mathrm{d}}$ [$\mu$s]", "(a) 随漂移时间的条件趋势"),
        (axes[1], radius, r"半径 $r$ [mm]", "(b) 随半径的条件趋势"),
    ]:
        for values, label, color, marker in [
            (s1, r"$S1$ 条件中位数", CORAL, "o"),
            (s2, r"$S2_{\mathrm{B}}$ 条件中位数", TEAL, "s"),
        ]:
            centers, medians, err_low, err_high, counts = binned_median(x, values, 12, rng)
            ax.errorbar(
                centers,
                medians,
                yerr=np.vstack([err_low, err_high]),
                color=color,
                marker=marker,
                ms=5.0,
                lw=1.7,
                capsize=2.5,
                label=label,
            )
        ax.axhline(1.0, color=DEEP_BLUE, lw=1.0, ls="--", alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("归一化条件中位数")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False)

    fig.suptitle("输入通道的条件中位数趋势（描述性统计）", fontsize=14.5, fontweight="bold", color=DEEP_BLUE)
    fig.text(
        0.5,
        0.01,
        "等频分箱；误差条为分箱中位数的 68% bootstrap 区间。趋势可能同时包含探测器响应、源分布与上游选择效应。",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.83, wspace=0.27)
    save_figure(fig, "04_conditional_trends_cn")


def write_summary(frame: pd.DataFrame, pearson: float, spearman: float) -> None:
    radius = np.hypot(frame["xS2T_max"], frame["yS2T_max"])
    dt_us = frame["dt"].to_numpy(float) / 1000.0
    fully_missing = int(frame.isna().all(axis=0).sum())
    numeric = frame.select_dtypes(include=[np.number])
    columns_with_inf = int(np.isinf(numeric.to_numpy(float)).any(axis=0).sum())

    lines = [
        f"data_file={DATA}",
        f"events={len(frame)}",
        f"source_fields={frame.attrs['source_field_count']}",
        f"read_fields={frame.shape[1] - 1}",
        f"source_row_unique={frame['sourceRow'].is_unique}",
        f"run_numbers={','.join(map(str, sorted(frame['runNumber'].unique())))}",
        f"fully_missing_fields={fully_missing}",
        f"fields_with_infinity={columns_with_inf}",
        f"dt_us_min={np.min(dt_us):.3f}",
        f"dt_us_median={np.median(dt_us):.3f}",
        f"dt_us_max={np.max(dt_us):.3f}",
        f"radius_mm_q25={np.quantile(radius, 0.25):.3f}",
        f"radius_mm_median={np.median(radius):.3f}",
        f"radius_mm_q75={np.quantile(radius, 0.75):.3f}",
        f"s1_s2_pearson={pearson:.6f}",
        f"s1_s2_spearman={spearman:.6f}",
    ]
    (HERE / "data_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    frame = load_data()
    make_overview(frame)
    pearson, spearman = make_anticorrelation(frame)
    make_spatial_coverage(frame)
    make_conditional_trends(frame)
    write_summary(frame, pearson, spearman)
    print(f"已从 {DATA.name} 生成阶段汇报与原始数据报告图，输出目录：{OUT}")


if __name__ == "__main__":
    main()
