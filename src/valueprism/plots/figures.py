"""Draw the seven paper figures.

The layouts follow values-analysis ``value_profiles.ipynb`` (cells 14, 16,
18, and 28), ``benchmark_properties.ipynb`` (cell 9), ``explore.ipynb``
(the agreement matrix), and ``scripts/calibrate_temperature.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import gaussian_kde

from valueprism.calibration import ALPHAS, Calibration
from valueprism.plots.common import MODEL_COLORS, display_name, save_figure

# Within-group diversity curves, then the temperature sweep.
LLM_COLOR = "#3B7DD8"
PHYSICIAN_COLOR = "#E8913A"
DIFFERENCE_COLOR = "#6B4C8A"
TEMPERATURE_COLORS = {
    0.3: "#B84A44",
    0.5: "#D4813F",
    1.0: "#3F8FC9",
    3.0: "#7A3E7A",
    10.0: "#4F8F5A",
}
AGREEMENT_CMAP = LinearSegmentedColormap.from_list(
    "agreement",
    ["#c75b5b", "#f3f2ed", "#4f7f5f"],
    N=256,
)


def _lower_heatmap(
    values: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    cmap,
    norm,
    figsize: tuple[float, float],
    tick_size: float,
    x_tick_size: float | None,
    linewidth: float,
    annotate: np.ndarray | None,
    annot_size: float,
    cbar_label: str,
    cbar_shrink: float,
    cbar_label_size: float | None,
    cbar_tick_size: float | None,
    cbar_label_left: bool,
) -> None:
    """Lower triangle, first row at the top. ``annotate`` is text per cell."""
    n = len(labels)
    masked = np.ma.array(values, mask=np.triu(np.ones((n, n), dtype=bool), k=1))
    cmap = cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    image = ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    if linewidth:
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=linewidth)
        ax.tick_params(which="minor", length=0)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    x_size = tick_size if x_tick_size is None else x_tick_size
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=x_size, color="black")
    ax.set_yticklabels(labels, fontsize=tick_size, color="black")
    ax.tick_params(which="major", length=0, colors="black")
    ax.grid(False)

    if annotate is not None:
        for i in range(n):
            for j in range(i):
                ax.text(
                    j,
                    i,
                    f"{int(annotate[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color="black",
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=cbar_shrink, pad=0.02)
    colorbar.set_label(cbar_label, fontsize=cbar_label_size)
    if cbar_label_left:
        # Ticks stay on the right; the title sits between the matrix and the bar.
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.yaxis.set_label_position("left")
    if cbar_tick_size is not None:
        colorbar.ax.tick_params(labelsize=cbar_tick_size, colors="black")
    colorbar.ax.yaxis.label.set_color("black")
    save_figure(fig, path)


def plot_jsd_matrix(matrix: pd.DataFrame, path: Path) -> None:
    """Lower triangle of pairwise JSD. ``matrix`` is indexed by decision maker."""
    labels = [display_name(str(maker)) for maker in matrix.index]
    _lower_heatmap(
        matrix.to_numpy(dtype=float),
        labels,
        path,
        cmap=plt.get_cmap("YlOrRd"),
        norm=plt.Normalize(vmin=0, vmax=0.6),
        figsize=(12, 10),
        tick_size=10,
        x_tick_size=None,
        linewidth=0.5,
        annotate=None,
        annot_size=10,
        cbar_label="Jensen-Shannon divergence",
        cbar_shrink=0.7,
        cbar_label_size=None,
        cbar_tick_size=None,
        cbar_label_left=True,
    )


def plot_agreement(rates: pd.DataFrame, path: Path) -> None:
    """Lower triangle of decision agreement, in percent. Diagonal is unlabelled."""
    labels = [display_name(str(maker)) for maker in rates.index]
    percent = np.rint(rates.to_numpy(dtype=float) * 100).astype(int)
    _lower_heatmap(
        percent,
        labels,
        path,
        cmap=AGREEMENT_CMAP,
        norm=TwoSlopeNorm(vmin=30, vcenter=50, vmax=100),
        figsize=(14, 12),
        tick_size=12,
        x_tick_size=8,
        linewidth=0,
        annotate=percent,
        annot_size=10,
        cbar_label="Agreement Rate",
        cbar_shrink=0.8,
        cbar_label_size=20,
        cbar_tick_size=18,
        cbar_label_left=False,
    )


def plot_distributional_pluralism(
    reference_jsds: np.ndarray,
    rows: list[tuple[str, float, float, float]],
    path: Path,
) -> None:
    """Reference density on top, each model's JSD and 95% interval below.

    ``rows`` are ``(label, jsd, ci_low, ci_high)``. The largest JSD is drawn
    at the bottom and takes the first model color.
    """
    ordered = sorted(rows, key=lambda row: -row[1])
    n_models = len(ordered)
    observed = [row[1] for row in ordered]
    percentile = float(np.percentile(reference_jsds, 95))
    ceiling = max(float(reference_jsds.max()), max(observed)) + 0.02

    fig, (ax_density, ax_forest) = plt.subplots(
        2,
        1,
        figsize=(8, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
        layout="constrained",
    )

    density = gaussian_kde(reference_jsds, bw_method=0.3)
    xs = np.linspace(0, ceiling, 500)
    ys = density(xs)
    ax_density.fill_between(xs, ys, alpha=0.35, color="#aaaaaa")
    ax_density.plot(xs, ys, color="#666666", lw=2)
    ax_density.axvspan(percentile, ceiling, alpha=0.08, color="#cc4444", zorder=0)
    ax_density.axvline(percentile, color="#444444", lw=1, ls=":", alpha=0.6)
    ax_density.text(
        percentile + 0.004,
        ys.max() * 0.95,
        "95th percentile",
        fontsize=12,
        color="#444444",
        va="top",
    )
    ax_density.set_yticks([])
    ax_density.tick_params(axis="x", bottom=False)
    ax_density.set_ylabel(
        "Bootstrapped\nreference density\nof individual\nphysicians",
        fontsize=13,
        rotation=0,
        linespacing=1.4,
        va="center",
        ha="right",
    )
    ax_density.spines[["top", "right"]].set_visible(False)
    ax_density.grid(False)
    ax_density.set_ylim(0, 15)

    ax_forest.axvspan(percentile, ceiling, alpha=0.06, color="#cc4444", zorder=0)
    ax_forest.axvline(percentile, color="#444444", lw=1, ls=":", alpha=0.6)
    for i, (_label, jsd, low, high) in enumerate(ordered):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax_forest.plot([low, high], [i, i], color=color, lw=2.5, solid_capstyle="round", zorder=3)
        ax_forest.scatter(
            jsd,
            i,
            color=color,
            s=85,
            zorder=5,
            edgecolors="white",
            linewidths=1.0,
            marker="D",
        )
    ax_forest.set_yticks(range(n_models))
    ax_forest.set_yticklabels([row[0] for row in ordered], fontsize=12)
    ax_forest.set_xlabel("JS divergence from pooled physician profile", fontsize=15, labelpad=10)
    ax_forest.spines[["top", "right", "bottom"]].set_visible(False)
    ax_forest.set_ylim(-0.5, n_models - 0.5)
    ax_forest.grid(axis="x", alpha=0.2, lw=0.5)
    ax_forest.set_axisbelow(True)
    ax_forest.tick_params(axis="y", length=0)
    ax_forest.tick_params(axis="x", labelsize=10)
    ax_forest.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    ax_forest.set_xlim(0, 0.23)
    save_figure(fig, path, tight=False)


def plot_monoculture_distribution(
    llm_samples: np.ndarray,
    physician_samples: np.ndarray,
    difference: np.ndarray,
    observed_llm: float,
    observed_physician: float,
    path: Path,
) -> None:
    """Bootstrap densities of within-group D̄.

    A vertical line marks each group's observed D̄. The bar is the
    percentile interval of the bootstrap difference, and the diamond
    and label are the observed D̄_LLM − D̄_phys.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    peak = 0.0
    for samples, color, label, observed in (
        (llm_samples, LLM_COLOR, "LLMs", observed_llm),
        (physician_samples, PHYSICIAN_COLOR, "Physicians", observed_physician),
    ):
        density = gaussian_kde(samples, bw_method=0.3)
        xs = np.linspace(samples.min(), samples.max(), 400)
        ys = density(xs)
        peak = max(peak, float(ys.max()))
        ax.fill_between(xs, ys, alpha=0.22, color=color)
        ax.plot(xs, ys, color=color, lw=2, label=label)
        ax.axvline(observed, color=color, lw=1.5, ls="--", zorder=3)

    low, high = np.percentile(difference, [2.5, 97.5])
    observed_diff = float(observed_llm - observed_physician)
    bar_y = peak * 1.12
    ax.plot([low, high], [bar_y, bar_y], color=DIFFERENCE_COLOR, lw=3, solid_capstyle="round", zorder=4)
    ax.scatter(
        observed_diff,
        bar_y,
        color=DIFFERENCE_COLOR,
        s=55,
        zorder=5,
        edgecolors="white",
        linewidths=1.0,
        marker="D",
    )
    ax.text(
        high + 0.002,
        bar_y,
        f"  Δ = {observed_diff:.3f}  [{low:.3f}, {high:.3f}]",
        va="center",
        ha="left",
        fontsize=11,
        color=DIFFERENCE_COLOR,
    )
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlabel("Mean pairwise Jensen-Shannon divergence", fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_yticks([])
    # The notebook's fixed ceiling of 28 clips this panel: the physician
    # density peaks just above it, and the interval bar is drawn above the peak.
    ax.set_ylim(0, bar_y * 1.28)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, path, tight=False)


def plot_monoculture_test(null: np.ndarray, observed: float, path: Path) -> None:
    """Permutation null of |Δ|, with the observed value marked."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        null,
        bins=80,
        color="#aaaaaa",
        edgecolor="white",
        alpha=0.8,
        label="Null distribution",
    )
    ax.axvline(
        observed,
        color="#B84A44",
        lw=2,
        label=f"Observed |Δ| = {observed:.4f}",
    )
    ax.set_xlabel("| mean JSD(LLMs) − mean JSD(Physicians) |", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Permutation test for equal within-group diversity",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, path, tight=False)


def plot_case_entropy(
    panels: list[tuple[str, np.ndarray, np.ndarray, float, float]],
    path: Path,
) -> None:
    """Physician entropy against model entropy, one panel per model.

    ``panels`` are ``(label, physician, model, spearman_rho, p_value)``,
    in reading order. Points are jittered by the same seed as the notebook.
    """
    rng = np.random.default_rng(42)
    jitter = 0.012
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.5, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for (label, physician, model, rho, pvalue), ax in zip(panels, axes.flat):
        n = len(physician)
        x = physician + rng.uniform(-jitter, jitter, size=n)
        y = model + rng.uniform(-jitter, jitter, size=n)
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#b0b0b0", zorder=0)
        ax.scatter(
            x,
            y,
            s=38,
            alpha=0.4,
            color="#1f4e79",
            edgecolors="white",
            linewidths=0.55,
            zorder=2,
            rasterized=True,
        )
        ax.set_title(label)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.text(
            0.03,
            0.97,
            rf"$\rho_s = {rho:.2f}$" + "\n" + rf"$p = {pvalue:.4f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d8d8d8",
                "linewidth": 0.6,
                "alpha": 0.92,
            },
        )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Physician entropy (bits)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Model entropy (bits)")
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    save_figure(fig, path)


def plot_temperature(result: Calibration, path: Path) -> None:
    """Mean reconstruction JSD against temperature, one curve per concentration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for alpha in ALPHAS:
        mean = result.mean_curve(alpha)
        se = result.se_curve(alpha)
        color = TEMPERATURE_COLORS[alpha]
        ax.plot(result.temperatures, mean, linewidth=2, color=color, label=f"α = {alpha}")
        ax.fill_between(result.temperatures, mean - se, mean + se, alpha=0.15, color=color)
        index = int(np.argmin(mean))
        ax.scatter(
            [result.temperatures[index]],
            [mean[index]],
            color=color,
            s=50,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Softmax Temperature  T", fontsize=12)
    ax.set_ylabel("Mean JS Divergence (recovered vs. ground truth)", fontsize=12)
    ax.set_title("Optimal Temperature by Dirichlet Concentration α", fontsize=14)
    ax.legend(fontsize=11, frameon=False, title="Concentration", title_fontsize=11)
    ax.grid(False)
    ax.set_xlim(result.temperatures[0], result.temperatures[-1])

    optima = []
    for alpha in ALPHAS:
        temperature = result.optimal_temperature(alpha)
        if temperature not in optima:
            optima.append(temperature)
    for temperature, height in zip(optima, (0.95, 0.88)):
        ax.axvline(temperature, color="grey", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.text(
            temperature,
            ax.get_ylim()[1] * height,
            f"T = {temperature:.4f}",
            ha="center",
            va="top",
            fontsize=9,
            color="grey",
        )
    fig.tight_layout()
    save_figure(fig, path, tight=False)
