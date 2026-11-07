"""Radar charts of softmax value profiles.

Angles run counterclockwise from the right, in the order A, B, N, J.
Dashed rings sit at 0.25, 0.5, 0.75, and 1. The polygon is the profile,
closed back on A. This is the value-profiles notebook's panel, without
the hand crop that split one grid into the three paper figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

LETTERS = ("A", "B", "N", "J")
RINGS = (0.25, 0.5, 0.75, 1.0)
RING_LABELS = ("0.25", "0.5", "0.75", "1")

# Notebook palette, in sorted model-id order. That order is the LLM figure.
LLM_COLORS = (
    "#B84A44",
    "#3F8FC9",
    "#7A3E7A",
    "#4F8F5A",
    "#2C6DA4",
    "#D6B85A",
    "#3FA7A3",
    "#D9823B",
    "#8A9A3A",
    "#C86A9A",
    "#C6A26A",
    "#2E4E86",
)
CONSENSUS_COLOR = "#1a1a1a"
PHYSICIAN_COLOR = "#888888"

_ANGLES = np.linspace(0, 2 * np.pi, len(LETTERS), endpoint=False)
_CLOSED = np.append(_ANGLES, _ANGLES[0])
_CIRCLE = np.linspace(0, 2 * np.pi, 100)


def draw_radar(
    ax: Axes,
    profile,
    color: str,
    title: str,
    title_size: float,
    title_pad: float,
) -> None:
    """Draw one profile. ``profile`` is four numbers in A, B, N, J order."""
    values = np.append(np.asarray(profile, dtype=float), profile[0])
    ax.set_ylim(0, 1)
    ax.set_yticks(list(RINGS))
    ax.set_yticklabels(list(RING_LABELS), fontsize=7, color="#555555")
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    for ring in RINGS:
        ax.plot(_CIRCLE, np.full(100, ring), color="#888888", lw=0.7, ls="--", zorder=1)
    ax.plot(_CLOSED, values, marker="o", markersize=5, lw=1.8, color=color, zorder=3)
    ax.fill(_CLOSED, values, alpha=0.4, color=color, zorder=2)
    ax.set_thetagrids(np.degrees(_ANGLES), [""] * len(LETTERS))
    for angle, letter in zip(_ANGLES, LETTERS):
        ax.text(
            angle,
            0.87,
            letter,
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=title_pad)


def radar_figure(
    profiles,
    colors,
    titles,
    n_cols: int,
    figsize: tuple[float, float],
    title_size: float,
    title_pad: float,
) -> Figure:
    """A grid of radars, row-major. Unused cells in the last row are hidden."""
    n = len(profiles)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        subplot_kw={"polar": True},
    )
    flat = axes.ravel()
    for ax, profile, color, title in zip(flat, profiles, colors, titles):
        draw_radar(ax, profile, color, title, title_size, title_pad)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=3.0)
    return fig


def centered_radar_figure(
    profile,
    color: str,
    title: str,
    canvas_px: tuple[int, int],
    radius_px: float,
    title_size: float,
    dpi: int = 300,
    center_from_top: float | None = None,
    title_top_px: float = 14,
) -> Figure:
    """One radar on a canvas of exactly ``canvas_px``.

    ``center_from_top`` is the ring center in pixels from the top of the
    canvas. The default is the vertical midpoint. The title is placed
    independently, ``title_top_px`` from the top. ``radius_px`` is the ring
    radius, so the radar is not resized. Saving without a tight bbox keeps
    the pixel size.
    """
    width, height = canvas_px
    if center_from_top is None:
        center_from_top = height / 2
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    center_from_bottom = height - center_from_top
    left = (width / 2 - radius_px) / width
    bottom = (center_from_bottom - radius_px) / height
    ax = fig.add_axes(
        [left, bottom, 2 * radius_px / width, 2 * radius_px / height],
        polar=True,
    )
    draw_radar(ax, profile, color, "", title_size, 0)
    fig.text(
        0.5,
        1 - title_top_px / height,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
    )
    return fig


def save_radar_figure(fig: Figure, path: Path, dpi: int = 300, *, tight: bool = True) -> None:
    """Write ``fig`` at ``dpi``. A tenth of an inch of pad matches the paper crops.

    ``tight=False`` keeps the figure's own size, which the centered consensus
    panel needs so the canvas stays 959 × 1148.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    else:
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
