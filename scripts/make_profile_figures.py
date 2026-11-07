#!/usr/bin/env python
"""Write the three value-profile radar figures.

The value-profiles notebook drew one grid; the paper figures are the
three layouts noted at the bottom of that cell. LLMs are in sorted
model-id order, which is the paper's left-to-right order. Physicians
are P01–P25. Public ids were assigned by sorting source ids, and the
old figure sorted its panels the same way, so this is that order on
the current panel.

Row height is 4 inches, as in the notebook grid. The consensus panel is
sized to the cropped file rather than the notebook's 16-inch note.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights
from valueprism.data import load_cases, load_decisions
from valueprism.plots.radar import (
    CONSENSUS_COLOR,
    LLM_COLORS,
    PHYSICIAN_COLOR,
    centered_radar_figure,
    radar_figure,
    save_radar_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "results" / "figures"
# The paper crop. Saving without a tight bbox keeps this exact size.
CONSENSUS_CANVAS = (959, 1148)

# model id, paper display name. Sorted by id, which is the figure order.
LLMS = (
    ("anthropic/claude-opus-4.5", "Claude Opus 4.5"),
    ("baidu/ernie-4.5-vl-424b-a47b", "Baidu Ernie 4.5 VL"),
    ("deepseek/deepseek-chat", "DeepSeek Chat"),
    ("google/gemini-3-pro-preview", "Gemini 3 Pro"),
    ("meta-llama/llama-4-maverick", "Meta Llama 4 Maverick"),
    ("mistralai/mistral-large-2512", "Mistral AI Large"),
    ("moonshotai/kimi-k2-thinking", "Moonshot AI Kimi K2"),
    ("openai/gpt-5.2", "OpenAI GPT 5.2"),
    ("perplexity/sonar-pro", "Perplexity Sonar Pro"),
    ("qwen/qwen3-max", "Qwen 3 Max"),
    ("x-ai/grok-4", "X-AI Grok 4"),
    ("z-ai/glm-4.6", "Zhipu AI GLM 4.6"),
)


def _profile(delta, decisions):
    weights = value_weights(delta, decisions)
    return softmax_profile(weights, T_STAR).loc[list(COLUMNS)].to_numpy()


def _ring_radius_px(profile, dpi: int = 300) -> float:
    """Ring radius of the historical 3.33 × 3.85 consensus crop, in pixels."""
    fig = radar_figure(
        [profile],
        [CONSENSUS_COLOR],
        ["Pooled Physicians"],
        n_cols=1,
        figsize=(3.33, 3.85),
        title_size=14,
        title_pad=35,
    )
    fig.set_dpi(dpi)
    fig.canvas.draw()
    ax = fig.axes[0]
    center = ax.transData.transform((0.0, 0.0))
    edge = ax.transData.transform((0.0, 1.0))
    radius = abs(float(edge[0] - center[0]))
    plt.close(fig)
    return radius


def _llm_row_midpoint(image: Image.Image) -> float:
    """Y of the point halfway between the two radar-row centers, in pixels.

    The rings are the dashed #888888 circles. Each column is fitted twice,
    once per row, and the two mean centers are averaged.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    mask = np.all(np.abs(rgb.astype(np.int16) - 136) <= 12, axis=2)
    # Column centers of this 2×6 grid, measured on values-llms.png.
    columns = (595, 1788, 2980, 4173, 5365, 6558)
    rows = (657, 1828)
    fitted = []
    for cx in columns:
        for cy in rows:
            y0, y1 = max(0, cy - 520), min(height, cy + 520)
            x0, x1 = max(0, cx - 520), min(width, cx + 520)
            ys, xs = np.where(mask[y0:y1, x0:x1])
            xs = xs.astype(float) + x0
            ys = ys.astype(float) + y0
            distance = np.hypot(xs - cx, ys - cy)
            keep = (distance > 380) & (distance < 560)
            xs, ys = xs[keep], ys[keep]
            if len(xs) < 30:
                raise SystemExit(f"could not find an LLM radar near ({cx}, {cy})")
            design = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
            solution, *_ = np.linalg.lstsq(design, xs**2 + ys**2, rcond=None)
            center_x, center_y, constant = solution
            fitted.append(float(center_y))
    top = float(np.mean(fitted[0::2]))
    bottom = float(np.mean(fitted[1::2]))
    return (top + bottom) / 2


def write_layout_check(consensus_path: Path, llm_path: Path, dest: Path) -> None:
    """Consensus at 0.21 and the LLM grid at 0.77, top-aligned.

    A horizontal line marks the midpoint between the two LLM rows.
    """
    consensus = Image.open(consensus_path).convert("RGB")
    llms = Image.open(llm_path).convert("RGB")
    line_y = _llm_row_midpoint(llms)
    # Keep the LLM pixels and scale the consensus panel to the paper ratio.
    consensus_width = round(llms.width * 0.21 / 0.77)
    consensus_height = round(consensus_width * consensus.height / consensus.width)
    consensus = consensus.resize(
        (consensus_width, consensus_height), Image.Resampling.LANCZOS
    )
    canvas = Image.new(
        "RGB",
        (consensus_width + llms.width, max(consensus_height, llms.height)),
        "white",
    )
    canvas.paste(consensus, (0, 0))
    canvas.paste(llms, (consensus_width, 0))
    draw = ImageDraw.Draw(canvas)
    y = round(line_y)
    draw.line([(0, y), (canvas.width - 1, y)], fill=(176, 0, 32), width=2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)


def main() -> None:
    # The notebook calls seaborn set_theme, which prefers Arial. The cropped
    # figures are Arial Bold, narrower than DejaVu Sans.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    _cases, delta = load_cases()
    decisions = load_decisions()

    llm_ids = [model_id for model_id, _name in LLMS]
    present = set(decisions.loc[decisions["group"] == "llm", "decision_maker"])
    missing = [model_id for model_id in llm_ids if model_id not in present]
    if missing:
        raise SystemExit(f"missing LLM decisions: {missing}")

    llm_profiles = [
        _profile(delta, decisions.loc[decisions["decision_maker"] == model_id])
        for model_id in llm_ids
    ]
    physicians = sorted(
        decisions.loc[decisions["group"] == "physician", "decision_maker"].unique()
    )
    physician_profiles = [
        _profile(delta, decisions.loc[decisions["decision_maker"] == physician_id])
        for physician_id in physicians
    ]
    pooled = _profile(delta, decisions.loc[decisions["group"] == "physician"])

    # Consensus keeps the historical ring size on the 959×1148 crop. Its center
    # is the midpoint between the two LLM rows once the panels are top-aligned
    # at 0.21 and 0.77 of the text width. The title lines up with the LLM titles.
    # LLMs: 6 columns, width 24, fontsize 27. Physicians: 5 columns, width 16,
    # fontsize 22. Pad 20 on both grids.
    consensus_path = FIGURE_DIR / "values-consensus.png"
    llm_path = FIGURE_DIR / "values-llms.png"
    llm_image = Image.open(llm_path)
    # Same scale factor that maps an LLM-image y to a consensus-image y.
    scale = (0.77 / 0.21) * (CONSENSUS_CANVAS[0] / llm_image.width)
    consensus = centered_radar_figure(
        pooled,
        CONSENSUS_COLOR,
        "Pooled Physicians",
        CONSENSUS_CANVAS,
        _ring_radius_px(pooled),
        title_size=14,
        center_from_top=_llm_row_midpoint(llm_image) * scale,
        title_top_px=14,
    )
    save_radar_figure(consensus, consensus_path, tight=False)
    print(f"wrote {consensus_path.relative_to(REPO_ROOT)}")

    figures = {
        "values-llms.png": radar_figure(
            llm_profiles,
            LLM_COLORS,
            [name for _model_id, name in LLMS],
            n_cols=6,
            figsize=(24, 8),
            title_size=27,
            title_pad=20,
        ),
        "values-physicians.png": radar_figure(
            physician_profiles,
            [PHYSICIAN_COLOR] * len(physicians),
            list(physicians),
            n_cols=5,
            figsize=(16, 20),
            title_size=22,
            title_pad=20,
        ),
    }
    for name, fig in figures.items():
        path = FIGURE_DIR / name
        save_radar_figure(fig, path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    check_path = FIGURE_DIR / "compare" / "fig2_layout.png"
    write_layout_check(consensus_path, FIGURE_DIR / "values-llms.png", check_path)
    print(f"wrote {check_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
