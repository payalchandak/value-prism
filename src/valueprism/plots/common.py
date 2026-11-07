"""Display names, the model palette, and saving figures."""

from pathlib import Path

import matplotlib.pyplot as plt

# Names on the paper figures. Physician ids stay the public labels P01–P25.
DISPLAY_NAMES = {
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "baidu/ernie-4.5-vl-424b-a47b": "Baidu Ernie 4.5 VL",
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "meta-llama/llama-4-maverick": "Meta Llama 4 Maverick",
    "mistralai/mistral-large-2512": "Mistral AI Large",
    "moonshotai/kimi-k2-thinking": "Moonshot AI Kimi K2",
    "openai/gpt-5.2": "OpenAI GPT 5.2",
    "perplexity/sonar-pro": "Perplexity Sonar Pro",
    "qwen/qwen3-max": "Qwen 3 Max",
    "x-ai/grok-4": "X-AI Grok 4",
    "z-ai/glm-4.6": "Zhipu AI GLM 4.6",
    "human_consensus": "Pooled Physicians",
}

# One color per model, in the order used by the forest plot: rank 0 is the
# largest divergence and is drawn at the bottom.
MODEL_COLORS = [
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
]

DPI = 300


def display_name(model_id: str) -> str:
    """Short name for an axis label. Unknown ids are returned unchanged."""
    if model_id in DISPLAY_NAMES:
        return DISPLAY_NAMES[model_id]
    short = model_id.split("/")[-1]
    return DISPLAY_NAMES.get(short, short)


def save_figure(fig, path: Path, *, tight: bool = True) -> None:
    """Write ``fig`` at 300 dpi and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": DPI, "facecolor": "white"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **kwargs)
    plt.close(fig)
