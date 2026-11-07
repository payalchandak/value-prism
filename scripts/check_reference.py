#!/usr/bin/env python
"""Check the bootstrap reference distribution against value_profiles.ipynb.

The notebook drew physicians from its participant ids sorted ascending.
Public ids are assigned in that order, so this script resamples the
physicians sorted by public id.
"""

from valueprism.data import load_cases, load_decisions
from valueprism.reference import N_BOOTSTRAP, SEED, bootstrap_reference

# model id -> name in the notebook preview
NAMES = {
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "mistralai/mistral-large-2512": "Mistral Large",
    "deepseek/deepseek-chat": "DeepSeek",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "baidu/ernie-4.5-vl-424b-a47b": "Ernie 4.5 VL",
    "z-ai/glm-4.6": "GLM 4.6",
    "qwen/qwen3-max": "Qwen 3 Max",
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "moonshotai/kimi-k2-thinking": "Kimi K2",
    "perplexity/sonar-pro": "Sonar Pro",
    "x-ai/grok-4": "Grok 4",
    "openai/gpt-5.2": "GPT 5.2",
}

# name -> JSD, CI low, CI high, p, as printed to 4 decimals
PREVIEW = {
    "Gemini 3 Pro": ("0.0161", "0.0083", "0.0292", "0.9318"),
    "Mistral Large": ("0.0434", "0.0270", "0.0655", "0.6215"),
    "DeepSeek": ("0.0933", "0.0750", "0.1148", "0.1893"),
    "Llama 4 Maverick": ("0.0940", "0.0667", "0.1263", "0.1868"),
    "Ernie 4.5 VL": ("0.0947", "0.0709", "0.1228", "0.1834"),
    "GLM 4.6": ("0.1082", "0.0790", "0.1424", "0.1407"),
    "Qwen 3 Max": ("0.1087", "0.0823", "0.1399", "0.1393"),
    "Claude Opus 4.5": ("0.1278", "0.1036", "0.1540", "0.0949"),
    "Kimi K2": ("0.1493", "0.1158", "0.1869", "0.0447"),
    "Sonar Pro": ("0.1503", "0.1227", "0.1825", "0.0421"),
    "Grok 4": ("0.1673", "0.1317", "0.2066", "0.0110"),
    "GPT 5.2": ("0.1758", "0.1400", "0.2159", "0.0045"),
}
PREVIEW_REFERENCE = ("250,000", "0.0627", "0.0530", "0.1471")

flags: list[str] = []


def check(label: str, got: str, expected: str) -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    print(f"  {label}: {got}    notebook {expected}    {mark}")
    if got != expected:
        flags.append(f"{label}: got {got}, notebook {expected}")


def main() -> None:
    _cases, delta = load_cases()
    decisions = load_decisions()
    order = sorted(decisions.loc[decisions["group"] == "physician", "decision_maker"].unique())

    print(f"bootstrap: B = {N_BOOTSTRAP}, seed = {SEED}")
    print("resampling physicians sorted by public id")
    print(f"  {' '.join(order)}")

    result = bootstrap_reference(
        delta,
        decisions,
        order,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    print("reference")
    got_reference = (
        f"{result.n:,}",
        f"{result.mean:.4f}",
        f"{result.median:.4f}",
        f"{result.percentile_95:.4f}",
    )
    summary = (
        f"Reference: n = {got_reference[0]}, mean {got_reference[1]}, "
        f"median {got_reference[2]}, 95th percentile {got_reference[3]}."
    )
    print(summary)
    labels = ("n", "mean", "median", "95th percentile")
    for label, got, expected in zip(labels, got_reference, PREVIEW_REFERENCE):
        check(label, got, expected)

    print("JSD, 95% CI and p for each model")
    rows = []
    seen = []
    for row in result.models:
        name = NAMES.get(row.decision_maker)
        if name is None:
            flags.append(f"unexpected model {row.decision_maker}")
            print(f"  {row.decision_maker}: no preview name")
            continue
        seen.append(name)
        jsd, low, high, p_value = (
            f"{row.jsd:.4f}",
            f"{row.ci_low:.4f}",
            f"{row.ci_high:.4f}",
            f"{row.p_value:.4f}",
        )
        text = f"{name} {jsd} [{low}, {high}] p {p_value}"
        expected = PREVIEW.get(name)
        if expected is None:
            flags.append(f"{name}: no preview row")
            print(f"  {text}  autonomy {row.autonomy:.3f}    no preview row")
            continue
        expected_text = f"{name} {expected[0]} [{expected[1]}, {expected[2]}] p {expected[3]}"
        mark = "ok" if (jsd, low, high, p_value) == expected else "DIFFERENT"
        print(f"  {text}  autonomy {row.autonomy:.3f}    {mark}")
        if mark != "ok":
            flags.append(f"{name}: got {text}, notebook {expected_text}")
            flags.append(
                f"{name} raw: jsd {row.jsd:.8f} "
                f"ci [{row.ci_low:.8f}, {row.ci_high:.8f}] p {row.p_value:.8f}"
            )
        rows.append((name, jsd, low, high, p_value))
    missing = sorted(set(PREVIEW) - set(seen))
    if missing:
        flags.append(f"missing models: {missing}")

    print("latex")
    print(r"\begin{tabular}{lccc}")
    print(r"model & JSD & CI & p \\")
    for name, jsd, low, high, p_value in rows:
        print(f"{name} & {jsd} & [{low}, {high}] & {p_value} \\\\")
    print(r"\end{tabular}")

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
