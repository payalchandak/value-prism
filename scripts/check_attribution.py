#!/usr/bin/env python
"""Check value profiles and the uniform-weight likelihood-ratio test.

LLM profiles are compared to the table printed by values-analysis
src/analysis/value_profiles.ipynb, which is the paper table. Physician
labels are the public P-ids.
"""

import pandas as pd

from valueprism.attribution import COLUMNS, T_STAR, lrt_uniform, softmax_profile, value_weights
from valueprism.data import load_cases, load_decisions

# model id -> (paper name, A, B, N, J), in the paper's autonomy order.
PAPER_LLMS = {
    "google/gemini-3-pro-preview": ("Gemini 3 Pro", (0.312, 0.355, 0.222, 0.111)),
    "deepseek/deepseek-chat": ("DeepSeek Chat", (0.286, 0.158, 0.111, 0.445)),
    "mistralai/mistral-large-2512": ("Mistral AI Large", (0.221, 0.327, 0.211, 0.241)),
    "anthropic/claude-opus-4.5": ("Claude Opus 4.5", (0.205, 0.218, 0.526, 0.051)),
    "qwen/qwen3-max": ("Qwen 3 Max", (0.191, 0.665, 0.063, 0.081)),
    "baidu/ernie-4.5-vl-424b-a47b": ("Baidu Ernie 4.5 VL", (0.173, 0.270, 0.160, 0.397)),
    "meta-llama/llama-4-maverick": ("Meta Llama 4 Maverick", (0.129, 0.459, 0.156, 0.256)),
    "perplexity/sonar-pro": ("Perplexity Sonar Pro", (0.128, 0.161, 0.320, 0.391)),
    "z-ai/glm-4.6": ("Zhipu AI GLM 4.6", (0.111, 0.519, 0.152, 0.218)),
    "x-ai/grok-4": ("X-AI Grok 4", (0.081, 0.431, 0.085, 0.403)),
    "moonshotai/kimi-k2-thinking": ("Moonshot AI Kimi K2", (0.069, 0.585, 0.169, 0.177)),
    "openai/gpt-5.2": ("OpenAI GPT 5.2", (0.061, 0.676, 0.121, 0.142)),
}
PAPER_POOLED = (0.437, 0.286, 0.148, 0.129)
# Notebook p-values for the two LLMs that do not reject at 0.05.
PAPER_LRT_P = {
    "baidu/ernie-4.5-vl-424b-a47b": 0.05363242,
    "mistralai/mistral-large-2512": 0.6230429,
}
# Public ids of the 11 physicians who reject the uniform null.
PAPER_REJECT = {
    "P01",
    "P03",
    "P04",
    "P05",
    "P06",
    "P14",
    "P15",
    "P17",
    "P20",
    "P21",
    "P22",
}
# Notebook profiles for the 25 physicians, keyed by public id, autonomy order.
PAPER_PHYSICIANS = {
    "P17": (0.813, 0.081, 0.050, 0.055),
    "P04": (0.792, 0.058, 0.137, 0.013),
    "P06": (0.778, 0.127, 0.079, 0.017),
    "P20": (0.665, 0.171, 0.043, 0.121),
    "P15": (0.628, 0.162, 0.089, 0.120),
    "P22": (0.582, 0.204, 0.069, 0.146),
    "P05": (0.538, 0.260, 0.164, 0.038),
    "P23": (0.507, 0.164, 0.284, 0.045),
    "P03": (0.483, 0.150, 0.352, 0.015),
    "P12": (0.371, 0.268, 0.218, 0.143),
    "P02": (0.354, 0.185, 0.263, 0.199),
    "P01": (0.351, 0.410, 0.037, 0.202),
    "P13": (0.344, 0.123, 0.319, 0.214),
    "P21": (0.338, 0.637, 0.011, 0.015),
    "P14": (0.337, 0.509, 0.022, 0.132),
    "P07": (0.326, 0.181, 0.087, 0.406),
    "P18": (0.316, 0.357, 0.165, 0.162),
    "P11": (0.300, 0.136, 0.383, 0.181),
    "P09": (0.291, 0.383, 0.097, 0.228),
    "P08": (0.281, 0.517, 0.138, 0.064),
    "P19": (0.260, 0.450, 0.091, 0.199),
    "P10": (0.248, 0.447, 0.153, 0.153),
    "P24": (0.242, 0.332, 0.180, 0.246),
    "P25": (0.231, 0.373, 0.304, 0.092),
    "P16": (0.167, 0.545, 0.129, 0.159),
}
LATEX_HEADERS = ["Autonomy", "Beneficence", "Nonmaleficence", "Justice"]

flags: list[str] = []


def check(label: str, got, expected) -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    print(f"  {label}: {got}    paper {expected}    {mark}")
    if got != expected:
        flags.append(f"{label}: got {got}, paper {expected}")


def slash(values) -> str:
    return " / ".join(f"{value:.3f}" for value in values)


def profile_text(profile: pd.Series) -> str:
    return slash(profile[column] for column in COLUMNS)


def main() -> None:
    _cases, delta = load_cases()
    decisions = load_decisions()
    llm_ids = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    physician_ids = sorted(
        decisions.loc[decisions["group"] == "physician", "decision_maker"].unique()
    )

    print("LLM profiles")
    check("models", llm_ids, sorted(PAPER_LLMS))
    fitted = {}
    for model_id in llm_ids:
        rows = decisions.loc[decisions["decision_maker"] == model_id]
        fitted[model_id] = softmax_profile(value_weights(delta, rows), T_STAR)
    for model_id, (name, expected) in PAPER_LLMS.items():
        if model_id not in fitted:
            continue
        check(name, profile_text(fitted[model_id]), slash(expected))

    print("pooled physician profile")
    physician_rows = decisions.loc[decisions["group"] == "physician"]
    pooled = softmax_profile(value_weights(delta, physician_rows), T_STAR)
    check("physicians", profile_text(pooled), slash(PAPER_POOLED))

    print("physician trial counts")
    answered = physician_rows.loc[physician_rows["choice"].notna()]
    per_case = answered.groupby(["decision_maker", "case_id"]).size()
    check("max n_i", int(per_case.max()), 1)
    print(f"  missing physician responses: {int(physician_rows['choice'].isna().sum())}")

    print("likelihood-ratio test, LLMs")
    llm_lrt = {}
    for model_id, (name, _expected) in PAPER_LLMS.items():
        rows = decisions.loc[decisions["decision_maker"] == model_id]
        result = lrt_uniform(delta, rows)
        llm_lrt[model_id] = result
        reject = "reject" if result["p_value"] < 0.05 else "do not reject"
        print(
            f"  {name}: χ² = {result['lrt_statistic']:.2f}"
            f"    p = {result['p_value']:.4f}    df = {result['df']}    {reject}"
        )
    quiet = sorted(model_id for model_id, result in llm_lrt.items() if result["p_value"] >= 0.05)
    check("LLMs that do not reject", quiet, sorted(PAPER_LRT_P))
    for model_id, expected_p in PAPER_LRT_P.items():
        name = PAPER_LLMS[model_id][0]
        got = llm_lrt[model_id]["p_value"]
        # Baidu is quoted to four decimals, Mistral to three.
        places = 4 if expected_p < 0.1 else 3
        check(f"{name} p", round(got, places), round(expected_p, places))

    print("likelihood-ratio test, physicians")
    physician_lrt = []
    for physician_id in physician_ids:
        rows = decisions.loc[decisions["decision_maker"] == physician_id]
        result = lrt_uniform(delta, rows)
        physician_lrt.append((physician_id, result))
    physician_lrt.sort(key=lambda item: item[1]["lrt_statistic"], reverse=True)
    for physician_id, result in physician_lrt:
        reject = "reject" if result["p_value"] < 0.05 else "do not reject"
        print(
            f"  {physician_id}: χ² = {result['lrt_statistic']:.2f}"
            f"    p = {result['p_value']:.4f}    df = {result['df']}    {reject}"
        )
    rejecting_ids = sorted(
        physician_id
        for physician_id, result in physician_lrt
        if result["p_value"] < 0.05
    )
    check("physicians who reject", len(rejecting_ids), 11)
    check("ids of those physicians", set(rejecting_ids), PAPER_REJECT)
    print(f"  rejecting P-ids: {', '.join(rejecting_ids)}")

    print("individual physician profiles, sorted by autonomy")
    profiles = []
    for physician_id in physician_ids:
        rows = decisions.loc[decisions["decision_maker"] == physician_id]
        profile = softmax_profile(value_weights(delta, rows), T_STAR)
        profiles.append((physician_id, profile))
    profiles.sort(key=lambda item: (-item[1]["A"], item[0]))
    for physician_id, profile in profiles:
        expected = PAPER_PHYSICIANS.get(physician_id)
        if expected is None:
            flags.append(f"{physician_id} has no notebook profile")
            print(f"  {physician_id}: {profile_text(profile)}    no notebook row")
            continue
        check(physician_id, profile_text(profile), slash(expected))

    print("latex")
    print(r"\begin{tabular}{lcccc}")
    header = " & ".join([r"\textbf{Physician}"] + [rf"\textbf{{{name}}}" for name in LATEX_HEADERS])
    print(header + r" \\")
    for physician_id, profile in profiles:
        cells = " & ".join(f"{profile[column]:.3f}" for column in COLUMNS)
        print(f"{physician_id} & {cells} \\\\")
    print(r"\end{tabular}")

    degrees = {result["df"] for result in llm_lrt.values()}
    degrees.update(result["df"] for _physician_id, result in physician_lrt)
    print("degrees of freedom")
    check("df", degrees, {3})

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
