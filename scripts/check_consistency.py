#!/usr/bin/env python
"""Print the consistency figures from the paper and the 25-physician preview."""

import numpy as np

from valueprism.consistency import (
    case_entropy_correlation,
    cases_below_endorsement,
    entropy_statistics,
    fleiss_kappa,
    llm_cases,
    physician_cases,
    value_complexity,
)
from valueprism.data import load_cases, load_decisions

# model id -> name used when these figures were reported
NAMES = {
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
}

# Paper figures for the 12 LLMs. The three names are called out in the text.
PAPER_P75_ZERO = {
    "anthropic/claude-opus-4.5",
    "google/gemini-3-pro-preview",
    "openai/gpt-5.2",
}
PAPER_COMPLEXITY = {2: (11, "0.240"), 3: (21, "0.233"), 4: (18, "0.176")}
PAPER_SPEARMAN = {
    "mean_entropy": ("-0.13", "0.38"),
    "agreement_share": ("0.18", "0.20"),
}

flags: list[str] = []


def check(label: str, got, expected, source: str = "paper", detail: str = "") -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    suffix = f"    ({detail})" if detail else ""
    print(f"  {label}: {got}    {source} {expected}    {mark}{suffix}")
    if got != expected:
        flags.append(f"{label}: got {got}, {source} {expected}")


def name_of(model: str) -> str:
    return NAMES.get(model, model)


def main() -> None:
    _cases, delta = load_cases()
    decisions = load_decisions()
    models = llm_cases(decisions)
    panel = physician_cases(decisions)

    print("LLM entropy and near-perfect agreement")
    summary = []
    shares = []
    for model, group in models.groupby("decision_maker", sort=True):
        stats = entropy_statistics(group["entropy"])
        share = float(group["near_perfect"].mean()) * 100
        shares.append(share)
        summary.append((model, stats))
        print(
            f"  {name_of(model):24s}  median {stats.median:.3f}"
            f"  p75 {stats.p75:.3f}  near-perfect {share:.0f}%  n={stats.n}"
        )

    n_models = len(summary)
    n_median_zero = sum(stats.median == 0 for _model, stats in summary)
    n_p75_zero = sum(stats.p75 == 0 for _model, stats in summary)
    zero_p75 = [model for model, stats in summary if stats.p75 == 0]
    share_stats = entropy_statistics(shares)
    check("models with median entropy 0", f"{n_median_zero} of {n_models}", "11 of 12")
    check("models with p75 entropy 0", f"{n_p75_zero} of {n_models}", "6 of 12")
    print("  p75 entropy is 0: " + ", ".join(name_of(model) for model in zero_p75))
    included = PAPER_P75_ZERO <= set(zero_p75)
    check(
        "p75 = 0 includes Claude Opus 4.5, Gemini 3 Pro, and GPT 5.2",
        "yes" if included else "no",
        "yes",
    )
    check(
        "median share of cases with near-perfect agreement",
        f"{share_stats.median:.0f}%",
        "82%",
    )
    check(
        "75th percentile share of cases with near-perfect agreement",
        f"{share_stats.p75:.0f}%",
        "86%",
        detail=f"median {share_stats.median:.4f}, p75 {share_stats.p75:.4f}",
    )

    print("value complexity")
    table, tests = value_complexity(delta, models)
    for row in table.itertuples(index=False):
        n_cases, mean = PAPER_COMPLEXITY[int(row.k)]
        got = f"{int(row.n_cases)} cases, {row.mean_entropy:.3f}"
        check(
            f"k={int(row.k)}",
            got,
            f"{n_cases} cases, {mean}",
            detail=f"mean {row.mean_entropy:.6f}",
        )
    labels = {
        "mean_entropy": "Spearman ρ, mean model entropy vs engaged values",
        "agreement_share": "Spearman ρ, share with near-perfect agreement vs engaged values",
    }
    for row in tests.itertuples(index=False):
        rho_text, p_text = PAPER_SPEARMAN[row.outcome]
        check(
            labels[row.outcome],
            f"{row.rho:.2f} (p={row.p:.2f})",
            f"{rho_text} (p={p_text})",
            detail=f"ρ={row.rho:.6f}, p={row.p:.6f}",
        )

    print("physicians")
    n_physicians = decisions.loc[decisions["group"] == "physician", "decision_maker"].nunique()
    votes = int((panel["n1"] + panel["n2"]).median())
    print(f"  panel: {n_physicians} physicians, {votes} votes on each of {len(panel)} cases")
    kappa = fleiss_kappa(panel)
    n_below = cases_below_endorsement(panel)
    phys_stats = entropy_statistics(panel["entropy"])
    check("Fleiss' κ", f"{kappa:.4f}", "0.2444", source="preview", detail=f"{kappa:.6f}")
    check(
        "cases below 70%",
        f"{n_below} of {len(panel)}",
        "24 of 50",
        source="preview",
    )
    iqr = f"{phys_stats.median:.3f} [{phys_stats.p25:.3f}, {phys_stats.p75:.3f}]"
    check(
        "entropy median [IQR]",
        iqr,
        "0.855 [0.529, 0.943]",
        source="preview",
        detail=f"{phys_stats.median:.6f} [{phys_stats.p25:.6f}, {phys_stats.p75:.6f}]",
    )

    print("Spearman ρ, physician case entropy vs model case entropy")
    reference = panel.set_index("case_id")["entropy"]
    rhos = []
    pvalues = []
    for model, group in models.groupby("decision_maker", sort=True):
        rho, pvalue, n_cases = case_entropy_correlation(
            reference, group.set_index("case_id")["entropy"]
        )
        rhos.append(rho)
        pvalues.append(pvalue)
        print(f"  {name_of(model):24s}  ρ={rho:+.6f}  p={pvalue:.6f}  n={n_cases}")
    mean_rho = float(np.mean(rhos))
    check(
        "Spearman ρ mean [range]",
        f"{mean_rho:.3f} [{min(rhos):.3f}, {max(rhos):.3f}]",
        "-0.005 [-0.181, 0.291]",
        source="preview",
        detail=f"mean {mean_rho:.6f}, min {min(rhos):.6f}, max {max(rhos):.6f}",
    )
    check(
        "minimum Spearman p",
        f"{min(pvalues):.4f}",
        "0.0402",
        source="preview",
        detail=f"{min(pvalues):.6f}",
    )

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
