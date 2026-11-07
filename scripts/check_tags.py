#!/usr/bin/env python
"""Check re-annotation agreement, the majority-tag refit, and tag erosion.

Agreement uses every rater in the ratings file. Erosion is unconditional,
seeded as in tag_noise_robustness.ipynb.
"""

import numpy as np
from scipy.stats import kendalltau

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights
from valueprism.data import DATA_DIR, load_cases, load_decisions
from valueprism.robustness.tags import (
    SEED,
    benchmark_agreement,
    distance_table,
    erosion_shifts,
    erosion_tau,
    krippendorff_alpha,
    load_ratings,
    majority_alignment,
    majority_split,
    neutral_disagreements,
    pairwise_exact,
    prepare,
    production_labels,
    rating_units,
    value_order,
)

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
# JSD to 4 decimals, benchmark tags then physician-majority tags.
PAPER_JSD = {
    "google/gemini-3-pro-preview": ("0.0161", "0.0264"),
    "mistralai/mistral-large-2512": ("0.0434", "0.0623"),
    "deepseek/deepseek-chat": ("0.0933", "0.0971"),
    "meta-llama/llama-4-maverick": ("0.0940", "0.1006"),
    "baidu/ernie-4.5-vl-424b-a47b": ("0.0947", "0.1278"),
    "z-ai/glm-4.6": ("0.1082", "0.1140"),
    "qwen/qwen3-max": ("0.1087", "0.0685"),
    "anthropic/claude-opus-4.5": ("0.1278", "0.1627"),
    "moonshotai/kimi-k2-thinking": ("0.1493", "0.1439"),
    "perplexity/sonar-pro": ("0.1503", "0.1650"),
    "x-ai/grok-4": ("0.1673", "0.1635"),
    "openai/gpt-5.2": ("0.1758", "0.2145"),
}
# Models whose four-value order, and whose top value, survive the majority tags.
PAPER_ORDER = {
    "openai/gpt-5.2",
    "z-ai/glm-4.6",
}
PAPER_TOP = PAPER_ORDER | {
    "anthropic/claude-opus-4.5",
    "baidu/ernie-4.5-vl-424b-a47b",
    "deepseek/deepseek-chat",
    "qwen/qwen3-max",
}
# Mean |Δπ| in percentage points, A / B / N / J / mean, 20 runs. 50 cases.
PAPER_SHIFT = {
    0.05: (1.6, 3.3, 1.9, 2.6, 2.3),
    0.10: (2.2, 4.9, 3.1, 3.7, 3.5),
    0.20: (3.1, 5.9, 4.5, 5.2, 4.7),
    0.30: (4.6, 9.6, 7.0, 8.7, 7.5),
}
# Kendall τ of the distance ordering, 100 runs. 30% is quoted to two decimals.
PAPER_TAU = {0.05: ("0.791", 3), 0.30: ("0.55", 2)}
CLOSEST = {"Gemini 3 Pro", "Mistral Large"}
FURTHEST = {"GPT 5.2", "Grok 4", "Sonar Pro"}

flags: list[str] = []


def check(label: str, got, expected, source: str = "paper") -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    print(f"  {label}: {got}    {source} {expected}    {mark}")
    if got != expected:
        flags.append(f"{label}: got {got}, {source} {expected}")


def name_of(model: str) -> str:
    return NAMES.get(model, model)


def coverage_line(ratings) -> str:
    per_case = ratings.groupby("case_id")["rater"].nunique()
    twos = int((per_case == 2).sum())
    threes = int((per_case == 3).sum())
    return f"{len(per_case)} cases, {twos} with two ratings, {threes} with three"


def ordering(rows):
    return sorted(rows, key=lambda row: (row[2], row[0]))


def print_order(title, rows) -> None:
    print(title)
    for rank, (model, _profile, divergence) in enumerate(ordering(rows), start=1):
        print(f"  {rank:2d}  {divergence:.4f}  {name_of(model)}")


def main() -> None:
    cases, delta = load_cases()
    decisions = load_decisions()
    ratings = load_ratings(DATA_DIR / "value_annotations" / "ratings.csv")
    case_ids, alignment, llms, physicians = prepare(cases, decisions)

    print(f"temperature T* = {T_STAR}, erosion seed = {SEED}")
    print("raters")
    unknown = sorted(set(ratings["case_id"]) - set(case_ids))
    check("cases outside the benchmark", unknown, [])
    raters = sorted(ratings["rater"].unique())
    print(f"  {len(raters)} raters in the file: {' '.join(raters)}")
    per_rater = ratings.groupby("rater")["case_id"].nunique()
    for rater in raters:
        print(f"  {rater}: {int(per_rater[rater])} cases")
    contributed = int((per_rater > 0).sum())
    check("raters with at least one rating on the 50 cases", contributed, 8)
    print(f"  {coverage_line(ratings)}")
    per_case = ratings.groupby("case_id")["rater"].nunique()
    check("cases with two ratings", int((per_case == 2).sum()), 43)
    check("cases with three ratings", int((per_case == 3).sum()), 7)

    print("agreement")
    exact, n_pairs = pairwise_exact(ratings)
    check("mean pairwise agreement", f"{exact:.1%}", "59.0%")
    print(f"  pairwise pairs: {n_pairs}")
    units = [labels for _key, labels in rating_units(ratings)]
    alpha = krippendorff_alpha(units, metric="ordinal")
    check("ordinal Krippendorff α", f"{alpha:.3f}", "0.527")
    production = production_labels(cases)
    with_benchmark = [labels + [production[key]] for key, labels in rating_units(ratings)]
    alpha_benchmark = krippendorff_alpha(with_benchmark, metric="ordinal")
    check("ordinal α including benchmark tags", f"{alpha_benchmark:.3f}", "0.620")
    agree = benchmark_agreement(ratings, cases)
    check("agreement with benchmark tags", f"{agree:.0%}", "70%")
    print(f"  agreement with benchmark tags, unrounded: {agree:.1%}")
    boundary, disagreed = neutral_disagreements(ratings)
    share = boundary / disagreed
    check("disagreements that are neutral vs non-neutral", f"{share:.0%}", "83%")
    print(f"  those disagreements: {boundary} of {disagreed}")
    same, different, tied = majority_split(ratings, cases)
    check("cells", same + different + tied, 400)
    check("majority same as benchmark", same, 220)
    check("majority different from benchmark", different, 38)
    check("no majority", tied, 142)

    print("re-annotation refit")
    print("  profiles are fit on the 50 released cases")
    before = distance_table(alignment, llms, physicians)
    after = distance_table(majority_alignment(alignment, case_ids, ratings), llms, physicians)
    by_id_before = {model: (profile, divergence) for model, profile, divergence in before}
    by_id_after = {model: (profile, divergence) for model, profile, divergence in after}
    print_order("distance before, benchmark tags (near to far)", before)
    print_order("distance after, physician majority (near to far)", after)
    for model, (expected_before, expected_after) in PAPER_JSD.items():
        check(
            f"{name_of(model)} before",
            f"{by_id_before[model][1]:.4f}",
            expected_before,
            source="notebook",
        )
        check(
            f"{name_of(model)} after",
            f"{by_id_after[model][1]:.4f}",
            expected_after,
            source="notebook",
        )

    order_kept = []
    top_kept = []
    for model, _counts in llms:
        old = by_id_before[model][0]
        new = by_id_after[model][0]
        kept_order = value_order(old) == value_order(new)
        kept_top = COLUMNS[int(np.argmax(old))] == COLUMNS[int(np.argmax(new))]
        if kept_order:
            order_kept.append(model)
        if kept_top:
            top_kept.append(model)
        print(
            f"  {name_of(model):16s}  top {'kept' if kept_top else 'changed':7s}"
            f"  order {'kept' if kept_order else 'changed':7s}"
            f"  {value_order(old)} -> {value_order(new)}"
        )
    check("top value preserved", f"{len(top_kept)} of 12", "6 of 12")
    check("full ordering preserved", f"{len(order_kept)} of 12", "2 of 12")
    check("models whose top value is kept", set(top_kept), PAPER_TOP)
    check("models whose ordering is kept", set(order_kept), PAPER_ORDER)
    ids = [model for model, _counts in llms]
    tau = float(
        kendalltau(
            [by_id_before[model][1] for model in ids],
            [by_id_after[model][1] for model in ids],
        ).statistic
    )
    check("Kendall τ of the distance ordering", f"{tau:.3f}", "0.788")

    before_names = [name_of(model) for model, _profile, _jsd in ordering(before)]
    after_names = [name_of(model) for model, _profile, _jsd in ordering(after)]
    claim_before = set(before_names[:2]) == CLOSEST and set(before_names[-3:]) == FURTHEST
    claim_after = set(after_names[:2]) == CLOSEST and set(after_names[-3:]) == FURTHEST
    print(
        f"  claim before: closest {before_names[:2]}, furthest {before_names[-3:]}    "
        f"{'holds' if claim_before else 'does not hold'}"
    )
    print(
        f"  claim after: closest {after_names[:2]}, furthest {after_names[-3:]}    "
        f"{'holds' if claim_after else 'does not hold'}"
    )
    check("claim holds before and after", claim_before and claim_after, True)
    kimi = "moonshotai/kimi-k2-thinking"
    for label, names, table in (
        ("before", before_names, by_id_before),
        ("after", after_names, by_id_after),
    ):
        rank = names.index(NAMES[kimi]) + 1
        print(f"  Kimi K2 {label}: rank {rank} of {len(names)}, JSD {table[kimi][1]:.4f}")

    print("fit against attribution.value_weights")
    widest = 0.0
    for model, _counts in llms:
        rows = decisions.loc[decisions["decision_maker"] == model]
        fitted = softmax_profile(value_weights(delta, rows), T_STAR)
        gap = np.max(np.abs(fitted[list(COLUMNS)].to_numpy() - by_id_before[model][0]))
        widest = max(widest, float(gap))
    print(f"  max |profile − attribution| = {widest:.2e}")
    check("profiles match attribution.value_weights", widest <= 1e-8, True)

    print("erosion toward neutral, truly random, 12 LLMs, 20 runs")
    print("  50 released cases")
    for p, per_value, mean, realized in erosion_shifts(
        alignment, llms, tuple(PAPER_SHIFT), n_runs=20
    ):
        got = tuple(round(float(value), 1) for value in list(per_value) + [mean])
        text = " / ".join(f"{value:.1f}" for value in got)
        expected = " / ".join(f"{value:.1f}" for value in PAPER_SHIFT[p])
        check(f"{p:.0%} A/B/N/J/mean", text, expected)
        print(f"  {p:.0%} realized erosion: {realized:.3f}")

    print("erosion, Kendall τ of the distance ordering, 100 runs")
    for p, tau_mean, realized in erosion_tau(
        alignment, llms, physicians, tuple(PAPER_TAU), n_runs=100
    ):
        places = PAPER_TAU[p][1]
        check(f"Kendall τ at {p:.0%}", f"{tau_mean:.{places}f}", PAPER_TAU[p][0])
        print(f"  {p:.0%} unrounded τ {tau_mean:.3f}, realized erosion {realized:.3f}")

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
