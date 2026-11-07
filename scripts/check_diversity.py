#!/usr/bin/env python
"""Within-group value diversity of the 12 LLMs and 25 physicians.

The four headline lines are the corrected bootstrap means and 95%
percentile intervals, plus the permutation test from values-analysis
``value_profiles.ipynb``. A bootstrap draw still resamples decision
makers with replacement. The mean inside a draw skips pairs of the same
original unit. Both calls use 10,000 resamples and seed 42. Group A is
the LLMs. The signed difference is D̄_LLM − D̄_phys.

The observed lines are the plug-in D̄ on the original profiles.

The bootstrap resamples by position. Public ids follow the notebook's
sorted participant ids, so the physician list sorted by public id is
that sequence.

The pairwise matrix written to results/pairwise_jsd.csv leads with the
pooled physician consensus. D̄, the permutation test, and the agreement
ranges are the 37 individuals.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights
from valueprism.data import load_cases, load_decisions
from valueprism.diversity import (
    bootstrap_mean_jsd,
    mean_pairwise_jsd,
    pairwise_jsd,
    permutation_test_jsd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "results" / "pairwise_jsd.csv"
N_RESAMPLES = 10_000
SEED = 42

# Corrected bootstrap mean and percentile interval. A minus sign is U+2212.
PREVIEW = [
    "LLMs 0.0999 [0.0577, 0.1464]",
    "Physicians 0.1134 [0.0860, 0.1415]",
    "Difference −0.0135 [−0.0642, 0.0399]",
    "Observed |Δ| = 0.0137, permutation p = 0.6911",
]

# Plug-in D̄ on the original profiles, four decimals.
OBSERVED = {
    "LLMs": "0.0998",
    "Physicians": "0.1135",
    "Difference": "−0.0137",
}

flags: list[str] = []


def show(value: float) -> str:
    """Four decimals, with U+2212 for a negative number."""
    return f"{value:.4f}".replace("-", "−")


def profiles_for(delta: pd.DataFrame, decisions: pd.DataFrame, ids: list[str]) -> dict:
    """softmax(w / T*) for each id, value order A, B, N, J."""
    fitted = {}
    for maker in ids:
        rows = decisions.loc[decisions["decision_maker"] == maker]
        weights = value_weights(delta, rows)
        profile = softmax_profile(weights, T_STAR)
        fitted[maker] = {name: float(profile[name]) for name in COLUMNS}
    return fitted


def majority_choices(decisions: pd.DataFrame, case_ids: list[str], ids: list[str]) -> pd.DataFrame:
    """One column per decision maker and one row per case.

    An LLM's choice is its majority over the non-missing runs. A tie,
    including an even split, is choice 1, matching ``RunSummary.majority_choice``.
    A physician has one run. A case with no non-missing run is left missing.
    """
    chosen = {}
    for maker, maker_rows in decisions.groupby("decision_maker", sort=False):
        by_case = {}
        for case_id, rows in maker_rows.groupby("case_id", sort=False):
            answered = rows.loc[rows["choice"].notna(), "choice"]
            n1 = int(answered.eq(1).sum())
            n2 = int(answered.eq(2).sum())
            if n1 + n2 == 0:
                by_case[case_id] = np.nan
            else:
                by_case[case_id] = 1.0 if n1 >= n2 else 2.0
        chosen[maker] = by_case
    frame = pd.DataFrame(chosen).reindex(index=case_ids, columns=ids)
    return frame


def agreement_range(choices: pd.DataFrame, left: list[str], right: list[str]) -> tuple[float, float]:
    """Min and max share of cases on which the two sides pick the same option.

    The denominator is every case in ``choices``. A missing choice does not
    agree. Pairs inside one group are unordered and exclude the diagonal.
    """
    rates = []
    same_group = left == right
    for i, maker_a in enumerate(left):
        partners = right[i + 1 :] if same_group else right
        column_a = choices[maker_a].to_numpy()
        for maker_b in partners:
            column_b = choices[maker_b].to_numpy()
            rates.append(float(np.mean(column_a == column_b)))
    return min(rates), max(rates)


def main() -> None:
    cases, delta = load_cases()
    decisions = load_decisions()
    llm_ids = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    physician_ids = sorted(
        decisions.loc[decisions["group"] == "physician", "decision_maker"].unique()
    )
    if len(llm_ids) != 12 or len(physician_ids) != 25:
        raise SystemExit(f"expected 12 LLMs and 25 physicians, got {len(llm_ids)} and {len(physician_ids)}")
    if len(cases) != 50:
        raise SystemExit(f"expected 50 cases, got {len(cases)}")

    # Resampling order is the public ids sorted ascending, which is the
    # notebook's order. The matrix below is stored in that same order.
    resample_physicians = physician_ids
    fitted = profiles_for(delta, decisions, llm_ids + physician_ids)

    boot = bootstrap_mean_jsd(
        fitted,
        group_a_ids=llm_ids,
        group_b_ids=resample_physicians,
        n_bootstrap=N_RESAMPLES,
        seed=SEED,
    )
    perm = permutation_test_jsd(
        fitted,
        group_a_ids=llm_ids,
        group_b_ids=resample_physicians,
        n_permutations=N_RESAMPLES,
        seed=SEED,
    )

    lines = []
    for label, key in (
        ("LLMs", "group_a_mean"),
        ("Physicians", "group_b_mean"),
        ("Difference", "difference"),
    ):
        result = boot[key]
        low, high = result.ci(95)
        lines.append(f"{label} {show(result.mean)} [{show(low)}, {show(high)}]")
    lines.append(
        f"Observed |Δ| = {perm['observed_diff']:.4f}, permutation p = {perm['p_value']:.4f}"
    )
    for got, expected in zip(lines, PREVIEW):
        print(got)
        if got != expected:
            flags.append(f"{expected}    got {got}")

    # Consensus leads the stored matrix, matching the JSD figure. The
    # means below index the individual blocks and leave that row out.
    consensus = softmax_profile(
        value_weights(delta, decisions.loc[decisions["group"] == "physician"]),
        T_STAR,
    )
    stored = {"human_consensus": {name: float(consensus[name]) for name in COLUMNS}}
    for maker in llm_ids + physician_ids:
        stored[maker] = fitted[maker]
    matrix = pairwise_jsd(stored)
    direct = {
        "LLMs": mean_pairwise_jsd(matrix.loc[llm_ids, llm_ids].to_numpy()),
        "Physicians": mean_pairwise_jsd(matrix.loc[physician_ids, physician_ids].to_numpy()),
    }
    direct["Difference"] = direct["LLMs"] - direct["Physicians"]
    if abs(abs(direct["Difference"]) - perm["observed_diff"]) > 1e-12:
        flags.append(
            f"original |D̄_LLM − D̄_phys| {abs(direct['Difference']):.6f} "
            f"disagrees with the permutation statistic {perm['observed_diff']:.6f}"
        )

    print("D̄ on the original profiles")
    for label in ("LLMs", "Physicians", "Difference"):
        got = f"{label} {show(direct[label])}"
        print(got)
        if show(direct[label]) != OBSERVED[label]:
            flags.append(f"observed {label} {OBSERVED[label]}    got {show(direct[label])}")

    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(MATRIX_PATH, float_format="%.10f")
    print(f"pairwise JSD matrix  results/pairwise_jsd.csv  {matrix.shape[0]} × {matrix.shape[1]}")

    case_ids = cases["case_id"].tolist()
    choices = majority_choices(decisions, case_ids, llm_ids + physician_ids)
    missing = int(choices.isna().sum().sum())
    if missing:
        flags.append(f"{missing} decision-maker/case cells have no majority choice")
    print("decision agreement")
    for label, left, right in (
        ("physician–physician", physician_ids, physician_ids),
        ("LLM–LLM", llm_ids, llm_ids),
        ("LLM–physician", llm_ids, physician_ids),
    ):
        low, high = agreement_range(choices, left, right)
        print(f"{label} min {low:.4f} max {high:.4f}")

    if flags:
        print("mismatch")
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
