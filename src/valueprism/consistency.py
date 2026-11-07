"""Entropy, agreement, and Fleiss' κ for decisions on the benchmark.

Refusals are removed per case before any statistic below is computed.
The definitions match values-analysis.

Binary Shannon entropy is the base-2 entropy of the choice split.
Near-perfect agreement means at least 90% of the remaining responses
select the same choice. Fleiss' κ is the statistic implemented in
``statsmodels.stats.inter_rater.fleiss_kappa``. Value complexity groups
cases by how many columns of Δ are nonzero.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa as fleiss_kappa_table

# At least 90% of the non-refusal responses. For 10 runs, 9 or 10.
NEAR_PERFECT = 0.9

LETTERS = ["A", "B", "N", "J"]


@dataclass(frozen=True)
class EntropyStatistics:
    """Summaries of one list of case entropies.

    ``p25`` and ``p75`` are numpy's linear percentiles, and ``n`` counts
    only cases that still had a non-refusal response.
    """

    mean: float
    median: float
    p25: float
    p75: float
    n: int


def binary_entropy(k: int, n: int) -> float:
    """Shannon entropy, in bits, of ``k`` choice-1 votes out of ``n``.

    The result is 0 when ``n`` is 0 or every vote selected the same choice.
    """
    if n == 0:
        return 0.0
    p = k / n
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _counts(choice: pd.Series) -> tuple[int, int]:
    """Choice-1 and choice-2 counts. Missing choices are refusals."""
    return int((choice == 1).sum()), int((choice == 2).sum())


def _splits(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        n1, n2 = _counts(group["choice"])
        rows.append((*key, n1, n2))
    return pd.DataFrame(rows, columns=[*keys, "n1", "n2"])


def _with_entropy(counts: pd.DataFrame) -> pd.DataFrame:
    entropy = []
    near_perfect = []
    majority = []
    for n1, n2 in zip(counts["n1"], counts["n2"]):
        n = int(n1) + int(n2)
        if n == 0:
            entropy.append(math.nan)
            near_perfect.append(math.nan)
            majority.append(math.nan)
            continue
        top = max(int(n1), int(n2))
        entropy.append(binary_entropy(int(n1), n))
        near_perfect.append(float(top / n >= NEAR_PERFECT))
        majority.append(top / n)
    out = counts.copy()
    out["entropy"] = entropy
    out["near_perfect"] = near_perfect
    out["majority"] = majority
    return out


def llm_cases(decisions: pd.DataFrame) -> pd.DataFrame:
    """One row per model and case: vote counts, entropy, near-perfect flag.

    ``near_perfect`` is 1 when at least 90% of that model's non-refusal
    responses on the case agree, and missing when the model refused every run.
    """
    models = decisions.loc[decisions["group"] == "llm"]
    return _with_entropy(_splits(models, ["decision_maker", "case_id"]))


def physician_cases(decisions: pd.DataFrame) -> pd.DataFrame:
    """One row per case: the pooled physician votes, entropy, and majority share."""
    physicians = decisions.loc[decisions["group"] == "physician"]
    return _with_entropy(_splits(physicians, ["case_id"]))


def entropy_statistics(values) -> EntropyStatistics:
    """Mean, median, and quartiles of case entropies, ignoring missing values."""
    array = np.asarray(pd.Series(values).dropna(), dtype=float)
    if array.size == 0:
        raise ValueError("no valid entropy values")
    return EntropyStatistics(
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        p25=float(np.percentile(array, 25)),
        p75=float(np.percentile(array, 75)),
        n=int(array.size),
    )


def fleiss_kappa(cases: pd.DataFrame) -> float:
    """Fleiss' κ for a table of choice-1 and choice-2 counts.

    Cases with no non-refusal vote are left out. The number is
    ``statsmodels.stats.inter_rater.fleiss_kappa`` on the remaining rows,
    which requires the same number of votes on every case.
    """
    voted = cases.loc[cases["n1"] + cases["n2"] > 0, ["n1", "n2"]]
    return float(fleiss_kappa_table(voted.to_numpy()))


def cases_below_endorsement(cases: pd.DataFrame, threshold: float = 0.70) -> int:
    """Cases where neither choice reaches ``threshold`` of the non-refusal votes.

    The comparison is strict, matching the 70% count in
    ``benchmark_properties.ipynb``: the majority share is below the threshold.
    """
    majority = cases["majority"].dropna()
    return int((majority < threshold).sum())


def case_entropy_correlation(reference: pd.Series, model: pd.Series) -> tuple[float, float, int]:
    """Spearman ρ between two per-case entropy series, aligned on the index.

    Cases missing either entropy are dropped. Returns ρ, the two-sided
    p-value, and the number of cases used.
    """
    joined = pd.concat({"reference": reference, "model": model}, axis=1).dropna()
    if len(joined) < 3:
        raise ValueError(f"need at least 3 cases with both entropies, got {len(joined)}")
    rho, pvalue = spearmanr(joined["reference"], joined["model"])
    return float(rho), float(pvalue), int(len(joined))


def value_complexity(delta: pd.DataFrame, cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify model entropy by the number of engaged values.

    An engaged value is a nonzero column of Δ. The first frame has one row
    per count ``k``: how many cases, and the mean entropy over every model
    on those cases. The second frame has two Spearman tests of ``k`` against
    a per-case outcome: the mean model entropy, and the share of models with
    near-perfect agreement.
    """
    engaged = (delta[LETTERS] != 0).sum(axis=1)
    valid = cases.dropna(subset=["entropy"]).copy()
    valid["k"] = valid["case_id"].map(engaged)
    if valid["k"].isna().any():
        missing = sorted(set(valid.loc[valid["k"].isna(), "case_id"]))
        raise ValueError(f"cases missing from Δ: {missing}")
    valid["k"] = valid["k"].astype(int)

    table = (
        valid.groupby("k", sort=True)
        .agg(n_cases=("case_id", "nunique"), mean_entropy=("entropy", "mean"))
        .reset_index()
    )
    per_case = valid.groupby("case_id", sort=False).agg(
        k=("k", "first"),
        mean_entropy=("entropy", "mean"),
        agreement_share=("near_perfect", "mean"),
    )
    tests = []
    for outcome in ("mean_entropy", "agreement_share"):
        rho, pvalue = spearmanr(per_case["k"], per_case[outcome])
        tests.append({"outcome": outcome, "rho": float(rho), "p": float(pvalue)})
    return table, pd.DataFrame(tests)
