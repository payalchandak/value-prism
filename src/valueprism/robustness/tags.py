"""Agreement of the re-annotations, and how the profiles move when tags change.

The re-annotation figures follow values-analysis ``value_annotations.py`` and
``value_annotation_agreement.ipynb``. Erosion follows ``tag_noise.py`` and
``tag_noise_robustness.ipynb``: each non-neutral tag is set to neutral with
probability p, in one draw, with no validity check. The draw is seeded
``42 + 1000 * p_index + k``. ``p_index`` is the rate's place in the notebook
grid, which starts at 0 and includes rates that are not reported here. Cases
are walked in sorted case-id order, and each case draws eight uniform numbers.
"""

from __future__ import annotations

import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from scipy.stats import kendalltau
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationWarning,
)

from valueprism.attribution import COLUMNS, T_STAR
from valueprism.data import TAG_SCORE

FIELDS = ["autonomy", "beneficence", "nonmaleficence", "justice"]
# tag_noise_robustness.ipynb P_GRID, then the shorter grid used for Kendall τ.
SHIFT_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
TAU_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
SEED = 42
_LEVELS = (-1, 0, 1)


def load_ratings(path) -> pd.DataFrame:
    """Long ratings: rater, case_id, choice (1 or 2), value, tag, label."""
    frame = pd.read_csv(path, dtype={"rater": str, "case_id": str, "value": str, "tag": str})
    expected = ["rater", "case_id", "choice", "value", "tag"]
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(f"ratings file is missing {missing}")
    frame["choice"] = frame["choice"].astype(int)
    frame["tag"] = frame["tag"].str.strip().str.lower()
    frame["label"] = frame["tag"].map(TAG_SCORE)
    if frame["label"].isna().any() or not frame["choice"].isin([1, 2]).all():
        raise ValueError("ratings have an unknown tag or choice")
    if not frame["value"].isin(FIELDS).all():
        raise ValueError("ratings have an unknown value")
    frame["label"] = frame["label"].astype(int)
    return frame


def prepare(cases: pd.DataFrame, decisions: pd.DataFrame):
    """Alignment tensor, one count vector per LLM, and the pooled physicians.

    ``alignment`` has shape (n_cases, 2, 4) in sorted case-id order. Counts are
    aligned to that order. A case with no non-missing response has a zero trial
    count and is left out of the fit.
    """
    cases = cases.sort_values("case_id").reset_index(drop=True)
    case_ids = cases["case_id"].tolist()
    alignment = np.zeros((len(case_ids), 2, len(FIELDS)), dtype=np.int8)
    for index, field in enumerate(FIELDS):
        for side, column in ((0, f"{field}_1"), (1, f"{field}_2")):
            scores = cases[column].map(TAG_SCORE)
            if scores.isna().any():
                raise ValueError(f"benchmark has an unknown {field} tag")
            alignment[:, side, index] = scores.to_numpy()

    def counts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        choice_1 = np.zeros(len(case_ids), dtype=np.float64)
        trials = np.zeros(len(case_ids), dtype=np.float64)
        answered = frame.loc[frame["choice"].notna(), ["case_id", "choice"]]
        position = {case_id: i for i, case_id in enumerate(case_ids)}
        for case_id, choice in zip(answered["case_id"], answered["choice"]):
            row = position.get(case_id)
            if row is None:
                raise ValueError(f"unknown case_id: {case_id}")
            trials[row] += 1.0
            if choice == 1:
                choice_1[row] += 1.0
        return choice_1, trials

    llms = []
    makers = decisions.loc[decisions["group"] == "llm", "decision_maker"].unique()
    for maker in sorted(makers):
        rows = decisions.loc[decisions["decision_maker"] == maker]
        llms.append((maker, counts(rows)))
    physicians = counts(decisions.loc[decisions["group"] == "physician"])
    return case_ids, alignment, llms, physicians


def deltas(alignment: np.ndarray) -> np.ndarray:
    """Δ = tag(choice 1) − tag(choice 2), shape (n_cases, 4)."""
    return alignment[:, 0, :].astype(np.float64) - alignment[:, 1, :].astype(np.float64)


def _profile(delta: np.ndarray, counts) -> np.ndarray:
    """softmax(w / T*) for one decision maker. A failed fit is uniform."""
    choice_1, trials = counts
    keep = trials > 0
    design = delta[keep]
    n = trials[keep]
    proportions = choice_1[keep] / n
    if (
        proportions.size == 0
        or np.all(proportions == 0.0)
        or np.all(proportions == 1.0)
        or np.all(design == 0.0)
    ):
        weights = np.zeros(design.shape[1] if design.size else 4, dtype=np.float64)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", HessianInversionWarning)
                warnings.simplefilter("ignore", PerfectSeparationWarning)
                family = sm.families.Binomial(link=sm.families.links.Logit())
                result = sm.GLM(
                    proportions, design, family=family, freq_weights=n
                ).fit(disp=False, cov_type="HC3")
            weights = np.asarray(result.params, dtype=np.float64)
        except Exception:
            weights = np.zeros(design.shape[1], dtype=np.float64)
    return softmax(np.asarray(weights, dtype=np.float64) / T_STAR)


def jsd(profile: np.ndarray, reference: np.ndarray) -> float:
    """Squared base-2 Jensen-Shannon divergence."""
    return float(jensenshannon(profile, reference, base=2.0) ** 2)


def distance_table(alignment: np.ndarray, llms, physicians):
    """Each LLM's profile and its JSD to the pooled physician profile.

    LLMs are returned in sorted id order. The JSD vector in that order is what
    Kendall's τ is computed on.
    """
    delta = deltas(alignment)
    pooled = _profile(delta, physicians)
    rows = []
    for maker, counts in llms:
        profile = _profile(delta, counts)
        rows.append((maker, profile, jsd(profile, pooled)))
    return rows


def rating_units(ratings: pd.DataFrame):
    grouped = ratings.groupby(["case_id", "choice", "value"], sort=False)["label"]
    return [(key, [int(label) for label in labels]) for key, labels in grouped]


def production_labels(cases: pd.DataFrame) -> dict:
    labels = {}
    for row in cases.itertuples(index=False):
        for choice, side in ((1, "1"), (2, "2")):
            for field in FIELDS:
                tag = getattr(row, f"{field}_{side}")
                score = TAG_SCORE.get(tag)
                if score is None:
                    raise ValueError(f"unknown tag {tag!r}")
                labels[(row.case_id, choice, field)] = score
    return labels


def krippendorff_alpha(units, metric: str = "ordinal") -> float:
    """Ordinal or nominal α. ``units`` is a list of label lists."""
    if metric not in ("nominal", "ordinal"):
        raise ValueError(f"metric must be 'nominal' or 'ordinal', got {metric!r}")
    level = {lab: i for i, lab in enumerate(_LEVELS)}
    observed = np.zeros((3, 3), dtype=np.float64)
    for labels in units:
        m = len(labels)
        if m < 2:
            continue
        counts = np.zeros(3, dtype=np.float64)
        for label in labels:
            counts[level[label]] += 1.0
        for left in range(3):
            for right in range(3):
                if left == right:
                    observed[left, right] += counts[left] * (counts[left] - 1.0) / (m - 1.0)
                else:
                    observed[left, right] += counts[left] * counts[right] / (m - 1.0)
    marginal = observed.sum(axis=1)
    n = float(marginal.sum())
    if n == 0.0:
        return float("nan")
    if metric == "nominal":
        difference = 1.0 - np.eye(3)
    else:
        difference = np.zeros((3, 3), dtype=np.float64)
        for left in range(3):
            for right in range(3):
                lo, hi = (left, right) if left <= right else (right, left)
                span = marginal[lo : hi + 1].sum() - (marginal[left] + marginal[right]) / 2.0
                difference[left, right] = span * span
    observed_disagreement = float((observed * difference).sum() / n)
    expected = float((np.outer(marginal, marginal) * difference).sum() / (n * (n - 1.0)))
    if expected == 0.0:
        return 1.0
    return 1.0 - observed_disagreement / expected


def pairwise_exact(ratings: pd.DataFrame) -> tuple[float, int]:
    """Pooled fraction of co-rating pairs with the same label, and the pair count."""
    exact = pairs = 0
    for _key, labels in rating_units(ratings):
        for left, right in combinations(labels, 2):
            pairs += 1
            exact += left == right
    if pairs == 0:
        return float("nan"), 0
    return exact / pairs, pairs


def benchmark_agreement(ratings: pd.DataFrame, cases: pd.DataFrame) -> float:
    """Fraction of individual ratings that equal the benchmark tag."""
    production = production_labels(cases)
    hits = 0
    for row in ratings.itertuples(index=False):
        hits += row.label == production[(row.case_id, int(row.choice), row.value)]
    return hits / len(ratings) if len(ratings) else float("nan")


def neutral_disagreements(ratings: pd.DataFrame) -> tuple[int, int]:
    """Pairs that disagree, and how many of those are neutral against a side."""
    disagreed = boundary = 0
    for _key, labels in rating_units(ratings):
        for left, right in combinations(labels, 2):
            if left == right:
                continue
            disagreed += 1
            boundary += abs(left - right) == 1
    return boundary, disagreed


def majority_split(ratings: pd.DataFrame, cases: pd.DataFrame) -> tuple[int, int, int]:
    """Cells whose strict majority matches the benchmark, differs, or ties."""
    production = production_labels(cases)
    same = different = tied = 0
    for key, labels in rating_units(ratings):
        winner = _majority(labels, None)
        if winner is None:
            tied += 1
        elif winner == production[key]:
            same += 1
        else:
            different += 1
    return same, different, tied


def _majority(labels: list[int], fallback: int | None) -> int | None:
    counts = Counter(labels)
    top = max(counts.values())
    winners = [label for label, count in counts.items() if count == top]
    if len(winners) == 1:
        return int(winners[0])
    return fallback


def majority_alignment(alignment: np.ndarray, case_ids: list[str], ratings: pd.DataFrame):
    """Physician majority. A tie, or a cell with no rating, keeps the benchmark tag."""
    out = alignment.copy()
    position = {case_id: i for i, case_id in enumerate(case_ids)}
    value_index = {field: i for i, field in enumerate(FIELDS)}
    grouped = ratings.groupby(["case_id", "choice", "value"])["label"].apply(list)
    for (case_id, choice, value), labels in grouped.items():
        row = position[case_id]
        choice_index = int(choice) - 1
        column = value_index[value]
        out[row, choice_index, column] = _majority(
            [int(label) for label in labels], int(alignment[row, choice_index, column])
        )
    return out


def value_order(profile: np.ndarray) -> tuple[str, ...]:
    """Values from largest weight to smallest. Ties keep A, B, N, J order."""
    ranked = sorted(range(len(COLUMNS)), key=lambda i: -profile[i])
    return tuple(COLUMNS[i] for i in ranked)


def erode(alignment: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Set each non-neutral tag to neutral with probability p. One unconditional draw."""
    if p <= 0.0:
        return alignment.copy()
    mask = (rng.random(size=alignment.shape) < p) & (alignment != 0)
    out = alignment.copy()
    out[mask] = 0
    return out


def realized_erosion(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Fraction of non-neutral tags that became neutral."""
    engaged = clean != 0
    total = int(engaged.sum())
    if total == 0:
        return 0.0
    return int(((noisy == 0) & engaged).sum()) / total


def _seed(grid: tuple[float, ...], p: float, draw: int) -> int:
    return SEED + 1000 * grid.index(p) + draw


def erosion_shifts(alignment, llms, rates, n_runs: int = 20):
    """Mean absolute profile change, in percentage points, over the LLMs.

    Returns one tuple per rate: (p, per-value means in A B N J order, mean of
    those four, mean realized erosion).
    """
    clean = [_profile(deltas(alignment), counts) for _maker, counts in llms]
    rows = []
    for p in rates:
        total = np.zeros(4, dtype=np.float64)
        realized = []
        for draw in range(n_runs):
            rng = np.random.default_rng(_seed(SHIFT_GRID, p, draw))
            noisy = erode(alignment, p, rng)
            realized.append(realized_erosion(alignment, noisy))
            delta = deltas(noisy)
            for base, (_maker, counts) in zip(clean, llms):
                total += np.abs(_profile(delta, counts) - base)
        mean = 100.0 * total / (n_runs * len(llms))
        rows.append((p, mean, float(mean.mean()), float(np.mean(realized))))
    return rows


def erosion_tau(alignment, llms, physicians, rates, n_runs: int = 100):
    """Mean Kendall τ of the LLM distance ordering, and mean realized erosion.

    τ compares the noisy JSD vector to the clean one. Both vectors are in
    sorted model-id order, which is the order of ``llms``.
    """
    clean_rows = distance_table(alignment, llms, physicians)
    clean = np.array([row[2] for row in clean_rows])
    rows = []
    for p in rates:
        taus = []
        realized = []
        for draw in range(n_runs):
            rng = np.random.default_rng(_seed(TAU_GRID, p, draw))
            noisy = erode(alignment, p, rng)
            realized.append(realized_erosion(alignment, noisy))
            jsds = np.array([row[2] for row in distance_table(noisy, llms, physicians)])
            taus.append(float(kendalltau(clean, jsds).statistic))
        rows.append((p, float(np.mean(taus)), float(np.mean(realized))))
    return rows
