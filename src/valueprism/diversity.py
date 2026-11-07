"""Mean pairwise JSD within a group, and tests that compare two groups.

D̄(G) is the mean pairwise Jensen-Shannon divergence inside G. The
divergence is base 2: ``scipy`` returns the distance, and squaring it
puts the value in [0, 1]. A pair of identical profiles has divergence 0.

The bootstrap draws decision makers with replacement, independently in
each group. A pair of draws that land on the same original unit has
divergence 0, and that pair is left out of the within-draw average.
The permutation test pools the decision makers and reassigns them to
two groups of the original sizes; every pair stays in its average.
Both read a JSD matrix built once; the resampling only indexes that
matrix. Pass ``seed`` for a fixed ``numpy`` Generator. The
values-analysis notebook uses 42.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


class BootstrapResult:
    """Mean and percentile interval of one bootstrap sample."""

    def __init__(self, samples: np.ndarray):
        self.samples = np.asarray(samples, dtype=np.float64)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    def ci(self, confidence: float = 95) -> tuple[float, float]:
        alpha = (100 - confidence) / 2
        lower = float(np.percentile(self.samples, alpha))
        upper = float(np.percentile(self.samples, 100 - alpha))
        return lower, upper


def _jsd_matrix(profiles: dict[str, dict[str, float]], ids: list[str]) -> np.ndarray:
    """Square pairwise JSD, in the order of ``ids``. Diagonal is 0."""
    value_names = list(profiles[ids[0]].keys())
    vectors = np.array(
        [[profiles[mid][name] for name in value_names] for mid in ids],
        dtype=np.float64,
    )
    n = len(ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            divergence = jensenshannon(vectors[i], vectors[j], base=2.0) ** 2
            matrix[i, j] = divergence
            matrix[j, i] = divergence
    return matrix


def _mean_within_jsd(
    jsd_mat: np.ndarray,
    indices: np.ndarray,
    triu: tuple[np.ndarray, np.ndarray],
) -> float:
    """Mean of the upper triangle of ``jsd_mat`` at ``indices``.

    A repeated index contributes the zero on that pair. The permutation
    test passes each original unit once, so every pair is two units.
    The triangle is the fixed index pair of a group of this size.
    """
    sub = jsd_mat[np.ix_(indices, indices)]
    return float(sub[triu].mean())


def _mean_cross_unit_jsd(
    jsd_mat: np.ndarray,
    indices: np.ndarray,
    triu: tuple[np.ndarray, np.ndarray],
) -> float:
    """Mean JSD of upper-triangle pairs that name different original units.

    ``triu`` is the upper triangle of a group of this size. A bootstrap
    draw repeats units, and a pair of positions that both landed on one
    unit is left out. Pairs of different units keep the multiplicity of
    the draw. A draw with no such pair has no within-group divergence
    and returns NaN.
    """
    left = indices[triu[0]]
    right = indices[triu[1]]
    keep = left != right
    if not np.any(keep):
        return float("nan")
    return float(jsd_mat[left[keep], right[keep]].mean())


def pairwise_jsd(profiles: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Pairwise JSD for ``profiles``, in insertion order.

    The frame is symmetric and its diagonal is 0. Labels are the
    decision-maker ids.
    """
    ids = list(profiles.keys())
    frame = pd.DataFrame(_jsd_matrix(profiles, ids), index=ids, columns=ids)
    frame.index.name = "decision_maker"
    return frame


def mean_pairwise_jsd(matrix: np.ndarray) -> float:
    """D̄ of one square JSD matrix: the mean of the strict upper triangle."""
    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 profiles, got {n}")
    return float(matrix[np.triu_indices(n, k=1)].mean())


def bootstrap_mean_jsd(
    profiles: dict[str, dict[str, float]],
    group_a_ids: list[str],
    group_b_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Bootstrap CIs for D̄(A), D̄(B), and D̄(A) − D̄(B).

    Each draw resamples that group's ids with replacement. The mean
    inside a draw averages JSD over pairs drawn from different original
    units. The returned ``BootstrapResult.mean`` is the mean of those
    draws, not D̄ on the original profiles.

    Keys are ``group_a_mean``, ``group_b_mean``, and ``difference``.
    Either group with fewer than 2 members raises ``ValueError``.
    """
    n_a, n_b = len(group_a_ids), len(group_b_ids)
    if n_a < 2:
        raise ValueError(f"group_a must have >= 2 members, got {n_a}")
    if n_b < 2:
        raise ValueError(f"group_b must have >= 2 members, got {n_b}")

    all_ids = list(dict.fromkeys(group_a_ids + group_b_ids))
    id_to_idx = {mid: i for i, mid in enumerate(all_ids)}
    jsd_mat = _jsd_matrix(profiles, all_ids)
    idx_a = np.array([id_to_idx[mid] for mid in group_a_ids])
    idx_b = np.array([id_to_idx[mid] for mid in group_b_ids])
    triu_a = np.triu_indices(n_a, k=1)
    triu_b = np.triu_indices(n_b, k=1)

    rng = np.random.default_rng(seed)
    samples_a = np.empty(n_bootstrap)
    samples_b = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        boot_a = idx_a[rng.integers(0, n_a, size=n_a)]
        boot_b = idx_b[rng.integers(0, n_b, size=n_b)]
        samples_a[draw] = _mean_cross_unit_jsd(jsd_mat, boot_a, triu_a)
        samples_b[draw] = _mean_cross_unit_jsd(jsd_mat, boot_b, triu_b)

    return {
        "group_a_mean": BootstrapResult(samples=samples_a),
        "group_b_mean": BootstrapResult(samples=samples_b),
        "difference": BootstrapResult(samples=samples_a - samples_b),
    }


def permutation_test_jsd(
    profiles: dict[str, dict[str, float]],
    group_a_ids: list[str],
    group_b_ids: list[str],
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Permutation test of D̄(A) = D̄(B).

    The statistic is |D̄(A) − D̄(B)| on the original profiles. Each
    permutation shuffles the pooled ids in place and splits them into
    groups of the original sizes. The p-value is the share of permuted
    statistics at least as large as the observed one.

    Keys are ``observed_diff``, ``p_value``, and ``null_distribution``.
    Either group with fewer than 2 members raises ``ValueError``.
    """
    n_a, n_b = len(group_a_ids), len(group_b_ids)
    if n_a < 2:
        raise ValueError(f"group_a must have >= 2 members, got {n_a}")
    if n_b < 2:
        raise ValueError(f"group_b must have >= 2 members, got {n_b}")

    all_ids = list(dict.fromkeys(group_a_ids + group_b_ids))
    id_to_idx = {mid: i for i, mid in enumerate(all_ids)}
    jsd_mat = _jsd_matrix(profiles, all_ids)
    idx_a = np.array([id_to_idx[mid] for mid in group_a_ids])
    idx_b = np.array([id_to_idx[mid] for mid in group_b_ids])
    triu_a = np.triu_indices(n_a, k=1)
    triu_b = np.triu_indices(n_b, k=1)
    observed_diff = abs(
        _mean_within_jsd(jsd_mat, idx_a, triu_a) - _mean_within_jsd(jsd_mat, idx_b, triu_b)
    )

    pooled = np.concatenate([idx_a, idx_b])
    rng = np.random.default_rng(seed)
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(pooled)
        null_dist[i] = abs(
            _mean_within_jsd(jsd_mat, pooled[:n_a], triu_a)
            - _mean_within_jsd(jsd_mat, pooled[n_a:], triu_b)
        )

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(np.mean(null_dist >= observed_diff)),
        "null_distribution": null_dist,
    }
