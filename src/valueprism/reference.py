"""Bootstrap reference distribution for value-profile divergence.

The procedure matches values-analysis ``src/analysis/value_profiles.ipynb``.
Each of 10,000 iterations resamples the physicians with replacement, at
panel size. ``numpy.random.Generator.choice`` indexes the physician ids
sorted ascending, with seed 42. Public ids are assigned in that same
order, so the sorted ids are the sequence the notebook drew.

For each position j the pooled GLM is refit on the distinct physicians
in the panel other than the physician at j, and JSD(π_j, π_LOO) is
recorded. The same iteration refits the pooled profile on the distinct
physicians who were drawn and records each model's JSD to it. A
physician drawn more than once contributes one count vector to a fit.
The divergence is still recorded once per position, so a repeated draw
is a repeated value in the reference distribution.

JSD is the base-2 Jensen-Shannon divergence, the square of
``scipy.spatial.distance.jensenshannon``. A model's p-value is the
fraction of reference JSDs at least as large as its JSD to the pooled
profile of the original panel. The 95% interval is the 2.5 and 97.5
percentiles of that model's draws.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationWarning,
)

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights

N_BOOTSTRAP = 10_000
SEED = 42


@dataclass(frozen=True)
class ModelReference:
    """One model's divergence from the original physician profile."""

    decision_maker: str
    jsd: float
    ci_low: float
    ci_high: float
    p_value: float
    autonomy: float


@dataclass(frozen=True)
class BootstrapReference:
    """Summaries of one run. ``models`` is ordered by increasing JSD.

    ``reference_jsds`` is every physician draw, in resampling order.
    """

    n: int
    mean: float
    median: float
    percentile_95: float
    models: tuple[ModelReference, ...]
    reference_jsds: np.ndarray


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(jensenshannon(left, right, base=2.0) ** 2)


def _tallies(
    delta: pd.DataFrame, decisions: pd.DataFrame, physician_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Choice-1 counts and trial counts, one row per physician, cases sorted."""
    if len(set(physician_ids)) != len(physician_ids):
        raise ValueError("physician_ids has duplicates")
    present = set(decisions.loc[decisions["group"] == "physician", "decision_maker"])
    if present != set(physician_ids):
        missing = sorted(present - set(physician_ids))
        extra = sorted(set(physician_ids) - present)
        raise ValueError(
            f"physician panel does not match the decisions: missing {missing}, extra {extra}"
        )

    design = delta[COLUMNS].sort_index()
    case_at = {case_id: i for i, case_id in enumerate(design.index)}
    index_of = {pid: i for i, pid in enumerate(physician_ids)}
    choice_1 = np.zeros((len(physician_ids), len(case_at)), dtype=np.float64)
    trials = np.zeros_like(choice_1)
    answered = decisions.loc[
        (decisions["group"] == "physician") & decisions["choice"].notna(),
        ["decision_maker", "case_id", "choice"],
    ]
    for maker, case_id, choice in zip(
        answered["decision_maker"], answered["case_id"], answered["choice"]
    ):
        column = case_at.get(case_id)
        if column is None:
            raise ValueError(f"unknown case_id: {case_id}")
        row = index_of[maker]
        trials[row, column] += 1.0
        if choice == 1:
            choice_1[row, column] += 1.0

    empty = [pid for pid, row in index_of.items() if trials[row].sum() == 0]
    if empty:
        raise ValueError(f"no non-missing responses: {empty}")
    return choice_1, trials, design.to_numpy(dtype=np.float64), index_of


def _profile_fn(choice_1, trials, design, index_of):
    """Softmax profile of a physician subset, cached on the member set."""
    family = sm.families.Binomial(link=sm.families.links.Logit())
    cache: dict[frozenset[str], np.ndarray] = {}

    def profile(members: frozenset[str]) -> np.ndarray:
        cached = cache.get(members)
        if cached is not None:
            return cached
        if not members:
            raise ValueError("physician subset is empty")
        rows = [index_of[pid] for pid in members]
        n = trials[rows].sum(axis=0)
        keep = n > 0
        if not np.any(keep):
            raise ValueError("physician subset has no non-missing responses")
        nn = n[keep]
        proportions = choice_1[rows].sum(axis=0)[keep] / nn
        used = design[keep]
        # The notebook's fit returns zeros, and a uniform profile, when every
        # proportion is 0 or 1, the design is constant, or the GLM raises.
        if np.all(proportions == 0.0) or np.all(proportions == 1.0) or np.all(used == 0.0):
            weights = np.zeros(used.shape[1], dtype=np.float64)
        else:
            try:
                result = sm.GLM(proportions, used, family=family, freq_weights=nn).fit(disp=False)
                weights = np.asarray(result.params, dtype=np.float64)
            except Exception:
                weights = np.zeros(used.shape[1], dtype=np.float64)
        fitted = softmax(np.asarray(weights, dtype=np.float64) / T_STAR)
        cache[members] = fitted
        return fitted

    return profile


def bootstrap_reference(
    delta: pd.DataFrame,
    decisions: pd.DataFrame,
    physician_ids: list[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> BootstrapReference:
    """Draw the reference distribution and each model's interval.

    Physicians are resampled in sorted id order. ``physician_ids`` must
    list each physician in ``decisions`` once. The order of the argument
    is not used.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    physician_ids = sorted(physician_ids)
    choice_1, trials, design, index_of = _tallies(delta, decisions, physician_ids)
    model_ids = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    if len(model_ids) == 0:
        raise ValueError("no LLM decisions")

    model_profiles: dict[str, np.ndarray] = {}
    autonomy: dict[str, float] = {}
    for model_id in model_ids:
        rows = decisions.loc[decisions["decision_maker"] == model_id]
        fitted = softmax_profile(value_weights(delta, rows), T_STAR)
        model_profiles[model_id] = fitted[COLUMNS].to_numpy(dtype=np.float64)
        autonomy[model_id] = float(fitted["A"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", HessianInversionWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        profile = _profile_fn(choice_1, trials, design, index_of)
        individual = {pid: profile(frozenset({pid})) for pid in physician_ids}
        pooled = profile(frozenset(physician_ids))
        observed = {
            model_id: _jsd(vector, pooled) for model_id, vector in model_profiles.items()
        }

        n_physicians = len(physician_ids)
        reference = np.empty(n_bootstrap * n_physicians, dtype=np.float64)
        model_draws = {
            model_id: np.empty(n_bootstrap, dtype=np.float64) for model_id in model_ids
        }
        rng = np.random.default_rng(seed)
        cursor = 0
        for draw in range(n_bootstrap):
            panel = [
                str(pid)
                for pid in rng.choice(physician_ids, size=n_physicians, replace=True)
            ]
            members = frozenset(panel)
            for physician in panel:
                reference[cursor] = _jsd(individual[physician], profile(members - {physician}))
                cursor += 1
            resampled = profile(members)
            for model_id, vector in model_profiles.items():
                model_draws[model_id][draw] = _jsd(vector, resampled)

    models = []
    for model_id in model_ids:
        jsd = observed[model_id]
        low, high = np.percentile(model_draws[model_id], [2.5, 97.5])
        models.append(
            ModelReference(
                decision_maker=model_id,
                jsd=float(jsd),
                ci_low=float(low),
                ci_high=float(high),
                p_value=float(np.mean(reference >= jsd)),
                autonomy=autonomy[model_id],
            )
        )
    models.sort(key=lambda row: (row.jsd, row.decision_maker))
    return BootstrapReference(
        n=int(reference.size),
        mean=float(reference.mean()),
        median=float(np.median(reference)),
        percentile_95=float(np.percentile(reference, 95)),
        models=tuple(models),
        reference_jsds=reference,
    )
