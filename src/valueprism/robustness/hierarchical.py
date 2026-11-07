"""Partially pooled value weights.

One binomial logit for every decision maker and case, with no intercept:

    logit p = (w_group + u) · Δ + a
    u = τ z
    a = σ Q z_case

w is a group mean, one vector for the LLMs and one for the physicians.
The subject term is non-centered: z is standard normal and τ, the scale
for that group and value, is HalfNormal(10). Q is an orthonormal basis
for the null space of Δᵀ. On this benchmark that basis has 46 columns,
so Δᵀ a = 0 and a case effect cannot imitate a value weight. w is
Normal(0, 10) and σ is HalfNormal(10).

Sampling is NUTS: 4 chains, 2000 tuning draws, 2000 retained draws,
target acceptance 0.9, seed 0. Those are the settings in
hierarchical_profiles.ipynb.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import softmax

from valueprism.attribution import COLUMNS, T_STAR

GROUP_ORDER = ("LLM", "physician")
DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.9
SEED = 0
W_SD = 10.0
TAU_SD = 10.0
SIGMA_CASE_SD = 10.0

# Public physician ids were assigned by sorting the source ids, which is
# the order the notebook passed to the sampler. Subject rows follow that
# order: model ids sorted ascending, then P01–P25.
GROUP_MEAN = "physician_group_mean"


class HierarchicalFit:
    """One NUTS fit.

    ``w`` has shape (draws, groups, values), groups in ``GROUP_ORDER`` and
    values in A, B, N, J. ``u`` has shape (draws, subjects, values), subjects
    in ``subjects`` order. Both arrays stack chains and then draws.
    """

    def __init__(
        self,
        subjects: list[str],
        group_of: dict[str, str],
        w: np.ndarray,
        u: np.ndarray,
        n_case_effects: int,
        diagnostics: dict,
        summary: pd.DataFrame,
    ):
        self.subjects = subjects
        self.group_of = group_of
        self.w = w
        self.u = u
        self.n_case_effects = n_case_effects
        self.diagnostics = diagnostics
        self.summary = summary


def observation_frame(delta: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """One row per decision maker and case with a non-missing response.

    Cases are sorted by id. Within a case the rows are the LLMs in id
    order, then the physicians in public-id order. ``successes`` counts
    choice 1. ``n_trials`` counts non-missing runs. Refusals and
    incoherent runs are already missing in ``decisions``.
    """
    missing = [column for column in COLUMNS if column not in delta.columns]
    if missing:
        raise ValueError(f"Δ is missing {missing}")
    answered = decisions.loc[
        decisions["choice"].notna(), ["decision_maker", "group", "case_id", "choice"]
    ]
    if answered.empty:
        raise ValueError("no non-missing responses")

    llm_ids = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    physician_ids = sorted(
        decisions.loc[decisions["group"] == "physician", "decision_maker"].unique()
    )
    if set(llm_ids) & set(physician_ids):
        raise ValueError("a decision maker is in both groups")

    trials: dict[tuple[str, str], int] = {}
    choice_1: dict[tuple[str, str], int] = {}
    makers = answered["decision_maker"].to_numpy()
    cases = answered["case_id"].to_numpy()
    choices = answered["choice"].astype(int).to_numpy()
    for maker, case_id, choice in zip(makers, cases, choices):
        key = (str(maker), str(case_id))
        trials[key] = trials.get(key, 0) + 1
        if choice == 1:
            choice_1[key] = choice_1.get(key, 0) + 1

    case_ids = sorted({case_id for _maker, case_id in trials})
    unknown = [case_id for case_id in case_ids if case_id not in delta.index]
    if unknown:
        raise ValueError(f"unknown case_id: {unknown[:5]}")

    rows = []
    subjects = [(maker, "LLM") for maker in llm_ids] + [
        (maker, "physician") for maker in physician_ids
    ]
    for case_id in case_ids:
        differences = delta.loc[case_id, list(COLUMNS)]
        for maker, group in subjects:
            n_trials = trials.get((maker, case_id), 0)
            if n_trials == 0:
                continue
            successes = choice_1.get((maker, case_id), 0)
            rows.append(
                {
                    "subject": maker,
                    "case": case_id,
                    "group": group,
                    "successes": successes,
                    "n_trials": n_trials,
                    **{column: float(differences[column]) for column in COLUMNS},
                }
            )
    if not rows:
        raise ValueError("no (subject, case) observations with a response")
    return pd.DataFrame(rows)


def _case_basis(
    delta_rows: np.ndarray, case_index: np.ndarray, n_cases: int
) -> np.ndarray:
    """Orthonormal columns spanning the null space of the case-level Δᵀ."""
    n_values = delta_rows.shape[1]
    delta_case = np.empty((n_cases, n_values), dtype=np.float64)
    for index in range(n_cases):
        rows = delta_rows[case_index == index]
        if len(rows) == 0:
            raise ValueError(f"case {index} has no rows")
        delta_case[index] = rows[0]
        if not np.allclose(rows, rows[0], atol=1e-12, rtol=1e-12):
            raise ValueError("Δ must be constant within a case")

    _u, singular_values, vh = np.linalg.svd(delta_case.T, full_matrices=True)
    tolerance = (
        max(delta_case.T.shape) * np.finfo(np.float64).eps * singular_values.max(initial=0.0)
    )
    rank = int(np.sum(singular_values > tolerance))
    if rank != n_values:
        raise ValueError(f"Δ must have rank {n_values}, found {rank}")
    if n_cases <= rank:
        raise ValueError(f"need more cases than values, found {n_cases}")

    basis = vh[rank:].T
    if not np.allclose(delta_case.T @ basis, 0.0, atol=1e-10):
        raise RuntimeError("case-effect basis is not orthogonal to Δ")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
        raise RuntimeError("case-effect basis is not orthonormal")
    return basis


def _diagnostics(idata) -> tuple[dict, pd.DataFrame]:
    """R-hat, effective sample size, and divergences of the free parameters."""
    import arviz as az

    names = [name for name in ("w", "tau", "sigma_case", "z_subj", "z_case") if name in idata.posterior]
    rhat = az.rhat(idata, var_names=names)
    ess_bulk = az.ess(idata, var_names=names, method="bulk")
    ess_tail = az.ess(idata, var_names=names, method="tail")

    def _extreme(dataset, reducer) -> float:
        values = [reducer(np.asarray(dataset[name].values, dtype=np.float64)) for name in names]
        return float(reducer(values))

    n_divergences = 0
    if "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].values.sum())
    diagnostics = {
        "max_rhat": _extreme(rhat, np.nanmax),
        "min_ess_bulk": _extreme(ess_bulk, np.nanmin),
        "min_ess_tail": _extreme(ess_tail, np.nanmin),
        "n_divergences": n_divergences,
    }
    summary = az.summary(idata, var_names=["w", "tau", "sigma_case"], hdi_prob=0.95)
    return diagnostics, summary


def fit_hierarchical(
    delta: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    target_accept: float = TARGET_ACCEPT,
    seed: int = SEED,
) -> HierarchicalFit:
    """Fit the crossed binomial logit and return the weight draws.

    The chain starts at zero weights, so the linear predictor starts at 0
    and every binomial probability starts at 1/2. A diffuse half-normal
    draw for τ or σ saturates the logit and the sampler never leaves the
    start. Jitter is still applied around that centre.
    """
    frame = observation_frame(delta, decisions)
    subjects = list(dict.fromkeys(frame["subject"]))
    case_ids = list(dict.fromkeys(frame["case"]))
    group_of = {subject: group for subject, group in zip(frame["subject"], frame["group"])}
    group_index = {name: index for index, name in enumerate(GROUP_ORDER)}
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}

    subject_row = frame["subject"].map(subject_index).to_numpy(dtype=np.intp)
    case_row = frame["case"].map(case_index).to_numpy(dtype=np.intp)
    group_row = frame["group"].map(group_index).to_numpy(dtype=np.intp)
    delta_rows = frame[list(COLUMNS)].to_numpy(dtype=np.float64)
    successes = frame["successes"].to_numpy(dtype=np.intp)
    n_trials = frame["n_trials"].to_numpy(dtype=np.intp)
    basis = _case_basis(delta_rows, case_row, len(case_ids))

    group_of_subject = np.array([group_index[group_of[subject]] for subject in subjects], dtype=np.intp)
    # The value columns and the case columns have to be full rank together.
    # Otherwise a group mean is still aliased with the case effects.
    group_design = np.concatenate(
        [delta_rows * (group_row == index)[:, None] for index in range(len(GROUP_ORDER))],
        axis=1,
    )
    constrained = np.column_stack([group_design, basis[case_row]])
    if np.linalg.matrix_rank(constrained) != constrained.shape[1]:
        raise ValueError("group values and case effects are still rank deficient")

    coords = {
        "values": list(COLUMNS),
        "groups": list(GROUP_ORDER),
        "subjects": subjects,
        "cases": case_ids,
        "case_residuals": np.arange(basis.shape[1]),
    }
    n_groups, n_values, n_subjects = len(GROUP_ORDER), len(COLUMNS), len(subjects)
    with pm.Model(coords=coords):
        w = pm.Normal("w", 0.0, W_SD, dims=("groups", "values"))
        tau = pm.HalfNormal("tau", TAU_SD, dims=("groups", "values"))
        z_subject = pm.Normal("z_subj", 0.0, 1.0, dims=("subjects", "values"))
        u = pm.Deterministic(
            "u", tau[group_of_subject, :] * z_subject, dims=("subjects", "values")
        )
        sigma_case = pm.HalfNormal("sigma_case", SIGMA_CASE_SD)
        z_case = pm.Normal("z_case", 0.0, 1.0, dims="case_residuals")
        a = pm.Deterministic("a", sigma_case * pt.dot(basis, z_case), dims="cases")
        eta = pt.sum((w[group_row, :] + u[subject_row, :]) * delta_rows, axis=1) + a[case_row]
        pm.Binomial("y", n=n_trials, logit_p=eta, observed=successes)

        initvals = {
            "w": np.zeros((n_groups, n_values)),
            "tau": np.full((n_groups, n_values), 0.5),
            "z_subj": np.zeros((n_subjects, n_values)),
            "sigma_case": np.array(0.5),
            "z_case": np.zeros(basis.shape[1]),
        }
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            initvals=initvals,
        )

    diagnostics, summary = _diagnostics(idata)
    diagnostics["n_case_effects"] = int(basis.shape[1])
    diagnostics["n_rows"] = int(len(frame))
    diagnostics["n_subjects"] = n_subjects
    diagnostics["n_cases"] = len(case_ids)
    w_draws = np.asarray(idata.posterior["w"].values, dtype=np.float64)
    u_draws = np.asarray(idata.posterior["u"].values, dtype=np.float64)
    return HierarchicalFit(
        subjects=subjects,
        group_of=group_of,
        w=w_draws.reshape((-1,) + w_draws.shape[2:]),
        u=u_draws.reshape((-1,) + u_draws.shape[2:]),
        n_case_effects=int(basis.shape[1]),
        diagnostics=diagnostics,
        summary=summary,
    )


def _softmax(weights: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return softmax(np.asarray(weights, dtype=np.float64) / temperature, axis=-1)


def _theta(fit: HierarchicalFit) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw subject weights, and the physician group weight.

    A subject's weight is its group mean plus its deviation. The group
    weight is the physician mean alone.
    """
    group_index = np.array(
        [GROUP_ORDER.index(fit.group_of[subject]) for subject in fit.subjects],
        dtype=np.intp,
    )
    theta = fit.w[:, group_index, :] + fit.u
    physician = GROUP_ORDER.index("physician")
    return theta, fit.w[:, physician, :]


def mean_weight_profiles(fit: HierarchicalFit, temperature: float = T_STAR) -> dict[str, pd.Series]:
    """softmax of the posterior-mean weight. This is the published profile.

    The mean is taken on the logit scale, then passed through the softmax.
    ``GROUP_MEAN`` is the physician group weight, with no subject deviation.
    """
    theta, physician = _theta(fit)
    profiles = _softmax(theta.mean(axis=0), temperature)
    out = {
        subject: pd.Series(profiles[index], index=list(COLUMNS))
        for index, subject in enumerate(fit.subjects)
    }
    out[GROUP_MEAN] = pd.Series(_softmax(physician.mean(axis=0), temperature), index=list(COLUMNS))
    return out


def posterior_profiles(fit: HierarchicalFit, temperature: float = T_STAR) -> pd.DataFrame:
    """Mean and equal-tailed 95% interval of the per-draw softmax profile.

    Each draw is softmax(θ / T). The reported mean is the average of those
    profiles. It is not the softmax of the average weight, which is what
    ``mean_weight_profiles`` returns and what the paper table uses.
    """
    theta, physician = _theta(fit)
    subject_profiles = _softmax(theta, temperature)
    group_profile = _softmax(physician, temperature)
    low_subject, high_subject = np.quantile(subject_profiles, [0.025, 0.975], axis=0)
    low_group, high_group = np.quantile(group_profile, [0.025, 0.975], axis=0)
    mean_subject = subject_profiles.mean(axis=0)
    mean_group = group_profile.mean(axis=0)

    llms = [subject for subject in fit.subjects if fit.group_of[subject] == "LLM"]
    physicians = [subject for subject in fit.subjects if fit.group_of[subject] == "physician"]
    order = [(subject, "llm") for subject in llms]
    order.append((GROUP_MEAN, "physician_group_mean"))
    order.extend((subject, "physician") for subject in physicians)
    index_of = {subject: index for index, subject in enumerate(fit.subjects)}

    rows = []
    for maker, kind in order:
        if kind == "physician_group_mean":
            mean, low, high = mean_group, low_group, high_group
        else:
            index = index_of[maker]
            mean, low, high = mean_subject[index], low_subject[index], high_subject[index]
        for value_index, value in enumerate(COLUMNS):
            rows.append(
                {
                    "decision_maker": maker,
                    "kind": kind,
                    "value": value,
                    "mean": float(mean[value_index]),
                    "ci_low": float(low[value_index]),
                    "ci_high": float(high[value_index]),
                }
            )
    return pd.DataFrame(rows)


def _jsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Squared base-2 Jensen-Shannon divergence over the last axis."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    midpoint = 0.5 * (left + right)

    def _kl(weights: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = weights * (np.log2(weights) - np.log2(midpoint))
        return np.where(weights > 0.0, term, 0.0).sum(axis=-1)

    return 0.5 * _kl(left) + 0.5 * _kl(right)


def outlier_probabilities(fit: HierarchicalFit, temperature: float = T_STAR) -> pd.Series:
    """Posterior probability an LLM sits past the physicians' 95th percentile.

    On each draw, every profile is softmax(θ / T) and the reference point
    is the physician group-mean profile on that same draw. The threshold
    is the 95th percentile of the 25 physician divergences from that
    profile. The probability is the share of draws on which the LLM's
    divergence is larger.
    """
    theta, physician = _theta(fit)
    profiles = _softmax(theta, temperature)
    group = _softmax(physician, temperature)
    divergence = _jsd(profiles, group[:, None, :])
    is_physician = np.array([fit.group_of[subject] == "physician" for subject in fit.subjects])
    threshold = np.percentile(divergence[:, is_physician], 95, axis=1)
    llm_ids = [subject for subject in fit.subjects if fit.group_of[subject] == "LLM"]
    llm_index = [fit.subjects.index(subject) for subject in llm_ids]
    exceed = divergence[:, llm_index] > threshold[:, None]
    return pd.Series(exceed.mean(axis=0), index=llm_ids, name="posterior_probability")
