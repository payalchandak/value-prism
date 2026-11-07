"""Attribute choices to the four values.

The model is a binomial GLM with a logit link and no intercept:

    logit p_i = w_A Δ_iA + w_B Δ_iB + w_N Δ_iN + w_J Δ_iJ

p_i is the share of non-missing responses on case i that pick choice 1.
n_i, the number of those responses, is the binomial trial count.
Refusals and incoherent runs are missing and are left out of both.

Pass the rows for one decision maker. An LLM's n_i is its non-refusal
runs. One physician has a single run on a case, so n_i is 1. All of the
physician rows together are the pooled profile: the counts are summed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import softmax
from scipy.stats import chi2

COLUMNS = ["A", "B", "N", "J"]

# scripts/calibrate_temperature.py. Four of five concentrations select this value.
T_STAR = 0.262


def _counts(decisions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Choice-1 counts and trial counts, indexed by case_id.

    Cases with no non-missing response are absent.
    """
    answered = decisions.loc[decisions["choice"].notna(), ["case_id", "choice"]]
    if answered.empty:
        raise ValueError("no non-missing responses")
    trials = answered.groupby("case_id", sort=True).size()
    choice_1 = answered["choice"].eq(1).groupby(answered["case_id"], sort=True).sum()
    return choice_1.loc[trials.index], trials


def _matrices(delta: pd.DataFrame, decisions: pd.DataFrame):
    """Δ rows, choice-1 proportions, and trial counts, cases in id order."""
    choice_1, trials = _counts(decisions)
    unknown = trials.index.difference(delta.index)
    if len(unknown):
        raise ValueError(f"unknown case_id: {sorted(unknown)}")
    delta_rows = delta.loc[trials.index, COLUMNS].to_numpy(dtype=np.float64)
    n = trials.to_numpy(dtype=np.float64)
    proportions = choice_1.to_numpy(dtype=np.float64) / n
    return delta_rows, proportions, n


def _fit(proportions, delta_rows, n):
    """Binomial GLM. freq_weights = n makes the response the choice-1 counts."""
    family = sm.families.Binomial(link=sm.families.links.Logit())
    model = sm.GLM(proportions, delta_rows, family=family, freq_weights=n)
    return model.fit(disp=False)


def value_weights(delta: pd.DataFrame, decisions: pd.DataFrame) -> pd.Series:
    """w for the runs in ``decisions``. Index is A, B, N, J."""
    delta_rows, proportions, n = _matrices(delta, decisions)
    result = _fit(proportions, delta_rows, n)
    return pd.Series(np.asarray(result.params, dtype=np.float64), index=COLUMNS)


def softmax_profile(w: pd.Series, T: float) -> pd.Series:
    """softmax(w / T), in the same order as ``w``. The entries sum to 1."""
    if T <= 0:
        raise ValueError(f"temperature must be > 0, got {T}")
    order = list(w.index)
    scaled = np.asarray([w[name] for name in order], dtype=np.float64) / T
    return pd.Series(softmax(scaled), index=order)


def lrt_uniform(delta: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    """Test equal weights against the four-weight model.

    Null: w_A = w_B = w_N = w_J. Its one predictor is the row sum of Δ.
    Λ = 2 (ℓ_full − ℓ_null), clamped at 0, is referred to χ² with 3 degrees
    of freedom.
    """
    delta_rows, proportions, n = _matrices(delta, decisions)
    full = _fit(proportions, delta_rows, n)
    row_sum = delta_rows.sum(axis=1, keepdims=True)
    null = _fit(proportions, row_sum, n)
    df = delta_rows.shape[1] - 1
    lam = max(2.0 * (full.llf - null.llf), 0.0)
    return {
        "lrt_statistic": float(lam),
        "p_value": float(chi2.sf(lam, df)),
        "df": int(df),
        "ll_null": float(null.llf),
        "ll_alt": float(full.llf),
    }
