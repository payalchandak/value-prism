"""Ridge and Firth refits of the no-intercept binomial GLM.

Both use the four Δ columns and the trial counts of
:func:`valueprism.attribution.value_weights`. Pass the rows for one
decision maker, or every physician row for the pooled profile.

Ridge is ``statsmodels`` ``GLM.fit_regularized`` with ``L1_wt=0``.
That penalty is ``-loglike / n_rows + alpha * ||w||^2 / 2``, and
``statsmodels`` divides by the number of rows. The value passed in is
``alpha * mean(n)``, so a fixed ``alpha`` is the same per-trial penalty
for every decision maker. ``alpha = 0`` is the unpenalised fit.

Firth is ``firthlogist`` 0.5.0 with no intercept, the Jeffreys penalty
and no tuning constant. Each case becomes ``n_i`` binary rows, of which
``round(p_i * n_i)`` are labeled 1.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from firthlogist import FirthLogisticRegression

from valueprism.attribution import COLUMNS, _matrices


def _degenerate(delta_rows: np.ndarray, proportions: np.ndarray) -> bool:
    """True when the design is constant or the response is all 0s or all 1s."""
    return bool(
        np.all(delta_rows == 0) or np.all(proportions == 0) or np.all(proportions == 1)
    )


def _zeros() -> pd.Series:
    return pd.Series(0.0, index=COLUMNS)


def ridge_weights(delta: pd.DataFrame, decisions: pd.DataFrame, alpha: float) -> pd.Series:
    """L2-penalised weights. Index is A, B, N, J."""
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    delta_rows, proportions, n = _matrices(delta, decisions)
    if _degenerate(delta_rows, proportions):
        return _zeros()

    family = sm.families.Binomial(link=sm.families.links.Logit())
    model = sm.GLM(proportions, delta_rows, family=family, freq_weights=n)
    if alpha == 0:
        params = model.fit(disp=False).params
    else:
        penalty = alpha * float(n.sum()) / len(proportions)
        params = model.fit_regularized(alpha=penalty, L1_wt=0.0).params
    return pd.Series(np.asarray(params, dtype=np.float64), index=COLUMNS)


def _binary(delta_rows: np.ndarray, proportions: np.ndarray, n: np.ndarray):
    """One binary row per trial. ``round(p_i * n_i)`` of the rows are 1."""
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for xi, pi, ni in zip(delta_rows, proportions, n):
        ni = int(ni)
        ones = int(round(float(pi) * ni))
        ones = min(max(ones, 0), ni)
        rows.extend([xi] * ni)
        labels.extend([1] * ones + [0] * (ni - ones))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.intp)


def firth_weights(delta: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.Series, int]:
    """Jeffreys-penalised weights, and the Newton–Raphson iteration count."""
    delta_rows, proportions, n = _matrices(delta, decisions)
    if _degenerate(delta_rows, proportions):
        return _zeros(), 0

    binary_rows, labels = _binary(delta_rows, proportions, n)
    model = FirthLogisticRegression(fit_intercept=False, skip_ci=True)
    model.fit(binary_rows, labels)
    weights = pd.Series(np.asarray(model.coef_, dtype=np.float64), index=COLUMNS)
    return weights, int(model.n_iter_)
