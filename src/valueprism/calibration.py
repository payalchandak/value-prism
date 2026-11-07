"""Calibrate the softmax temperature on synthetic agents.

The sweep is the one in values-analysis
``src/analysis/synthetic_agents.ipynb``. For each concentration α in
{0.3, 0.5, 1.0, 3.0, 10.0}, one Generator seeded with 42 draws 100
weight vectors from Dir(α, α, α, α). Agent i, counting from 0 across
that list, scores every case by s = w · Δ, draws Binomial(100, σ(s))
answers with seed 42 + i, and is refit with the binomial GLM.

JSD is base 2: ``jensenshannon(..., base=2) ** 2``. The 50 temperatures
are ``logspace(-1.5, 1, 50)``, from about 0.032 to 10. Cases are taken
in case-id order.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import expit, softmax

from valueprism.attribution import COLUMNS, _fit

ALPHAS = (0.3, 0.5, 1.0, 3.0, 10.0)
AGENTS_PER_ALPHA = 100
N_RUNS = 100
N_TEMPERATURES = 50
SEED = 42


def temperature_grid() -> np.ndarray:
    """Fifty log-spaced temperatures, from 10^-1.5 to 10."""
    return np.logspace(-1.5, 1, N_TEMPERATURES)


def draw_agents(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Dirichlet weights, shape (500, 4), and the α of each agent.

    The Generator is consumed one concentration at a time, which is the
    notebook's ``sweep_rng``.
    """
    rng = np.random.default_rng(seed)
    weights = []
    alphas = []
    concentration = np.empty(len(COLUMNS), dtype=np.float64)
    for alpha in ALPHAS:
        concentration.fill(alpha)
        weights.append(rng.dirichlet(concentration, size=AGENTS_PER_ALPHA))
        alphas.append(np.full(AGENTS_PER_ALPHA, alpha))
    return np.vstack(weights), np.concatenate(alphas)


def _design(delta: pd.DataFrame) -> np.ndarray:
    missing = [name for name in COLUMNS if name not in delta.columns]
    if missing:
        raise ValueError(f"Δ is missing columns: {missing}")
    if len(delta) == 0:
        raise ValueError("Δ has no cases")
    return delta[COLUMNS].sort_index().to_numpy(dtype=np.float64)


def recover_weights(
    delta: pd.DataFrame,
    weights: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    """Refit ŵ for each row of ``weights``.

    Row i uses ``default_rng(seed + i)``, then one binomial draw per
    case. The draw order is case-id order.
    """
    design = _design(delta)
    if weights.ndim != 2 or weights.shape[1] != design.shape[1]:
        raise ValueError(f"weights must have shape (n, {design.shape[1]}), got {weights.shape}")
    recovered = np.empty_like(weights, dtype=np.float64)
    trials = np.full(design.shape[0], float(N_RUNS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, w in enumerate(weights):
            rng = np.random.default_rng(seed + i)
            proportions = rng.binomial(N_RUNS, expit(design @ w)) / N_RUNS
            result = _fit(proportions, design, trials)
            recovered[i] = np.asarray(result.params, dtype=np.float64)
    return recovered


def divergence_matrix(
    weights: np.ndarray,
    recovered: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Base-2 JSD between softmax(ŵ / T) and w. Shape (agents, temperatures)."""
    matrix = np.empty((len(weights), len(temperatures)), dtype=np.float64)
    for j, temperature in enumerate(temperatures):
        profiles = softmax(recovered / temperature, axis=1)
        for i in range(len(weights)):
            matrix[i, j] = jensenshannon(weights[i], profiles[i], base=2.0) ** 2
    return matrix


def normal_interval(values: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    """Mean and a normal interval. The standard error uses ``std / sqrt(n)``.

    NumPy's default divisor is the one in the notebook's ±1 SE bands.
    ``z = 1.96`` is the 95% critical value.
    """
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std() / np.sqrt(values.size))
    return mean, mean - z * se, mean + z * se


@dataclass(frozen=True)
class Calibration:
    """JSD of every agent at every temperature. ``alphas`` aligns with rows."""

    temperatures: np.ndarray
    alphas: np.ndarray
    jsd: np.ndarray

    def _subset(self, alpha: float | None) -> np.ndarray:
        if alpha is None:
            return self.jsd
        return self.jsd[self.alphas == alpha]

    def mean_curve(self, alpha: float | None = None) -> np.ndarray:
        return self._subset(alpha).mean(axis=0)

    def se_curve(self, alpha: float | None = None) -> np.ndarray:
        rows = self._subset(alpha)
        return rows.std(axis=0) / np.sqrt(len(rows))

    def optimal_temperature(self, alpha: float | None = None) -> float:
        """Argmin of the mean JSD. ``alpha`` restricts the mean to that group."""
        curve = self.mean_curve(alpha)
        return float(self.temperatures[int(np.argmin(curve))])

    def groups(self) -> list[dict]:
        """One row per α: its optimal temperature, and the mean and std there."""
        rows = []
        for alpha in ALPHAS:
            mask = self.alphas == alpha
            curve = self.jsd[mask].mean(axis=0)
            index = int(np.argmin(curve))
            rows.append(
                {
                    "alpha": alpha,
                    "temperature": float(self.temperatures[index]),
                    "mean_jsd": float(curve[index]),
                    "std_jsd": float(self.jsd[mask][:, index].std()),
                }
            )
        return rows

    def adopted_temperature(self) -> float:
        """Temperature selected by the most concentrations.

        The argmin of the pooled mean sits lower, because α = 0.3 prefers
        a cooler temperature. The paper uses the majority value.
        """
        chosen = [row["temperature"] for row in self.groups()]
        best = chosen[0]
        best_count = 0
        for temperature in chosen:
            count = sum(temperature == other for other in chosen)
            if count > best_count:
                best = temperature
                best_count = count
        return best

    def at_temperature(self, temperature: float) -> np.ndarray:
        """Per-agent JSD at one grid temperature."""
        index = int(np.argmin(np.abs(self.temperatures - temperature)))
        if self.temperatures[index] != temperature:
            raise ValueError(f"{temperature} is not on the temperature grid")
        return self.jsd[:, index]


def calibrate(delta: pd.DataFrame, seed: int = SEED) -> Calibration:
    """Draw the 500 agents, refit them, and score every temperature."""
    temperatures = temperature_grid()
    weights, alphas = draw_agents(seed)
    recovered = recover_weights(delta, weights, seed)
    return Calibration(temperatures, alphas, divergence_matrix(weights, recovered, temperatures))
