#!/usr/bin/env python
"""Calibrate the softmax temperature and save the curve.

The adopted temperature is the one selected by the most Dirichlet
concentrations. The figure is mean JSD against T for each concentration,
with ±1 SE bands, on a log temperature axis.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from valueprism.calibration import calibrate, normal_interval
from valueprism.data import load_cases
from valueprism.plots import plot_temperature

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = REPO_ROOT / "results" / "figures" / "temperature.png"

# Base-2 mean JSD at the adopted T* = 0.262, on the 50 cases.
# The interval is the mean ± 1.96 standard errors.
PAPER_T_STAR = "0.262"
PAPER_BY_ALPHA = {0.3: "0.233", 0.5: "0.262", 1.0: "0.262", 3.0: "0.262", 10.0: "0.262"}
PAPER_JSD = "0.0126"
PAPER_CI = ("0.0113", "0.0139")


def _ci_text(values: np.ndarray) -> str:
    mean, low, high = normal_interval(values)
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def main() -> None:
    flags: list[str] = []
    _cases, delta = load_cases()
    result = calibrate(delta)
    groups = result.groups()
    pooled_t = result.optimal_temperature()
    adopted = result.adopted_temperature()
    n_adopted = sum(row["temperature"] == adopted for row in groups)

    print(f"{len(result.alphas)} agents, {len(delta)} cases")
    print(
        f"{len(result.temperatures)} temperatures, "
        f"from {result.temperatures[0]:.4f} to {result.temperatures[-1]:.4f}"
    )
    print(f"{'α':>6} {'optimal T':>10} {'mean JSD':>12} {'std JSD':>12}")
    for row in groups:
        rendered = f"{row['temperature']:.3f}"
        expected = PAPER_BY_ALPHA[row["alpha"]]
        print(
            f"{row['alpha']:6.1f} {row['temperature']:10.4f} "
            f"{row['mean_jsd']:12.6f} {row['std_jsd']:12.6f}"
        )
        if rendered != expected:
            flags.append(f"α = {row['alpha']}: optimal T {rendered}, paper {expected}")

    print(
        f"Pooled argmin: T = {pooled_t:.4f}, "
        f"mean JSD = {result.mean_curve()[np.argmin(result.mean_curve())]:.6f}, "
        f"std = {result.at_temperature(pooled_t).std():.6f}"
    )
    print(f"Adopted T* = {adopted:.3f}  ({n_adopted} of {len(groups)} concentrations)")
    if f"{adopted:.3f}" != PAPER_T_STAR:
        flags.append(f"adopted T* {adopted:.3f}, paper {PAPER_T_STAR}")

    print("Mean JSD, base 2, 95% CI = mean ± 1.96 SE")
    pooled_text = _ci_text(result.at_temperature(pooled_t))
    adopted_text = _ci_text(result.at_temperature(adopted))
    best_text = _ci_text(result.jsd.min(axis=1))
    print(f"  at pooled argmin:      {pooled_text}")
    print(f"  at adopted T*:         {adopted_text}")
    print(f"  at each agent's best:  {best_text}")
    nats = np.log(2.0)
    mean, low, high = normal_interval(result.at_temperature(pooled_t))
    nats_text = f"{mean * nats:.4f} [{low * nats:.4f}, {high * nats:.4f}]"
    print(f"  pooled argmin, in nats (base-2 JSD × ln 2): {nats_text}")
    paper_text = f"{PAPER_JSD} [{PAPER_CI[0]}, {PAPER_CI[1]}]"
    print(f"Expected reconstruction JSD, base 2, at adopted T* on 50 cases: {paper_text}")
    if adopted_text != paper_text:
        flags.append(f"reconstruction JSD at adopted T*: {adopted_text}, expected {paper_text}")

    plot_temperature(result, FIGURE_PATH)
    print(f"wrote {FIGURE_PATH.relative_to(REPO_ROOT)}")

    if flags:
        print("mismatch")
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
