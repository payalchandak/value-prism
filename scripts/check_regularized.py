#!/usr/bin/env python
"""Firth and ridge profiles against the plain binomial GLM.

The average is the 12 LLMs and the pooled physician. Individual
physicians are not in it. JSD is base 2, the squared Jensen–Shannon
distance, between softmax(w / T*) profiles.

Paper figures, at the precision they are printed:

  Firth mean JSD 9.3e-6, range 4.8e-7 to 2.4e-5.
  Every Firth fit converges in 3–6 iterations, without warnings.
  Ridge mean JSD: 0.01 → 0.00048, 0.03 → 0.0023, 0.1 → 0.0083,
  0.3 → 0.019, 1 → 0.036, 3 → 0.051.
  No optimizer warnings at alpha 0.1.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights
from valueprism.data import load_cases, load_decisions
from valueprism.robustness import firth_weights, ridge_weights

PAPER_FIRTH_MEAN = 9.3e-6
PAPER_FIRTH_MIN = 4.8e-7
PAPER_FIRTH_MAX = 2.4e-5
# alpha -> mean JSD printed in the paper
PAPER_RIDGE = {
    0.01: 0.00048,
    0.03: 0.0023,
    0.1: 0.0083,
    0.3: 0.019,
    1.0: 0.036,
    3.0: 0.051,
}

flags: list[str] = []


def check(label: str, got, expected) -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    print(f"  {label}: {got}    paper {expected}    {mark}")
    if got != expected:
        flags.append(f"{label}: got {got}, paper {expected}")


def two_sig(value: float) -> str:
    """The paper quotes these JSDs to two significant figures."""
    return f"{value:.2g}"


def jsd(left: pd.Series, right: pd.Series) -> float:
    p = np.array([left[name] for name in COLUMNS], dtype=np.float64)
    q = np.array([right[name] for name in COLUMNS], dtype=np.float64)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def capture(fn):
    """Call ``fn`` and return its value plus every warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, list(caught)


def warning_line(item) -> str:
    return f"{item.category.__name__}: {item.message}"


def profiles(delta: pd.DataFrame, decisions: pd.DataFrame):
    """GLM profile, then one (profile, iterations, warnings) per maker."""
    llm_ids = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    makers = [(model_id, decisions.loc[decisions["decision_maker"] == model_id]) for model_id in llm_ids]
    makers.append(("pooled physician", decisions.loc[decisions["group"] == "physician"]))

    glm = {}
    firth = {}
    for name, rows in makers:
        glm[name] = softmax_profile(value_weights(delta, rows), T_STAR)
        (weights, iterations), caught = capture(lambda rows=rows: firth_weights(delta, rows))
        firth[name] = (softmax_profile(weights, T_STAR), iterations, caught)
    return makers, glm, firth


def main() -> None:
    _cases, delta = load_cases()
    decisions = load_decisions()
    makers, glm, firth = profiles(delta, decisions)
    names = [name for name, _rows in makers]

    print(f"decision makers in each average: {len(names)}")
    check("who", names[-1], "pooled physician")
    check("count", len(names), 13)

    print("Firth, JSD from the GLM profile")
    distances = []
    iterations = []
    firth_warnings = []
    for name in names:
        profile, n_iter, caught = firth[name]
        distance = jsd(glm[name], profile)
        distances.append(distance)
        iterations.append(n_iter)
        for item in caught:
            firth_warnings.append(f"{name}: {warning_line(item)}")
        print(f"  {name}: JSD {distance:.8g}    iterations {n_iter}")

    mean = float(np.mean(distances))
    low = float(np.min(distances))
    high = float(np.max(distances))
    print(f"  mean {mean:.8g}    min {low:.8g}    max {high:.8g}")
    check("Firth mean JSD", two_sig(mean), two_sig(PAPER_FIRTH_MEAN))
    check("Firth min JSD", two_sig(low), two_sig(PAPER_FIRTH_MIN))
    check("Firth max JSD", two_sig(high), two_sig(PAPER_FIRTH_MAX))
    span = f"{min(iterations)}–{max(iterations)}"
    print(f"  iterations {span}    counts {iterations}")
    check("iterations in 3–6", all(3 <= n <= 6 for n in iterations), True)
    if firth_warnings:
        print("  warnings:")
        for line in firth_warnings:
            print(f"    {line}")
    else:
        print("  warnings: none")
    check("Firth warnings", len(firth_warnings), 0)

    print("Ridge, mean JSD from the GLM profile")
    warnings_at_0_1 = []
    for alpha, expected in PAPER_RIDGE.items():
        distances = []
        caught_here = []
        for name, rows in makers:
            weights, caught = capture(lambda rows=rows, alpha=alpha: ridge_weights(delta, rows, alpha))
            distances.append(jsd(glm[name], softmax_profile(weights, T_STAR)))
            for item in caught:
                caught_here.append(f"{name}: {warning_line(item)}")
        mean = float(np.mean(distances))
        print(f"  alpha {alpha:g}: mean JSD {mean:.8g}")
        check(f"ridge alpha {alpha:g}", two_sig(mean), two_sig(expected))
        if alpha == 0.1:
            warnings_at_0_1 = caught_here
        elif caught_here:
            print(f"    warnings at alpha {alpha:g}:")
            for line in caught_here:
                print(f"      {line}")

    optimizer = [line for line in warnings_at_0_1 if "converg" in line.lower() or "optim" in line.lower()]
    if warnings_at_0_1:
        print("  warnings at alpha 0.1:")
        for line in warnings_at_0_1:
            print(f"    {line}")
    else:
        print("  warnings at alpha 0.1: none")
    check("optimizer warnings at alpha 0.1", len(optimizer), 0)

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
