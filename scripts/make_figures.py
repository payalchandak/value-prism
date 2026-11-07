#!/usr/bin/env python
"""Write the paper figures and a side-by-side comparison with the current PNGs.

The JSD matrix is Pooled Physicians, then the 12 models, then P01–P25.
The agreement matrix is the 12 models, then Pooled Physicians, then
P01–P25. Those are the orders in value_profiles.ipynb cell 14 and in
explore.ipynb's agreement matrix. Diversity, the permutation test, and
the three agreement ranges use the 37 individuals. The reference
bootstrap and the diversity resamples use 10,000 draws and seed 42.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights
from valueprism.calibration import calibrate
from valueprism.consistency import (
    case_entropy_correlation,
    llm_cases,
    physician_cases,
)
from valueprism.data import load_cases, load_decisions
from valueprism.diversity import (
    bootstrap_mean_jsd,
    mean_pairwise_jsd,
    pairwise_jsd,
    permutation_test_jsd,
)
from valueprism.plots import (
    plot_agreement,
    plot_case_entropy,
    plot_distributional_pluralism,
    plot_jsd_matrix,
    plot_monoculture_distribution,
    plot_monoculture_test,
    plot_temperature,
)
from valueprism.plots.common import display_name
from valueprism.reference import N_BOOTSTRAP, SEED, bootstrap_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "results" / "figures"
COMPARE_DIR = FIGURE_DIR / "compare"
PAPER_DIR = (
    REPO_ROOT.parents[1]
    / "writing"
    / "hvp neurips overleaf"
    / "699c5bded44b80eaf78f99b6"
    / "figures"
)
N_PANEL = 37
CONSENSUS_ID = "human_consensus"
# Rounded pooled physician profile, in column order A, B, N, J.
CONSENSUS_PROFILE = (0.437, 0.286, 0.148, 0.129)
MATRIX_PATH = REPO_ROOT / "results" / "pairwise_jsd.csv"


def profiles(delta: pd.DataFrame, decisions: pd.DataFrame, ids: list[str]) -> dict:
    """softmax(w / T*) for each id. Value order is A, B, N, J."""
    fitted = {}
    for maker in ids:
        rows = decisions.loc[decisions["decision_maker"] == maker]
        weights = value_weights(delta, rows)
        profile = softmax_profile(weights, T_STAR)
        fitted[maker] = {name: float(profile[name]) for name in COLUMNS}
    return fitted


def majority_choices(decisions: pd.DataFrame, case_ids: list[str], ids: list[str]) -> pd.DataFrame:
    """One column per decision maker. A tie, including an even split, is choice 1."""
    chosen = {}
    for maker in ids:
        rows = decisions.loc[decisions["decision_maker"] == maker]
        by_case = {}
        for case_id, group in rows.groupby("case_id", sort=False):
            answered = group.loc[group["choice"].notna(), "choice"]
            n1 = int(answered.eq(1).sum())
            n2 = int(answered.eq(2).sum())
            if n1 + n2 == 0:
                by_case[case_id] = np.nan
            else:
                by_case[case_id] = 1.0 if n1 >= n2 else 2.0
        chosen[maker] = by_case
    return pd.DataFrame(chosen).reindex(index=case_ids, columns=ids)


def physician_consensus(decisions: pd.DataFrame, case_ids: list[str]) -> tuple[pd.Series, int]:
    """Majority physician vote on each case, and how many cases have no majority.

    Each non-missing physician run is one vote. The votes are summed across
    physicians, which is the human_consensus branch of values-analysis
    ``agreement_rate``. Choice 1 wins only when it has strictly more votes
    than choice 2. A tie, including a case with no votes, is missing.
    """
    physicians = decisions.loc[decisions["group"] == "physician"]
    by_case = {}
    for case_id, group in physicians.groupby("case_id", sort=False):
        answered = group.loc[group["choice"].notna(), "choice"]
        n1 = int(answered.eq(1).sum())
        n2 = int(answered.eq(2).sum())
        if n1 > n2:
            by_case[case_id] = 1.0
        elif n2 > n1:
            by_case[case_id] = 2.0
        else:
            by_case[case_id] = np.nan
    series = pd.Series(by_case, dtype=float).reindex(case_ids)
    return series, int(series.isna().sum())


def agreement_rates(
    choices: pd.DataFrame, skip_ids: set[str] | None = None
) -> pd.DataFrame:
    """Share of cases with the same choice.

    The denominator is every case, and a missing choice does not agree.
    An id in ``skip_ids`` is the exception: a missing choice there is left
    out of the denominator. Physician consensus is that id, because a tie
    has no decision.
    """
    skip_ids = set() if skip_ids is None else skip_ids
    ids = list(choices.columns)
    values = choices.to_numpy(dtype=float)
    n = len(ids)
    rates = np.eye(n)
    for i in range(n):
        column = values[:, i]
        for j in range(i + 1, n):
            other = values[:, j]
            if ids[i] in skip_ids or ids[j] in skip_ids:
                mask = np.isfinite(column) & np.isfinite(other)
                if not mask.any():
                    raise ValueError(f"no cases where {ids[i]} and {ids[j]} both choose")
                rate = float(np.mean(column[mask] == other[mask]))
            else:
                rate = float(np.mean(column == other))
            rates[i, j] = rate
            rates[j, i] = rate
    return pd.DataFrame(rates, index=ids, columns=ids)


def _pct(low: float, high: float) -> str:
    """Integer percent range, with an en dash."""
    return f"{round(100 * low)}–{round(100 * high)}%"


def _span(rates: pd.DataFrame, left: list[str], right: list[str]) -> tuple[float, float]:
    """Min and max agreement. Pairs inside one group skip the diagonal."""
    same_group = left == right
    values = []
    for i, maker in enumerate(left):
        partners = right[i + 1 :] if same_group else right
        for other in partners:
            values.append(float(rates.loc[maker, other]))
    return min(values), max(values)


def write_matrices(
    cases: pd.DataFrame,
    delta: pd.DataFrame,
    decisions: pd.DataFrame,
    llm_ids: list[str],
    physician_ids: list[str],
    fitted: dict,
) -> None:
    """Write the two 38 × 38 matrices, the JSD table, and the range lines.

    ``fitted`` is the 37 individual profiles. The consensus profile is the
    pooled physician fit and is not added to ``fitted``.
    """
    weights = value_weights(delta, decisions.loc[decisions["group"] == "physician"])
    pooled = softmax_profile(weights, T_STAR)
    consensus_profile = {name: float(pooled[name]) for name in COLUMNS}
    rounded = tuple(round(consensus_profile[name], 3) for name in COLUMNS)
    if rounded != CONSENSUS_PROFILE:
        raise SystemExit(f"pooled physician profile {rounded}, expected {CONSENSUS_PROFILE}")
    print("pooled physician profile " + " / ".join(f"{value:.3f}" for value in rounded))

    jsd_order = [CONSENSUS_ID] + llm_ids + physician_ids
    profiles_for_jsd = {CONSENSUS_ID: consensus_profile}
    for maker in llm_ids + physician_ids:
        profiles_for_jsd[maker] = fitted[maker]
    matrix = pairwise_jsd(profiles_for_jsd)
    if list(matrix.index) != jsd_order or matrix.shape != (N_PANEL + 1, N_PANEL + 1):
        raise SystemExit(f"expected a {N_PANEL + 1} × {N_PANEL + 1} JSD matrix, got {matrix.shape}")

    if MATRIX_PATH.is_file():
        previous = pd.read_csv(MATRIX_PATH, index_col=0)
        individual = llm_ids + physician_ids
        common = [maker for maker in individual if maker in previous.index]
        if len(common) == N_PANEL:
            block = matrix.loc[common, common].to_numpy()
            prior = previous.loc[common, common].to_numpy(dtype=float)
            gap = float(np.max(np.abs(block - prior)))
            if gap > 5e-10:
                raise SystemExit(f"individual JSD block moved by {gap}")

    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(MATRIX_PATH, float_format="%.10f")
    print(f"pairwise JSD matrix  results/pairwise_jsd.csv  {matrix.shape[0]} × {matrix.shape[1]}")

    case_ids = cases["case_id"].tolist()
    consensus, n_ties = physician_consensus(decisions, case_ids)
    print(
        f"physician-consensus ties {n_ties} of {len(case_ids)} "
        "(a tie is left out of the agreement rate)"
    )
    choices = majority_choices(decisions, case_ids, llm_ids + physician_ids)
    choices[CONSENSUS_ID] = consensus
    agree_order = llm_ids + [CONSENSUS_ID] + physician_ids
    choices = choices.reindex(columns=agree_order)
    rates = agreement_rates(choices, skip_ids={CONSENSUS_ID})

    spans = {
        "physician–physician": _span(rates, physician_ids, physician_ids),
        "LLM–LLM": _span(rates, llm_ids, llm_ids),
        "LLM–physician": _span(rates, llm_ids, physician_ids),
    }
    expected = {
        "physician–physician": "38–82%",
        "LLM–LLM": "60–92%",
        "LLM–physician": "28–76%",
    }
    print(
        "The agreement ranges, unchanged: "
        + ", ".join(f"{name} {_pct(*span)}" for name, span in spans.items())
        + "."
    )
    for name, span in spans.items():
        got = _pct(*span)
        if got != expected[name]:
            raise SystemExit(f"{name} agreement {got}, expected {expected[name]}")

    consensus_llm = _span(rates, [CONSENSUS_ID], llm_ids)
    consensus_physician = _span(rates, [CONSENSUS_ID], physician_ids)
    jsd_llm = matrix.loc[CONSENSUS_ID, llm_ids].to_numpy()
    jsd_physician = matrix.loc[CONSENSUS_ID, physician_ids].to_numpy()
    print(f"consensus–LLM agreement {_pct(*consensus_llm)}")
    print(f"consensus–physician agreement {_pct(*consensus_physician)}")
    print(f"consensus–LLM JSD {jsd_llm.min():.4f}–{jsd_llm.max():.4f}")
    print(f"consensus–physician JSD {jsd_physician.min():.4f}–{jsd_physician.max():.4f}")

    plot_jsd_matrix(matrix, FIGURE_DIR / "jsd-matrix.png")
    plot_agreement(rates, FIGURE_DIR / "agreement-rate.png")


def entropy_panels(decisions: pd.DataFrame) -> list[tuple]:
    """One row per model: label, physician entropy, model entropy, ρ, p."""
    reference = physician_cases(decisions).set_index("case_id")["entropy"]
    panels = []
    for maker, group in llm_cases(decisions).groupby("decision_maker", sort=True):
        model = group.set_index("case_id")["entropy"]
        joined = pd.concat({"physician": reference, "model": model}, axis=1).dropna()
        joined = joined.sort_index()
        rho, pvalue, _n = case_entropy_correlation(reference, model)
        panels.append(
            (
                display_name(str(maker)),
                joined["physician"].to_numpy(dtype=float),
                joined["model"].to_numpy(dtype=float),
                rho,
                pvalue,
            )
        )
    return panels


def side_by_side(old: Path, new: Path, dest: Path) -> None:
    """Place the paper PNG and the new PNG on one canvas, labelled old and new."""
    left = Image.open(old).convert("RGB")
    right = Image.open(new).convert("RGB")
    height = min(left.height, right.height, 1800)

    def fit(image: Image.Image) -> Image.Image:
        if image.height == height:
            return image
        width = max(1, round(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    left, right = fit(left), fit(right)
    gap = 16
    header = 40
    canvas = Image.new("RGB", (left.width + gap + right.width, header + height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 22)
    draw.text((12, 8), "old", fill="black", font=font)
    draw.text((left.width + gap + 12, 8), "new", fill="black", font=font)
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width + gap, header))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)


def main() -> None:
    cases, delta = load_cases()
    decisions = load_decisions()
    llm_ids = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    physician_ids = sorted(
        decisions.loc[decisions["group"] == "physician", "decision_maker"].unique()
    )
    order = llm_ids + physician_ids
    if len(order) != N_PANEL:
        raise SystemExit(f"expected {N_PANEL} decision makers, got {len(order)}")

    fitted = profiles(delta, decisions, order)
    write_matrices(cases, delta, decisions, llm_ids, physician_ids, fitted)

    print(f"diversity bootstrap, n = {N_BOOTSTRAP}, seed = {SEED}")
    individual = pairwise_jsd(fitted)
    observed_llm = mean_pairwise_jsd(individual.loc[llm_ids, llm_ids].to_numpy())
    observed_physician = mean_pairwise_jsd(
        individual.loc[physician_ids, physician_ids].to_numpy()
    )
    boot = bootstrap_mean_jsd(
        fitted,
        group_a_ids=llm_ids,
        group_b_ids=physician_ids,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    perm = permutation_test_jsd(
        fitted,
        group_a_ids=llm_ids,
        group_b_ids=physician_ids,
        n_permutations=N_BOOTSTRAP,
        seed=SEED,
    )
    panels = entropy_panels(decisions)

    print(f"reference bootstrap, n = {N_BOOTSTRAP}, seed = {SEED}")
    reference = bootstrap_reference(
        delta,
        decisions,
        physician_ids,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    rows = [
        (display_name(row.decision_maker), row.jsd, row.ci_low, row.ci_high)
        for row in reference.models
    ]

    print("temperature sweep")
    calibration = calibrate(delta)

    names = (
        "distributional-pluralism.png",
        "jsd-matrix.png",
        "monoculture-disttribution.png",
        "monoculture-test.png",
        "agreement-rate.png",
        "case-level-entropy.png",
        "temperature.png",
    )
    plot_distributional_pluralism(
        reference.reference_jsds, rows, FIGURE_DIR / "distributional-pluralism.png"
    )
    plot_monoculture_distribution(
        boot["group_a_mean"].samples,
        boot["group_b_mean"].samples,
        boot["difference"].samples,
        observed_llm,
        observed_physician,
        FIGURE_DIR / "monoculture-disttribution.png",
    )
    plot_monoculture_test(
        perm["null_distribution"],
        perm["observed_diff"],
        FIGURE_DIR / "monoculture-test.png",
    )
    plot_case_entropy(panels, FIGURE_DIR / "case-level-entropy.png")
    plot_temperature(calibration, FIGURE_DIR / "temperature.png")

    compare = PAPER_DIR.is_dir()
    if not compare:
        print("skipped side-by-side: paper figure directory is not present")
    for name in names:
        path = FIGURE_DIR / name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        print(path.relative_to(REPO_ROOT))
        if not compare:
            continue
        old = PAPER_DIR / name
        if not old.is_file():
            raise SystemExit(f"missing paper figure {old}")
        side_by_side(old, path, COMPARE_DIR / name)
        print((COMPARE_DIR / name).relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
