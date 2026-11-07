#!/usr/bin/env python
"""Check the hierarchical profiles against hierarchical_profiles.ipynb.

The published profiles are softmax of the posterior-mean weight at T*.
The new table and figure instead softmax every posterior draw, then
report the mean and the equal-tailed 95% interval of those profiles.

NUTS uses 4 chains, 2000 tuning draws, 2000 retained draws, target
acceptance 0.9, and seed 0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from valueprism.attribution import COLUMNS, T_STAR, softmax_profile, value_weights
from valueprism.data import load_cases, load_decisions
from valueprism.plots.common import display_name, save_figure
from valueprism.robustness.hierarchical import (
    CHAINS,
    DRAWS,
    GROUP_MEAN,
    SEED,
    TARGET_ACCEPT,
    TUNE,
    W_SD,
    fit_hierarchical,
    mean_weight_profiles,
    observation_frame,
    outlier_probabilities,
    posterior_profiles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = REPO_ROOT / "results" / "hierarchical_profiles.csv"
FIGURE_PATH = REPO_ROOT / "results" / "figures" / "hierarchical-profiles.png"

VALUE_LABELS = {
    "A": "Autonomy",
    "B": "Beneficence",
    "N": "Nonmaleficence",
    "J": "Justice",
}
# Notebook printout, three decimals read as percents.
PAPER_GROUP = (55.8, 26.2, 12.9, 5.0)
PAPER_POOLED = (43.7, 28.6, 14.8, 12.9)
PAPER_JSD = {"LLM": 0.0091, "physician": 0.0241}
PAPER_CONSENSUS_JSD = 0.0191
PAPER_SHIFT = {"A": 12.1, "J": -7.9}
KIND_COLOR = {
    "llm": "#3F8FC9",
    "physician": "#888888",
    "physician_group_mean": "#1a1a1a",
}

flags: list[str] = []


def check(label: str, got, expected, source: str = "paper") -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    print(f"  {label}: {got}    {source} {expected}    {mark}")
    if got != expected:
        flags.append(f"{label}: got {got}, {source} {expected}")


def slash(values, digits: int) -> str:
    return " / ".join(f"{value:.{digits}f}" for value in values)


def percent(profile: pd.Series) -> tuple[float, ...]:
    """One decimal, from rounding the probability to three decimals."""
    return tuple(float(np.round(100.0 * profile[column], 1)) for column in COLUMNS)


def show_pp(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def jsd(left: pd.Series, right: pd.Series) -> float:
    return float(
        jensenshannon(left[list(COLUMNS)], right[list(COLUMNS)], base=2.0) ** 2
    )


def plot_profiles(table: pd.DataFrame, path: Path) -> None:
    """One row per decision maker and one panel per value."""
    makers = list(dict.fromkeys(table["decision_maker"]))
    kind_of = table.drop_duplicates("decision_maker").set_index("decision_maker")["kind"]
    labels = [
        "Physician group mean" if maker == GROUP_MEAN else display_name(maker) for maker in makers
    ]
    y = np.arange(len(makers))
    band = makers.index(GROUP_MEAN)

    fig, axes = plt.subplots(
        1, 4, figsize=(13.2, 12.2), sharey=True, layout="constrained"
    )
    for ax, value in zip(axes, COLUMNS):
        rows = table.loc[table["value"] == value].set_index("decision_maker").loc[makers]
        ax.axhspan(band - 0.5, band + 0.5, color="#f3f1ea", zorder=0)
        for maker, position in zip(makers, y):
            row = rows.loc[maker]
            color = KIND_COLOR[kind_of[maker]]
            width = 2.4 if maker == GROUP_MEAN else 1.5
            ax.plot(
                [row["ci_low"], row["ci_high"]],
                [position, position],
                color=color,
                lw=width,
                solid_capstyle="round",
                zorder=2,
            )
            # Point: posterior mean of softmax(θ / T*) across draws.
            # A person's θ is the group weight plus that person's deviation.
            # The physician group mean uses the physician group weight alone.
            # Bar: equal-tailed 95% interval of those softmax draws.
            ax.scatter(
                row["mean"],
                position,
                s=28 if maker == GROUP_MEAN else 16,
                color=color,
                zorder=3,
            )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel(VALUE_LABELS[value], fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    for label, maker in zip(axes[0].get_yticklabels(), makers):
        if maker == GROUP_MEAN:
            label.set_fontweight("bold")
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    save_figure(fig, path)


def main() -> None:
    import arviz
    import pymc

    _cases, delta = load_cases()
    decisions = load_decisions()
    frame = observation_frame(delta, decisions)

    print(
        f"pymc {pymc.__version__}    arviz {arviz.__version__}    numpy {np.__version__}"
    )
    print(
        f"NUTS draws={DRAWS} tune={TUNE} chains={CHAINS} "
        f"target_accept={TARGET_ACCEPT} seed={SEED}"
    )
    print(f"priors: w ~ Normal(0, {W_SD:g}), τ and σ ~ HalfNormal({W_SD:g})")
    print(f"T* = {T_STAR}")
    print(
        f"frame rows={len(frame)}  subjects={frame['subject'].nunique()}  "
        f"cases={frame['case'].nunique()}"
    )
    print("rows by group:", frame.groupby("group").size().to_dict())
    trial_range = frame.groupby("group")["n_trials"].agg(["min", "max"])
    print("n_trials by group:")
    print(trial_range.to_string())
    check("rows", len(frame), 12 * 50 + 25 * 50, source="expected")
    check("cases", int(frame["case"].nunique()), 50, source="expected")
    check("subjects", int(frame["subject"].nunique()), 37, source="expected")
    if flags:
        print("flags")
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)

    fit = fit_hierarchical(delta, decisions)
    diagnostics = fit.diagnostics
    print("sampler diagnostics")
    print(f"  case-effect dimension: {fit.n_case_effects}")
    check("null-space dimension", fit.n_case_effects, 46)
    for key in ("max_rhat", "min_ess_bulk", "min_ess_tail"):
        print(f"  {key}: {diagnostics[key]:.4f}")
    print(f"  n_divergences: {diagnostics['n_divergences']}")
    converged = (
        diagnostics["max_rhat"] < 1.01
        and diagnostics["min_ess_bulk"] > 400
        and diagnostics["min_ess_tail"] > 400
        and diagnostics["n_divergences"] == 0
    )
    print(f"  convergence gate: {converged}")
    print(fit.summary.round(3).to_string())

    profiles = mean_weight_profiles(fit, T_STAR)
    llm_ids = [subject for subject in fit.subjects if fit.group_of[subject] == "LLM"]
    physician_ids = [subject for subject in fit.subjects if fit.group_of[subject] == "physician"]

    print("mean JSD, hierarchical softmax(E[θ]/T*) vs unpooled GLM")
    grouped = {"LLM": [], "physician": []}
    for maker in llm_ids + physician_ids:
        rows = decisions.loc[decisions["decision_maker"] == maker]
        unpooled = softmax_profile(value_weights(delta, rows), T_STAR)
        group = "LLM" if maker in llm_ids else "physician"
        grouped[group].append(jsd(profiles[maker], unpooled))
    for group, paper_key in (("LLM", "LLM"), ("physician", "physician")):
        mean = float(np.mean(grouped[group]))
        label = "LLMs" if group == "LLM" else "physicians"
        print(f"  {label} raw {mean:.8f}")
        check(f"{label} mean JSD", f"{mean:.4f}", f"{PAPER_JSD[paper_key]:.4f}")

    print("physician group mean vs pooled-vote GLM")
    pooled_rows = decisions.loc[decisions["group"] == "physician"]
    pooled = softmax_profile(value_weights(delta, pooled_rows), T_STAR)
    group_profile = profiles[GROUP_MEAN]
    group_percent = percent(group_profile)
    pooled_percent = percent(pooled)
    consensus_jsd = jsd(group_profile, pooled)
    print(f"  group mean probabilities: {slash(group_profile[list(COLUMNS)], 4)}")
    print(f"  pooled probabilities: {slash(pooled[list(COLUMNS)], 4)}")
    check("group mean %", slash(group_percent, 1), slash(PAPER_GROUP, 1))
    check("pooled GLM %", slash(pooled_percent, 1), slash(PAPER_POOLED, 1))
    print(f"  JSD raw {consensus_jsd:.8f}")
    check("JSD", f"{consensus_jsd:.4f}", f"{PAPER_CONSENSUS_JSD:.4f}")
    print("percentage-point shift, group mean minus pooled GLM")
    for column, label in VALUE_LABELS.items():
        shift = float(np.round(group_percent[COLUMNS.index(column)] - pooled_percent[COLUMNS.index(column)], 1))
        shown = show_pp(shift)
        if column in PAPER_SHIFT:
            check(label, shown, show_pp(PAPER_SHIFT[column]))
        else:
            print(f"  {label}: {shown}")

    table = posterior_profiles(fit, T_STAR)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    plot_profiles(table, FIGURE_PATH)
    print(f"wrote {TABLE_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {FIGURE_PATH.relative_to(REPO_ROOT)}")
    print(
        "figure point: posterior mean of softmax(θ / T*) on each draw; "
        "bar: equal-tailed 95% interval"
    )

    print(
        "posterior probability that an LLM's JSD to the physician group mean "
        "exceeds the 95th percentile of the physician-to-group-mean JSDs "
        "on the same draw"
    )
    probabilities = outlier_probabilities(fit, T_STAR)
    n_draws = fit.w.shape[0]
    for maker, probability in probabilities.items():
        hits = int(round(probability * n_draws))
        print(f"  {display_name(maker)}: {probability:.4f}    ({hits} / {n_draws})")

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
