#!/usr/bin/env python
"""Print the benchmark checks cited in the paper, and compare choice counts
to values-analysis's loader on the values-analysis data."""

import sys
import types
from pathlib import Path

import pandas as pd

from valueprism.data import DATA_DIR, load_cases, load_decisions
from valueprism.schemas import DecisionRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT.parent / "values-analysis"

PAPER_ROWS = {"physician": 25 * 50, "llm": 12 * 50 * 10}
# model id -> (name used in the paper, refusal count)
PAPER_REFUSALS = {
    "meta-llama/llama-4-maverick": ("Llama 4 Maverick", 1),
    "moonshotai/kimi-k2-thinking": ("Kimi K2", 1),
    "z-ai/glm-4.6": ("GLM 4.6", 7),
}
PAPER_REFUSAL_TOTAL = 9
# Pair order is the paper's order, then B-J, which the paper does not give.
PAPER_TENSIONS = {("A", "N"): 28, ("A", "B"): 23, ("B", "N"): 22, ("A", "J"): 17, ("J", "N"): 12}
TENSION_PAIRS = [("A", "N"), ("A", "B"), ("B", "N"), ("A", "J"), ("J", "N"), ("B", "J")]
PAPER_ENGAGED = {2: 11, 3: 21, 4: 18}
VALUE_NAMES = ["autonomy", "beneficence", "nonmaleficence", "justice"]

flags: list[str] = []


def check(label: str, got, expected) -> None:
    mark = "ok" if got == expected else "DIFFERENT"
    print(f"  {label}: {got}    paper {expected}    {mark}")
    if got != expected:
        flags.append(f"{label}: got {got}, paper {expected}")


def count_parsed(folder: str) -> tuple[dict[str, int], dict[str, int]]:
    refusals: dict[str, int] = {}
    incoherent: dict[str, int] = {}
    for path in sorted((DATA_DIR / folder).glob("*.json")):
        record = DecisionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for maker, model in record.models.items():
            for run in model.runs:
                if run.parsed_choice == "REFUSAL":
                    refusals[maker] = refusals.get(maker, 0) + 1
                elif run.parsed_choice == "INCOHERENT":
                    incoherent[maker] = incoherent.get(maker, 0) + 1
    return refusals, incoherent


def choice_counts(frame: pd.DataFrame, choice: int) -> dict[tuple[str, str], int]:
    picked = frame["choice"].eq(choice)
    grouped = picked.groupby([frame["decision_maker"], frame["case_id"]], sort=False).sum()
    return {(maker, case_id): int(n) for (maker, case_id), n in grouped.items()}


def analysis_choice_counts():
    """Choice counts from values-analysis's own loader and summarize().

    src.analysis.__init__ imports the regression stack. Register the package
    first so Python loads loader.py and summary.py and skips that stack.
    """
    if not ANALYSIS_ROOT.is_dir():
        raise SystemExit(f"values-analysis repo not found: {ANALYSIS_ROOT}")
    sys.path.insert(0, str(ANALYSIS_ROOT))
    import src

    package = types.ModuleType("src.analysis")
    package.__path__ = [str(ANALYSIS_ROOT / "src" / "analysis")]
    package.__package__ = "src.analysis"
    src.analysis = package
    sys.modules["src.analysis"] = package

    from src.analysis.loader import load_human_decisions, load_llm_decisions
    from src.analysis.summary import summarize

    human_records = load_human_decisions()
    # Same rule as the export: source ids sorted ascending become P01–P25.
    old_ids = sorted(
        {
            maker.removeprefix("human/")
            for record in human_records
            for maker in record.models
            if maker.startswith("human/")
        }
    )
    if len(old_ids) != 25:
        raise SystemExit(f"expected 25 source physicians, found {len(old_ids)}")
    id_map = {old: f"P{number:02d}" for number, old in enumerate(old_ids, start=1)}

    def counts(records, public_name):
        choice_1: dict[tuple[str, str], int] = {}
        choice_2: dict[tuple[str, str], int] = {}
        for record in records:
            for maker, model in record.models.items():
                summary = summarize(model)
                key = (public_name(maker), record.case_id)
                choice_1[key] = summary.choice_1_count
                choice_2[key] = summary.choice_2_count
        return choice_1, choice_2

    def physician_name(maker: str) -> str:
        old = maker.removeprefix("human/")
        if old not in id_map:
            raise SystemExit(f"{old} is not a source physician id")
        return id_map[old]

    human_1, human_2 = counts(human_records, physician_name)
    model_1, model_2 = counts(load_llm_decisions(), lambda maker: maker)
    return human_1, human_2, model_1, model_2


def mismatches(release, source, case_ids: set[str]) -> list[tuple]:
    found = []
    keys = {key for key in release if key[1] in case_ids}
    keys |= {key for key in source if key[1] in case_ids}
    for maker, case_id in sorted(keys):
        got = release.get((maker, case_id))
        expected = source.get((maker, case_id))
        if got != expected:
            found.append((maker, case_id, got, expected))
    return found


def print_mismatches(label: str, found: list[tuple]) -> None:
    print(f"  {label}: {len(found)}")
    for maker, case_id, got, expected in found[:20]:
        print(f"    {maker} {case_id}: release {got}    analysis {expected}")
    if found:
        flags.append(f"{label}: {len(found)}")


def main() -> None:
    cases, delta = load_cases()
    decisions = load_decisions()

    print("rows")
    for group, expected in PAPER_ROWS.items():
        check(group, int((decisions["group"] == group).sum()), expected)
    print(f"  cases: {len(cases)}")
    print(f"  physicians: {decisions.loc[decisions.group == 'physician', 'decision_maker'].nunique()}")
    print(f"  models: {decisions.loc[decisions.group == 'llm', 'decision_maker'].nunique()}")

    print("refusals")
    refusals, incoherent = count_parsed("llm_decisions")
    models = sorted(decisions.loc[decisions["group"] == "llm", "decision_maker"].unique())
    for model in models:
        got = refusals.get(model, 0)
        if model in PAPER_REFUSALS:
            paper_name, expected = PAPER_REFUSALS[model]
            check(f"{paper_name} ({model})", got, expected)
        else:
            check(model, got, 0)
    check("total", sum(refusals.values()), PAPER_REFUSAL_TOTAL)
    incoherent_total = sum(incoherent.values())
    print(f"  incoherent: {incoherent_total}")
    if incoherent_total:
        flags.append(f"incoherent runs: {incoherent_total}")

    # Opposite signs, same test as values-analysis value_tension_pairs.
    print("value pairs in tension (Δ components with opposite signs)")
    for left, right in TENSION_PAIRS:
        got = int((delta[left] * delta[right] < 0).sum())
        label = f"{left}-{right}"
        if (left, right) in PAPER_TENSIONS:
            check(label, got, PAPER_TENSIONS[(left, right)])
        else:
            print(f"  {label}: {got}    paper (not stated)")

    print("cases by number of engaged values")
    n_engaged = pd.Series(0, index=cases.index)
    for name in VALUE_NAMES:
        engaged = (cases[f"{name}_1"] != "neutral") | (cases[f"{name}_2"] != "neutral")
        n_engaged = n_engaged + engaged.astype(int)
    counted = {int(k): int(v) for k, v in n_engaged.value_counts().items()}
    for k in sorted(set(PAPER_ENGAGED) | set(counted)):
        check(str(k), counted.get(k, 0), PAPER_ENGAGED.get(k, 0))

    print("choice counts vs values-analysis loader")
    if not ANALYSIS_ROOT.is_dir():
        print("  skipped: values-analysis repo is not present")
    else:
        human_1, human_2, model_1, model_2 = analysis_choice_counts()
        case_ids = set(cases["case_id"])
        physicians = decisions[decisions["group"] == "physician"]
        llms = decisions[decisions["group"] == "llm"]
        print_mismatches(
            "physician choice-1 mismatches",
            mismatches(choice_counts(physicians, 1), human_1, case_ids),
        )
        print_mismatches(
            "physician choice-2 mismatches",
            mismatches(choice_counts(physicians, 2), human_2, case_ids),
        )
        print_mismatches(
            "model choice-1 mismatches",
            mismatches(choice_counts(llms, 1), model_1, case_ids),
        )
        print_mismatches(
            "model choice-2 mismatches",
            mismatches(choice_counts(llms, 2), model_2, case_ids),
        )
        extra_cases = sorted({case_id for _maker, case_id in model_1 if case_id not in case_ids})
        print(f"  analysis model cases not in the release: {len(extra_cases)}")
        for case_id in extra_cases:
            print(f"    {case_id}")

    print("flags")
    if flags:
        for flag in flags:
            print(f"  {flag}")
        raise SystemExit(1)
    print("  none")


if __name__ == "__main__":
    main()
