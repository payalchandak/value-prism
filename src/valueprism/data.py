"""Read the benchmark table and the decision files shipped in data/."""

from pathlib import Path

import pandas as pd

from valueprism.schemas import DecisionRecord

# src/valueprism/data.py -> repository root is parents[2].
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Same scores as values-analysis metrics._get_alignment.
TAG_SCORE = {"promotes": 1, "neutral": 0, "violates": -1}

# CSV header stem, row column, Δ column. Δ column order is A, B, N, J.
VALUES = [
    ("Autonomy", "autonomy", "A"),
    ("Beneficence", "beneficence", "B"),
    ("Nonmaleficence", "nonmaleficence", "N"),
    ("Justice", "justice", "J"),
]


def load_cases(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per case, and the value-difference matrix Δ.

    The case table has the eight tags, four values on each choice.
    Δ has shape (n_cases, 4) and columns A, B, N, J. Its index is case_id.
    Δ = tag(choice 1) − tag(choice 2), with promotes = +1, neutral = 0,
    violates = −1.
    """
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    raw = pd.read_csv(data_dir / "benchmark" / "cases.csv")

    cases = pd.DataFrame(
        {
            "case_id": raw["Case ID"],
            "vignette": raw["Vignette"],
            "choice_1": raw["Choice 1"],
            "choice_2": raw["Choice 2"],
        }
    )
    delta = pd.DataFrame({"case_id": raw["Case ID"]})
    for header, name, letter in VALUES:
        tag_1 = raw[f"{header} C1"]
        tag_2 = raw[f"{header} C2"]
        score_1 = tag_1.map(TAG_SCORE)
        score_2 = tag_2.map(TAG_SCORE)
        unknown = sorted(set(tag_1[score_1.isna()]) | set(tag_2[score_2.isna()]))
        if unknown:
            raise ValueError(f"unknown {header} tag: {unknown}")
        cases[f"{name}_1"] = tag_1
        cases[f"{name}_2"] = tag_2
        delta[letter] = score_1.astype(int) - score_2.astype(int)

    return cases, delta.set_index("case_id")


def load_decisions(data_dir: Path | None = None) -> pd.DataFrame:
    """Return one row per run.

    Columns are decision_maker, group (``llm`` or ``physician``), case_id,
    run, and choice. ``run`` is the run's position in the file, starting at 0.
    ``choice`` is 1, 2, or missing when parsed_choice is REFUSAL or INCOHERENT.
    Physician ids are the public labels P01–P25.
    """
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    rows = []
    folders = (("physician", "human_decisions"), ("llm", "llm_decisions"))
    for group, folder in folders:
        for path in sorted((data_dir / folder).glob("*.json")):
            text = path.read_text(encoding="utf-8")
            record = DecisionRecord.model_validate_json(text)
            for maker in sorted(record.models):
                decision_maker = maker.removeprefix("human/") if group == "physician" else maker
                for run_index, run in enumerate(record.models[maker].runs):
                    if run.parsed_choice == "choice_1":
                        choice = 1
                    elif run.parsed_choice == "choice_2":
                        choice = 2
                    elif run.parsed_choice in ("REFUSAL", "INCOHERENT"):
                        choice = None
                    else:
                        raise ValueError(f"{path.name} {maker}: {run.parsed_choice}")
                    rows.append(
                        {
                            "decision_maker": decision_maker,
                            "group": group,
                            "case_id": record.case_id,
                            "run": run_index,
                            "choice": choice,
                        }
                    )

    frame = pd.DataFrame(rows, columns=["decision_maker", "group", "case_id", "run", "choice"])
    frame["choice"] = frame["choice"].astype("Int64")
    frame = frame.sort_values(["group", "decision_maker", "case_id", "run"])
    return frame.reset_index(drop=True)
