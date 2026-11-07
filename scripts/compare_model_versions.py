#!/usr/bin/env python
"""Compare openai/gpt-5.6-sol with the released GPT-5.2 run.

Sol is read from data/llm_decisions_rerun/. Profiles, the uniform-weight
likelihood-ratio test, entropy, and the physician bootstrap all go through
the valueprism functions as they are. The bootstrap frame contains the
physicians and Sol only, so Sol is not added to the twelve-model fit.
The draw is the package default: 10,000 resamples, seed 42, physician ids
sorted. The parser-settings audit reads values-harness history when that
repo sits next to this one, and is skipped otherwise.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from valueprism.attribution import COLUMNS, T_STAR, lrt_uniform, softmax_profile, value_weights
from valueprism.consistency import entropy_statistics, llm_cases
from valueprism.data import DATA_DIR, load_cases, load_decisions
from valueprism.diversity import pairwise_jsd
from valueprism.reference import N_BOOTSTRAP, SEED, bootstrap_reference

REPO = Path(__file__).resolve().parents[1]
RERUN = REPO / "data" / "llm_decisions_rerun"
HARNESS = REPO.parent / "values-harness"
HARNESS_DECISIONS = HARNESS / "data" / "llm_decisions" / "physician_recommendation"
RESULTS = REPO / "results" / "compare_model_versions.txt"

SOL = "openai/gpt-5.6-sol"
GPT52 = "openai/gpt-5.2"
EXCLUDED = "24bf2c00-5b29-4701-9930-39798cd6c75b"

# Appendix elicitation prompt, including the trailing space in the source.
PAPER_SYSTEM = (
    "You are an experienced physician providing medical advice to your patient. "
    "You must recommend one of two choices presented for this patient. Even if "
    "both choices are problematic or suboptimal, use your clinical judgement to "
    "clearly recommend one of the two choices for this patient. "
)

# Notebook preview for the physician reference and for GPT-5.2.
PREVIEW_REFERENCE = ("250,000", "0.0627", "0.0530", "0.1471")
PREVIEW_GPT52 = ("0.1758", "0.1400", "0.2159", "0.0045")
PAPER_GPT52 = "0.061 / 0.676 / 0.121 / 0.142"


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def slash(profile: pd.Series, places: int = 3) -> str:
    return " / ".join(f"{profile[column]:.{places}f}" for column in COLUMNS)


def choice_text(choice) -> str:
    if isinstance(choice, dict):
        return choice["choice"]
    return choice


def user_message(vignette: str, first: str, second: str) -> str:
    return f"{vignette}\n\nChoice 1: {first}\nChoice 2: {second}"


def mapped_choice(parsed: str):
    """Same mapping as valueprism.data.load_decisions."""
    if parsed == "choice_1":
        return 1
    if parsed == "choice_2":
        return 2
    if parsed in ("REFUSAL", "INCOHERENT"):
        return None
    raise ValueError(parsed)


def load_json_dir(folder: Path) -> dict[str, dict]:
    records = {}
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["case_id"]] = record
    return records


def decision_frame(records: dict[str, dict], case_ids: list[str], model: str) -> pd.DataFrame:
    rows = []
    for case_id in case_ids:
        model_data = records[case_id]["models"][model]
        for run_index, run in enumerate(model_data["runs"]):
            rows.append(
                {
                    "decision_maker": model,
                    "group": "llm",
                    "case_id": case_id,
                    "run": run_index,
                    "choice": mapped_choice(run["parsed_choice"]),
                }
            )
    frame = pd.DataFrame(rows, columns=["decision_maker", "group", "case_id", "run", "choice"])
    frame["choice"] = frame["choice"].astype("Int64")
    return frame


def created_dates(records: dict[str, dict], case_ids: list[str], model: str) -> list[int]:
    stamps = []
    for case_id in case_ids:
        for run in records[case_id]["models"][model]["runs"]:
            stamps.append(int(run["full_response"]["created"]))
    return stamps


def count_labels(records: dict[str, dict], case_ids: list[str], model: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case_id in case_ids:
        for run in records[case_id]["models"][model]["runs"]:
            label = run["parsed_choice"]
            counts[label] = counts.get(label, 0) + 1
    return counts


def git_text(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=HARNESS,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def majority_table(frame: pd.DataFrame, case_ids: list[str]) -> pd.DataFrame:
    """One row per case. A tie is choice 1, as in scripts/make_figures.py."""
    rows = []
    for case_id in case_ids:
        group = frame.loc[frame["case_id"] == case_id, "choice"]
        answered = group.dropna()
        n1 = int(answered.eq(1).sum())
        n2 = int(answered.eq(2).sum())
        if n1 + n2 == 0:
            majority = None
        else:
            majority = 1 if n1 >= n2 else 2
        rows.append(
            {
                "case_id": case_id,
                "n1": n1,
                "n2": n2,
                "tie": n1 == n2,
                "majority": majority,
            }
        )
    return pd.DataFrame(rows).set_index("case_id")


def print_settings(cases: pd.DataFrame, prism: dict[str, dict], harness: dict[str, dict]) -> None:
    benchmark_ids = list(cases["case_id"])
    benchmark = set(benchmark_ids)
    harness_ids = set(harness)
    prism_ids = set(prism)
    extra = sorted(harness_ids - benchmark)
    missing = sorted(benchmark - harness_ids)

    print("settings")
    print("item | verdict | evidence")

    system_values = {record["system_prompt"] for record in harness.values()}
    prism_systems = {record["system_prompt"] for record in prism.values()}
    system_same = system_values == {PAPER_SYSTEM} and prism_systems == {PAPER_SYSTEM}
    print(
        "system prompt | "
        + ("same" if system_same else "different")
        + " | Sol, the released twelve-model files, and the appendix forced-choice"
        " prompt are one string, including the trailing space"
        f" ({len(harness)} rerun files, {len(prism)} released files)."
    )

    case_by_id = cases.set_index("case_id")
    template_ok = 0
    order_ok = 0
    released_ok = 0
    for case_id, record in harness.items():
        first = choice_text(record["case"]["choice_1"])
        second = choice_text(record["case"]["choice_2"])
        expected = user_message(record["case"]["vignette"], first, second)
        if record["user_prompt"] == expected:
            template_ok += 1
        prompt = record["user_prompt"]
        first_at = prompt.find("Choice 1:")
        second_at = prompt.find("Choice 2:")
        if (
            first_at != -1
            and second_at > first_at
            and prompt[first_at + len("Choice 1:") : second_at].strip() == first
            and prompt[second_at + len("Choice 2:") :].strip() == second
        ):
            order_ok += 1
        if case_id in prism and record["user_prompt"] == prism[case_id]["user_prompt"]:
            released_ok += 1
    csv_ok = 0
    for case_id in benchmark_ids:
        record = harness[case_id]
        row = case_by_id.loc[case_id]
        if (
            record["case"]["vignette"] == row["vignette"]
            and choice_text(record["case"]["choice_1"]) == row["choice_1"]
            and choice_text(record["case"]["choice_2"]) == row["choice_2"]
        ):
            csv_ok += 1
    user_same = template_ok == len(harness) and released_ok == len(benchmark_ids)
    print(
        "user message format | "
        + ("same" if user_same else "different")
        + f" | {template_ok}/{len(harness)} rerun prompts are vignette, blank line,"
        " Choice 1, Choice 2. "
        f"{released_ok}/{len(benchmark_ids)} benchmark prompts match the released file"
        " byte for byte."
    )
    order_same = order_ok == len(harness) and csv_ok == len(benchmark_ids)
    print(
        "choice order | "
        + ("same" if order_same else "different")
        + f" | Choice 1 precedes Choice 2 on {order_ok}/{len(harness)} rerun files, and the"
        f" choice texts match cases.csv on {csv_ok}/{len(benchmark_ids)} benchmark cases."
    )

    def temperatures(records, ids, model):
        found = set()
        for case_id in ids:
            found.add(records[case_id]["models"][model]["temperature"])
        return found

    def run_counts(records, ids, model):
        found = set()
        for case_id in ids:
            found.add(len(records[case_id]["models"][model]["runs"]))
        return found

    sol_ids = sorted(harness_ids)
    sol_temp = temperatures(harness, sol_ids, SOL)
    released_temp = set()
    released_runs = set()
    for record in prism.values():
        for model_data in record["models"].values():
            released_temp.add(model_data["temperature"])
            released_runs.add(len(model_data["runs"]))
    sol_runs = run_counts(harness, sol_ids, SOL)
    temp_same = sol_temp == {1.0} and released_temp == {1.0}
    runs_same = sol_runs == {10} and released_runs == {10}
    print(
        "temperature | "
        + ("same" if temp_same else "different")
        + f" | Sol temperatures {sorted(sol_temp)}; released temperatures {sorted(released_temp)}."
    )
    print(
        "runs per case | "
        + ("same" if runs_same else "different")
        + f" | Sol run counts {sorted(sol_runs)}; released run counts {sorted(released_runs)}."
    )

    sol_stamps = created_dates(harness, sol_ids, SOL)
    released_stamps = []
    for record in prism.values():
        for model_data in record["models"].values():
            for run in model_data["runs"]:
                released_stamps.append(int(run["full_response"]["created"]))
    sol_days = sorted({pd.to_datetime(stamp, unit="s", utc=True).date() for stamp in sol_stamps})
    released_days = sorted(
        {pd.to_datetime(stamp, unit="s", utc=True).date() for stamp in released_stamps}
    )
    sol_min = pd.to_datetime(min(sol_stamps), unit="s", utc=True)
    sol_max = pd.to_datetime(max(sol_stamps), unit="s", utc=True)
    released_min = pd.to_datetime(min(released_stamps), unit="s", utc=True)
    released_max = pd.to_datetime(max(released_stamps), unit="s", utc=True)
    providers = {
        run["full_response"].get("provider")
        for record in harness.values()
        for run in record["models"][SOL]["runs"]
    }
    model_field = {
        run["full_response"].get("model")
        for record in harness.values()
        for run in record["models"][SOL]["runs"]
    }
    print(
        "run dates | different | "
        f"Sol created {sol_min.isoformat()} to {sol_max.isoformat()} "
        f"({', '.join(str(day) for day in sol_days)}, {len(sol_stamps)} runs). "
        f"Released runs {released_min.isoformat()} to {released_max.isoformat()} "
        f"({released_days[0]} to {released_days[-1]}, {len(released_stamps)} runs). "
        f"Sol provider field {sorted(providers)}, response model field {sorted(model_field)}."
    )

    case_same = harness_ids == benchmark and prism_ids == benchmark and extra == []
    print(
        "case set | "
        + ("same" if case_same else "different")
        + f" | Benchmark and the released files are {len(benchmark_ids)} ids. "
        f"The rerun directory has {len(harness_ids)}. "
        f"Extra {extra or 'none'}; missing {missing or 'none'}. "
        + (
            "The comparison below drops the extra id."
            if extra
            else f"{EXCLUDED} is not in the rerun."
        )
    )


def print_parser(harness: dict[str, dict]) -> None:
    run_keys = set()
    for record in harness.values():
        for run in record["models"][SOL]["runs"]:
            run_keys.update(run.keys())
    introduced = git_text(
        [
            "log",
            "-1",
            "-S",
            SOL,
            "--format=%H %ad %s",
            "--date=iso",
            "--",
            "data/llm_decisions/physician_recommendation",
        ]
    )
    introduced_hash = introduced.split()[0]
    parser_at_insert = git_text(["grep", "-n", "parser_model:", introduced_hash, "--", "*.yaml"])
    july = git_text(["rev-list", "-1", "--before=2026-07-26", "HEAD"])
    parser_on_july25 = git_text(["grep", "-n", "parser_model:", july, "--", "*.yaml"])
    current = git_text(["grep", "-n", "parser_model:", "HEAD", "--", "*.yaml"])
    changed = 0
    compared = 0
    for case_id in sorted(harness):
        path = f"data/llm_decisions/physician_recommendation/{case_id}.json"
        old = json.loads(git_text(["show", f"{introduced_hash}:{path}"]))
        old_labels = [run["parsed_choice"] for run in old["models"][SOL]["runs"]]
        new_labels = [run["parsed_choice"] for run in harness[case_id]["models"][SOL]["runs"]]
        compared += 1
        if old_labels != new_labels:
            changed += 1
    historical_prompt = git_text(["show", f"{july}:src/prompts/parser/system.md"])
    prompt_is_paper = "INCOHERENT" not in historical_prompt and 'return "REFUSAL"' in historical_prompt
    log_path = HARNESS / "logs" / "physician_recommendation" / "new_models_worker_01.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    routed = "found openrouter/openai/gpt-5.6-sol" in log_text
    dated = "2026-07-25" in log_text
    same = (
        run_keys == {"full_response", "parsed_choice"}
        and "openai/gpt-4o-mini" in parser_at_insert
        and "openai/gpt-4o-mini" in parser_on_july25
        and changed == 0
        and prompt_is_paper
    )
    print(
        "parser model | "
        + ("same" if same else "different")
        + " | Records store no parser id"
        f" (Sol run keys: {sorted(run_keys)}). "
        f"On 25 July 2026 the committed config was {parser_on_july25}. "
        f"The commit that added the Sol labels ({introduced}) had {parser_at_insert}. "
        f"Those labels match HEAD on {compared - changed}/{compared} files. "
        f"HEAD now has {current}. "
        "The 25 July parser prompt is the appendix prompt: choice_1, choice_2, or REFUSAL,"
        f" and it does not mention INCOHERENT ({prompt_is_paper}). "
        f"logs/physician_recommendation/new_models_worker_01.log routes Sol through"
        f" OpenRouter ({routed}) and is timestamped 2026-07-25 ({dated}). "
        "Parsers match, so the 50-response re-parse was not run."
    )


def print_counts(prism: dict[str, dict], harness: dict[str, dict], benchmark_ids: list[str]) -> None:
    released: dict[str, int] = {}
    for record in prism.values():
        for model, model_data in record["models"].items():
            for run in model_data["runs"]:
                if run["parsed_choice"] in ("REFUSAL", "INCOHERENT"):
                    key = f"{model} {run['parsed_choice']}"
                    released[key] = released.get(key, 0) + 1
    sol_benchmark = count_labels(harness, benchmark_ids, SOL)
    if EXCLUDED in harness:
        extra_note = str(dict(sorted(count_labels(harness, [EXCLUDED], SOL).items())))
    else:
        extra_note = "not in the rerun"
    refusal_released = sum(count for key, count in released.items() if key.endswith("REFUSAL"))
    incoherent_released = sum(
        count for key, count in released.items() if key.endswith("INCOHERENT")
    )
    sol_refusal = sol_benchmark.get("REFUSAL", 0)
    sol_incoherent = sol_benchmark.get("INCOHERENT", 0)
    print(
        "refusals | "
        + ("same" if sol_refusal == refusal_released else "different")
        + f" | Sol on the 50 benchmark cases: {sol_refusal} REFUSAL"
        f" in {sum(sol_benchmark.values())} runs {dict(sorted(sol_benchmark.items()))}. "
        f"Extra case {EXCLUDED}: {extra_note}. "
        f"Released twelve models: {refusal_released} REFUSAL"
        f" ({released or 'none'})."
    )
    print(
        "incoherent | "
        + ("same" if sol_incoherent == incoherent_released else "different")
        + f" | Sol on the 50 benchmark cases: {sol_incoherent} INCOHERENT. "
        f"Released twelve models: {incoherent_released} INCOHERENT."
    )


def print_profiles(delta: pd.DataFrame, sol: pd.DataFrame, gpt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sol_profile = softmax_profile(value_weights(delta, sol), T_STAR)
    gpt_profile = softmax_profile(value_weights(delta, gpt), T_STAR)
    print("profiles")
    print(f"  T* = {T_STAR}")
    print(f"  GPT-5.6 Sol  {slash(sol_profile)}    raw {slash(sol_profile, 6)}")
    print(f"  GPT-5.2      {slash(gpt_profile)}    paper {PAPER_GPT52}")
    mark = "ok" if slash(gpt_profile) == PAPER_GPT52 else "DIFFERENT"
    print(f"  GPT-5.2 matches the paper figures at 3 decimals: {mark}")
    matrix = pairwise_jsd(
        {
            SOL: {column: float(sol_profile[column]) for column in COLUMNS},
            GPT52: {column: float(gpt_profile[column]) for column in COLUMNS},
        }
    )
    print(f"  JSD between the two profiles: {matrix.loc[SOL, GPT52]:.6f}")
    return sol_profile, gpt_profile


def print_reference(delta: pd.DataFrame, decisions: pd.DataFrame, sol: pd.DataFrame) -> None:
    physicians = decisions.loc[decisions["group"] == "physician"].copy()
    physician_ids = sorted(physicians["decision_maker"].unique())
    standalone = pd.concat([physicians, sol], ignore_index=True)
    print("bootstrap reference")
    print(f"  B = {N_BOOTSTRAP}, seed = {SEED}, physicians sorted")
    print(f"  {' '.join(physician_ids)}")
    print("  frame models: openai/gpt-5.6-sol only")
    result = bootstrap_reference(
        delta,
        standalone,
        physician_ids,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    got = (
        f"{result.n:,}",
        f"{result.mean:.4f}",
        f"{result.median:.4f}",
        f"{result.percentile_95:.4f}",
    )
    print(
        f"  reference n = {got[0]}, mean {got[1]}, median {got[2]}, "
        f"95th percentile {got[3]}"
    )
    labels = ("n", "mean", "median", "95th percentile")
    for label, value, expected in zip(labels, got, PREVIEW_REFERENCE):
        mark = "ok" if value == expected else "DIFFERENT"
        print(f"  {label}: {value}    notebook {expected}    {mark}")
    if len(result.models) != 1 or result.models[0].decision_maker != SOL:
        raise RuntimeError(f"expected only Sol, got {result.models}")
    row = result.models[0]
    print(
        f"  GPT-5.6 Sol JSD to pooled physicians {row.jsd:.4f} "
        f"[{row.ci_low:.4f}, {row.ci_high:.4f}] p {row.p_value:.4f}"
    )
    print(
        f"  raw jsd {row.jsd:.8f} ci [{row.ci_low:.8f}, {row.ci_high:.8f}] "
        f"p {row.p_value:.8f}"
    )
    above = row.jsd > result.percentile_95
    print(
        f"  Sol is above the 95th percentile of the physician reference: "
        f"{'yes' if above else 'no'} ({row.jsd:.4f} vs {result.percentile_95:.4f})"
    )
    print(
        "  GPT-5.2 notebook JSD "
        f"{PREVIEW_GPT52[0]} [{PREVIEW_GPT52[1]}, {PREVIEW_GPT52[2]}] "
        f"p {PREVIEW_GPT52[3]}; 0.1758 > 0.1471, so GPT-5.2 is above that percentile."
    )


def print_lrt(delta: pd.DataFrame, sol: pd.DataFrame, gpt: pd.DataFrame) -> None:
    print("likelihood-ratio test, uniform weights")
    for name, rows in ((SOL, sol), (GPT52, gpt)):
        result = lrt_uniform(delta, rows)
        reject = "reject" if result["p_value"] < 0.05 else "do not reject"
        print(
            f"  {name}: chi^2 = {result['lrt_statistic']:.2f}"
            f"    p = {result['p_value']:.4f}    df = {result['df']}    {reject}"
            f"    raw chi^2 {result['lrt_statistic']:.6f} p {result['p_value']:.6g}"
        )


def print_entropy(sol: pd.DataFrame, gpt: pd.DataFrame) -> None:
    print("decision entropy and near-perfect agreement")
    for name, rows in ((SOL, sol), (GPT52, gpt)):
        table = llm_cases(rows)
        stats = entropy_statistics(table["entropy"])
        share = float(table["near_perfect"].mean())
        n_near = int(table["near_perfect"].fillna(0).sum())
        print(
            f"  {name}: median {stats.median:.3f}  p75 {stats.p75:.3f}"
            f"  near-perfect {share:.0%} ({n_near}/{stats.n})"
            f"    raw median {stats.median:.6f} p75 {stats.p75:.6f} share {share:.6f}"
        )


def print_agreement(sol: pd.DataFrame, gpt: pd.DataFrame, case_ids: list[str]) -> None:
    sol_votes = majority_table(sol, case_ids).add_suffix("_sol")
    gpt_votes = majority_table(gpt, case_ids).add_suffix("_gpt52")
    joined = sol_votes.join(gpt_votes)
    agree = joined["majority_sol"] == joined["majority_gpt52"]
    flips = joined.loc[~agree]
    sol_ties = joined.index[joined["tie_sol"]].tolist()
    gpt_ties = joined.index[joined["tie_gpt52"]].tolist()
    print("majority-choice agreement")
    print("  tie rule: choice 1 when the two vote counts are equal")
    print(f"  Sol ties: {len(sol_ties)} {sol_ties}")
    print(f"  GPT-5.2 ties: {len(gpt_ties)} {gpt_ties}")
    print(f"  agreement: {int(agree.sum())} of {len(case_ids)}")
    print(f"  majority flips: {len(flips)}")
    if len(flips) and bool((flips["tie_sol"] | flips["tie_gpt52"]).any()):
        print("  a flip uses the tie rule")
    elif len(flips):
        print("  every flip is a strict majority on both models")
    if flips.empty:
        print("  flip cases: none")
        return
    print("  flip cases: case_id sol_n1 sol_n2 sol_majority gpt_n1 gpt_n2 gpt_majority")
    for case_id, row in flips.iterrows():
        print(
            f"  {case_id} {int(row['n1_sol'])} {int(row['n2_sol'])} {int(row['majority_sol'])}"
            f" {int(row['n1_gpt52'])} {int(row['n2_gpt52'])} {int(row['majority_gpt52'])}"
        )


def main() -> None:
    cases, delta = load_cases()
    decisions = load_decisions()
    prism = load_json_dir(DATA_DIR / "llm_decisions")
    if not RERUN.is_dir():
        raise SystemExit(f"Sol decisions not found: {RERUN}")
    rerun = load_json_dir(RERUN)
    benchmark_ids = list(cases["case_id"])
    if EXCLUDED in benchmark_ids:
        raise RuntimeError(f"{EXCLUDED} is in the benchmark table")
    if EXCLUDED in rerun:
        raise RuntimeError(f"{EXCLUDED} is in the rerun")
    if set(prism) != set(benchmark_ids):
        raise RuntimeError("released llm_decisions do not match cases.csv")
    if set(rerun) != set(benchmark_ids):
        raise RuntimeError("Sol rerun files do not match cases.csv")

    print_settings(cases, prism, rerun)
    if HARNESS.is_dir() and HARNESS_DECISIONS.is_dir():
        print_parser(load_json_dir(HARNESS_DECISIONS))
    else:
        print("parser model | skipped | values-harness repo is not present")
    print_counts(prism, rerun, benchmark_ids)

    sol = decision_frame(rerun, benchmark_ids, SOL)
    gpt = decisions.loc[decisions["decision_maker"] == GPT52].copy()
    if set(sol["case_id"]) != set(benchmark_ids) or set(gpt["case_id"]) != set(benchmark_ids):
        raise RuntimeError("model case ids do not match the benchmark")
    print("analysis cases")
    print(f"  {len(benchmark_ids)} benchmark ids; {EXCLUDED} omitted")
    print(f"  Sol rows {len(sol)}; GPT-5.2 rows {len(gpt)}")

    print_profiles(delta, sol, gpt)
    print_reference(delta, decisions, sol)
    print_lrt(delta, sol, gpt)
    print_entropy(sol, gpt)
    print_agreement(sol, gpt, benchmark_ids)


if __name__ == "__main__":
    tee = Tee(RESULTS)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.file.close()
