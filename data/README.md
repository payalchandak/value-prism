# Data

De-identified Value Prism benchmark. Physician names, emails, expertise, and timestamps are not included. Physicians are labeled P01–P25. Licensed under CC BY 4.0; see ../LICENSE-DATA.

## benchmark/

`cases.csv` is the 50-case benchmark, one row per case. Columns are Case ID, Vignette, Choice 1, the four value tags for that choice (Autonomy, Beneficence, Nonmaleficence, Justice), Choice 2, and the same four tags. Tag cells are `promotes`, `violates`, or `neutral`.

## human_decisions/

One `{case_id}.json` file per case, in the DecisionRecord shape shared with the model files: `case_id`, `case` (vignette and both choices), and `models`. Keys are `human/P01` through `human/P25`. Each physician has a single run with `parsed_choice` (`choice_1` or `choice_2`) and `full_response.response_text` (the text of the chosen option).

## llm_decisions/

One `{case_id}.json` file per case. Same DecisionRecord shape, plus the `system_prompt` and `user_prompt` shown to the model. `models` holds twelve models, ten runs each. A run has `parsed_choice` and `full_response` (the provider completion payload).

## value_annotations/

`ratings.csv` is an independent re-annotation of the benchmark tags. One row per rater, case, choice, and value. Raters are R1–R8. There is no key from these labels back to a name. Every case has two or three ratings. `choice` is 1 or 2 in the same order as `benchmark/cases.csv`. `value` is autonomy, beneficence, nonmaleficence, or justice. `tag` is `promotes`, `violates`, or `neutral`. The source workbook's summary sheet has no ratings and is not included.
