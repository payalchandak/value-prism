# Value Prism

## Overview

ValuePrism is a benchmark of 50 clinical cases. Each case offers two choices, tagged for how they bear on autonomy, beneficence, nonmaleficence, and justice, together with decisions from 25 physicians and 12 language models. The attribution method fits a binomial GLM that maps those tag differences onto choice probabilities, then turns the fitted weights into a value profile. This repository is the public code and data release for the NeurIPS 2026 paper “Do Language Models Share Your Doctor’s Priorities? Measuring Values from Clinical Decisions”.

```bibtex
@inproceedings{valueprism2026,
  title={Do Language Models Share Your Doctor's Priorities? Measuring Values from Clinical Decisions},
  author={Payal Chandak, Victoria Alkin, David Wu, Maya Dagan, Taposh Dutta Roy, Maria Clara Saad Menezes, Ayush Noori, Nirali Somia, John S. Brownstein, Ran Balicer, Rebecca W. Brendel, Noa Dagan, Isaac S. Kohane, Gabriel A. Brat},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Contents

The data are described in [data/README.md](data/README.md):

- `data/benchmark/` — the 50 cases (`cases.csv`)
- `data/human_decisions/` — one DecisionRecord per case for physicians P01–P25
- `data/llm_decisions/` — the same shape for twelve models, ten runs each
- `data/llm_decisions_rerun/` — GPT-5.6 Sol on the same 50 cases, ten runs each
- `data/value_annotations/` — independent tag ratings from annotators R1–R8

`src/valueprism/` is the analysis package:

- `data` and `schemas` load the benchmark and the decision records
- `attribution` fits the value weights, the profiles, and the likelihood-ratio test
- `consistency` measures agreement and entropy of the decisions
- `reference` compares a model with the physician panel
- `diversity` measures within-group spread and runs the permutation test
- `calibration` selects the softmax temperature
- `robustness/regularized`, `robustness/hierarchical`, and `robustness/tags` are the ridge, Firth, hierarchical, and tag-sensitivity checks
- `plots` writes the paper figures

Scripts:

- `scripts/check_data.py` — benchmark counts
- `scripts/check_consistency.py` — decision consistency
- `scripts/check_attribution.py` — value profiles and the likelihood-ratio test
- `scripts/check_reference.py` — physician-panel comparison
- `scripts/check_diversity.py` — diversity and the permutation test
- `scripts/calibrate_temperature.py` — temperature calibration
- `scripts/check_regularized.py` — ridge and Firth
- `scripts/check_hierarchical.py` — hierarchical model
- `scripts/check_tags.py` — tag sensitivity
- `scripts/compare_model_versions.py` — GPT-5.6 Sol against the released GPT-5.2 run
- `scripts/make_figures.py` and `scripts/make_profile_figures.py` — figures

`tests/` holds the package tests.

## Install

```bash
uv sync
```

## Reproducing the paper

| Paper result | Script |
| --- | --- |
| Decision consistency (§4) | `scripts/check_consistency.py` |
| Value profiles and LRT (§5) | `scripts/check_attribution.py` |
| Physician-panel comparison and p-values (§6, App. G) | `scripts/check_reference.py` |
| Diversity and permutation test (§9, App. I) | `scripts/check_diversity.py` |
| Temperature calibration (App. F) | `scripts/calibrate_temperature.py` |
| Ridge/Firth, hierarchical model, tag sensitivity (§7, App. F) | `scripts/check_regularized.py`, `scripts/check_hierarchical.py`, `scripts/check_tags.py` |
| GPT-5.6 Sol comparison (App. C) | `scripts/compare_model_versions.py` |
| All figures | `scripts/make_figures.py`, `scripts/make_profile_figures.py` |

## Notes

The released model decisions were parsed with GPT-4o-mini, as in the paper.

MCMC summaries may differ in the last digit on other platforms. The pinned lockfile reproduces the paper.

Physicians are anonymized as P01–P25 and annotators as R1–R8. No identifying information is included.

The benchmark is the 50 cases in `data/benchmark/cases.csv`.

## Licenses

Code is under the MIT license ([LICENSE](LICENSE)). Data are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)).
