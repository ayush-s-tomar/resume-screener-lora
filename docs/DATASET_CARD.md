# resume-screener-synthetic

Synthetic training and test data for [resume-screener-lora](https://github.com/ayush-s-tomar/resume-screener-lora), a LoRA adapter that reads a resume and a target role and returns a structured JSON verdict (adapter: https://huggingface.co/Kus-hal/resume-screener-lora). Every resume here is synthetic or fictional; no real person's resume or personal data was used.

## Contents

| File | Split | Rows | What it is |
|---|---|---|---|
| `train.jsonl` | train | 800 | Training examples written by a template-based generator with a written rubric. |
| `eval.jsonl` | test | 96 | Held-out examples from the same generator (the held-out set in the project's evaluation). |
| `handwritten_eval.jsonl` | test, config `handwritten` | 36 | Fictional resumes in layouts the generator never produces, labelled against a written rubric. |

## Fields

`train.jsonl` and `eval.jsonl` have two fields: `prompt` (the instruction "Screen this resume for a {role} position and return a structured verdict." followed by the resume text) and `completion` (a JSON string with the keys `ats_score`, `matched_skills`, `missing_skills`, `role`, `verdict`, `years_experience`).

`handwritten_eval.jsonl` has the fields `core_hit`, `id`, `label`, `resume`, `role`, `same_field`, `years`. `label` is `strong_match`, `moderate_match` or `weak_match`.

## Verdict distribution

**train.jsonl** (800 rows)

| Verdict | Rows | Score range in the data |
|---|---|---|
| `weak_match` | 295 | 8 to 48 |
| `moderate_match` | 319 | 50 to 74 |
| `strong_match` | 186 | 75 to 100 |

**eval.jsonl** (96 rows)

| Verdict | Rows | Score range in the data |
|---|---|---|
| `weak_match` | 38 | 8 to 48 |
| `moderate_match` | 41 | 50 to 74 |
| `strong_match` | 17 | 76 to 100 |

**handwritten_eval.jsonl** (36 rows): 12 `weak_match`, 12 `moderate_match`, 12 `strong_match`.

Roles in `train.jsonl`: AI Engineer (100), Backend Engineer (100), Data Scientist (100), DevOps Engineer (100), Frontend Developer (100), Full-Stack Developer (100), ML Engineer (100), Product Manager (100).

Roles in `handwritten_eval.jsonl`: AI Engineer (6), Backend Engineer (6), Data Engineer (6), DevOps Engineer (6), Frontend Engineer (6), Software Developer (6).

## How it was made

`train.jsonl` and `eval.jsonl` were produced by a template-based generator with a written rubric; the verdict follows from the score. The generator and the rubric are in the project repository. `handwritten_eval.jsonl` was drafted with an AI assistant against the rubric in [HANDWRITTEN_RUBRIC.md](https://github.com/ayush-s-tomar/resume-screener-lora/blob/main/data/HANDWRITTEN_RUBRIC.md) and has not been checked by humans.

## Intended use

- Reproducing the evaluations in the project, and studying how a small model behaves on a task whose labels come from a rubric.
- Teaching and demonstration.

Not for training or evaluating tools that screen real candidates, and not a measure of what makes a good candidate.

## Limitations

- The data is templated and synthetic, and the labels are the generator's rubric, not human judgments. A model can learn the rubric and look accurate: the adapter reaches 88.5% on `eval.jsonl` but 44.4% on `handwritten_eval.jsonl` (see the project README for the current numbers).
- The labels of `handwritten_eval.jsonl` are rubric labels drafted with an AI assistant. Independent human labels have not been collected yet.
- English only, a handful of roles and under a thousand rows, so any number computed on it carries wide uncertainty.
- Verdicts follow score bands, so examples close to a band edge are ambiguous.

## Links and citation

- Code, evaluation and results: https://github.com/ayush-s-tomar/resume-screener-lora
- Adapter and model card: https://huggingface.co/Kus-hal/resume-screener-lora

```bibtex
@dataset{tomar2026resumescreenersynthetic,
  author = {Tomar, Ayush Singh},
  title  = {resume-screener-synthetic},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Kus-hal/resume-screener-synthetic}
}
```

Generated from the data files by `docs/build_dataset_card.py` on 2026-11-27.
