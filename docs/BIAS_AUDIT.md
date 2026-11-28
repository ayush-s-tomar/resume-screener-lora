# Name sensitivity checks (this is not a bias audit)

> Written by `docs/build_docs.py` on 2026-11-25 from `results/name_swap_v1.json` and `results/name_swap_boundary_v1.json`.

These checks ask one narrow question: does the adapter's output change when only the candidate's name at the top of the resume changes? They are a sensitivity check on one adapter, synthetic resumes and a handful of names I chose. They say nothing about whether the adapter is fair to any group of people.

## Test 1: 15 held-out resumes, 8 names

Each resume was scored once as it is and once with each name placed at the top. A score change is the score with the name minus the score without it.

| Prefix | Resumes | Mean score change | Mean absolute change | Verdict flips |
|---|---|---|---|---|
| Emily Walsh | 15 | +0.7 | 0.7 | 0 of 15 |
| Greg Baker | 15 | -0.3 | 1.1 | 0 of 15 |
| Lakisha Washington | 15 | +0.1 | 1.2 | 1 of 15 |
| Jamal Robinson | 15 | +0.7 | 0.7 | 0 of 15 |
| Priya Sharma | 15 | +0.4 | 0.4 | 0 of 15 |
| Rahul Verma | 15 | -0.1 | 0.9 | 1 of 15 |
| Mei Chen | 15 | +0.0 | 1.3 | 0 of 15 |
| Mohammed Hassan | 15 | +0.1 | 1.2 | 1 of 15 |
| Candidate A (control) | 15 | +0.4 | 0.4 | 0 of 15 |

Across the names, the mean absolute score change was 0.9 points (largest single change 10), and 3 of 120 verdicts flipped.
The flips came from 1 resume(s): resume 5 (base score 76, 1.5 points from the nearest verdict boundary): 3 name(s).

## Test 2: the 8 resumes closest to a verdict boundary, names against neutral prefixes

The first test's flips sat on a resume close to a verdict boundary, so this test picked the held-out resumes whose adapter score is closest to a boundary (within 1.5 points) and compared the names with neutral prefixes such as "Applicant 17" and "Reference 4821". Several neutral prefixes contain digits, which may disturb the model more than a name does.

| Prefix type | Verdict flips | Mean absolute score change |
|---|---|---|
| Names | 3 of 64 (4.7%) | 1.1 points |
| Neutral prefixes | 13 of 64 (20.3%) | 3.8 points |

Across the resumes, names flipped more verdicts on 0, neutral prefixes on 6, and they tied on 2 (sign test p = 0.03; read it loosely, the pairs are not independent).

Neutral prefixes flipped more verdicts than names did, so near a boundary the adapter is fragile to changes at the top of the prompt in general. That does not show that names are harmless.

## What these checks do not show

- They do not measure outcomes for any group. There are no real candidates, no selection rates and no impact ratios, which is what bias audits for hiring tools normally report.
- A name is one proxy among many, and a weak one for most protected characteristics (age, disability, caste, religion, nationality). Other cues in a resume, such as a school, a gap or a language, were not tested.
- The names were chosen by me, a small set was used, and the resumes are synthetic. One adapter was tested.
- No difference found in a small sample is not evidence of no difference.

## What a real audit would need

1. Real resumes (with consent) and the real decisions or outcomes the tool would influence, in the setting where it would be used.
2. Demographic categories for those candidates, collected lawfully, with selection rates and impact ratios by group, including intersections.
3. Statistical tests that account for small groups, and a plan for what to do when a gap is found.
4. An independent auditor where the law requires one, repeated on a schedule and after every model change, with results published where required.

See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for the wider picture.
