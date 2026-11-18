# Error analysis

Computed by `error_analysis.py` on 2026-11-18 from the saved result files (`results/heldout_v1.json`, `results/handwritten_v1.json`) and the data files, with no model run. Every number can be recomputed with `python error_analysis.py`. The hand-authored labels were drafted with an AI assistant and have not been checked by humans, and the held-out labels come from the generator, so both describe agreement with a rubric, not with human judgment.

## 1. Held-out set (96 synthetic examples)

The adapter got 85 right (88.5%) and 11 wrong.

| Example | Rubric score | Adapter score | Rubric verdict | Adapter verdict | Rubric score's distance to nearest boundary |
|---|---|---|---|---|---|
| 17 | 78 | 74 | strong | moderate | 3.5 |
| 28 | 53 | 42 | moderate | weak | 4.0 |
| 29 | 50 | 46 | moderate | weak | 1.0 |
| 30 | 71 | 86 | moderate | strong | 3.5 |
| 37 | 42 | 54 | weak | moderate | 7.0 |
| 41 | 80 | 72 | strong | moderate | 5.5 |
| 57 | 71 | 86 | moderate | strong | 3.5 |
| 71 | 52 | 46 | moderate | weak | 3.0 |
| 77 | 48 | 56 | weak | moderate | 1.0 |
| 85 | 58 | 86 | moderate | strong | 9.0 |
| 88 | 42 | 68 | weak | moderate | 7.0 |

Within 6 score points of a verdict boundary: **8 of the 11 errors** (72.7%), against 34 of all 96 examples (35.4%). Errors that skip a whole band (weak against strong): 0.

Mean absolute score error: 4.5 on the 85 correct verdicts and 12.5 on the 11 wrong ones.

## 2. Hand-authored set (36 resumes)

Verdict accuracy 44.4% (16 of 36); 1 output were not valid JSON and count as wrong.

### Confusion matrix (rows: rubric label, columns: adapter verdict)

|  | weak | moderate | strong | invalid |
|---|---|---|---|---|
| weak | 11 | 1 | 0 | 0 |
| moderate | 10 | 2 | 0 | 0 |
| strong | 2 | 6 | 3 | 1 |

### Per class

| Rubric class | Resumes | Recall | Precision |
|---|---|---|---|
| weak | 12 | 11/12 (91.7%) | 11/23 (47.8%) |
| moderate | 12 | 2/12 (16.7%) | 2/9 (22.2%) |
| strong | 12 | 3/12 (25.0%) | 3/3 (100.0%) |

### Direction of the errors

| Adapter vs rubric | Resumes |
|---|---|
| two bands lower | 2 |
| one band lower | 16 |
| same | 16 |
| one band higher | 1 |
| two bands higher | 0 |

Of the 19 wrong verdicts with a valid output, **18 are under-predictions** and 1 over-predictions.

### By role

| Role | Resumes | Correct |
|---|---|---|
| AI Engineer | 6 | 2 (33.3%) |
| Backend Engineer | 6 | 2 (33.3%) |
| Data Engineer | 6 | 3 (50.0%) |
| DevOps Engineer | 6 | 4 (66.7%) |
| Frontend Engineer | 6 | 3 (50.0%) |
| Software Developer | 6 | 2 (33.3%) |

### By other fields in the data file (exploratory)

| same_field | Resumes | Correct |
|---|---|---|
| False | 6 | 5 (83.3%) |
| True | 30 | 11 (36.7%) |

### By resume length (exploratory)

| Group | Length | Correct |
|---|---|---|
| shortest third | 35 to 48 words | 11 (91.7%) |
| middle third | 49 to 56 words | 4 (33.3%) |
| longest third | 57 to 77 words | 1 (8.3%) |

### Adapter score by rubric label

| Rubric label | Scored resumes | Mean adapter score | Range |
|---|---|---|---|
| weak | 12 | 38.8 | 20 to 56 |
| moderate | 12 | 40.7 | 24 to 52 |
| strong | 11 | 64.0 | 46 to 84 |

## 3. Would moving the score thresholds fix it?

The adapter's score is the number behind the verdict, so one cheap question is whether the verdict gap is mostly a shifted threshold. On the 35 resumes with a score I searched for the two thresholds (weak/moderate and moderate/strong) that give the highest accuracy, and also checked them with leave-one-out validation, which refits the thresholds without each resume in turn and tests on that resume.

| Thresholds | Accuracy on the 35 scored resumes |
|---|---|
| Training bands (49 and 74.5) | 42.9% |
| Best on these resumes (weak below 50.0, strong from 57.0), in-sample | 60.0% |
| Same search under leave-one-out validation | 31.4% |

With 35 resumes the in-sample figure is optimistic by construction; the leave-one-out figure is the fairer estimate, and it is itself noisy.

## 4. What this suggests

- **Held-out errors:** 8 of 11 sit within 6 points of a verdict boundary, against 35.4% of all examples, which fits the boundary fragility seen in the name-swap boundary test.
- **Hand-authored errors:** 18 of 19 are under-predictions, so on differently written resumes the adapter mostly rates the candidate lower than the rubric does, not randomly.
- **Thresholds:** leave-one-out accuracy moves from 42.9% to 31.4%. That is not a trustworthy gain, so recalibrating the thresholds alone would not fix the verdict gap.

## 5. What this cannot tell

- It cannot say which part of a resume caused an error; that needs interpretability work or controlled edits of the resumes.
- The hand-authored groups have 6 to 12 resumes, so one resume moves a group's accuracy by 8 to 17 points. Treat the breakdowns as leads, not findings.
- Both label sets are rubric labels. Independent human labels (see `labeling/`) are needed to tell an adapter problem from a rubric problem.
