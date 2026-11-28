# Responsible use

> Written by `docs/build_docs.py` on 2026-11-25 from the files in `results/`; regenerate it with `python docs/build_docs.py`. It states what this project is and is not for. It is not legal advice.

## In one paragraph

resume-screener-lora is a research and portfolio project. Its labels are synthetic, it has never been validated on real resumes or on real hiring outcomes, and it has not been audited for bias. It must not be used to make, or to support, decisions about real people. The hosted demo exists to show the output format and the evaluation work, not to screen anyone.

## What the evidence shows, and what it does not

| Check | Result | Where |
|---|---|---|
| Held-out set (96 synthetic examples from the same generator as the training data) | 88.5% verdict accuracy | [comparison_results.json](../comparison_results.json) |
| Hand-authored set (36 fictional resumes in other layouts, labelled against an AI-drafted rubric and not checked by humans) | 44.4% (16/36) verdict accuracy, 2 severe errors (weak against strong), scores ordered correctly in 80.4% of pairs | [handwritten_v1.json](../results/handwritten_v1.json) |
| Name swap (15 held-out resumes, 8 names) | 3 of 120 verdicts flipped, mean absolute score change 0.9 points, all flips on 1 resume(s) | [BIAS_AUDIT.md](BIAS_AUDIT.md) |
| Boundary check (the 8 resumes closest to a verdict boundary) | names flipped 3 of 64 verdicts, neutral prefixes 13 of 64 | [BIAS_AUDIT.md](BIAS_AUDIT.md) |
| Skill lists (held-out set, against the generator's lists) | matched F1 1.00, missing F1 0.91; 4.3% of the skills listed as missing appear in the resume text | [skill_lists_v1.json](../results/skill_lists_v1.json) |

What this adds up to: the adapter reproduces its generator's rubric well on data from that generator (88.5%) and much less well on resumes written differently (44.4%), where it mostly rates candidates lower than my rubric does. Nothing here measures whether a score predicts job performance, whether the rubric itself is fair, or how the adapter treats groups of people. "Accuracy" always means agreement with a rubric.

## Known failure modes

- **Unfamiliar layouts.** On the hand-authored set 18 of the 19 wrong verdicts with a valid output are under-predictions (the adapter rated the resume lower than the rubric) and 1 are over-predictions.
- **Verdicts near a boundary are unstable.** Small changes at the top of the prompt flipped verdicts on the resumes whose score sits close to a verdict threshold, with names and with neutral prefixes alike.
- **Invalid output.** 1 of 36 outputs on the hand-authored set were not valid JSON.
- **Skill lists.** 4.3% of the skills listed as missing appear in the resume text, so a missing-skills list can be wrong.
- **Narrow scope.** English only, six roles, synthetic training data, one small model.
- **Protected characteristics.** Only a name check was run. Nothing was measured for age, gender, disability, caste, nationality or any other characteristic, and names are a weak proxy for most of them.

## Uses this project does not support

- Automatically rejecting, ranking or shortlisting real candidates, or using a score as the only basis for a decision about a person.
- Scoring people who have not been told that software is screening them.
- Submitting real people's resumes to the hosted demo: the text is sent to a third-party API (see [PRIVACY.md](PRIVACY.md)).
- Roles, languages or resume layouts that the evaluation did not cover.
- Treating the borderline flag as a safeguard: its margin (6 points) was chosen on 96 synthetic examples, and it only partly carries over to the hand-authored set (see [SELECTIVE_PREDICTION.md](SELECTIVE_PREDICTION.md)).

## What would be needed before anyone used something like this for real

| Condition | Status in this project |
|---|---|
| A defined purpose, and a human who makes every decision and can overrule the score | Not applicable: the project has no real use |
| Validation on real resumes from the actual setting, labelled by several qualified people, with their agreement reported | Not done; a labelling form exists in `labeling/`, but no labels have been collected |
| An adverse-impact analysis by group, using real selection outcomes, by an independent auditor where the law requires one | Not done; the name checks are not a substitute |
| Advance notice to candidates, and a way to ask for human review | Not done |
| Data minimisation and protection: mask identifiers, prefer local inference, short retention, a data-protection assessment | Partly: the repository can mask emails, links, phone numbers and ID numbers before scoring; no assessment was done |
| Monitoring for drift and complaints, and a way to withdraw the tool | Not done |
| Documentation and review | Partly: a model card in [MODEL_CARD.md](MODEL_CARD.md), a changelog and these documents |

## Rules that apply to hiring tools (pointers, not advice)

Checked on 2026-11-25 against public summaries, not against the legal texts. Rules and dates change; read the current primary sources and ask a lawyer before relying on any of this.

- **New York City, Local Law 144 of 2021.** Employers and employment agencies that use an automated employment decision tool to screen candidates in the city need a bias audit by an independent auditor within a year before use, must publish a summary of the results, and must notify candidates in advance (at least 10 business days). The audit looks at disparate impact across sex and race/ethnicity categories.
- **EU AI Act.** Annex III lists AI systems used to analyse and filter job applications and to evaluate candidates as high-risk. Under the Digital Omnibus on AI (Regulation (EU) 2026/1744), the date from which the obligations for stand-alone Annex III high-risk systems apply moved from 2 August 2026 to 2 December 2027. The delay is not a repeal.
- **Data protection.** The EU and UK GDPR (including its rule on decisions based solely on automated processing) and India's Digital Personal Data Protection Act, 2023, apply to personal data in resumes. Check their current status and rules.

## Reporting a concern

If you think this project is being used in a way it should not be, or you find a mistake in these documents, open an issue at https://github.com/ayush-s-tomar/resume-screener-lora/issues or use the private route in [SECURITY.md](../SECURITY.md).
