# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This is a research project, so a minor version adds evaluation, tooling or demo changes, and a major version would mean a new adapter.

## [Unreleased]

## [1.1] - 2026-11-18

### Highlights

- 95% Wilson confidence intervals on the 96-example held-out evaluation
- A name-swap sensitivity test, and a boundary-case follow-up with neutral-prefix controls
- An evaluation of the matched and missing skill lists (F1 1.00 and 0.91 against the generator's lists)
- A latency and footprint benchmark of the local adapter against the hosted 20B demo
- A 36-resume hand-authored test set; accuracy falls to 44.4% there, and the README reports it
- A form and script for collecting independent human labels
- Unit tests, a CI test workflow, Dockerfiles and a frozen requirements lock
- Demo hardening: response cache, per-visitor rate limit, daily cap, body-size limit and security headers
- Faster CI: the model evaluation is skipped on docs-only pushes, with dependency and model caches (a code push takes about 2 minutes instead of about 7)
- Community files: contributing guide, security policy, code of conduct, citation file, issue and pull request templates
- A model card in docs/MODEL_CARD.md (intended use, evaluation, limits and regulatory notes)
- A refreshed demo screenshot and GIF, and a social preview image
- A Colab notebook that runs the real adapter

The adapter weights are unchanged from v1.0: everything below is a new or corrected measurement, tool or document around it.

<details>
<summary>All 29 commits since v1.0</summary>

#### Evaluation

- Add a hand-authored test set (36 resumes) and report the adapter's accuracy on it (`0047e8f`)
- Add a human-labelling kit and aggregation for the hand-authored test set (`1e980e7`)
- Compare names with neutral prefixes on near-boundary resumes (`927208a`)
- Measure latency and footprint of the local adapter against the hosted 20B demo (`a318ed8`)
- Measure the matched and missing skill lists on the held-out set (`c00d2cc`)
- Add a name-swap sensitivity test for the adapter (`788e385`)
- Add a larger prompted baseline (gpt-oss-20b on Groq) to the held-out comparison (`078c12f`)

#### Demo

- Response cache, per-visitor rate limit, daily cap, body-size limit and security headers (`c4c019a`)
- Run the actual LoRA adapter in a Hugging Face Space and link it from the README (`ecf897e`)
- Fix the Colab notebook (remove the stale torchao, load the model on old and new transformers) (`85cfc0d`)
- Add a Colab notebook that runs the actual LoRA adapter (`fcc5ac4`)

#### CI

- Run the informational pylint step only on scheduled and manual runs (`8165c37`)
- Smaller evaluation on pushes, pylint on both cores, loud failures on scheduled runs (`fb48e88`)
- Skip the model evaluation on docs-only pushes, cache dependencies and models, add a fast lint workflow (`757d721`)

#### Maintenance

- Unit tests, a CI test workflow, Dockerfiles and a frozen requirements lock (`1877b64`)
- Remove the unused Hugging Face Space files (Gradio Spaces need a PRO subscription) (`c82ca47`)

#### Documentation

- Add a model card (intended use, evaluation, limits and regulatory notes) (`2ffd7f9`)
- Add community health files (contributing, security, code of conduct, citation, issue and PR templates, dependabot) (`fcef423`)
- Clarify that the hosted accuracy figure comes from the baseline prompt, not the demo prompt (`5b2297b`)
- Note that the neutral prefixes contain digits, so the boundary check shows fragility, not that names are harmless (`941ad17`)
- Note that all name-swap verdict flips came from one resume near a verdict boundary (`3d36293`)
- Note that all name-swap verdict flips came from one resume near a verdict boundary (`7e5fff9`)
- Add 95% confidence intervals to the held-out evaluation (`9d0f03f`)
- Add a social preview image (`d7e6d7e`)
- Refresh the demo screenshot and GIF from the current live demo (`67822a4`)
- Note that the matched and missing skill lists are not evaluated and can be wrong (`1d8fef3`)
- Clarify that the accuracy gap measures agreement with the generator's rubric (`59512df`)
- Mark training data as synthetic, fix section references and run instructions (`468fc18`)

#### Other

- Fix demo README, hide notebook from language stats (`56fc4b4`)

</details>

## [1.0] - 2026-10-23

- Initial release: the LoRA adapter, the held-out evaluation against zero-shot baselines, the inference script and the hosted demo.

[Unreleased]: https://github.com/ayush-s-tomar/resume-screener-lora/compare/v1.1...HEAD
[1.1]: https://github.com/ayush-s-tomar/resume-screener-lora/compare/v1.0...v1.1
[1.0]: https://github.com/ayush-s-tomar/resume-screener-lora/releases/tag/v1.0
