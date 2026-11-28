# Contributing

Thanks for your interest in resume-screener-lora. It is a small research and portfolio project: a LoRA fine-tuned Qwen2.5-0.5B that returns a structured JSON verdict for a resume and a target role. Issues, questions and focused pull requests are welcome.

## Ground rules

- **No real resumes or personal data** in issues, pull requests or test fixtures. Use fictional resumes only.
- **Keep the numbers honest.** If a change affects a metric, update the matching file in `results/` and the README section that quotes it in the same pull request, and say how you measured it. Negative results are welcome. `tests/test_readme_numbers.py` checks on every push that the numbers in the README and the model card still match `results/`.
- **Keep pull requests small and focused.** One change per pull request is easier to review.

## Development setup

    git clone https://github.com/ayush-s-tomar/resume-screener-lora.git
    cd resume-screener-lora
    python -m venv venv
    venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
    pip install -r requirements.txt
    pip install pytest ruff

The Python version, the extra packages the demo tests need and the exact commands CI runs are in `.github/workflows/tests.yml` and `.github/workflows/lint.yml`. Running the same commands locally before you open a pull request avoids surprises.

## Checks

    python -m pytest
    ruff check --select E9,F821 .

The model evaluation (`python compare_baseline_vs_finetuned.py`) needs the adapter in `resume-screener-lora-adapter/` and takes several minutes on CPU. CI runs a reduced version on pushes (set by `EVAL_LIMIT`) and a larger one on a weekly schedule.

## Where things are

- `inference.py`: score one resume with the local adapter.
- `compare_baseline_vs_finetuned.py`, `eval_groq_baseline.py`, `eval_adapter.py`: the held-out evaluation and its baselines.
- `name_swap_test.py`, `name_swap_boundary.py`, `eval_skill_lists.py`, `benchmark_latency.py`: sensitivity, skill-list and latency checks.
- `render-deploy/`: the hosted demo.
- `labeling/` and `aggregate_human_labels.py`: the form and script for collecting human labels.
- `results/`: the raw numbers behind every table in the README.
- `tests/`: unit tests.

## Ways to help

- **Label resumes.** Open `labeling/label_resumes.html`, label the 36 fictional resumes and send the downloaded file back (open an issue to coordinate). Independent labels are the most useful contribution, because the training labels are synthetic.
- Write new **fictional** resumes in layouts the project has not seen, with a target role and your own judgment of the fit.
- Report bugs, fix documentation, or add tests.

## Responsible use

This project is a research artifact. It must not be used to make real hiring decisions; see the limitations in the README.
