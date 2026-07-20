# Resume Screener — LoRA Fine-Tuned (Qwen2.5-0.5B)

Fine-tuned Qwen2.5-0.5B-Instruct using LoRA to output structured JSON verdicts
for resume screening, instead of relying on prompting alone.

![Resume Screener demo](Resume%20Screener%20Lora.gif)

## Demo

https://github.com/user-attachments/assets/3328d75d-4e17-41aa-95d9-09d8749e6c41

![Resume Screener app screenshot](Resume%20Screener%20Lora%20Demo.png)

## Why

Prompting a base instruct model for resume screening produces unstructured,
inconsistent prose — not something you can pipe into an ATS or scoring
pipeline. This project fine-tunes a small open model so structured JSON
output becomes the model's default behavior, not something you have to
coax out with prompt engineering.

## Approach

- Base model: Qwen2.5-0.5B-Instruct
- Method: LoRA (r=16, alpha=32, dropout=0.05) targeting q/k/v/o projections
- Trainable params: 2,162,688 / 496,195,456 total (0.44%)
- Dataset: 800 train / 96 eval examples, resume text -> structured JSON verdict
- 3 epochs on a free Colab T4 GPU

## Results

| Epoch | Training Loss | Validation Loss |
|-------|--------------|------------------|
| 1     | 0.4078       | 0.2784           |
| 2     | 0.1956       | 0.1883           |
| 3     | 0.1713       | 0.1714           |

Validation loss tracked training loss closely with no divergence -- no overfitting.

## Before vs After

**Prompt:** Screen this resume for a Backend Engineer position and return a
structured verdict. Resume: Bachelor's in CS, 4 years distributed backend
experience, skilled in Python, PostgreSQL, Docker, Kubernetes.

**Base model (no fine-tuning):** unstructured prose with markdown headers
(Summary, Key Skills, Experience) -- not machine-parseable.

**Fine-tuned model:**
```json
{"role": "Backend Engineer", "ats_score": 56, "verdict": "moderate_match", "matched_skills": ["Python", "PostgreSQL", "Docker", "Kubernetes"], "missing_skills": ["Java", "REST APIs", "Redis"], "years_experience": 4}
```

## Repo structure

- `finetune_lora.ipynb` - full training notebook
- `data/` - train/eval JSONL datasets + dataset generator script
- `resume-screener-lora-adapter/` - trained LoRA adapter weights

## Stack

PEFT/LoRA, Hugging Face Transformers, TRL, PyTorch, Colab T4 GPU
