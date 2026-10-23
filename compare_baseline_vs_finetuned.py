"""
compare_baseline_vs_finetuned.py
----------------------------------
Runs the eval set through three configurations of the SAME base model:
  1. Zero-shot: base model, prompted to return JSON, no fine-tuning
  2. Zero-shot + rubric: the same prompt, plus the verdict -> ats_score ranges
     measured on the TRAINING set (so the baseline is told the rubric the
     adapter learned from, without any fine-tuning)
  3. Fine-tuned: base model + LoRA adapter trained on our synthetic dataset

Compares each against ground truth on:
  - Valid JSON rate (does it even parse?)
  - Exact verdict match rate (strong/moderate/weak_match correct?)
  - ATS score mean absolute error

Run this AFTER downloading resume-screener-lora-final.zip from Colab and
unzipping it into this project folder.

Usage:
    python compare_baseline_vs_finetuned.py

Env vars:
    EVAL_LIMIT       Optional. If set to a positive integer, only the first N
                     eval examples are used. Useful for CI, where running the
                     full eval set on CPU runners would take too long.
                     Leave unset (or "0") to run the full eval set.
    RUBRIC_BASELINE  Optional. Set to "0" to skip the zero-shot + rubric pass
                     (saves about a third of the runtime). Default: run it.
"""

import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-adapter"
TRAIN_PATH = "./data/train.jsonl"
EVAL_PATH = "./data/eval.jsonl"
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))  # 0 = no limit, use full eval set
RUN_RUBRIC = os.environ.get("RUBRIC_BASELINE", "1") != "0"

ZERO_SHOT_INSTRUCTION = (
    "You are a resume screening assistant. Given a resume, respond with ONLY "
    "a JSON object with these exact keys: role (string), ats_score (0-100 "
    "integer), verdict (one of: strong_match, moderate_match, weak_match), "
    "matched_skills (list of strings), missing_skills (list of strings), "
    "years_experience (integer). Do not include any text outside the JSON."
)


def load_eval_set(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_rubric(train_path):
    """Min/max ats_score per verdict, measured on the training set only."""
    ranges = {}
    for ex in load_eval_set(train_path):
        label = json.loads(ex["completion"])
        score = label["ats_score"]
        lo, hi = ranges.get(label["verdict"], (score, score))
        ranges[label["verdict"]] = (min(lo, score), max(hi, score))
    return dict(sorted(ranges.items(), key=lambda kv: -kv[1][0]))


def rubric_instruction(ranges):
    spec = ", ".join(
        f"{verdict}: ats_score {lo}-{hi}" for verdict, (lo, hi) in ranges.items()
    )
    return (
        ZERO_SHOT_INSTRUCTION
        + " The verdict must agree with ats_score using these ranges: "
        + spec
        + "."
    )


def extract_json(text):
    """Pull the first {...} block out of a model response, tolerating extra text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate(model, tokenizer, messages, max_new_tokens=200):
    # return_dict=True forces a proper BatchEncoding with input_ids + attention_mask,
    # instead of relying on the (version-dependent) default, which can return either
    # a raw tensor or a dict depending on the transformers version installed.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def run_pass(model, tokenizer, instruction, examples, tag):
    preds = []
    for i, ex in enumerate(examples, 1):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": ex["prompt"]},
        ]
        preds.append(generate(model, tokenizer, messages))
        print(f"  {tag} {i}/{len(examples)}", flush=True)
    return preds


def score_predictions(predictions, ground_truths):
    valid_json_count = 0
    verdict_correct = 0
    score_errors = []

    for pred_text, truth in zip(predictions, ground_truths):
        parsed = extract_json(pred_text)
        if parsed is None:
            continue
        valid_json_count += 1

        if parsed.get("verdict") == truth["verdict"]:
            verdict_correct += 1

        if isinstance(parsed.get("ats_score"), (int, float)):
            score_errors.append(abs(parsed["ats_score"] - truth["ats_score"]))

    n = len(ground_truths)
    return {
        "valid_json_rate": round(valid_json_count / n, 3),
        "verdict_accuracy": round(verdict_correct / n, 3),
        "mean_absolute_score_error": round(sum(score_errors) / len(score_errors), 2) if score_errors else None,
    }


def main():
    eval_examples = load_eval_set(EVAL_PATH)

    if EVAL_LIMIT:
        eval_examples = eval_examples[:EVAL_LIMIT]
        print(f"EVAL_LIMIT set: using first {len(eval_examples)} eval examples", flush=True)

    ground_truths = [json.loads(ex["completion"]) for ex in eval_examples]

    print(f"Loaded {len(eval_examples)} eval examples\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )

    # --- Zero-shot baseline ---
    print("Running zero-shot baseline...", flush=True)
    zero_shot_preds = run_pass(base_model, tokenizer, ZERO_SHOT_INSTRUCTION, eval_examples, "zero-shot")
    zero_shot_results = score_predictions(zero_shot_preds, ground_truths)

    # --- Zero-shot + rubric baseline (must run BEFORE the adapter is attached) ---
    rubric_results = None
    rubric_ranges = None
    if RUN_RUBRIC:
        rubric_ranges = build_rubric(TRAIN_PATH)
        print("Running zero-shot + rubric baseline...", flush=True)
        print(f"  rubric measured on the train set: {rubric_ranges}", flush=True)
        rubric_preds = run_pass(
            base_model, tokenizer, rubric_instruction(rubric_ranges), eval_examples, "zero-shot+rubric"
        )
        rubric_results = score_predictions(rubric_preds, ground_truths)

    # --- Fine-tuned model ---
    print("Loading fine-tuned adapter and running eval...", flush=True)
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    finetuned_preds = []
    for i, ex in enumerate(eval_examples, 1):
        messages = [{"role": "user", "content": ex["prompt"]}]
        finetuned_preds.append(generate(finetuned_model, tokenizer, messages))
        print(f"  fine-tuned {i}/{len(eval_examples)}", flush=True)

    finetuned_results = score_predictions(finetuned_preds, ground_truths)

    def cell(results, key):
        return "n/a" if results is None else str(results[key])

    rows = [
        ("Valid JSON rate", "valid_json_rate"),
        ("Verdict accuracy", "verdict_accuracy"),
        ("Mean abs score error", "mean_absolute_score_error"),
    ]
    print("\n" + "=" * 64, flush=True)
    print("RESULTS", flush=True)
    print("=" * 64, flush=True)
    print(f"{'Metric':<28} {'Zero-shot':<12} {'Fine-tuned':<12} {'ZS+rubric':<12}", flush=True)
    print("-" * 64, flush=True)
    for label, key in rows:
        print(
            f"{label:<28} {cell(zero_shot_results, key):<12} "
            f"{cell(finetuned_results, key):<12} {cell(rubric_results, key):<12}",
            flush=True,
        )

    out = {"zero_shot": zero_shot_results, "fine_tuned": finetuned_results}
    if rubric_results is not None:
        out["zero_shot_rubric"] = rubric_results
        out["rubric_ranges"] = {k: list(v) for k, v in rubric_ranges.items()}
    out["n_examples"] = len(eval_examples)

    with open("comparison_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved full results to comparison_results.json", flush=True)


if __name__ == "__main__":
    main()